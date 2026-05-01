# GMMXX

`GMMXX` is a PyTorch Gaussian Mixture Model package built with flash-kmeans-style streaming and Triton kernels.

Current scope:

- Spherical, diagonal, tied, and full-covariance GMM
- Batched input support: `(B, N, D)` and `(N, D)`
- Chunked PyTorch EM fallback that avoids materializing the full responsibility tensor during fitting
- Optional `flash-kmeans` initialization when `flash-kmeans` is installed or available in a local `third_party/flash-kmeans` checkout
- Copied-and-modified Triton kernels for spherical, diagonal, tied, and small-D full-covariance EM fitting on CUDA
- Exact fused single-K-tile Triton E/M update for supported spherical, diagonal, and tied shapes
- Optional approximate top-k EM training mode for very large component counts, with a fused Triton spherical path for supported CUDA shapes
- Triton inference for spherical, diagonal, tied score/proba, and supported small-D full-covariance `predict`, `predict_proba`, and `score_samples`

The public estimator follows the familiar scikit-learn shape: `fit`, `predict`, `predict_proba`, `score`, `score_samples`, `aic`, `bic`, `get_params`, and `set_params`. The fastest path is still spherical covariance. Diagonal and tied covariance have streamed and fused Triton E/M paths for supported shapes; full covariance is intentionally conservative because its core cost scales as `N*K*D^2`.

## Target stack

- Python 3.12
- PyTorch 2.11 / CUDA 13.0 wheels
- Windows: `triton-windows` 3.6 (`import triton`)

## Installation

Install from this checkout:

```powershell
python -m pip install -e .
```

Install with benchmark and sklearn helpers:

```powershell
python -m pip install -e ".[benchmark]"
```

Build a wheel and source distribution:

```powershell
python -m pip install -e ".[dev]"
python -m build
```

The wheel contains the canonical `gmmxx` package. `third_party/` is kept as local reference code and is excluded from distributions. If you want `init_params="kmeans"` to use the external initializer after installing from a wheel, install `flash-kmeans` separately:

```powershell
python -m pip install ".[kmeans]"
```

## Quick Start

```python
import torch
from gmmxx import GMMXX

x = torch.randn(8192, 128, device="cuda", dtype=torch.float32)

gmm = GMMXX(
    n_components=64,
    max_iter=50,
    tol=1e-4,
    random_state=0,
    init_params="kmeans",
    covariance_type="spherical",  # "spherical", "diag", "tied", or "full"
)

gmm.fit(x)
labels = gmm.predict(x)
probs = gmm.predict_proba(x[:256])
logp = gmm.score_samples(x[:256])
```

Learned attributes follow sklearn naming where practical:

- `means_`, `weights_`, `covariances_`, `labels_`
- `lower_bound_`, `lower_bound_history_`, `n_iter_`
- runtime diagnostics such as `triton_estep_enabled_`, `triton_fused_update_enabled_`, `triton_approx_topk_enabled_`, and `last_fallback_reason_`

Backward-compatible names are still supported: `d`, `k`, `niter`, and `seed` map to `n_features`, `n_components`, `max_iter`, and `random_state`.

## Notes

- On Windows, the distribution name is `triton-windows`, while the Python import stays `triton`.
- `covariance_type="spherical"` stores variances as `(B, K)` and uses the Triton path where profitable.
- `covariance_type="diag"` stores diagonal variances as `(B, K, D)` and uses a Triton streamed E/M path for large CUDA shapes up to `D <= 64`, `K <= 512`.
- `covariance_type="tied"` stores one shared covariance matrix as `(B, D, D)` and uses projected or native fused Triton E/M paths for large CUDA shapes up to `D <= 64`, `K <= 512`.
- `covariance_type="full"` stores per-component covariance matrices as `(B, K, D, D)`. Full fit/update uses Triton only for profitable `D <= 8` shapes; full inference supports `D <= 16`.
- `use_triton=True` is the single runtime switch. CUDA runs validated Triton paths where profitable; unsupported shapes and runtime compile/cache failures use PyTorch/cuBLAS automatically.
- `fit()` avoids materializing full `(B, N, K)` responsibilities on all covariance types. Prediction/probability helpers use validated Triton paths for spherical, diagonal, tied score/proba, and supported small-D full covariance. Tied labels intentionally remain on the exact PyTorch path because projected logits can differ on near-tie assignments.
- Fused E/M update is exact, not approximate. It is enabled internally when `K` fits one Triton K tile: currently `D <= 32, K <= 128` or `D <= 64, K <= 64`.
- `approx_top_k=N` is an explicit approximation for training only. Each E-step keeps the top `N` component logits per sample, normalizes responsibilities over that subset, and updates full sufficient statistics. `None` keeps exact EM, and values `>= K` are treated as exact. Spherical CUDA fits use a fused Triton top-k update for supported shapes; other shapes fall back to the PyTorch approximate path.
- `matmul_precision="high"` or `"medium"` forwards to `torch.set_float32_matmul_precision(...)` before GMMXX operations. This is opt-in because it can change floating-point results slightly.
- `compute_labels_on_fit=False` skips the final label assignment inside `fit()`. Use `fit_predict()` or `predict()` when labels are needed.
- `init_params="kmeans"` uses greedy k-means++ seeding for moderate component counts before running the local `flash-kmeans` initializer. This improves clustering quality without adding another public init mode.
- If Triton is installed but cannot compile or use its cache at runtime on Windows, the code falls back to the PyTorch path instead of failing the whole operation. The most recent recorded fallback is exposed as `GMMXX.last_fallback_reason_`.
- The implementation targets all sklearn-style covariance types. Full covariance is intentionally conservative because each component needs matrix quadratic forms and second-moment reductions.

## Benchmarking

Install the benchmark extras for the standard CPU baseline:

```powershell
python -m pip install -e ".[benchmark]"
```

Run a synthetic spherical benchmark:

```powershell
python benchmarks\benchmark_gmm.py --dataset blobs --n-samples 65536 --n-features 128 --n-components 64 --device cuda --baselines flash-auto flash-torch sklearn-spherical
```

Run the top-k approximate EM path for large component counts:

```powershell
python benchmarks\benchmark_gmm.py --dataset blobs --n-samples 65536 --n-features 32 --n-components 512 --device cuda --max-iter 2 --init-params random --skip-fit-labels --approx-top-k 16 --baselines flash-auto flash-torch
```

Run a diagonal covariance benchmark:

```powershell
python benchmarks\benchmark_gmm.py --dataset anisotropic-blobs --n-samples 131072 --n-features 32 --n-components 64 --device cuda --baselines flash-diag flash-diag-torch sklearn-diag
```

Run all covariance baselines on a small full-covariance-friendly shape:

```powershell
python benchmarks\benchmark_gmm.py --dataset anisotropic-blobs --n-samples 131072 --n-features 8 --n-components 32 --device cuda --baselines flash-diag flash-diag-torch flash-tied flash-tied-torch flash-full flash-full-torch sklearn-diag sklearn-tied sklearn-full
```

Validate numerical equivalence before timing:

```powershell
python benchmarks\validate_equivalence.py --device cuda
```

Run the low-level Triton module benchmark used for kernel tuning:

```powershell
python .autotune\bench_triton_modules.py --profile standard --repeats 7
```

Run the size-coverage sweep against the PyTorch path:

```powershell
python benchmarks\validate_size_sweep.py --device cuda --profile standard
```

Validate clustering quality on labeled synthetic datasets:

```powershell
python benchmarks\validate_quality.py --device cuda
```

Benchmark accuracy and likelihood quality without failing on low-ARI cases:

```powershell
python benchmarks\benchmark_accuracy.py --device cuda --dataset anisotropic-blobs --n-samples 32768 --n-features 16 --n-components 16
```

Add `--include-sklearn` for a CPU sklearn quality baseline, and add
`--fail-on-low-ari` when you want the accuracy benchmark to behave like a
regression gate.

For larger supported shapes and fallback boundaries:

```powershell
python benchmarks\validate_size_sweep.py --device cuda --profile large --warmup-runs 1
```

For a custom grid:

```powershell
python benchmarks\validate_size_sweep.py --device cuda --cartesian --n-values 256,4096,65536 --d-values 1,32,64,128,129,256 --k-values 1,16,64,256,2048,2049
```

Recommended baselines:

- `flash-auto`: this package with the default auto fit policy.
- `flash-torch`: this package with `use_triton=False`; this isolates the value of the Triton path.
- `flash-diag`: this package with `covariance_type="diag"`.
- `flash-diag-torch`: this package with `covariance_type="diag"` and `use_triton=False`.
- `flash-tied`: this package with `covariance_type="tied"`.
- `flash-tied-torch`: this package with `covariance_type="tied"` and `use_triton=False`.
- `flash-full`: this package with `covariance_type="full"`.
- `flash-full-torch`: this package with `covariance_type="full"` and `use_triton=False`.
- `sklearn-spherical`: `sklearn.mixture.GaussianMixture(covariance_type="spherical")`; this is the standard CPU correctness and quality baseline.
- `sklearn-diag`: standard CPU baseline for diagonal covariance.
- `sklearn-tied`: standard CPU baseline for tied covariance.
- `sklearn-full`: standard CPU baseline for full covariance.
- `torchgmm-spherical`: optional PyTorch Lightning GPU baseline from `torchgmm`; install separately with `python -m pip install torchgmm`.

More implementation references are collected in `docs/high_performance_gmm_references.md`.

Recent local CUDA benchmark notes on RTX 4090, Python 3.12, `torch 2.11.0+cu130`, `triton-windows 3.6.0.post26`. Timings use warm caches and at least one warmup run, so first-time Triton compilation is excluded.

- `N=524288, D=32, K=64, 2 iters`: `flash-auto 0.0192s`, `flash-torch 0.0571s`.
- `N=524288, D=128, K=64, 2 iters`: `flash-auto 0.0448s`, `flash-torch 0.0596s`.
- `N=1048576, D=32, K=64, 2 iters`: `flash-auto 0.0302s`, `flash-torch 0.1006s`.
- `N=1048576, D=128, K=64, 2 iters`: sequential isolated run with two warmups: `flash-auto 0.0750s`, `flash-torch 0.1257s`.
- `N=1048576, D=128, K=128, 2 iters`: sequential isolated run with two warmups: `flash-auto 0.0840s`, `flash-torch 0.1437s`.
- `N=2097152, D=32, K=64, 2 iters`: `flash-auto 0.0786s`, `flash-torch 0.2306s`.
- `N=2097152, D=128, K=64, 2 iters`: `flash-auto 0.2011s`, `flash-torch 0.2500s`.
- Diagonal Triton fit benchmark on anisotropic blobs: `N=131072, D=32, K=64, 3 iters`: `flash-diag 0.0073s`, `flash-diag-torch 0.0097s`.
- Tied Triton fit benchmark on anisotropic blobs: `N=131072, D=32, K=64, 3 iters`: `flash-tied 0.0107s`, `flash-tied-torch 0.0122s`.
- Full small-D Triton fit benchmark on blobs: `N=131072, D=8, K=32, 3 iters`: `flash-full 0.0089s`, `flash-full-torch 0.0132s`.
- Larger diagonal Triton fit benchmark on anisotropic blobs: `N=1048576, D=32, K=64, 3 iters`: `flash-diag 0.0284s`, `flash-diag-torch 0.0419s`.
- Larger tied Triton fit benchmark on anisotropic blobs: `N=1048576, D=32, K=64, 3 iters`: `flash-tied 0.0289s`, `flash-tied-torch 0.0331s`.
- Larger full small-D Triton fit benchmark on blobs: `N=1048576, D=8, K=32, 3 iters`: `flash-full 0.0330s`, `flash-full-torch 0.0689s`.
- Diagonal covariance sanity benchmark on anisotropic blobs: `N=32768, D=32, K=16, 5 iters`: `flash-auto 0.0105s`, `flash-diag 0.0208s`, `sklearn-spherical 0.1328s`, `sklearn-diag 0.1310s`.
- All-covariance sanity benchmark on anisotropic blobs: `N=32768, D=8, K=8, 5 iters`: `flash-diag 0.0178s`, `flash-tied 0.0209s`, `flash-full 0.0196s`, `sklearn-diag 0.0567s`, `sklearn-tied 0.1132s`, `sklearn-full 0.1418s`. This benchmark is not a quality-equivalence run because the libraries may initialize differently.
- Larger all-covariance sanity benchmark on anisotropic blobs: `N=131072, D=8, K=8, 5 iters`: `flash-diag 0.0223s`, `flash-tied 0.0292s`, `flash-full 0.0261s`, `sklearn-diag 0.2796s`, `sklearn-tied 0.3837s`, `sklearn-full 0.5519s`.
- Larger-K covariance sanity benchmark on anisotropic blobs: `N=65536, D=16, K=64, 3 iters`: `flash-diag 0.0100s`, `flash-tied 0.0135s`, `flash-full 0.0221s`.
- External random-init comparison with `--batch-size 8192` for TorchGMM: `N=131072, D=128, K=64, 3 iters`: `flash-auto 0.0134s`, `torchgmm-spherical 0.1443s`, `sklearn-spherical 1.5874s`. This is a speed sanity check, not a strict quality-equivalence run, because each library initializes and parameterizes training differently.
- Fused E/M update sanity benchmark on RTX 4090, random data, `N=131072`, `2 iters`, labels skipped: spherical `D=32,K=64` `2.29x` over `flash-torch`; spherical `D=32,K=128` `3.44x`; diagonal `D=16,K=128` `2.71x`; tied `D=32,K=64` `3.44x`; full `D=8,K=32` streamed Triton `3.53x`.
- Top-k approximate EM sanity benchmark on RTX 4090, blobs, `N=32768, D=32, K=512, 2 iters`, labels skipped, one warmup: `flash-auto approx_top_k=16` `0.0059s`, `flash-torch approx_top_k=16` `0.0226s` (`3.83x`). A random-data isolated timing measured `approx-triton` at about `19.8x` over the PyTorch approximate path and `21.8x` over exact PyTorch EM for the same shape. This is an approximate objective; compare quality for your dataset before using it as a replacement for exact EM.

Datasets to use:

- `blobs`: synthetic isotropic Gaussian blobs from `sklearn.datasets.make_blobs`; best first speed and correctness benchmark for spherical GMM.
- `anisotropic-blobs`: transformed Gaussian blobs; useful for showing where spherical GMM quality is expected to lose to diagonal/full covariance methods.
- `iris`, `wine`, `digits`: small standard scikit-learn datasets for sanity checks and regression tests.
- MNIST via OpenML is a useful larger public dataset, but it is not included in the local benchmark script yet because it requires network download and caching.

## Acknowledgements

`GMMXX` is inspired by [`flash-kmeans`](https://github.com/svg-project/flash-kmeans), especially its IO-aware batched clustering design and Triton kernel structure. If this project is useful in your work, please also cite the Flash-KMeans paper:

```bibtex
@article{yang2026flash,
  title={Flash-KMeans: Fast and Memory-Efficient Exact K-Means},
  author={Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Li, Muyang and Fan, Xiaoze and Zhang, Jintao and Cai, Han and Lin, Yujun and Li, Xiuyu and Keutzer, Kurt and others},
  journal={arXiv preprint arXiv:2603.09229},
  year={2026}
}
```
