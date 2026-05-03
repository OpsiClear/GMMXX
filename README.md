# GMMXX

`GMMXX` is a PyTorch Gaussian Mixture Model package with flash-kmeans-style streaming, chunked EM updates, and Triton kernels for CUDA acceleration.

It exposes a scikit-learn-like estimator API while keeping GPU memory bounded by avoiding a materialized `(B, N, K)` responsibility tensor during fitting.

## Highlights

- Supports `covariance_type="spherical"`, `"diag"`, `"tied"`, and `"full"`.
- Accepts both `(N, D)` and batched `(B, N, D)` inputs.
- Provides `fit`, `predict`, `predict_proba`, `score`, `score_samples`, `aic`, `bic`, `get_params`, and `set_params`.
- Uses validated Triton paths on CUDA where profitable, with automatic PyTorch/cuBLAS fallback.
- Includes exact fused single-K-tile E/M updates for supported spherical, diagonal, and tied shapes.
- Includes optional approximate top-k EM training for very large component counts.
- Can use `flash-kmeans` initialization when installed or available under `third_party/flash-kmeans`.

The fastest path is spherical covariance. Diagonal and tied covariance have streamed and fused Triton E/M paths for supported shapes. Full covariance is intentionally conservative because its core update cost scales as `N*K*D^2`.

## Requirements

| Platform | Python | PyTorch | Triton package |
| --- | --- | --- | --- |
| Linux CUDA | `>=3.12` | `>=2.11` | `triton>=3.6,<3.7` |
| Windows CUDA | `>=3.12` | `>=2.11` | `triton-windows>=3.6,<3.7` |

The Python import is `triton` on both Linux and Windows. For CUDA 13.0, install the matching PyTorch wheel before installing `GMMXX`.

## Installation

Install from this checkout:

```powershell
python -m pip install -e .
```

Install benchmark and sklearn helpers:

```powershell
python -m pip install -e ".[benchmark]"
```

Install optional external baselines:

```powershell
python -m pip install -e ".[benchmark-gpu]"
```

Install optional external `flash-kmeans` initialization support:

```powershell
python -m pip install ".[kmeans]"
```

Build a wheel:

```powershell
python -m pip install -e ".[dev]"
python -m pip wheel . --no-deps -w dist
```

The wheel contains the canonical `gmmxx` package. `third_party/` is reference code and is excluded from distributions.

### CUDA backend (experimental)

`GMMXX` is migrating to a hand-written CUDA backend. **Spherical covariance is feature-complete on CUDA** (Plans 2–5). For `D ≤ 64, K ≤ 128` shapes the fused single-tile E/M kernel runs in one CTA pass per BLOCK_N rows — logits + softmax + per-cluster sufficient-statistic accumulation in registers/SMEM, four kernel launches reduced to one per EM iteration. For wider shapes (up to `D ≤ 128, K ≤ 2048`), the unfused pipeline (sorted-run atomic-coalesced M-step + sm_80 `mma.sync` E-step for fp16/bf16 + safe SIMT for fp32) is used. `predict()`, `predict_proba()`, `score_samples()`, `score()` all dispatch to CUDA when `backend="cuda"` and the shape is in the support window. Perf-gated to within 10% of Triton on all supported shapes. Diagonal is now on CUDA (Plan 6) — the safe-path E-step + per-token M-step + finalize support D ≤ 64, K ≤ 512 for fp32/fp16/bf16. Tied is now on CUDA (Plan 7) — uses projected coordinates `y = L⁻¹ x` to reduce the tied logit to a Euclidean distance, reusing the spherical CUDA kernels with `var=1`, then finalizes via host Cholesky factorization (`D ≤ 64, K ≤ 512`). Full is still on Triton/PyTorch — coming in Plan 8. See `docs/superpowers/specs/2026-05-02-gmmxx-cuda-backend-design.md` for the overall design and `docs/superpowers/plans/` for per-PR plans.

The CUDA path is selected automatically on hosts with a working build:

| Backend | Selected when |
| --- | --- |
| `cuda` | `gmmxx._C` is built AND compute capability ≥ 8.0 AND shape is supported |
| `triton` | CUDA path unsupported; Triton is installed; shape is in the Triton policy |
| `torch` | All else (always works as a fallback) |

Build prerequisites:

- CUDA Toolkit ≥ 12.8 (required for sm_100/sm_120 — older toolkits work but Blackwell archs are skipped automatically).
- C++17 compiler (MSVC 2019 16.5+ on Windows; gcc/clang on Linux).
- `nanobind>=2.0` (installed automatically via build deps).

```powershell
# Standard install (builds CUDA extension at install time):
uv pip install -e .

# Single-arch dev build (much faster):
$env:TORCH_CUDA_ARCH_LIST = "8.9"   # PowerShell — replace with your local arch
uv pip install -e .

# Skip the CUDA build entirely (Triton-only / CPU-only install):
$env:GMMXX_SKIP_CUDA = "1"
uv pip install -e ".[triton]"
```

Backend selection:

```python
from gmmxx import GMMXX

# Auto: pick CUDA when supported, else Triton, else PyTorch.
gmm = GMMXX(n_components=64, backend="auto")

# Pin to a specific backend:
gmm = GMMXX(n_components=64, backend="triton")

# Or via env var (kwarg wins when explicit):
import os
os.environ["GMMXX_BACKEND"] = "torch"
gmm = GMMXX(n_components=64)  # uses torch
```

After a `fit()`, inspect what actually ran:

```python
gmm.fit(x)
print(gmm.last_backend_used_)        # "cuda" / "triton" / "torch"
print(gmm.last_fallback_reason_)     # diagnostic string if a fallback fired
print(gmm.fit_info_["backend_breakdown"])  # mixed runs: {"cuda": 18, "triton": 2}
```

Deprecation note: `use_triton=True/False` constructor kwarg still works but emits a `DeprecationWarning`. Switch to `backend=`. The mapping is `use_triton=True → backend="auto"`; `use_triton=False → backend="auto"` with Triton filtered from the dispatch chain (so you still get CUDA when available — historically `use_triton=False` meant "no Triton JIT", not "no GPU"). Removed in v2.0.

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
- `triton_estep_enabled_`, `triton_fused_update_enabled_`, `triton_approx_topk_enabled_`, `last_fallback_reason_`

Backward-compatible constructor names are still supported: `d`, `k`, `niter`, and `seed` map to `n_features`, `n_components`, `max_iter`, and `random_state`.

## Execution Model

Use `use_triton=True` as the single runtime switch. Unsupported shapes, compile failures, cache issues, and non-profitable cases automatically use the PyTorch/cuBLAS path. The most recent fallback reason is available as `GMMXX.last_fallback_reason_`.

`fit()` avoids materializing full `(B, N, K)` responsibilities for all covariance types. Prediction helpers use Triton for supported spherical, diagonal, tied score/proba, and small-D full covariance inference. Tied labels intentionally remain on the exact PyTorch path because projected logits can differ on near-tie assignments.

### Covariance Coverage

| Covariance | Parameter shape | CUDA/Triton coverage |
| --- | --- | --- |
| `spherical` | `(B, K)` | Fastest path. Exact fused E/M supports up to `D <= 64, K <= 128`; broader streamed/inference paths are used where profitable. |
| `diag` | `(B, K, D)` | Streamed E/M supports large CUDA shapes up to `D <= 64, K <= 512`; exact fused high-D tile remains conservative at `D <= 64, K <= 64`. |
| `tied` | `(B, D, D)` | Uses projected or native fused Triton E/M up to `D <= 64, K <= 512`; exact fused tile supports up to `D <= 64, K <= 128`. |
| `full` | `(B, K, D, D)` | Fit/update uses Triton only for profitable `D <= 8` shapes; inference supports `D <= 16`. |

### Useful Options

| Option | Purpose |
| --- | --- |
| `init_params="kmeans"` | Uses greedy k-means++ seeding for moderate component counts before the local or installed `flash-kmeans` initializer. |
| `approx_top_k=N` | Approximate training mode. Each E-step keeps the top `N` component logits per sample and normalizes over that subset. `None` keeps exact EM; values `>= K` are treated as exact. |
| `compute_labels_on_fit=False` | Skips final label assignment during `fit()`. Use `fit_predict()` or `predict()` when labels are needed. |
| `matmul_precision="high"` or `"medium"` | Forwards to `torch.set_float32_matmul_precision(...)`; opt-in because it can slightly change floating-point results. |

Approximate top-k EM is training-only and should be quality-checked on your dataset before replacing exact EM.

## Validation

Run the unit tests:

```powershell
python -m pytest tests -q
```

Validate numerical equivalence against internal PyTorch paths and sklearn references:

```powershell
python benchmarks\validate_equivalence.py --device cuda
```

Run the standard size sweep:

```powershell
python benchmarks\validate_size_sweep.py --device cuda --profile standard
```

Run larger supported shapes and fallback boundaries:

```powershell
python benchmarks\validate_size_sweep.py --device cuda --profile large --warmup-runs 1
```

Validate clustering quality on labeled synthetic datasets:

```powershell
python benchmarks\validate_quality.py --device cuda
```

Benchmark likelihood and clustering quality:

```powershell
python benchmarks\benchmark_accuracy.py --device cuda --dataset anisotropic-blobs --n-samples 32768 --n-features 16 --n-components 16
```

Add `--include-sklearn` for a CPU sklearn quality baseline. Add `--fail-on-low-ari` when the accuracy benchmark should behave like a regression gate.

## Benchmarking

Install benchmark extras first:

```powershell
python -m pip install -e ".[benchmark]"
```

Common benchmark commands:

```powershell
# Spherical
python benchmarks\benchmark_gmm.py --dataset blobs --n-samples 65536 --n-features 128 --n-components 64 --device cuda --baselines flash-auto flash-torch sklearn-spherical

# Diagonal
python benchmarks\benchmark_gmm.py --dataset anisotropic-blobs --n-samples 131072 --n-features 32 --n-components 64 --device cuda --baselines flash-diag flash-diag-torch sklearn-diag

# Full-covariance-friendly shape
python benchmarks\benchmark_gmm.py --dataset anisotropic-blobs --n-samples 131072 --n-features 8 --n-components 32 --device cuda --baselines flash-diag flash-diag-torch flash-tied flash-tied-torch flash-full flash-full-torch sklearn-diag sklearn-tied sklearn-full

# Approximate top-k EM
python benchmarks\benchmark_gmm.py --dataset blobs --n-samples 65536 --n-features 32 --n-components 512 --device cuda --max-iter 2 --init-params random --skip-fit-labels --approx-top-k 16 --baselines flash-auto flash-torch
```

Low-level Triton module benchmark:

```powershell
python .autotune\bench_triton_modules.py --profile standard --repeats 7
```

Custom size grid:

```powershell
python benchmarks\validate_size_sweep.py --device cuda --cartesian --n-values 256,4096,65536 --d-values 1,32,64,128,129,256 --k-values 1,16,64,256,2048,2049
```

### Baselines

| Baseline | Meaning |
| --- | --- |
| `flash-auto` | GMMXX with the default auto CUDA policy. |
| `flash-torch` | GMMXX with `use_triton=False`; isolates Triton speedup. |
| `flash-diag`, `flash-tied`, `flash-full` | GMMXX with the corresponding covariance type. |
| `flash-diag-torch`, `flash-tied-torch`, `flash-full-torch` | Same covariance type with `use_triton=False`. |
| `sklearn-spherical`, `sklearn-diag`, `sklearn-tied`, `sklearn-full` | CPU sklearn correctness and quality baselines. |
| `torchgmm-spherical` | Optional PyTorch Lightning GPU baseline from `torchgmm`. |
| `tgmm-spherical`, `tgmm-diag`, `tgmm-tied`, `tgmm-full` | Optional PyTorch EM baselines from `tgmm`. |

Install external GPU baselines separately if needed:

```powershell
python -m pip install torchgmm tgmm
```

More implementation references are collected in [docs/high_performance_gmm_references.md](docs/high_performance_gmm_references.md).

### Local RTX 4090 Notes

Recent local CUDA notes were measured on RTX 4090, Python 3.12, `torch 2.11.0+cu130`, and `triton-windows 3.6.0.post26`. Timings use warm caches and exclude first-time Triton compilation.

| Shape / benchmark | Result |
| --- | --- |
| `N=524288, D=32, K=64, 2 iters` | `flash-auto 0.0192s`, `flash-torch 0.0571s` |
| `N=1048576, D=32, K=64, 2 iters` | `flash-auto 0.0302s`, `flash-torch 0.1006s` |
| `N=1048576, D=128, K=64, 2 iters` | `flash-auto 0.0750s`, `flash-torch 0.1257s` |
| `N=2097152, D=32, K=64, 2 iters` | `flash-auto 0.0786s`, `flash-torch 0.2306s` |
| `N=2097152, D=128, K=64, 2 iters` | `flash-auto 0.2011s`, `flash-torch 0.2500s` |
| Diagonal, `N=1048576, D=32, K=64, 3 iters` | `flash-diag 0.0284s`, `flash-diag-torch 0.0419s` |
| Tied, `N=1048576, D=32, K=64, 3 iters` | `flash-tied 0.0289s`, `flash-tied-torch 0.0331s` |
| Full, `N=1048576, D=8, K=32, 3 iters` | `flash-full 0.0330s`, `flash-full-torch 0.0689s` |
| External TorchGMM, `N=131072, D=128, K=64, 3 iters` | `flash-auto 0.0134s`, `torchgmm-spherical 0.1443s`, `sklearn-spherical 1.5874s` |
| Approx top-k, `N=32768, D=32, K=512, top_k=16, 2 iters` | `flash-auto 0.0059s`, `flash-torch 0.0226s` |

Fused E/M update speedups over `flash-torch`, random data, `N=131072`, `2 iters`, labels skipped:

| Mode | Shape | Speedup |
| --- | --- | --- |
| Spherical | `D=32, K=64` | `2.29x` |
| Spherical | `D=32, K=128` | `3.44x` |
| Spherical | `D=64, K=128` | `2.11x` |
| Diagonal | `D=16, K=128` | `2.71x` |
| Tied | `D=32, K=64` | `3.44x` |
| Tied | `D=64, K=128` | `1.09x` |
| Full streamed Triton | `D=8, K=32` | `3.53x` |

These are speed sanity checks, not strict quality-equivalence runs, because external libraries may initialize and parameterize training differently.

## Datasets

- `blobs`: isotropic Gaussian blobs from `sklearn.datasets.make_blobs`; best first speed and correctness benchmark for spherical GMM.
- `anisotropic-blobs`: transformed Gaussian blobs; useful for testing diagonal, tied, and full covariance behavior.
- `iris`, `wine`, `digits`: small standard sklearn datasets for sanity checks and regression tests.
- MNIST via OpenML is useful as a larger public dataset, but it is not included in the local benchmark script because it requires network download and caching.

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
