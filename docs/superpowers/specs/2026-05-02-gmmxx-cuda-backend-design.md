# GMMXX CUDA Backend — Design Spec

**Status:** Approved (2026-05-02). Brainstorming-derived design for a hand-written CUDA backend that ships alongside the existing Triton path inside the `gmmxx` package.

**Branch:** `GMMXX-cuda`. **Reference template:** `~/Projects/flash-kmeans-cuda` (sibling of `~/Projects/flash-kmeans`).

---

## 1. Goals & non-goals

### Goals

1. Add a hand-written CUDA backend to GMMXX that:
   - Preserves the public Python API exactly (`GMMXX` class, functional `batch_gmm_*`, all attributes).
   - Is selected automatically on supported hardware (`backend="auto"` default).
   - Beats the existing Triton path on every supported shape on sm_80+ GPUs.
   - Co-exists with Triton; users can pin to either backend or fall through to PyTorch.
2. Match the architectural pattern of `flash-kmeans-cuda` (sibling of `flash-kmeans`) so future maintainers can map between the two repos.
3. Cover all four covariance types — `spherical`, `diag`, `tied`, `full` — in Phase 1, matching current Triton coverage.

### Non-goals

- No prebuilt PyPI wheels in Phase 1 (source-build only). Wheel CI deferred to v1.0.
- No new public Python user-facing API beyond a small `gmmxx.cuda_ops` functional layer for power users.
- No optimized path for sm_70 / sm_75 (Volta / Turing). They lack `mma.sync.m16n8k16.f32.f16.f16.f32`. The safe kernel still compiles and runs there, but `_dispatch.resolve_backend` routes pre-sm_80 hardware to Triton/PyTorch unless the user explicitly requests `backend="cuda"`.
- No multi-GPU or large-N CPU-streaming changes. `large_n.py` continues to call into the chosen backend per chunk.
- No fp8 (E4M3/E5M2) support. Deferred to Phase 2 if user demand appears.

---

## 2. Architecture

```
                                ┌────────────────────────────────────────┐
                                │  gmmxx.GMMXX (interface.py — public)   │
                                │  fit / predict / score / score_samples │
                                │  predict_proba / aic / bic / params    │
                                └──────────────────┬─────────────────────┘
                                                   │  selects backend per call
                                                   ▼
                                ┌────────────────────────────────────────┐
                                │  gmmxx._dispatch (new)                 │
                                │  resolve_backend(req, cov, shape, dt)  │
                                │  shape gates  +  fallback chain        │
                                └─┬───────────────────┬──────────────────┘
                       ┌──────────┘                   │                    └────────────────┐
                       ▼                              ▼                                     ▼
              ┌─────────────────┐          ┌────────────────────┐                   ┌─────────────────┐
              │ gmmxx._cuda     │          │ gmmxx.*_triton     │                   │ gmmxx.torch_    │
              │ (Python wrap)   │          │ (existing JIT)     │                   │ fallback        │
              └────────┬────────┘          └────────────────────┘                   └─────────────────┘
                       │ nanobind
                       ▼
              ┌─────────────────────────────────────────────────────────┐
              │ gmmxx._C  —  compiled extension (csrc/)                 │
              │   bindings.cpp  →  NB_MODULE                            │
              │   estep/{spherical,diag,tied,full}_{safe,sm80}.cu       │
              │   mstep/{blocked_<cov>,finalize}.cu                     │
              │   fused/{fused_spherical,fused_diag,fused_tied}.cu      │
              │   approx/approx_topk_spherical.cu                       │
              │   common/{arch.cuh, ptx.cuh, reduce.cuh, nb_torch.h}    │
              └─────────────────────────────────────────────────────────┘
```

The `GMMXX` orchestrator does not change. All backend selection is encapsulated in the new `_dispatch` module. Each compute backend (CUDA, Triton, torch) is a peer; none knows about the others.

---

## 3. Public API

### Preserved (no changes)

- `GMMXX(...)` constructor — all existing kwargs.
- `.fit(x)`, `.train(x)`, `.predict(x)`, `.predict_proba(x)`, `.score(x)`, `.score_samples(x)`, `.aic(x)`, `.bic(x)`, `.fit_predict(x)`.
- `.get_params(deep=True)`, `.set_params(**kw)`.
- Attributes: `means_`, `weights_`, `covariances_`, `labels_`, `lower_bound_`, `lower_bound_history_`, `n_iter_`, `triton_estep_enabled_`, `triton_fused_update_enabled_`, `triton_approx_topk_enabled_`, `last_fallback_reason_`.
- Functional `batch_gmm_Spherical/Diagonal/Tied/Full(_torch_native)` and large-N CPU helpers in `gmmxx.large_n`.

### Additions

| Surface | Type | Purpose |
| --- | --- | --- |
| `backend: str = "auto"` constructor kwarg | `"auto" \| "cuda" \| "triton" \| "torch"` | Backend selection. |
| `GMMXX_BACKEND` env var | string, same domain | Global override; lower precedence than explicit kwarg. |
| `last_backend_used_` | str | Mirrors `last_fallback_reason_`; populated after each `fit()`/`predict()`. |
| `cuda_estep_enabled_` | bool | Was the CUDA E-step actually used. |
| `cuda_fused_update_enabled_` | bool | Was the CUDA fused single-tile path used. |
| `cuda_approx_topk_enabled_` | bool | Was the CUDA approx-topK path used. |
| `gmmxx.cuda_ops` module | functional | Low-level access to compiled kernels for power users. |

### Deprecated (with shims)

`use_triton: bool` constructor kwarg remains accepted:
- `use_triton=True` → equivalent to `backend="auto"`.
- `use_triton=False` → equivalent to `backend="torch"`.
- Emits `DeprecationWarning` recommending `backend=...`.
- Removed in v2.0.

### `gmmxx.cuda_ops` functional surface

Mirrors `flash_kmeans_cuda.ops` in shape and naming:

```python
spherical_assign(x, means, var, log_w, out=None) -> Tensor[int32]
spherical_logsumexp(x, means, var, log_w, out=None) -> Tensor[float32]
spherical_resp(x, means, var, log_w, log_norm, out=None) -> Tensor

diag_assign(x, prec, weighted_means, logdet, log_w, out=None) -> Tensor[int32]
diag_logsumexp(...)
diag_resp(...)

tied_assign(x, l_inv, weighted_means_proj, logdet, log_w, out=None) -> Tensor[int32]
tied_logsumexp(...)
tied_resp(...)

full_assign(x, prec, weighted_means, logdet, log_w, out=None) -> Tensor[int32]   # D ≤ 16 only
full_logsumexp(...)
full_resp(...)

blocked_update_<cov>(x, sorted_idx, sorted_ids, sums_out, sumsq_out, counts_out)
fused_<cov>(x, params..., partial_outs)
approx_topk_spherical(x, means, var, log_w, top_k, partial_outs)
centroid_finalize(sums, sumsq, counts, old_means, reg_covar, out=None) -> (means, vars)
```

All ops require contiguous CUDA tensors; type/shape/device validated in C++ via `TORCH_CHECK`.

---

## 4. CUDA kernel inventory (Phase 1)

| Covariance | E-step kernels | M-step (blocked) | Fused single-tile E/M | Approx top-K |
| --- | --- | --- | --- | --- |
| spherical | safe + sm80 mma | sorted-run + blocked | yes (D ≤ 64, K ≤ 128) | yes (training only) |
| diag | safe + sm80 mma | sorted-run + blocked | yes (D ≤ 64, K ≤ 64) | — |
| tied | safe + sm80 (projected) | sorted-run + blocked | yes (D ≤ 64, K ≤ 128) | — |
| full | safe only (D ≤ 16) | safe blocked (D ≤ 16) | — | — |

E-step exposes three kernels each (assign, logsumexp, resp). Total device kernels: ≈ 24 + finalize + utilities. All host launchers exposed through `bindings.cpp`.

Shape gates (compile-time + runtime) are documented in `gmmxx/_runtime.py` (extended) and mirrored in `_dispatch.py` so the orchestrator can decide before crossing the FFI boundary.

---

## 5. Optimization techniques

Applied per kernel where profitable. None blanket-applied.

- **mma.sync `m16n8k16.row.col.f32.f16.f16.f32`** for assign / logsumexp / blocked-update on sm_80+ when input dtype is fp16 or bf16. Tile defaults: `BLOCK_N=128`, `BLOCK_K=64`, `BLOCK_D=16`. Per-arch overrides in `arch.cuh`.
- **`cp.async.cg` double-buffered SMEM** for centroid tiles `(BLOCK_K × D)`. SMEM padded by 8 elements per row to eliminate 32-way bank conflicts on D = power-of-2.
- **Register-tiled fused min-over-K accumulator** in assign kernels: cross-product never materialized to SMEM. Per-thread `Best{dist,idx}` with first-occurrence tie-breaking.
- **Sorted-run atomic coalescing for M-step**: caller pre-sorts `cluster_ids` (Python side, `torch.argsort`); kernel walks runs and emits one `atomicAdd` per (run, feature) tuple. Estimated ~256× atomic-issue reduction vs. naive per-token approach.
- **Persistent kernels for E-step** when `N >= 256k`: one CTA per SM, work-stealing on a shared counter. Eliminates relaunch overhead across EM iterations.
- **Multi-stream E/M overlap**: M-step partials emitted on a side stream; final reduce + finalize sync. Opt-in via `GMMXX_OVERLAP_STREAMS=1` env var; default off in Phase 1 to keep determinism.
- **Per-arch occupancy tuning**: `__launch_bounds__(threads, ctas_per_sm)` set per kernel. sm_90+ uses larger SMEM and `BLOCK_N=256` variants where beneficial. Switched in `arch.cuh`.
- **Stable logsumexp**: per-row max via warp shuffle, subtract-max-then-exp-then-sum, all in fp32 regardless of input dtype.
- **Packed half2 / bfloat162 math** in fused single-tile path.
- **Vectorized loads**: `float4` / `__half2` / `__nv_bfloat162` when D-strides align to 8 bytes.
- **Empty-cluster handling**: `centroid_finalize` keeps the previous mean/variance when `count == 0`, then adds `reg_covar` to variance (matches `torch_fallback` semantics).

---

## 6. Numerical correctness contract

| Output | Tolerance vs `torch_fallback` reference |
| --- | --- |
| `means_`, `covariances_`, `weights_` | `rtol=1e-4, atol=1e-4` (fp32); `rtol=1e-2, atol=1e-2` (fp16/bf16) |
| `lower_bound_`, `score_samples` | `rtol=1e-4, atol=1e-4` (fp32); `rtol=1e-2` (fp16/bf16) |
| `labels_` | ≥ 99% agreement on separable data; ≥ 95% on near-degenerate clusters |

- All accumulations are fp32 regardless of input dtype.
- mma.sync uses fp32 accumulator (`f32.f16.f16.f32`). No fp16 accumulator path even on Ada (perf gain not worth the precision loss for EM).
- Logsumexp always subtracts the per-row max in fp32 before exponentiation.

---

## 7. Backend dispatch (`_dispatch.py`)

```python
def resolve_backend(requested: str, covariance: str, shape: tuple, dtype) -> str:
    """Returns one of "cuda", "triton", "torch" given the user request and the call shape."""
    if requested == "torch":
        return "torch"
    if requested == "triton":
        return "triton" if _triton_supported(covariance, shape, dtype) else "torch"
    if requested == "cuda":
        return "cuda" if _cuda_supported(covariance, shape, dtype) else "torch"
    # "auto"
    if _cuda_supported(covariance, shape, dtype): return "cuda"
    if _triton_supported(covariance, shape, dtype): return "triton"
    return "torch"
```

`_cuda_supported` checks (in order, short-circuit on first failure):

1. `torch.cuda.is_available()`.
2. Compute capability ≥ 8.0. (Below 8.0 the safe kernel works but is rarely a perf win; we route to Triton/torch unless `requested == "cuda"`.)
3. `gmmxx._C` module imported successfully (i.e., extension was built).
4. Input dtype in `{fp16, bf16, fp32}` (varies per kernel — full covariance is fp32-only).
5. Shape within compiled bounds (e.g., full requires D ≤ 16; fused spherical requires D ≤ 64 and K ≤ 128).
6. Tensors are contiguous and on a CUDA device (else `_dispatch` triggers `.contiguous()` and continues).

On any path failure, `last_fallback_reason_` is set with a structured string (e.g., `"cuda_unsupported_shape: full_covariance D=32 > 16"`).

`GMMXX_BACKEND` env var sets the default for `requested` when the kwarg is `"auto"` and the env var is one of the four values.

---

## 8. Source layout

```
gmmxx/
├── __init__.py                       (extended — exports cuda_ops)
├── interface.py                      (modified — backend kwarg, last_backend_used_, cuda_*_enabled_ attrs)
├── _dispatch.py                      (NEW — resolve_backend + cuda/triton/torch supported checks)
├── _runtime.py                       (extended — cuda_<cov>_supported helpers)
├── _cuda.py                          (NEW — Python wrappers around _C; allocates outputs, validates inputs)
├── cuda_ops.py                       (NEW — public re-export of _cuda functional surface)
├── csrc/
│   ├── bindings.cpp                  (NEW — NB_MODULE(_C, ...))
│   ├── nb_torch.h                    (NEW — verbatim copy from flash-kmeans-cuda)
│   ├── common/
│   │   ├── arch.cuh                  (NEW — FKC-style arch probes + dtype traits)
│   │   ├── ptx.cuh                   (NEW — cp.async, ldmatrix, mma.sync wrappers)
│   │   ├── reduce.cuh                (NEW — warp/block reductions, stable logsumexp helper)
│   │   └── torch_cuda_includes.h     (NEW)
│   ├── estep/
│   │   ├── estep.h
│   │   ├── estep_common.cuh
│   │   ├── spherical_safe.cu
│   │   ├── spherical_sm80.cu
│   │   ├── diag_safe.cu
│   │   ├── diag_sm80.cu
│   │   ├── tied_safe.cu              (uses spherical_sm80 on projected coords)
│   │   ├── tied_sm80.cu
│   │   └── full_safe.cu              (D ≤ 16, no MMA)
│   ├── mstep/
│   │   ├── mstep.h
│   │   ├── blocked_spherical.cu      (sorted-run atomic-coalesced)
│   │   ├── blocked_diag.cu
│   │   ├── blocked_tied.cu
│   │   ├── blocked_full.cu
│   │   └── finalize.cu
│   ├── fused/
│   │   ├── fused.h
│   │   ├── fused_spherical.cu        (D ≤ 64, K ≤ 128)
│   │   ├── fused_diag.cu             (D ≤ 64, K ≤ 64)
│   │   └── fused_tied.cu             (D ≤ 64, K ≤ 128)
│   └── approx/
│       ├── approx.h
│       └── approx_topk_spherical.cu
├── (existing Triton + torch_fallback modules unchanged)
└── ...
pyproject.toml                       (extended — build deps + nanobind)
setup.py                             (NEW — CUDAExtension build, mirrors flash-kmeans-cuda)
```

---

## 9. Build & dependencies

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=64", "wheel", "torch>=2.11", "nanobind>=2.0,<3"]
build-backend = "setuptools.build_meta"

[project]
# existing fields preserved
dependencies = [
  "torch>=2.11",
]

[project.optional-dependencies]
triton = [
  "triton>=3.6,<3.7; platform_system != 'Windows'",
  "triton-windows>=3.6,<3.7; platform_system == 'Windows'",
]
sklearn = ["numpy", "scikit-learn"]
# existing extras kept
```

### `setup.py`

Mirrors `flash_kmeans_cuda/setup.py`:

- `_common_nvcc_flags()`: `-O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -lineinfo` plus `-Xcompiler=/Zc:preprocessor` on Windows and `-Xcompiler=-fPIC` on Linux.
- `_gencode_flags()`: emits `-gencode arch=compute_XX,code=sm_XX` for `XX in {80, 86, 89, 90, 100, 120}`. Honors `TORCH_CUDA_ARCH_LIST` when set.
- `_common_cxx_flags()`: Windows `/O2 /std:c++17 /EHsc /bigobj /Zc:preprocessor`; Linux `-O3 -std=c++17 -fPIC -fvisibility=hidden`.
- `CUDAExtension(name="gmmxx._C", sources=[...], include_dirs=[...])` — sources are the four `.cu` directories plus `bindings.cpp` plus nanobind's `nb_combined.cpp`.
- Skip CUDA build via `GMMXX_SKIP_CUDA=1` env var so users without an nvcc toolchain (Triton-only, CPU-only) can still install.

### Dev workflow

- `uv pip install -e .` builds the extension once at install time. Subsequent `import gmmxx` is no-nvcc.
- `uv run pytest tests` runs the parametrized test suite (skips CUDA cases on CPU-only hosts).
- `uv run python benchmarks/benchmark_cuda_vs_triton.py` runs the perf gate.

---

## 10. Testing strategy

### Existing tests — extended

All tests in `tests/test_gmmxx.py` are parametrized via a pytest fixture:

```python
@pytest.fixture(params=["torch", "triton", "cuda"])
def backend(request, has_cuda, has_triton):
    if request.param == "cuda" and not has_cuda: pytest.skip("no CUDA")
    if request.param == "triton" and not has_triton: pytest.skip("no Triton")
    return request.param
```

Tests pass the fixture value to the `GMMXX(..., backend=backend)` constructor. No tolerance changes — the existing `rtol=1e-4` gates already hold per Section 6.

### New tests

| File | Purpose |
| --- | --- |
| `tests/test_cuda_kernels.py` | Direct unit tests on `gmmxx.cuda_ops`. One test class per (covariance, op). Per-kernel shape sweeps. Compares CUDA output element-wise vs `torch_fallback` reference. |
| `tests/test_dispatch.py` | Backend resolution truth table (kwarg × env var × hardware × shape). Exercises every fallback path. |
| `tests/test_cuda_build.py` | Smoke test: `import gmmxx._C` succeeds; `GMMXX_SKIP_CUDA=1` skips the import gracefully. |

### Benchmarks

| Script | Purpose |
| --- | --- |
| `benchmarks/benchmark_cuda_vs_triton.py` | Speedup matrix per `(covariance, N, D, K, dtype)`. Used as a perf gate: CUDA must not regress vs Triton on any shape inside the supported window. Runs in CI on a single GPU. |
| `benchmarks/validate_equivalence.py` (extended) | CUDA path added to the existing comparison harness. |

---

## 11. Phase 1 internal staging

The user requested a single big PR with full coverage. The staging below is **internal sequencing for the implementation plan**, not a multi-release ramp.

1. Build skeleton: `pyproject.toml` + `setup.py` + empty `_C` module that compiles and imports on Linux + Windows.
2. Common headers: `arch.cuh`, `ptx.cuh`, `nb_torch.h`, `bindings.cpp` skeleton, `_dispatch.py` wiring.
3. Spherical safe + sm80 (E-step + blocked M-step + fused). End-to-end first; canary for the whole architecture.
4. Diag + tied (E-step + blocked M-step + fused).
5. Full (safe only).
6. Approx top-K spherical.
7. Sorted-run M-step + persistent E-step kernels (perf pass).
8. Multi-stream, occupancy tuning, mma.sync verification on each gencode target.
9. Test parametrization, CI gating, perf benchmark harness.
10. Docs: README updates, `last_backend_used_` examples, deprecation note for `use_triton`.

---

## 12. Risk register

| Risk | Mitigation |
| --- | --- |
| nvcc compile time inflates dev install | Cap dev gencode list via `TORCH_CUDA_ARCH_LIST=8.9` env in dev docs; full fat-binary built only by release CI. |
| Triton autotuner beats hand CUDA on some shapes | `_cuda_supported()` is benchmark-driven, not aspirational; offending shapes return False so `auto` picks Triton. |
| nanobind ABI break | Pin `nanobind>=2.0,<3` in `pyproject.toml`. |
| Windows build breakage | Use template's `/Zc:preprocessor /bigobj` flags + `nb_torch.h` Windows macro fixes verbatim. |
| First-iteration JIT lag (Triton) becomes first-iter nvcc compile lag (CUDA) | Compiled `_C.pyd`/`.so` is built once at install time; no runtime nvcc. Better than Triton's per-shape JIT compile. |
| mma.sync correctness on fp16 | Tested element-wise vs `torch_fallback` at `rtol=1e-2`; fall through to safe kernel on shapes that trigger known mma instability. |
| Empty-cluster behavior drifts | `finalize` mirrors `torch_fallback` semantics: keep previous mean/variance, add `reg_covar`. Covered by tests in `test_cuda_kernels.py::TestEmptyCluster`. |
| `backend="auto"` regressions in the wild because Phase 1 defaults to CUDA | `GMMXX_BACKEND=triton` documented as the immediate escape hatch; `last_backend_used_` and `last_fallback_reason_` available for diagnosis. |

---

## 13. Out of scope (deferred)

- Prebuilt PyPI wheels — v1.0.
- sm_70/75 mma.sync (HMMA path with different tile shapes).
- Multi-GPU CUDA (covered by `large_n.py` CPU streaming today).
- fp8 (E4M3/E5M2) — Phase 2 if user demand appears.
- WGMMA (Hopper warpgroup matmul) — Phase 2; current sm_90 gencode falls back to mma.sync m16n8k16.
- Cosine / Dot variants — flash-kmeans-cuda dropped these; GMMXX has no equivalent today.
