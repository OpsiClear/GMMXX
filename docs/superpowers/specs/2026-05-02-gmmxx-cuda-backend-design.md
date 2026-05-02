# GMMXX CUDA Backend — Design Spec

**Status:** Approved (2026-05-02), revised after multi-agent review (2026-05-02). Brainstorming-derived design for a hand-written CUDA backend that ships alongside the existing Triton path inside the `gmmxx` package.

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
- No new public Python user-facing API beyond an experimental `gmmxx.cuda_ops` functional layer for power users.
- No optimized path for sm_70 / sm_75 (Volta / Turing). They lack `mma.sync.m16n8k16.f32.f16.f16.f32`. The safe kernel still compiles and runs there, but `_dispatch.resolve_backend` routes pre-sm_80 hardware to Triton/PyTorch unless the user explicitly requests `backend="cuda"`.
- No multi-GPU CUDA path. `large_n.py` continues to drive CPU-streaming, but is refactored in Phase 1 to dispatch into the chosen backend per chunk (see §7.5).
- No fp8 (E4M3/E5M2) support. Deferred to Phase 2 if user demand appears.
- No `torch.compile()` / Inductor interop. nanobind C extensions are not graph-traceable; users wrapping `gmmxx.cuda_ops.*` in compiled code get graph breaks.

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
                                │  try/except wrapper around CUDA FFI    │
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
              │   mstep/{blocked_<cov>,finalize_<cov>}.cu               │
              │   fused/{fused_spherical,fused_diag,fused_tied}.cu      │
              │   approx/approx_topk_spherical.cu                       │
              │   common/{arch.cuh, ptx.cuh, reduce.cuh, nb_torch.h}    │
              └─────────────────────────────────────────────────────────┘
```

The `GMMXX` orchestrator does not change beyond the new `backend` kwarg and attributes. All backend selection is encapsulated in the new `_dispatch` module. Each compute backend (CUDA, Triton, torch) is a peer; none knows about the others. `large_n.py` becomes a fourth caller of `_dispatch` (see §7.5).

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
| `last_backend_used_` | str | Set at the end of every `fit()`/`predict()` to the **dominant backend** (≥ 50 % of E-step iterations). |
| `fit_info_['backend_breakdown']` | `dict[str,int]` | Per-iteration counts for mixed-backend runs (e.g., `{"cuda": 18, "triton": 2}`). |
| `cuda_estep_enabled_` | bool | Was the CUDA E-step actually used at least once during the call. |
| `cuda_fused_update_enabled_` | bool | Was the CUDA fused single-tile path used. |
| `cuda_approx_topk_enabled_` | bool | Was the CUDA approx-topK path used. |
| `gmmxx.cuda_ops` module | functional, **experimental** | Low-level access to compiled kernels. Marked `Experimental: API may change before v1.0` in its docstring. |

### Deprecated (with shims)

`use_triton: bool` constructor kwarg remains accepted with the following semantics:

1. **Conflict.** Passing both `backend=` and `use_triton=` raises `ValueError`, mirroring the existing `_resolve_alias` pattern in `interface.py:100`.
2. **`use_triton=True`** → `backend="auto"`.
3. **`use_triton=False`** → `backend="auto"` with an internal `_legacy_no_triton=True` flag that filters Triton out of the resolution chain. The user gets CUDA when available, then PyTorch — never Triton. (This corrects an asymmetric mapping in the original spec; users who set `use_triton=False` historically wanted "no Triton JIT," not "no GPU at all.")
4. **`set_params(use_triton=...)`** routes through the same shim. **`get_params()`** returns the canonical `backend=` key only; `use_triton` is omitted from the dict so that `sklearn.base.clone()` round-trips cleanly.
5. `DeprecationWarning` emitted **once per estimator instance** (not per kernel call), recommending `backend=...`.
6. Removed in v2.0.

### `gmmxx.cuda_ops` functional surface (experimental)

Mirrors `flash_kmeans_cuda.ops` in shape and naming. Module-level docstring: *"Experimental low-level CUDA kernel surface. Subject to change before v1.0; the only API stability guarantee is the `GMMXX` class."*

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

# Caller-owned, MUST be zeroed before each call (see §5 zero-init contract).
blocked_update_<cov>(x, sorted_idx, sorted_ids, sums_out, sumsq_out, counts_out)
fused_<cov>(x, params..., partial_outs)
approx_topk_spherical(x, means, var, log_w, top_k, partial_outs)

# Per-covariance finalize. reg_covar handling differs per type — see §5.
finalize_spherical(sums, sumsq, counts, old_means, old_var, reg_covar) -> (means, var)
finalize_diag(sums, sumsq, counts, old_means, old_var, reg_covar) -> (means, var)
finalize_tied(sums, outer_sums, counts, old_means, old_chol, reg_covar) -> (means, chol)
finalize_full(sums, outer_sums, counts, old_means, old_chol, reg_covar) -> (means, chol)
```

All ops require contiguous CUDA tensors; type/shape/device validated in C++ via `TORCH_CHECK`. Internal Python wrappers in `gmmxx/_cuda.py` use the `_cuda` suffix convention (e.g., `spherical_assign_cuda`) to mirror the established `*_triton` naming; `gmmxx.cuda_ops` is the deduped public re-export.

---

## 4. CUDA kernel inventory (Phase 1)

| Covariance | E-step kernels | M-step (blocked) | Fused single-tile E/M | Approx top-K |
| --- | --- | --- | --- | --- |
| spherical | safe + sm80 mma | sorted-run + blocked | yes (D ≤ 64, K ≤ 128) | yes (training only) |
| diag | safe + sm80 mma | sorted-run + blocked | yes — three tile variants matching the Triton policy: `(D ≤ 16, K ≤ 128) ∪ (D ≤ 32, K ≤ 128) ∪ (D ≤ 64, K ≤ 64)` | — |
| tied | safe + sm80 (per-tile projection) | sorted-run + blocked | yes (D ≤ 64, K ≤ 128) | — |
| full | safe only (D ≤ 16) | safe blocked (D ≤ 16) | — | — |

E-step exposes three kernels each (assign, logsumexp, resp). Total device kernels: ≈ 24 + per-cov finalize + utilities. All host launchers exposed through `bindings.cpp`.

Shape gates (compile-time + runtime) are documented in `gmmxx/_runtime.py` (extended) and mirrored in `_dispatch.py` so the orchestrator can decide before crossing the FFI boundary.

---

## 5. Optimization techniques and host-side contracts

### 5a. Host-side launcher contract (non-optional)

Every host launcher in `csrc/estep/`, `csrc/mstep/`, `csrc/fused/`, and `csrc/approx/`:

1. Validates inputs via `TORCH_CHECK` (contiguity, dtype, device, shape).
2. Constructs `at::cuda::CUDAGuard guard(input.device())` as its first non-check statement. **Without this, multi-device safety is broken** — a kernel can launch on the wrong GPU when the caller's current device differs from the input tensor's device. FKC enforces this at every launcher (`assign_safe.cu:95`, `assign_sm80.cu:735`, `update_sorted.cu:213,279`, `update_finalize.cu:89`); GMMXX inherits the convention.
3. Uses `at::cuda::getCurrentCUDAStream()` for kernel launches.
4. Caches `cudaGetDeviceProperties` once per (process, device-index) pair via a function-local static — avoids redundant property reads inside EM loops.

### 5b. M-step zero-init contract (non-optional)

`blocked_update_<cov>` does **only** atomicAdd into `sums_out`, `sumsq_out`, `counts_out`. The caller MUST zero these tensors before every call. `gmmxx/_cuda.py` wrappers handle this via `tensor.zero_()` so users of the high-level `GMMXX` API and `gmmxx.cuda_ops` never see the contract directly. Without zero-init, sums leak across EM iterations — a silent correctness bug. Debug builds enable a sentinel pattern (write `0xDEADBEEF` after the kernel; assert during the next `zero_()` that the buffer hasn't been re-used uninitialized).

### 5c. Per-covariance finalize semantics

`reg_covar` handling differs per covariance type (matching `torch_fallback.py`):

- **Spherical** (`finalize_spherical`): `var = clamp_min((sumsq/n - mean²).mean(D), reg_covar)`; output shape `(B, K)`.
- **Diag** (`finalize_diag`): `var = clamp_min(sumsq/n - mean², reg_covar)`; output shape `(B, K, D)`.
- **Tied** (`finalize_tied`): outer-product sums divided by total N → covariance `Σ`; `Σ += reg_covar · I`; symmetrize; Cholesky-factor → `chol`. Output shape `(B, D, D)`.
- **Full** (`finalize_full`): per-cluster `Σ_k = outer_sums_k / n_k - mean_k mean_kᵀ`; `Σ_k += reg_covar · I`; symmetrize; per-cluster Cholesky → `chol_k`. Output shape `(B, K, D, D)`.

In all four, `count == 0` clusters keep their previous `(mean, var/chol)` tuple unchanged.

### 5d. Optimization techniques (applied per kernel where profitable)

- **mma.sync `m16n8k16.row.col.f32.f16.f16.f32`** for assign / logsumexp / blocked-update on sm_80+ when input dtype is fp16 or bf16. Tile defaults: `BLOCK_N=128`, `BLOCK_K=64`, `BLOCK_D=16`. Per-arch overrides in `arch.cuh`. fp32 inputs route to the safe kernel only; the `m16n8k16.f32.f32.f32.f32` opcode does not exist, and TF32 fallback (`m16n8k8.row.col.f32.tf32.tf32.f32`) is deferred to Phase 2 behind a `GMMXX_HAS_TF32_MMA` arch probe.
- **`cp.async.cg` double-buffered SMEM** for centroid tiles `(BLOCK_K × D)`. SMEM padded by 8 elements per row to eliminate 32-way bank conflicts on D = power-of-2. Reservation: `2 × BLOCK_K × (D + 8) × sizeof(dtype)` per kernel — included in `arch.cuh` SMEM accounting alongside register-tile working set.
- **Register-tiled fused min-over-K accumulator** in assign kernels: cross-product never materialized to SMEM. Per-thread `Best{dist,idx}` with first-occurrence tie-breaking.
- **Tied projection** is per-tile inside `tied_sm80.cu`: `mma.sync` against `L⁻¹` loaded into SMEM once per CTA; never materialized to global memory. `weighted_means_proj` is projected once per fit on the host (matches Triton precompute); `x` is projected on the fly. Mirrors `_fused_single_tile_tied_native_kernel` at `fused_update_triton.py:379`.
- **Sorted-run atomic coalescing for M-step**: caller pre-sorts `cluster_ids` (Python side, `torch.argsort`, executed inside `gmmxx/_cuda.py` only on the CUDA path); kernel walks runs and emits one `atomicAdd` per (run, feature) tuple. Estimated ~256× atomic-issue reduction within a CTA. Below `N · K < 2²¹` the dispatcher prefers a per-token `atomicAdd` kernel that skips the sort, since the sort cost dominates the coalescing win on small inputs.
- **Persistent kernels for E-step** when `N ≥ 256k`: one CTA per SM, work-stealing on a shared counter. Eliminates relaunch overhead across EM iterations. Persistent kernels declare `__launch_bounds__(<threads>, 1)`; non-persistent kernels target `__launch_bounds__(<threads>, 2)` on sm_80/86/89/90 with `BLOCK_N=128`, dropping to `(<threads>, 1)` on `BLOCK_N=256` variants. SMEM accounting per arch in `arch.cuh`.
- **Multi-stream E/M overlap**: M-step partials emitted on a side stream; final reduce + finalize sync. Opt-in via `GMMXX_OVERLAP_STREAMS=1` env var; default off because Phase 1 doesn't yet plumb `cudaEvent_t` handles between EM iterations to preserve the M-step→E-step happens-before. There is no determinism cost — both single-stream and multi-stream are nondeterministic at the bit level due to atomic ordering. Phase 2 plumbs events and unconditionally enables overlap.
- **Per-arch occupancy tuning**: `__launch_bounds__(threads, ctas_per_sm)` set per kernel as above. sm_90+ uses larger SMEM and `BLOCK_N=256` variants where beneficial.
- **Stable logsumexp**: per-row max via warp shuffle (`__shfl_xor_sync`), subtract-max-then-exp-then-sum, all in fp32 regardless of input dtype.
- **Packed half2 / bfloat162 math** in fused single-tile path.
- **Vectorized loads**: `float4` / `__half2` / `__nv_bfloat162` when D-strides align to 8 bytes.

---

## 6. Numerical correctness contract

Tolerances vs `torch_fallback` reference:

| Output | spherical / diag / tied (fp32 inputs) | spherical / diag / tied (fp16/bf16 inputs) | full covariance (any dtype) |
| --- | --- | --- | --- |
| `means_`, `weights_` | `rtol=1e-4, atol=1e-4` | `rtol=1e-3, atol=1e-3` | `rtol=1e-3, atol=1e-3` |
| `covariances_` | `rtol=1e-4, atol=1e-4` | `rtol=5e-3, atol=5e-3` | `rtol=1e-3, atol=1e-5` (outer-product cancellation) |
| `lower_bound_`, `score_samples` | `rtol=1e-4, atol=1e-4` | `rtol=1e-2` | `rtol=1e-3` |
| `labels_` | ≥ 99 % agreement on separable data; ≥ 95 % on near-degenerate clusters | same | same |

Notes:

- All accumulations are fp32 regardless of input dtype.
- mma.sync uses fp32 accumulator (`f32.f16.f16.f32`). No fp16 accumulator path even on Ada (perf gain not worth the precision loss for EM).
- Logsumexp always subtracts the per-row max in fp32 before exponentiation.
- fp32 inputs do NOT use the sm80 `mma.sync` path; they execute on the safe (SIMT fp32-fma) kernel. `_cuda_supported` returns False for shapes that would need the sm80 path with `dtype==fp32`, falling back to safe.
- Run-to-run determinism is **not guaranteed** at the bit level due to atomic ordering. A separate determinism harness bounds run-to-run drift independently of the reference comparison.

### 6.5. Batch handling

GMMXX supports both `(N, D)` and `(B, N, D)` user input. The CUDA backend:

- All kernels accept `(B, N, D)` with `B ≥ 1` and launch with `gridDim.z = B`.
- Unbatched `(N, D)` user input is unsqueezed to `(1, N, D)` before the FFI call (matches `_normalize_input` at `interface.py:309`).
- Phase 1 supports `B ≤ 8` with the same compiled binaries; larger batches loop on the Python side inside `_cuda.py`. Phase 2 widens the grid-Z bound after benchmarking.

---

## 7. Backend dispatch (`_dispatch.py`)

```python
def resolve_backend(requested: str, covariance: str, shape: tuple, dtype,
                    legacy_no_triton: bool = False) -> str:
    """Returns one of "cuda", "triton", "torch" given the user request and call shape.

    legacy_no_triton: True when called from a deprecated use_triton=False shim.
                      Filters Triton out of the resolution chain.
    """
    if requested == "torch":
        return "torch"
    if requested == "triton":
        if legacy_no_triton:
            raise ValueError("backend='triton' incompatible with use_triton=False")
        return "triton" if _triton_supported(covariance, shape, dtype) else "torch"
    if requested == "cuda":
        return "cuda" if _cuda_supported(covariance, shape, dtype) else "torch"
    # "auto"
    if _cuda_supported(covariance, shape, dtype):
        return "cuda"
    if (not legacy_no_triton) and _triton_supported(covariance, shape, dtype):
        return "triton"
    return "torch"
```

`_cuda_supported` checks (in order, short-circuit on first failure):

1. `torch.cuda.is_available()`.
2. Compute capability ≥ 8.0. (Below 8.0 the safe kernel works but is rarely a perf win; we route past unless `requested == "cuda"`.)
3. `gmmxx._C` module imported successfully (i.e., extension was built).
4. Input dtype in `{fp16, bf16, fp32}` (varies per kernel — full covariance is fp32-only; sm80 mma path is fp16/bf16-only, fp32 routes to safe).
5. Shape within compiled bounds (e.g., full requires D ≤ 16; fused spherical requires D ≤ 64 and K ≤ 128).
6. Tensors are contiguous and on a CUDA device (else `_dispatch` triggers `.contiguous()` and continues).

### 7a. Runtime error fallback

`gmmxx/_cuda.py` wraps each `gmmxx._C` invocation in `try / except (RuntimeError, torch.cuda.OutOfMemoryError, ImportError)`. On failure:

1. Sets `last_fallback_reason_ = f'cuda_runtime_error: {exc}'`.
2. Re-resolves with `requested='triton'` (or `'torch'` when `legacy_no_triton`).
3. Retries the same op on the fallback backend.

`GMMXX_CUDA_NO_FALLBACK=1` env var disables retry — useful in CI to make CUDA bugs loud instead of silently fallthrough. Explicit `backend='cuda'` with the extension unbuilt raises `RuntimeError('gmmxx._C extension not built; reinstall without GMMXX_SKIP_CUDA')` rather than silently downgrading.

`GMMXX_BACKEND` env var sets the default for `requested` when the kwarg is `"auto"` and the env var is one of the four values.

### 7.5. Large-N integration (`large_n.py` refactor)

`large_n.py` currently calls Triton kernels directly at ~15 sites (lines 650, 673, 683, 709, 740, 772, 795, 816, 828, 1369, 1380, 1390, 1497, 1540, 1635) and gates on `_HAS_TRITON`. This is incompatible with `backend="cuda"`.

Phase 1 refactors `large_n.py` to:

1. Accept `backend: str` from the calling `GMMXX` instance (replacing the existing `use_triton: bool` thread-through).
2. Replace each direct Triton call site with `_dispatch.dispatch_kernel(op_name, backend, *args)` — a thin helper in `_dispatch.py` that resolves the backend per chunk and dispatches into the right module.
3. Remove the `_HAS_TRITON` import-gating pattern in favor of `_dispatch.resolve_backend` outcomes.

A new `tests/test_large_n_dispatch.py` covers: chunked-CPU input flowing through CUDA, a chunk straddling the supported-shape boundary that falls through mid-stream, and an `_HAS_CUDA=False` host falling back to Triton or torch.

---

## 8. Source layout

```
gmmxx/
├── __init__.py                       (extended — exports cuda_ops; safe try/except on _C import)
├── interface.py                      (modified — backend kwarg, last_backend_used_, cuda_*_enabled_ attrs, deprecation shim)
├── _dispatch.py                      (NEW — resolve_backend + dispatch_kernel + cuda/triton/torch supported checks)
├── _runtime.py                       (extended — cuda_<cov>_supported helpers)
├── _cuda.py                          (NEW — Python wrappers around _C; allocates outputs, validates inputs, wraps FFI in try/except, performs sort+zero_init for blocked update)
├── cuda_ops.py                       (NEW — public re-export of _cuda functional surface; experimental docstring)
├── csrc/
│   ├── bindings.cpp                  (NEW — NB_MODULE(_C, ...). Skeleton: NB_MODULE block + per-op host fn + optional<at::Tensor> out + at::empty_like allocation + force_safe_path() function-local static reading GMMXX_FORCE_SAFE + tc_eligible(dtype, sm) dtype gate before sm80 dispatch)
│   ├── nb_torch.h                    (NEW — verbatim copy from flash-kmeans-cuda; no project-namespace identifiers, so no rename required)
│   ├── common/
│   │   ├── arch.cuh                  (NEW — GMMXX_-prefixed arch probes: GMMXX_CUDA_ARCH, GMMXX_HAS_F16_MMA, GMMXX_HAS_BF16_MMA, GMMXX_HAS_TF32_MMA (Phase 2 hook), GMMXX_HAS_CP_ASYNC; dtype traits)
│   │   ├── ptx.cuh                   (NEW — wrapper inventory: cp_async_cg, cp_async_commit, cp_async_wait_group<N>, ldmatrix_sync_x4, mma_m16n8k16_f32_f16, mma_m16n8k16_f32_bf16, mma_m16n8k8_f32_tf32 (Phase 2 hook), atomic_add_block, atomic_add_system, warp_shfl_xor_sync, warp_reduce_add_sync. Each documented with the inline-PTX it emits.)
│   │   ├── reduce.cuh                (NEW — warp/block reductions, stable logsumexp helper)
│   │   └── torch_cuda_includes.h     (NEW — Python.h kept out of kernel TUs)
│   ├── estep/
│   │   ├── estep.h
│   │   ├── estep_common.cuh
│   │   ├── spherical_safe.cu
│   │   ├── spherical_sm80.cu
│   │   ├── diag_safe.cu
│   │   ├── diag_sm80.cu
│   │   ├── tied_safe.cu
│   │   ├── tied_sm80.cu              (per-tile L⁻¹ projection in SMEM)
│   │   └── full_safe.cu              (D ≤ 16, no MMA, fp32 only)
│   ├── mstep/
│   │   ├── mstep.h
│   │   ├── blocked_spherical.cu      (sorted-run atomic-coalesced)
│   │   ├── blocked_diag.cu
│   │   ├── blocked_tied.cu           (outer-product accumulation)
│   │   ├── blocked_full.cu           (per-cluster outer-product accumulation)
│   │   ├── finalize_spherical.cu
│   │   ├── finalize_diag.cu
│   │   ├── finalize_tied.cu          (Cholesky factorization)
│   │   └── finalize_full.cu          (per-cluster Cholesky)
│   ├── fused/
│   │   ├── fused.h
│   │   ├── fused_spherical.cu        (D ≤ 64, K ≤ 128)
│   │   ├── fused_diag.cu             (three tile variants per §4)
│   │   └── fused_tied.cu             (D ≤ 64, K ≤ 128)
│   └── approx/
│       ├── approx.h
│       └── approx_topk_spherical.cu
├── (existing Triton + torch_fallback modules unchanged)
└── ...
.autotune/                            (extended — manual experiment logs per FKC convention: <host>-<arch>.json)
pyproject.toml                       (extended — build deps + nanobind + Python pin)
setup.py                             (NEW — CUDAExtension build, mirrors flash-kmeans-cuda)
```

`fused/` is a peer to `estep/`/`mstep/` because fused kernels carry their own per-cov tile policy and shape gates that don't compose with the standalone E/M variants; keeping them isolated makes the ABI gates self-contained. Departure from FKC's flat layout is intentional.

---

## 9. Build & dependencies

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=64", "wheel", "torch==2.11.*", "nanobind>=2.0,<3"]
build-backend = "setuptools.build_meta"

[project]
# existing fields preserved
requires-python = ">=3.12,<3.13"
dependencies = [
  "torch==2.11.*",   # libtorch C++ ABI is tied to the torch version
]

[project.optional-dependencies]
triton = [
  "triton>=3.6,<3.7; platform_system == 'Linux'",
  "triton-windows>=3.6,<3.7; platform_system == 'Windows'",
]
sklearn = ["numpy", "scikit-learn"]
# existing extras kept

[tool.uv.sources]
torch = { index = "pytorch-cu130" }    # CUDA 12.8+ toolchain required for sm_100/sm_120
```

### `setup.py`

Mirrors `flash_kmeans_cuda/setup.py`:

- `_common_nvcc_flags()`: `-O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -lineinfo` plus `-Xcompiler=/Zc:preprocessor` on Windows and `-Xcompiler=-fPIC` on Linux.
- `_gencode_flags()`: emits `-gencode arch=compute_XX,code=sm_XX` for `XX in {80, 86, 89, 90, 100, 120}`, with two safety rails: (a) **early-return `[]` when `TORCH_CUDA_ARCH_LIST` is set** in the env (matches FKC behavior; lets `cpp_extension` parse the env itself); (b) **detect `nvcc --version` at build time and silently drop sm_100/sm_120 when nvcc < 12.8**, so users on older toolchains don't hit cryptic compile errors. `GMMXX_BUILD_BLACKWELL=0` env var force-disables those targets.
- `_common_cxx_flags()`: Windows `/O2 /std:c++17 /EHsc /bigobj /Zc:preprocessor /utf-8 /Zc:__cplusplus`; Linux `-O3 -std=c++17 -fPIC -fvisibility=hidden`. (`/utf-8` avoids C4819 on non-EN locales for nanobind's UTF-8 string literals; `/Zc:__cplusplus` makes MSVC report the real `__cplusplus` value so nanobind's feature-detect macros enable the C++17 path.)
- **Do NOT pass `/MD` or `/MT` to nvcc/cl** — `CUDAExtension` injects the right MSVC runtime automatically; passing it raw is parsed as a positional input file.
- `CUDAExtension(name="gmmxx._C", sources=[...], include_dirs=[...])`. Sources are the `.cu` files in `csrc/{estep,mstep,fused,approx,common}/`, plus `bindings.cpp`, plus nanobind's bundled `nb_combined.cpp` (resolved from `nanobind.__file__.parent / 'src' / 'nb_combined.cpp'`). `include_dirs` adds `nanobind.include_dir()` AND `nanobind.__file__.parent / 'ext' / 'robin_map' / 'include'` — the robin_map path is required by nanobind 2.1+ and a frequent bring-up trip.
- Skip the entire CUDA build via `GMMXX_SKIP_CUDA=1` env var so users without an nvcc toolchain (Triton-only, CPU-only) can still install. `gmmxx/__init__.py` wraps `from . import _C` in `try/except ImportError`; on failure, sets `_HAS_CUDA = False` and `import gmmxx` still succeeds.

### Dev workflow

- `uv pip install -e .` builds the extension once at install time. Subsequent `import gmmxx` is no-nvcc.
- `uv run pytest tests` runs the parametrized test suite (skips CUDA cases on CPU-only hosts).
- `uv run python benchmarks/benchmark_cuda_vs_triton.py` runs the perf gate.
- Local dev with single-arch builds: `TORCH_CUDA_ARCH_LIST=8.9 uv pip install -e .` (compiles ~6× faster than the full fat binary).

---

## 10. Testing strategy

### 10a. Test refactor

`tests/test_gmmxx.py` is currently a single `unittest.TestCase` class with 17 test methods, several of which contain inline for-loops over covariance types (~40 logical cases total). A naive `@pytest.fixture` won't work — Phase 1 first **converts the file from `unittest.TestCase` to plain pytest functions**, then adds the `backend` fixture. Estimated total cases after refactor:

- ~110 cases on a CUDA host (3-way: torch, triton, cuda).
- ~55 cases on CPU-only (torch + triton skipped to torch).

```python
@pytest.fixture(params=["torch", "triton", "cuda"])
def backend(request, has_cuda, has_triton):
    if request.param == "cuda" and not has_cuda: pytest.skip("no CUDA")
    if request.param == "triton" and not has_triton: pytest.skip("no Triton")
    return request.param
```

Tests pass the fixture value to `GMMXX(..., backend=backend)`. Existing cross-backend equivalence tests (e.g., `test_spherical_approx_topk_triton_matches_torch` at `interface.py:204`) keep their explicit dual-construction; they are NOT covered by the `backend` fixture because they assert behavior comparing two specific backends.

### 10b. New test files

| File | Purpose |
| --- | --- |
| `tests/test_cuda_kernels.py` | Direct unit tests on `gmmxx.cuda_ops`. One test class per (covariance, op). Per-kernel shape sweeps. Compares CUDA element-wise vs `torch_fallback`. Includes `TestEmptyCluster` per-cov to verify finalize semantics. |
| `tests/test_cuda_vs_triton.py` | 3-way CUDA-vs-Triton oracle (mirrors FKC's `test_correctness.py`). Inside the Triton-supported window, asserts CUDA matches Triton at `rtol=5e-3` per dtype tier — catches drift between two GPU implementations even when both pass the torch-reference gate. |
| `tests/test_dispatch.py` | Backend resolution truth table (kwarg × env var × hardware × shape × `legacy_no_triton`). Exercises every fallback path including the `try/except` runtime-error fallback. Includes `test_clone_roundtrip` — `sklearn.base.clone(est)` returns identical `get_params()` and emits no `DeprecationWarning` on the cloned instance. |
| `tests/test_cuda_build.py` | Smoke test: `import gmmxx._C` succeeds; `GMMXX_SKIP_CUDA=1` skips the import gracefully. Subprocess-based pattern (mirrors FKC `test_persistent.py:24-83`) for env-var-dependent function-local statics like `force_safe_path()`. |
| `tests/test_large_n_dispatch.py` | CPU-streaming flows through `_dispatch.dispatch_kernel` — see §7.5. |

### 10c. Benchmarks & gates

| Script | Purpose |
| --- | --- |
| `benchmarks/benchmark_cuda_vs_triton.py` | Speedup matrix per `(covariance, N, D, K, dtype)`. Used as a perf gate: CUDA must not regress vs Triton on any shape inside the supported window. Runs in CI on a single GPU. |
| `benchmarks/validate_equivalence.py` (extended) | CUDA path added to the existing comparison harness. |

---

## 11. Phase 1 internal staging

Phase 1 lands as a sequence of incremental PRs on the `GMMXX-cuda` branch, then squash-merges to `main` as a single "GMMXX CUDA backend (Phase 1)" commit. Each PR keeps the branch green via `_dispatch` falling through to Triton for not-yet-implemented covariance types.

Per-PR scope:

1. **Build skeleton**: `pyproject.toml` + `setup.py` + empty `_C` module that compiles and imports on Linux + Windows. `_dispatch.py` skeleton with all-False support gates.
2. **Common headers**: `arch.cuh`, `ptx.cuh`, `nb_torch.h`, `bindings.cpp` skeleton, `_dispatch.py` wiring, `_cuda.py` helper.
3. **Spherical safe + sm80** (E-step + blocked M-step + fused + finalize). End-to-end first; canary for the whole architecture. Includes test-file pytest conversion and the `backend` fixture for spherical only.
4. **Diag** (E-step + blocked M-step + fused, three tile variants).
5. **Tied** (per-tile projection in SMEM + blocked + fused + Cholesky finalize).
6. **Full** (safe-only D ≤ 16 + per-cluster Cholesky finalize).
7. **Approx top-K spherical**.
8. **`large_n.py` refactor** to dispatch through `_dispatch.dispatch_kernel`.
9. **Sorted-run M-step + persistent E-step kernels** (perf pass driven by autotune logs in `.autotune/<host>-<arch>.json`).
10. **Multi-stream events plumbing** (still default-off behind env var); occupancy tuning; mma.sync verification on each gencode target; CI matrix gating.
11. **Docs**: README updates, `last_backend_used_` examples, deprecation note for `use_triton`, build-from-source instructions.

---

## 12. Risk register

| Risk | Mitigation |
| --- | --- |
| nvcc compile time inflates dev install | Cap dev gencode list via `TORCH_CUDA_ARCH_LIST=8.9` env in dev docs; full fat-binary built only by release CI. |
| Triton autotuner beats hand CUDA on some shapes | `_cuda_supported()` is benchmark-driven, not aspirational; offending shapes return False so `auto` picks Triton. |
| nanobind ABI break | Pin `nanobind>=2.0,<3` in `pyproject.toml`. |
| Windows build breakage | Use template's `/Zc:preprocessor /bigobj /utf-8 /Zc:__cplusplus` flags + `nb_torch.h` Windows macro fixes verbatim; do not pass `/MD` or `/MT` raw to nvcc. |
| sm_100/sm_120 fail to compile on older CUDA | Build-time `nvcc --version` detection drops Blackwell targets; `GMMXX_BUILD_BLACKWELL=0` opt-out. |
| First-iteration JIT lag (Triton) becomes first-iter nvcc compile lag (CUDA) | Compiled `_C.pyd`/`.so` is built once at install time; no runtime nvcc. Better than Triton's per-shape JIT compile. |
| mma.sync correctness on fp16 | Tested element-wise vs `torch_fallback` at `rtol=5e-3`; 3-way CUDA-vs-Triton oracle catches drift; `GMMXX_FORCE_SAFE=1` env forces the safe kernel as an escape hatch. |
| Empty-cluster behavior drifts | Per-cov `finalize_*` mirrors `torch_fallback` semantics; covered by `test_cuda_kernels.py::TestEmptyCluster` per covariance type. |
| Multi-device call lands on wrong GPU | `at::cuda::CUDAGuard` mandatory in every host launcher; enforced by code-review checklist. |
| Sums leak across EM iterations | Caller-owned zero-init contract per §5b; debug builds use sentinel pattern. |
| `large_n.py` still hardcodes Triton after merge | Phase 1 PR (8) refactors all ~15 call sites; covered by `tests/test_large_n_dispatch.py`. |
| `backend="auto"` regressions in the wild because Phase 1 defaults to CUDA | `GMMXX_BACKEND=triton` and `GMMXX_CUDA_NO_FALLBACK=0` documented; runtime `try/except` wraps each FFI call so CUDA bugs degrade to Triton silently in production while still being loud in CI. |

---

## 13. Out of scope (deferred)

- Prebuilt PyPI wheels — v1.0.
- sm_70/75 mma.sync (HMMA path with different tile shapes).
- Multi-GPU CUDA (covered by `large_n.py` CPU streaming refactored in §7.5).
- fp8 (E4M3/E5M2) — Phase 2 if user demand appears.
- WGMMA (Hopper warpgroup matmul) — Phase 2; current sm_90 gencode falls back to mma.sync m16n8k16.
- TF32 mma path for fp32 inputs — Phase 2 behind `GMMXX_HAS_TF32_MMA` arch probe.
- Multi-stream event plumbing across EM iterations — staged for Phase 2; Phase 1 ships single-stream by default with overlap as opt-in.
- `torch.compile()` / Inductor interop — out of scope; nanobind extensions are not graph-traceable.
- Cosine / Dot variants — flash-kmeans-cuda dropped these; GMMXX has no equivalent today.
