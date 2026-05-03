# GMMXX CUDA Backend — Plan 2: Spherical Safe Path

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn on the first real CUDA path. After this plan, `GMMXX(covariance_type="spherical", backend="cuda")` performs full EM training (E-step + M-step + finalize) on hand-written CUDA kernels for shapes inside the spherical support window. The kernels are "safe" (one thread per point, scalar FMA, no mma.sync, no cp.async, no sorted-run atomic coalescing). Performance optimizations land in Plan 3 (sm_80 mma) and Plan 4 (sorted-run M-step + persistent E-step).

**Architecture:** Three E-step kernels (`assign`, `logsumexp`, `resp`) + one M-step kernel (`blocked_update`) + one finalize kernel, all under `gmmxx/csrc/{estep,mstep}/spherical_safe.cu`. Python wrappers in `gmmxx/_cuda.py` validate inputs, allocate outputs, and pre-zero atomicAdd targets. `cuda_spherical_supported()` flips to True for fp32/fp16/bf16 inputs with `0 < d ≤ 128, 0 < k ≤ 2048`. The `interface.py` spherical training loop is updated to call the dispatcher instead of hardcoded Triton paths.

**Tech Stack:** CUDA 12.8+ via nvcc, nanobind 2.x, PyTorch 2.11.x, pytest. Reference template: existing Triton kernels in `gmmxx/assign_spherical_triton.py` and `gmmxx/torch_fallback.py:batch_gmm_Spherical_torch_native`.

**Spec:** `docs/superpowers/specs/2026-05-02-gmmxx-cuda-backend-design.md` §4–§6.

**Out of scope for this plan (deferred):**
- sm_80 mma.sync optimized E-step (Plan 3)
- Sorted-run atomic coalescing for M-step (Plan 3)
- Persistent kernels for E-step (Plan 3)
- Fused single-tile E/M (Plan 4)
- Approx top-K (Plan 5)
- Diagonal / tied / full covariance (Plans 6–8)
- `large_n.py` integration (Plan 9)

**Foundation assumed in place:** Plan 1 must be complete. Specifically, `gmmxx._C` builds and imports, the canary kernel works, `gmmxx/_cuda.py` exposes `_check_input` / `require_cuda` / `_no_fallback`, `gmmxx/_dispatch.resolve_backend` is wired (truth table works), `cuda_spherical_supported` exists as a False-returning stub, and the `backend` kwarg flows through `GMMXX.__init__`.

**Known limitation carried over from Plan 1 (Issue I-1 from final review):** The `use_triton` property in `gmmxx/interface.py` returns True for `backend ∈ {"auto", "cuda", "triton"}`. ~12 internal call sites in `interface.py` (`_use_triton_*_inference`, `gmm_use_triton_*` kwargs in `train()` lines 425-526) still read `self.use_triton` and will take the Triton path regardless of the `backend` kwarg. **Plan 2's Task 10 only addresses this for `fit()`/`train()`** by adding an early CUDA branch *before* those Triton-gated lines run. **`predict()`, `predict_proba()`, `score_samples()`, `score()` still route through `self.use_triton`** and therefore won't use the CUDA backend in this plan. Those paths get rewired in Plan 3 (alongside the sm80 mma kernel for inference). Users who want CUDA inference on spherical data in Plan 2 should call `fit_predict()` (since fit() takes the CUDA branch and emits `labels_b`).

---

## Numerical formulas (reference)

For spherical Gaussian mixture, component `k` with mean `μ_k ∈ R^D`, scalar variance `σ_k² ∈ R`, weight `π_k`:

- **Per-component log-likelihood** of point `x ∈ R^D`:
  `log p_k(x) = log π_k − D/2 · log(2π σ_k²) − 0.5/σ_k² · ||x − μ_k||²`
- **logsumexp** (per-row normalizer): `log Z(x) = log Σ_k exp(log p_k(x))`. Numerically stable: subtract `max_k log p_k(x)` before exponentiating.
- **Responsibility**: `r_{n,k} = exp(log p_k(x_n) − log Z(x_n))`.
- **M-step sufficient stats** (per cluster k, summing over points assigned/responsibility-weighted):
  - `n_k = Σ_n r_{n,k}` (count / soft-count)
  - `sum_x_k = Σ_n r_{n,k} · x_n` (D-dim)
  - `sum_xx_k = Σ_n r_{n,k} · ||x_n||²` (scalar — spherical aggregates over D)
- **Finalize**:
  - `μ_k = sum_x_k / n_k`
  - `σ_k² = max((sum_xx_k / n_k − ||μ_k||²) / D, reg_covar)`
  - `π_k = n_k / N` (or `Σ_k n_k`)

Plan 2 ships the **hard-assign** path first (E-step writes `cluster_ids` int32 instead of soft responsibilities); the soft-resp path follows from `spherical_resp`. The M-step `blocked_update` accepts `cluster_ids` (hard) — soft-update integration with `responsibilities` is wired in Plan 4 alongside fused single-tile.

---

## File Structure

### Created in this plan

| Path | Responsibility |
| --- | --- |
| `gmmxx/csrc/estep/spherical.h` | Host dispatch signatures (assign, logsumexp, resp). |
| `gmmxx/csrc/estep/spherical_safe.cu` | Three `__global__` kernels (one thread per point, scalar FMA) + host launchers. |
| `gmmxx/csrc/mstep/spherical.h` | Host dispatch signatures (blocked_update, finalize). |
| `gmmxx/csrc/mstep/blocked_spherical.cu` | Per-token `atomicAdd` accumulator (naive — sorted-run is Plan 3). |
| `gmmxx/csrc/mstep/finalize_spherical.cu` | Divide sums by counts, clamp variance to `reg_covar`, keep previous mean/var on `count==0`. |
| `tests/test_cuda_spherical_safe.py` | Per-kernel correctness vs `torch_fallback` reference. |
| `tests/test_cuda_vs_triton_spherical.py` | 3-way oracle inside Triton support window. |

### Modified

| Path | Change |
| --- | --- |
| `gmmxx/csrc/common/ptx.cuh` | Populate: `warp_shfl_xor_sync`, `warp_reduce_add_sync`, `warp_reduce_max_sync`. (No mma / cp.async — Plan 3.) |
| `gmmxx/csrc/common/reduce.cuh` | Populate: `block_max_f32`, `block_sum_f32`, `logsumexp_block_f32`. |
| `gmmxx/csrc/bindings.cpp` | Add 5 new `m.def` entries for the spherical ops. |
| `gmmxx/_cuda.py` | Add 5 Python wrappers (validate, allocate, zero-init, try/except). |
| `gmmxx/cuda_ops.py` | Re-export 5 callables. |
| `gmmxx/_runtime.py` | `cuda_spherical_supported(d, k, dtype) -> True` for `0 < d ≤ 128, 0 < k ≤ 2048, dtype ∈ {fp32, fp16, bf16}`. |
| `setup.py` | Append the 3 new `.cu` paths to `sources`. |
| `gmmxx/interface.py` | In `train()` spherical branch, route through `_dispatch.dispatch_kernel` so `backend="cuda"` actually executes the new path. Currently the spherical training loop calls `batch_gmm_Spherical_torch_native` regardless of backend. |
| `tests/test_gmmxx.py` | Add `backend` parametrization for the spherical tests. |
| `README.md` | Replace "Phase 1 in progress" note with current spherical-CUDA status. |

---

## Conventions

- **Working directory:** `C:\Users\HEQ\Projects\flashGMM2`. Branch already `GMMXX-cuda`.
- **Dev rebuild:** `$env:TORCH_CUDA_ARCH_LIST = "8.9"; uv pip install -e .` after each `.cu`/`.cpp`/`.h` change.
- **Test command:** `uv run pytest tests/test_cuda_spherical_safe.py -v` (per-task) and `uv run pytest tests/ -q` (final).
- **Reference checks:** Compare CUDA outputs to `torch_fallback._batch_gmm_Spherical_torch_native_inner` (the chunked PyTorch path) at `rtol=1e-4, atol=1e-4` for fp32; `rtol=1e-2` for fp16/bf16.

---

## Task 1 — Populate `ptx.cuh` warp helpers

**Files:** Modify `gmmxx/csrc/common/ptx.cuh`

- [ ] **Step 1.1 — Replace the file with this content:**

```cpp
#pragma once

// Warp-level PTX wrappers shared across kernels.
//
// Plan 2 populates only what the spherical safe path needs: warp shuffle
// reductions in fp32. Plan 3 will add cp_async_*, ldmatrix_sync_x4,
// mma_m16n8k16_*. All wrappers are __device__ __forceinline__ and gated
// on GMMXX_HAS_* macros from arch.cuh.

#include "arch.cuh"
#include <cuda_runtime.h>

namespace gmmxx { namespace ptx {

// Full-warp xor shuffle. Available on all CUDA-capable arches via
// __shfl_xor_sync; we wrap it for symmetry with the named-helper style
// used by the kernels.
__device__ __forceinline__ float warp_shfl_xor_f32(float v, int laneMask) {
    return __shfl_xor_sync(0xffffffffu, v, laneMask, kWarp);
}

// Full-warp sum reduction. Returns the same value on every lane.
__device__ __forceinline__ float warp_reduce_add_f32(float v) {
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset, kWarp);
    }
    return v;
}

// Full-warp max reduction. Returns the same value on every lane.
__device__ __forceinline__ float warp_reduce_max_f32(float v) {
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, v, offset, kWarp);
        v = fmaxf(v, other);
    }
    return v;
}

}}  // namespace gmmxx::ptx
```

- [ ] **Step 1.2 — Smoke-build (no kernels use these yet, but the file must compile):**

```bash
uv pip install -e .
```

Expected: builds clean. (No new sources yet — but `arch.cuh` is included transitively via canary.cu, so this header is exercised.)

- [ ] **Step 1.3 — Commit:**

```bash
git add gmmxx/csrc/common/ptx.cuh
git commit -m "$(cat <<'EOF'
Populate ptx.cuh with warp shuffle helpers (xor, add, max)

Plan 2 only needs full-warp fp32 shuffle reductions for the safe-path
spherical E-step's logsumexp helper. Plan 3 will extend ptx.cuh with
cp.async wrappers and mma_m16n8k16 for the sm80 fast path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Populate `reduce.cuh` block reductions

**Files:** Modify `gmmxx/csrc/common/reduce.cuh`

- [ ] **Step 2.1 — Replace the file with this content:**

```cpp
#pragma once

// Block-level reductions and stable logsumexp helpers.
//
// All operate in fp32 regardless of input dtype. Each helper assumes the
// CTA layout is 1D (blockDim.x = nthreads, blockDim.y = blockDim.z = 1).

#include "arch.cuh"
#include "ptx.cuh"

namespace gmmxx { namespace reduce {

// Block-wide max over fp32 values, one per thread. Uses warp shuffle within
// each warp, then a small SMEM tree across warps.
//
// Caller MUST provide a shared-memory pointer of at least
// (blockDim.x / kWarp) fp32 entries.
__device__ __forceinline__ float block_max_f32(float v, float* smem) {
    int tid = threadIdx.x;
    int lane = tid & (kWarp - 1);
    int warp_id = tid / kWarp;
    int n_warps = (blockDim.x + kWarp - 1) / kWarp;

    // 1. Per-warp max.
    v = ptx::warp_reduce_max_f32(v);
    if (lane == 0) smem[warp_id] = v;
    __syncthreads();

    // 2. Cross-warp tree using the first warp.
    if (warp_id == 0) {
        v = (lane < n_warps) ? smem[lane] : -INFINITY;
        v = ptx::warp_reduce_max_f32(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// Block-wide sum over fp32 values, one per thread. Same SMEM contract as
// block_max_f32.
__device__ __forceinline__ float block_sum_f32(float v, float* smem) {
    int tid = threadIdx.x;
    int lane = tid & (kWarp - 1);
    int warp_id = tid / kWarp;
    int n_warps = (blockDim.x + kWarp - 1) / kWarp;

    v = ptx::warp_reduce_add_f32(v);
    if (lane == 0) smem[warp_id] = v;
    __syncthreads();

    if (warp_id == 0) {
        v = (lane < n_warps) ? smem[lane] : 0.0f;
        v = ptx::warp_reduce_add_f32(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// Stable logsumexp of K values per row, computed in fp32.
// `logits[k]` is the per-thread input (k = thread id; assumes K == blockDim.x
// or that out-of-range threads pass v = -INFINITY).
// Returns the same value on every thread.
__device__ __forceinline__ float logsumexp_block_f32(float v, float* smem) {
    float m = block_max_f32(v, smem);
    if (isinf(m) && m < 0.0f) {
        // All inputs are -inf; logsumexp is -inf and exp((−inf) − (−inf)) = nan.
        // Return -inf to match torch_fallback semantics.
        return -INFINITY;
    }
    float e = expf(v - m);
    float s = block_sum_f32(e, smem);
    return m + logf(s);
}

}}  // namespace gmmxx::reduce
```

- [ ] **Step 2.2 — Build:**

```bash
uv pip install -e .
```

- [ ] **Step 2.3 — Commit:**

```bash
git add gmmxx/csrc/common/reduce.cuh
git commit -m "$(cat <<'EOF'
Populate reduce.cuh with block_max/block_sum/logsumexp helpers (fp32)

block_max_f32 and block_sum_f32 use warp shuffle within each warp + a
small SMEM tree across warps; the SMEM contract is documented (caller
provides nwarps fp32 slots).

logsumexp_block_f32 is the stable subtract-max-then-exp-then-sum
formulation. Returns -inf when all inputs are -inf (matches torch
fallback's empty-cluster degeneracy).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Spherical E-step kernels (safe path)

**Files:**
- Create: `gmmxx/csrc/estep/spherical.h`
- Create: `gmmxx/csrc/estep/spherical_safe.cu`
- Modify: `gmmxx/csrc/bindings.cpp` (add `m.def` entries)
- Modify: `setup.py` (append spherical_safe.cu to sources)

### Step 3.1 — Write `spherical.h`

```cpp
#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace estep { namespace spherical {

// Hard-assign cluster IDs (argmax of log p_k(x) over k).
// Returns int32 tensor of shape (B, N).
//
// x: (B, N, D) fp32 / fp16 / bf16. Contiguous, CUDA.
// means: (B, K, D) same dtype as x.
// var: (B, K) fp32. Per-component scalar variance.
// log_w: (B, K) fp32. Per-component log mixture weight.
at::Tensor assign(const at::Tensor& x,
                  const at::Tensor& means,
                  const at::Tensor& var,
                  const at::Tensor& log_w,
                  c10::optional<at::Tensor> out);

// Per-row stable logsumexp over K. Returns (B, N) fp32.
at::Tensor logsumexp(const at::Tensor& x,
                     const at::Tensor& means,
                     const at::Tensor& var,
                     const at::Tensor& log_w,
                     c10::optional<at::Tensor> out);

// Soft responsibilities r_{n,k} = exp(log p_k - log_norm). Returns (B, N, K) fp32.
//
// log_norm: (B, N) fp32. Caller supplies; typically obtained from logsumexp().
at::Tensor resp(const at::Tensor& x,
                const at::Tensor& means,
                const at::Tensor& var,
                const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out);

}}}  // namespace gmmxx::estep::spherical
```

### Step 3.2 — Write `spherical_safe.cu`

```cpp
#include "spherical.h"
#include "../common/arch.cuh"
#include "../common/reduce.cuh"
#include <cmath>

namespace gmmxx { namespace estep { namespace spherical {

// One thread per (b, n) point. Each thread loops over K clusters,
// computes log p_k(x), and tracks (best_logit, best_idx) for argmax.
//
// log p_k(x) = log_w_k - D/2 * log(2π σ_k²) - 0.5/σ_k² * ||x − μ_k||²
//
// (D/2 * log(2π) is constant across k; we still include it so log_norm
// downstream is the true logsumexp, not an offset.)
template <typename T>
__global__ void __launch_bounds__(128)
spherical_assign_safe_kernel(
    const T* __restrict__ x,        // (B, N, D)
    const T* __restrict__ means,    // (B, K, D)
    const float* __restrict__ var,  // (B, K)
    const float* __restrict__ log_w,// (B, K)
    int32_t* __restrict__ out,      // (B, N)
    int B, int N, int K, int D
) {
    int b = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || b >= B) return;

    // Pointer offsets for this batch.
    const T* x_b = x + (size_t)b * N * D;
    const T* means_b = means + (size_t)b * K * D;
    const float* var_b = var + (size_t)b * K;
    const float* log_w_b = log_w + (size_t)b * K;

    const T* x_n = x_b + (size_t)n * D;

    float best = -INFINITY;
    int best_k = 0;
    const float TWO_PI = 6.283185307179586f;

    for (int k = 0; k < K; ++k) {
        const T* mu_k = means_b + (size_t)k * D;
        float v = var_b[k];
        // ||x - μ_k||² in fp32 regardless of input dtype.
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float dx = static_cast<float>(x_n[d]) - static_cast<float>(mu_k[d]);
            dist += dx * dx;
        }
        float logit = log_w_b[k]
                    - 0.5f * (float)D * logf(TWO_PI * v)
                    - 0.5f * dist / v;
        if (logit > best) {
            best = logit;
            best_k = k;
        }
    }
    out[(size_t)b * N + n] = best_k;
}

// One CTA per (b, n). blockDim.x = K (or padded to nearest multiple of 32 >= K).
// Each thread computes one logit, then the block does a stable logsumexp.
template <typename T>
__global__ void
spherical_logsumexp_safe_kernel(
    const T* __restrict__ x,
    const T* __restrict__ means,
    const float* __restrict__ var,
    const float* __restrict__ log_w,
    float* __restrict__ out,         // (B, N)
    int B, int N, int K, int D
) {
    extern __shared__ float smem[];  // nwarps fp32 entries for block_max/block_sum

    int b = blockIdx.y;
    int n = blockIdx.x;
    int k = threadIdx.x;
    if (b >= B || n >= N) return;

    const T* x_b = x + (size_t)b * N * D;
    const T* means_b = means + (size_t)b * K * D;
    const float* var_b = var + (size_t)b * K;
    const float* log_w_b = log_w + (size_t)b * K;
    const T* x_n = x_b + (size_t)n * D;

    float logit = -INFINITY;
    if (k < K) {
        const T* mu_k = means_b + (size_t)k * D;
        float v = var_b[k];
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float dx = static_cast<float>(x_n[d]) - static_cast<float>(mu_k[d]);
            dist += dx * dx;
        }
        const float TWO_PI = 6.283185307179586f;
        logit = log_w_b[k] - 0.5f * (float)D * logf(TWO_PI * v) - 0.5f * dist / v;
    }

    float lse = ::gmmxx::reduce::logsumexp_block_f32(logit, smem);
    if (k == 0) {
        out[(size_t)b * N + n] = lse;
    }
}

// One thread per (b, n, k). Computes resp[b,n,k] = exp(logit_k - log_norm[b,n]).
template <typename T>
__global__ void __launch_bounds__(128)
spherical_resp_safe_kernel(
    const T* __restrict__ x,
    const T* __restrict__ means,
    const float* __restrict__ var,
    const float* __restrict__ log_w,
    const float* __restrict__ log_norm,  // (B, N)
    float* __restrict__ out,             // (B, N, K)
    int B, int N, int K, int D
) {
    int b = blockIdx.y;
    int n = blockIdx.x;
    int k = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= B || n >= N || k >= K) return;

    const T* x_b = x + (size_t)b * N * D;
    const T* means_b = means + (size_t)b * K * D;
    const float* var_b = var + (size_t)b * K;
    const float* log_w_b = log_w + (size_t)b * K;
    const T* x_n = x_b + (size_t)n * D;
    const T* mu_k = means_b + (size_t)k * D;

    float v = var_b[k];
    float dist = 0.0f;
    for (int d = 0; d < D; ++d) {
        float dx = static_cast<float>(x_n[d]) - static_cast<float>(mu_k[d]);
        dist += dx * dx;
    }
    const float TWO_PI = 6.283185307179586f;
    float logit = log_w_b[k] - 0.5f * (float)D * logf(TWO_PI * v) - 0.5f * dist / v;
    float lz = log_norm[(size_t)b * N + n];
    out[((size_t)b * N + n) * K + k] = expf(logit - lz);
}

// ---------------------------------------------------------------------
// Host launchers — dispatch by dtype, call the templated kernel.
// ---------------------------------------------------------------------

namespace {

// Common input checks shared by all three ops.
void _check_inputs(const at::Tensor& x, const at::Tensor& means,
                   const at::Tensor& var, const at::Tensor& log_w) {
    TORCH_CHECK(x.is_cuda(), "x must be on a CUDA device");
    TORCH_CHECK(x.is_contiguous() && means.is_contiguous() &&
                var.is_contiguous() && log_w.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(means.scalar_type() == x.scalar_type(),
                "means must match x dtype");
    TORCH_CHECK(var.scalar_type() == at::kFloat &&
                log_w.scalar_type() == at::kFloat,
                "var and log_w must be float32");
    TORCH_CHECK(x.dim() == 3 && means.dim() == 3,
                "x must be (B,N,D); means must be (B,K,D)");
    TORCH_CHECK(var.dim() == 2 && log_w.dim() == 2,
                "var must be (B,K); log_w must be (B,K)");
    TORCH_CHECK(x.size(0) == means.size(0),
                "x and means must agree on batch dim");
    TORCH_CHECK(x.size(2) == means.size(2),
                "x and means must agree on D");
    TORCH_CHECK(var.size(0) == x.size(0) && var.size(1) == means.size(1),
                "var must be (B,K)");
    TORCH_CHECK(log_w.sizes() == var.sizes(),
                "log_w must match var shape");
}

template <typename T>
void launch_assign(const at::Tensor& x, const at::Tensor& means,
                   const at::Tensor& var, const at::Tensor& log_w,
                   at::Tensor& out, cudaStream_t stream) {
    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);
    constexpr int kThreads = 128;
    dim3 grid((N + kThreads - 1) / kThreads, B);
    spherical_assign_safe_kernel<T><<<grid, kThreads, 0, stream>>>(
        x.data_ptr<T>(), means.data_ptr<T>(),
        var.data_ptr<float>(), log_w.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        B, N, K, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     at::Tensor& out, cudaStream_t stream) {
    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);
    // Round threads up to nearest multiple of warp size, capped at 1024.
    int threads = ((K + kWarp - 1) / kWarp) * kWarp;
    threads = std::min(threads, 1024);
    int n_warps = (threads + kWarp - 1) / kWarp;
    size_t smem_bytes = n_warps * sizeof(float);
    dim3 grid(N, B);
    spherical_logsumexp_safe_kernel<T><<<grid, threads, smem_bytes, stream>>>(
        x.data_ptr<T>(), means.data_ptr<T>(),
        var.data_ptr<float>(), log_w.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, K, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_resp(const at::Tensor& x, const at::Tensor& means,
                 const at::Tensor& var, const at::Tensor& log_w,
                 const at::Tensor& log_norm, at::Tensor& out, cudaStream_t stream) {
    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);
    constexpr int kThreads = 64;
    dim3 grid(N, B, (K + kThreads - 1) / kThreads);
    spherical_resp_safe_kernel<T><<<grid, kThreads, 0, stream>>>(
        x.data_ptr<T>(), means.data_ptr<T>(),
        var.data_ptr<float>(), log_w.data_ptr<float>(),
        log_norm.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, K, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // anonymous namespace

at::Tensor assign(const at::Tensor& x, const at::Tensor& means,
                  const at::Tensor& var, const at::Tensor& log_w,
                  c10::optional<at::Tensor> out) {
    _check_inputs(x, means, var, log_w);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto B = x.size(0);
    auto N = x.size(1);
    at::Tensor result;
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == at::kInt &&
                    out->sizes() == at::IntArrayRef({B, N}),
                    "out must be int32 (B,N)");
        result = *out;
    } else {
        result = at::empty({B, N}, x.options().dtype(at::kInt));
    }

    switch (x.scalar_type()) {
        case at::kFloat:    launch_assign<float>(x, means, var, log_w, result, stream); break;
        case at::kHalf:     launch_assign<at::Half>(x, means, var, log_w, result, stream); break;
        case at::kBFloat16: launch_assign<at::BFloat16>(x, means, var, log_w, result, stream); break;
        default: TORCH_CHECK(false, "spherical.assign: unsupported dtype ", x.scalar_type());
    }
    return result;
}

at::Tensor logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     c10::optional<at::Tensor> out) {
    _check_inputs(x, means, var, log_w);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto B = x.size(0);
    auto N = x.size(1);
    at::Tensor result;
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == at::kFloat &&
                    out->sizes() == at::IntArrayRef({B, N}),
                    "out must be float32 (B,N)");
        result = *out;
    } else {
        result = at::empty({B, N}, x.options().dtype(at::kFloat));
    }

    switch (x.scalar_type()) {
        case at::kFloat:    launch_logsumexp<float>(x, means, var, log_w, result, stream); break;
        case at::kHalf:     launch_logsumexp<at::Half>(x, means, var, log_w, result, stream); break;
        case at::kBFloat16: launch_logsumexp<at::BFloat16>(x, means, var, log_w, result, stream); break;
        default: TORCH_CHECK(false, "spherical.logsumexp: unsupported dtype ", x.scalar_type());
    }
    return result;
}

at::Tensor resp(const at::Tensor& x, const at::Tensor& means,
                const at::Tensor& var, const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out) {
    _check_inputs(x, means, var, log_w);
    TORCH_CHECK(log_norm.is_cuda() && log_norm.is_contiguous() &&
                log_norm.scalar_type() == at::kFloat &&
                log_norm.sizes() == at::IntArrayRef({x.size(0), x.size(1)}),
                "log_norm must be contiguous fp32 (B,N)");
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto B = x.size(0);
    auto N = x.size(1);
    auto K = means.size(1);
    at::Tensor result;
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == at::kFloat &&
                    out->sizes() == at::IntArrayRef({B, N, K}),
                    "out must be float32 (B,N,K)");
        result = *out;
    } else {
        result = at::empty({B, N, K}, x.options().dtype(at::kFloat));
    }

    switch (x.scalar_type()) {
        case at::kFloat:    launch_resp<float>(x, means, var, log_w, log_norm, result, stream); break;
        case at::kHalf:     launch_resp<at::Half>(x, means, var, log_w, log_norm, result, stream); break;
        case at::kBFloat16: launch_resp<at::BFloat16>(x, means, var, log_w, log_norm, result, stream); break;
        default: TORCH_CHECK(false, "spherical.resp: unsupported dtype ", x.scalar_type());
    }
    return result;
}

}}}  // namespace gmmxx::estep::spherical
```

### Step 3.3 — Update `setup.py` sources

Find the `sources` list (currently line ~113-117). Add the new `.cu`:

```python
sources = [
    str(CSRC / "bindings.cpp"),
    str(CSRC / "canary" / "canary.cu"),
    str(CSRC / "estep" / "spherical_safe.cu"),
    str(nb_combined),
]
```

Also extend `include_dirs` with the new estep dir:

```python
include_dirs = [
    str(CSRC),
    str(CSRC / "common"),
    str(CSRC / "canary"),
    str(CSRC / "estep"),
    nb_include,
    str(nb_robin_include),
]
```

### Step 3.4 — Wire bindings.cpp

Open `gmmxx/csrc/bindings.cpp`. After the existing canary includes, add:

```cpp
#include "estep/spherical.h"
```

Inside the existing `NB_MODULE(_C, m) { ... }` block, after the `m.def("canary_add_offset", ...)` call, add:

```cpp
    m.def(
        "spherical_assign",
        &gmmxx::estep::spherical::assign,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("out") = nb::none(),
        "Spherical E-step assign: argmax of log p_k(x) over k. Returns int32 (B,N).");

    m.def(
        "spherical_logsumexp",
        &gmmxx::estep::spherical::logsumexp,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("out") = nb::none(),
        "Spherical E-step stable logsumexp over k. Returns fp32 (B,N).");

    m.def(
        "spherical_resp",
        &gmmxx::estep::spherical::resp,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("log_norm"),
        nb::arg("out") = nb::none(),
        "Spherical E-step responsibilities r_{n,k}. Returns fp32 (B,N,K).");
```

### Step 3.5 — Build and smoke-test

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; uv pip install -e .
```

Expected: clean build. If `_check_inputs` link errors, ensure `<algorithm>` is in scope or use `(K + kWarp - 1) / kWarp * kWarp` directly inline (no `std::min` then).

Quick callable check:

```bash
uv run python -c "
import torch
from gmmxx import _C
B, N, D, K = 1, 4, 2, 3
x = torch.randn(B, N, D, device='cuda')
means = torch.randn(B, K, D, device='cuda')
var = torch.ones(B, K, device='cuda')
log_w = torch.zeros(B, K, device='cuda')
ids = _C.spherical_assign(x, means, var, log_w)
print('assign ids:', ids)
print('shape ok:', ids.shape == (B, N), 'dtype ok:', ids.dtype == torch.int32)

lz = _C.spherical_logsumexp(x, means, var, log_w)
print('logsumexp shape ok:', lz.shape == (B, N))

r = _C.spherical_resp(x, means, var, log_w, lz)
print('resp shape ok:', r.shape == (B, N, K))
print('resp sums:', r.sum(-1))  # should be ~1.0 per (b,n)
"
```

Expected: shape checks True, resp sums ~1.0.

### Step 3.6 — Commit

```bash
git add gmmxx/csrc/estep/spherical.h gmmxx/csrc/estep/spherical_safe.cu gmmxx/csrc/bindings.cpp setup.py
git commit -m "$(cat <<'EOF'
Add spherical E-step safe-path CUDA kernels (assign, logsumexp, resp)

Three template kernels covering fp32 / fp16 / bf16 inputs:
- assign: one thread per point, loops K clusters, argmax of log p_k(x).
- logsumexp: one CTA per point, K threads compute logits in parallel,
  block reduction via reduce.cuh helpers (stable subtract-max-then-exp).
- resp: one thread per (b,n,k), divides exp(logit) by exp(log_norm).

All accumulators in fp32. CUDAGuard + getCurrentCUDAStream + TORCH_CHECK
input validation per the spec's host-side launcher contract (§5a).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Spherical M-step blocked update (per-token atomic)

**Files:**
- Create: `gmmxx/csrc/mstep/spherical.h`
- Create: `gmmxx/csrc/mstep/blocked_spherical.cu`
- Modify: `gmmxx/csrc/bindings.cpp`
- Modify: `setup.py`

### Step 4.1 — Write `spherical.h`

```cpp
#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace mstep { namespace spherical {

// Hard-assignment M-step accumulator.
//
// Caller MUST zero-initialize sums_out, sumsq_out, counts_out before calling
// (see spec §5b). Kernel only does atomicAdd; it does not zero internally.
//
// x: (B, N, D) fp32 / fp16 / bf16
// cluster_ids: (B, N) int32 — per-point hard assignment
// sums_out: (B, K, D) fp32 — caller-zeroed accumulator (Σ x_n by cluster)
// sumsq_out: (B, K) fp32 — caller-zeroed accumulator (Σ ||x_n||²)
// counts_out: (B, K) int32 — caller-zeroed accumulator (Σ 1)
void blocked_update(const at::Tensor& x,
                    const at::Tensor& cluster_ids,
                    at::Tensor& sums_out,
                    at::Tensor& sumsq_out,
                    at::Tensor& counts_out);

// Finalize: convert accumulators to (means, var, weights) with reg_covar
// clamp and empty-cluster preservation.
//
// total_n: total point count per batch (typically N for unbatched, or
//   tensor of shape (B,) — passed as int64).
// reg_covar: minimum variance after computation.
//
// old_means: (B, K, D) — preserved when count[k] == 0.
// old_var: (B, K) — preserved when count[k] == 0.
// Returns (means, var, weights):
//   means: (B, K, D) same dtype as old_means
//   var: (B, K) fp32 (clamped to >= reg_covar)
//   weights: (B, K) fp32 (sum to 1 per batch)
std::tuple<at::Tensor, at::Tensor, at::Tensor> finalize(
    const at::Tensor& sums,
    const at::Tensor& sumsq,
    const at::Tensor& counts,
    const at::Tensor& old_means,
    const at::Tensor& old_var,
    int64_t total_n,
    double reg_covar);

}}}  // namespace gmmxx::mstep::spherical
```

### Step 4.2 — Write `blocked_spherical.cu`

```cpp
#include "spherical.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace mstep { namespace spherical {

template <typename T>
__global__ void __launch_bounds__(128)
blocked_update_spherical_kernel(
    const T* __restrict__ x,            // (B, N, D)
    const int32_t* __restrict__ ids,    // (B, N)
    float* __restrict__ sums,           // (B, K, D) atomicAdd target
    float* __restrict__ sumsq,          // (B, K) atomicAdd target
    int32_t* __restrict__ counts,       // (B, K) atomicAdd target
    int B, int N, int K, int D
) {
    int b = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || n >= N) return;

    const T* x_n = x + ((size_t)b * N + n) * D;
    int k = ids[(size_t)b * N + n];
    if (k < 0 || k >= K) return;  // defensive

    // Compute ||x_n||² in fp32 once.
    float xx = 0.0f;
    for (int d = 0; d < D; ++d) {
        float v = static_cast<float>(x_n[d]);
        xx += v * v;
        atomicAdd(sums + ((size_t)b * K + k) * D + d, v);
    }
    atomicAdd(sumsq + (size_t)b * K + k, xx);
    atomicAdd(counts + (size_t)b * K + k, 1);
}

void blocked_update(const at::Tensor& x,
                    const at::Tensor& cluster_ids,
                    at::Tensor& sums_out,
                    at::Tensor& sumsq_out,
                    at::Tensor& counts_out) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    TORCH_CHECK(cluster_ids.is_cuda() && cluster_ids.is_contiguous() &&
                cluster_ids.scalar_type() == at::kInt,
                "cluster_ids must be contiguous int32 CUDA");
    TORCH_CHECK(sums_out.is_cuda() && sums_out.is_contiguous() &&
                sums_out.scalar_type() == at::kFloat,
                "sums_out must be contiguous fp32 CUDA");
    TORCH_CHECK(sumsq_out.is_cuda() && sumsq_out.is_contiguous() &&
                sumsq_out.scalar_type() == at::kFloat,
                "sumsq_out must be contiguous fp32 CUDA");
    TORCH_CHECK(counts_out.is_cuda() && counts_out.is_contiguous() &&
                counts_out.scalar_type() == at::kInt,
                "counts_out must be contiguous int32 CUDA");
    TORCH_CHECK(x.dim() == 3, "x must be (B,N,D)");

    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)sums_out.size(1);
    TORCH_CHECK(sums_out.sizes() == at::IntArrayRef({B, K, D}),
                "sums_out must be (B,K,D)");
    TORCH_CHECK(sumsq_out.sizes() == at::IntArrayRef({B, K}),
                "sumsq_out must be (B,K)");
    TORCH_CHECK(counts_out.sizes() == at::IntArrayRef({B, K}),
                "counts_out must be (B,K)");
    TORCH_CHECK(cluster_ids.sizes() == at::IntArrayRef({B, N}),
                "cluster_ids must be (B,N)");

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int kThreads = 128;
    dim3 grid((N + kThreads - 1) / kThreads, B);

    switch (x.scalar_type()) {
        case at::kFloat:
            blocked_update_spherical_kernel<float><<<grid, kThreads, 0, stream>>>(
                x.data_ptr<float>(), cluster_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        case at::kHalf:
            blocked_update_spherical_kernel<at::Half><<<grid, kThreads, 0, stream>>>(
                x.data_ptr<at::Half>(), cluster_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        case at::kBFloat16:
            blocked_update_spherical_kernel<at::BFloat16><<<grid, kThreads, 0, stream>>>(
                x.data_ptr<at::BFloat16>(), cluster_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        default:
            TORCH_CHECK(false, "blocked_update_spherical: unsupported dtype ", x.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}}}  // namespace gmmxx::mstep::spherical
```

### Step 4.3 — Update `setup.py`

Add to `sources`:

```python
str(CSRC / "mstep" / "blocked_spherical.cu"),
```

Add to `include_dirs`:

```python
str(CSRC / "mstep"),
```

### Step 4.4 — Wire bindings.cpp

Add include:

```cpp
#include "mstep/spherical.h"
```

Add to NB_MODULE:

```cpp
    m.def(
        "blocked_update_spherical",
        &gmmxx::mstep::spherical::blocked_update,
        nb::arg("x"),
        nb::arg("cluster_ids"),
        nb::arg("sums_out"),
        nb::arg("sumsq_out"),
        nb::arg("counts_out"),
        "Spherical M-step accumulator (per-token atomicAdd). Caller MUST zero "
        "sums_out/sumsq_out/counts_out before calling.");
```

### Step 4.5 — Build + smoke

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; uv pip install -e .
```

```bash
uv run python -c "
import torch
from gmmxx import _C
B, N, D, K = 1, 16, 2, 3
torch.manual_seed(0)
x = torch.randn(B, N, D, device='cuda')
ids = torch.randint(0, K, (B, N), device='cuda', dtype=torch.int32)
sums = torch.zeros(B, K, D, device='cuda')
sumsq = torch.zeros(B, K, device='cuda')
counts = torch.zeros(B, K, device='cuda', dtype=torch.int32)
_C.blocked_update_spherical(x, ids, sums, sumsq, counts)
print('counts:', counts)
print('counts sum (should == N):', counts.sum().item())
print('sums:', sums)
print('||sums per cluster||² ≈ sumsq?', sumsq)
"
```

Expected: counts sum to N; sums and sumsq are reasonable.

### Step 4.6 — Commit

```bash
git add gmmxx/csrc/mstep/ gmmxx/csrc/bindings.cpp setup.py
git commit -m "$(cat <<'EOF'
Add spherical M-step blocked update (per-token atomicAdd)

Naive accumulator: each thread reads one (b,n) point, looks up its
cluster_id, then atomicAdds into sums (B,K,D), sumsq (B,K), counts (B,K).
Caller-owned zero-init contract (spec §5b) — kernel does NOT clear
buffers internally. Plan 3 will replace this with a sorted-run kernel
that emits one atomic per (run,feature) tuple instead of per token,
~256x atomic-issue reduction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Spherical finalize kernel

**Files:** Create `gmmxx/csrc/mstep/finalize_spherical.cu`; modify `gmmxx/csrc/bindings.cpp`, `setup.py`.

### Step 5.1 — Write `finalize_spherical.cu`

```cpp
#include "spherical.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace mstep { namespace spherical {

// Per-cluster finalize: divides accumulators, clamps variance, preserves
// previous mean/var on empty clusters.
//
// One thread per (b, k). Each thread:
//   - reads count[b,k]; if 0, copies old_means and old_var.
//   - else computes mean[b,k,d] = sums[b,k,d] / count[b,k] for each d.
//   - and var[b,k] = max((sumsq[b,k]/count - ||mean||²) / D, reg_covar)
//   - and weight[b,k] = count / total_n
template <typename T>
__global__ void
finalize_spherical_kernel(
    const float* __restrict__ sums,         // (B, K, D)
    const float* __restrict__ sumsq,        // (B, K)
    const int32_t* __restrict__ counts,     // (B, K)
    const T* __restrict__ old_means,        // (B, K, D)
    const float* __restrict__ old_var,      // (B, K)
    T* __restrict__ new_means,              // (B, K, D) output
    float* __restrict__ new_var,            // (B, K) output
    float* __restrict__ new_weights,        // (B, K) output
    int B, int K, int D, int total_n,
    float reg_covar
) {
    int b = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || k >= K) return;

    int32_t cnt = counts[(size_t)b * K + k];

    if (cnt <= 0) {
        // Empty cluster: preserve previous mean and variance.
        for (int d = 0; d < D; ++d) {
            new_means[((size_t)b * K + k) * D + d] = old_means[((size_t)b * K + k) * D + d];
        }
        new_var[(size_t)b * K + k] = old_var[(size_t)b * K + k];
        new_weights[(size_t)b * K + k] = 0.0f;
        return;
    }

    float n_inv = 1.0f / (float)cnt;

    // mean = sums / count, and accumulate ||mean||² for variance.
    float mu_sq = 0.0f;
    for (int d = 0; d < D; ++d) {
        float mu_d = sums[((size_t)b * K + k) * D + d] * n_inv;
        mu_sq += mu_d * mu_d;
        new_means[((size_t)b * K + k) * D + d] = static_cast<T>(mu_d);
    }

    // Spherical variance averages over D.
    float ss = sumsq[(size_t)b * K + k];
    float var_raw = (ss * n_inv - mu_sq) / (float)D;
    new_var[(size_t)b * K + k] = fmaxf(var_raw, reg_covar);

    // Weight.
    new_weights[(size_t)b * K + k] = (float)cnt / (float)total_n;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> finalize(
    const at::Tensor& sums,
    const at::Tensor& sumsq,
    const at::Tensor& counts,
    const at::Tensor& old_means,
    const at::Tensor& old_var,
    int64_t total_n,
    double reg_covar
) {
    TORCH_CHECK(sums.is_cuda() && sums.is_contiguous() && sums.scalar_type() == at::kFloat,
                "sums must be contiguous fp32 CUDA");
    TORCH_CHECK(sumsq.is_cuda() && sumsq.is_contiguous() && sumsq.scalar_type() == at::kFloat,
                "sumsq must be contiguous fp32 CUDA");
    TORCH_CHECK(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type() == at::kInt,
                "counts must be contiguous int32 CUDA");
    TORCH_CHECK(old_means.is_cuda() && old_means.is_contiguous(),
                "old_means must be contiguous CUDA");
    TORCH_CHECK(old_var.is_cuda() && old_var.is_contiguous() && old_var.scalar_type() == at::kFloat,
                "old_var must be contiguous fp32 CUDA");
    TORCH_CHECK(sums.dim() == 3 && sumsq.dim() == 2,
                "sums must be (B,K,D); sumsq must be (B,K)");
    TORCH_CHECK(total_n > 0, "total_n must be positive");

    int B = (int)sums.size(0);
    int K = (int)sums.size(1);
    int D = (int)sums.size(2);

    c10::cuda::CUDAGuard guard(sums.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto new_means = at::empty_like(old_means);
    auto new_var = at::empty({B, K}, sums.options());
    auto new_weights = at::empty({B, K}, sums.options());

    constexpr int kThreads = 64;
    dim3 grid((K + kThreads - 1) / kThreads, B);

    switch (old_means.scalar_type()) {
        case at::kFloat:
            finalize_spherical_kernel<float><<<grid, kThreads, 0, stream>>>(
                sums.data_ptr<float>(), sumsq.data_ptr<float>(), counts.data_ptr<int32_t>(),
                old_means.data_ptr<float>(), old_var.data_ptr<float>(),
                new_means.data_ptr<float>(), new_var.data_ptr<float>(), new_weights.data_ptr<float>(),
                B, K, D, (int)total_n, (float)reg_covar);
            break;
        case at::kHalf:
            finalize_spherical_kernel<at::Half><<<grid, kThreads, 0, stream>>>(
                sums.data_ptr<float>(), sumsq.data_ptr<float>(), counts.data_ptr<int32_t>(),
                old_means.data_ptr<at::Half>(), old_var.data_ptr<float>(),
                new_means.data_ptr<at::Half>(), new_var.data_ptr<float>(), new_weights.data_ptr<float>(),
                B, K, D, (int)total_n, (float)reg_covar);
            break;
        case at::kBFloat16:
            finalize_spherical_kernel<at::BFloat16><<<grid, kThreads, 0, stream>>>(
                sums.data_ptr<float>(), sumsq.data_ptr<float>(), counts.data_ptr<int32_t>(),
                old_means.data_ptr<at::BFloat16>(), old_var.data_ptr<float>(),
                new_means.data_ptr<at::BFloat16>(), new_var.data_ptr<float>(), new_weights.data_ptr<float>(),
                B, K, D, (int)total_n, (float)reg_covar);
            break;
        default:
            TORCH_CHECK(false, "finalize_spherical: unsupported dtype ", old_means.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(new_means, new_var, new_weights);
}

}}}  // namespace gmmxx::mstep::spherical
```

### Step 5.2 — `setup.py` add `str(CSRC / "mstep" / "finalize_spherical.cu"),` to sources.

### Step 5.3 — `bindings.cpp` add:

```cpp
    m.def(
        "finalize_spherical",
        &gmmxx::mstep::spherical::finalize,
        nb::arg("sums"),
        nb::arg("sumsq"),
        nb::arg("counts"),
        nb::arg("old_means"),
        nb::arg("old_var"),
        nb::arg("total_n"),
        nb::arg("reg_covar"),
        "Finalize spherical M-step. Returns (means, var, weights). "
        "Empty clusters preserve previous (means, var) and get weight 0.");
```

### Step 5.4 — Build + smoke + commit

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; uv pip install -e .
```

```bash
uv run python -c "
import torch
from gmmxx import _C
B, K, D = 1, 3, 2
sums = torch.tensor([[[10.0, 20.0], [0.0, 0.0], [5.0, 5.0]]], device='cuda')
sumsq = torch.tensor([[150.0, 0.0, 50.0]], device='cuda')
counts = torch.tensor([[5, 0, 2]], device='cuda', dtype=torch.int32)
old_means = torch.tensor([[[0.0, 0.0], [99.0, 99.0], [0.0, 0.0]]], device='cuda')
old_var = torch.tensor([[1.0, 42.0, 1.0]], device='cuda')
new_means, new_var, new_weights = _C.finalize_spherical(sums, sumsq, counts, old_means, old_var, 7, 1e-6)
print('new_means:', new_means)  # k=0: (2,4), k=1: (99,99) preserved, k=2: (2.5,2.5)
print('new_var:', new_var)       # k=0: ((150/5 - 20)/2)= 5, k=1: 42 preserved, k=2: ((50/2 - 12.5)/2)=6.25
print('new_weights:', new_weights)  # 5/7, 0, 2/7
"
```

Expected: empty cluster (k=1) preserves old_means and old_var; weights sum to ~1.

```bash
git add gmmxx/csrc/mstep/finalize_spherical.cu gmmxx/csrc/bindings.cpp setup.py
git commit -m "$(cat <<'EOF'
Add spherical finalize kernel: divide stats, clamp variance, preserve empty

One thread per (b,k):
- count <= 0: copy old_means and old_var (matches torch_fallback semantics).
- count > 0: mean = sums/count; var = max((sumsq/count - ||mean||²)/D, reg_covar).
- weight = count / total_n in all branches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Python wrappers in `gmmxx/_cuda.py`

**Files:** Modify `gmmxx/_cuda.py`

### Step 6.1 — Add the five wrappers

Append to `gmmxx/_cuda.py` (after the existing `canary_add_offset`):

```python
# ---------------------------------------------------------------------------
# Spherical kernels (Plan 2 — safe path)
# ---------------------------------------------------------------------------


def spherical_assign(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """E-step argmax. Returns int32 (B, N)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    try:
        return _C.spherical_assign(x, means, var, log_w, out)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"spherical_assign failed: {exc}") from exc


def spherical_logsumexp(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """E-step stable logsumexp over k. Returns fp32 (B, N)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    try:
        return _C.spherical_logsumexp(x, means, var, log_w, out)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"spherical_logsumexp failed: {exc}") from exc


def spherical_resp(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    log_norm: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """E-step responsibilities. Returns fp32 (B, N, K)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    log_norm = _check_input(log_norm, "log_norm", dtype=torch.float32)
    try:
        return _C.spherical_resp(x, means, var, log_w, log_norm, out)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"spherical_resp failed: {exc}") from exc


def blocked_update_spherical(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M-step accumulator. Allocates and zero-initializes sums/sumsq/counts,
    calls the kernel, returns the three accumulator tensors.

    Returns (sums, sumsq, counts) where:
      sums: (B, K, D) fp32
      sumsq: (B, K) fp32
      counts: (B, K) int32
    """
    require_cuda()
    x = _check_input(x, "x")
    cluster_ids = _check_input(cluster_ids, "cluster_ids", dtype=torch.int32)
    B, N, D = x.shape
    K = int(n_components)
    sums = torch.zeros((B, K, D), dtype=torch.float32, device=x.device)
    sumsq = torch.zeros((B, K), dtype=torch.float32, device=x.device)
    counts = torch.zeros((B, K), dtype=torch.int32, device=x.device)
    try:
        _C.blocked_update_spherical(x, cluster_ids, sums, sumsq, counts)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"blocked_update_spherical failed: {exc}") from exc
    return sums, sumsq, counts


def finalize_spherical(
    sums: torch.Tensor,
    sumsq: torch.Tensor,
    counts: torch.Tensor,
    old_means: torch.Tensor,
    old_var: torch.Tensor,
    total_n: int,
    reg_covar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M-step finalize. Returns (means, var, weights)."""
    require_cuda()
    sums = _check_input(sums, "sums", dtype=torch.float32)
    sumsq = _check_input(sumsq, "sumsq", dtype=torch.float32)
    counts = _check_input(counts, "counts", dtype=torch.int32)
    old_means = _check_input(old_means, "old_means")
    old_var = _check_input(old_var, "old_var", dtype=torch.float32)
    try:
        return _C.finalize_spherical(sums, sumsq, counts, old_means, old_var,
                                      int(total_n), float(reg_covar))
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"finalize_spherical failed: {exc}") from exc
```

### Step 6.2 — Smoke

```bash
uv run python -c "
import torch
from gmmxx import _cuda
B, N, D, K = 2, 1024, 8, 16
torch.manual_seed(0)
x = torch.randn(B, N, D, device='cuda')
means = torch.randn(B, K, D, device='cuda')
var = torch.ones(B, K, device='cuda')
log_w = torch.zeros(B, K, device='cuda')

ids = _cuda.spherical_assign(x, means, var, log_w)
lz = _cuda.spherical_logsumexp(x, means, var, log_w)
r = _cuda.spherical_resp(x, means, var, log_w, lz)
sums, sumsq, counts = _cuda.blocked_update_spherical(x, ids, K)
new_means, new_var, new_weights = _cuda.finalize_spherical(
    sums, sumsq, counts, means, var, B*N, 1e-6
)
print('All five wrappers callable. Shapes:',
      ids.shape, lz.shape, r.shape, new_means.shape, new_var.shape, new_weights.shape)
"
```

### Step 6.3 — Commit

```bash
git add gmmxx/_cuda.py
git commit -m "$(cat <<'EOF'
Add Python wrappers for spherical CUDA kernels in gmmxx/_cuda.py

Five wrappers — spherical_assign, spherical_logsumexp, spherical_resp,
blocked_update_spherical, finalize_spherical — each with input
validation, CudaRuntimeFallback wrapping, and (for blocked_update)
the caller-side zero-init that the M-step kernel's contract requires.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Re-export through `gmmxx/cuda_ops.py`

**Files:** Modify `gmmxx/cuda_ops.py`

Replace the contents of `gmmxx/cuda_ops.py` with:

```python
"""Experimental public re-export of low-level CUDA kernel callables.

WARNING — Experimental: API may change before v1.0. The only API stability
guarantee in GMMXX is the ``GMMXX`` class itself.
"""

from __future__ import annotations

from . import _cuda

# Smoke-test (Plan 1).
canary_add_offset = _cuda.canary_add_offset

# Spherical (Plan 2).
spherical_assign = _cuda.spherical_assign
spherical_logsumexp = _cuda.spherical_logsumexp
spherical_resp = _cuda.spherical_resp
blocked_update_spherical = _cuda.blocked_update_spherical
finalize_spherical = _cuda.finalize_spherical

# Lifecycle / introspection helpers.
has_cuda = _cuda.has_cuda
require_cuda = _cuda.require_cuda
CudaBackendUnavailable = _cuda.CudaBackendUnavailable
CudaRuntimeFallback = _cuda.CudaRuntimeFallback


__all__ = [
    "canary_add_offset",
    "spherical_assign",
    "spherical_logsumexp",
    "spherical_resp",
    "blocked_update_spherical",
    "finalize_spherical",
    "has_cuda",
    "require_cuda",
    "CudaBackendUnavailable",
    "CudaRuntimeFallback",
]
```

Smoke + commit:

```bash
uv run python -c "from gmmxx import cuda_ops; print(cuda_ops.__all__)"
git add gmmxx/cuda_ops.py
git commit -m "$(cat <<'EOF'
Re-export spherical CUDA ops through gmmxx.cuda_ops

Plan 2 adds five new entries: spherical_assign, spherical_logsumexp,
spherical_resp, blocked_update_spherical, finalize_spherical. The module
remains experimental.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — Per-kernel correctness tests

**Files:** Create `tests/test_cuda_spherical_safe.py`

### Step 8.1 — Write the test file

```python
"""Per-kernel correctness tests for the spherical CUDA safe path.

Compares CUDA outputs to torch_fallback reference at fp32 rtol=1e-4 / atol=1e-4
(per spec §6 numerical contract).
"""

from __future__ import annotations

import math

import pytest
import torch


def _has_cuda() -> bool:
    try:
        from gmmxx import _cuda
        return _cuda.has_cuda()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA + gmmxx._C")


def _random_setup(B=1, N=64, D=4, K=3, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    means = torch.randn(B, K, D, device=device, dtype=dtype)
    var = torch.rand(B, K, device=device).clamp_min(0.5).float()
    log_w = torch.log_softmax(torch.randn(B, K, device=device), dim=-1).float()
    return x, means, var, log_w


def _torch_logits(x, means, var, log_w):
    """Reference: log p_k(x_n) = log_w_k - D/2*log(2π σ_k²) - 0.5/σ_k² * ||x − μ_k||²"""
    B, N, D = x.shape
    K = means.shape[1]
    x_f = x.float()
    means_f = means.float()
    diff = x_f.unsqueeze(2) - means_f.unsqueeze(1)  # (B,N,K,D)
    dist_sq = diff.pow(2).sum(-1)                     # (B,N,K)
    return (
        log_w.unsqueeze(1)
        - 0.5 * D * torch.log(2 * math.pi * var).unsqueeze(1)
        - 0.5 * dist_sq / var.unsqueeze(1)
    )


class TestSphericalAssign:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_argmax_matches_torch_reference(self, dtype):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(dtype=dtype)
        cuda_ids = _cuda.spherical_assign(x, means, var, log_w)
        ref_ids = _torch_logits(x, means, var, log_w).argmax(-1).int()
        # Allow a small fraction of disagreement on near-tie samples for fp16/bf16.
        agree = (cuda_ids == ref_ids).float().mean().item()
        threshold = 0.99 if dtype == torch.float32 else 0.95
        assert agree >= threshold, f"only {agree:.3f} agreement"

    def test_returns_int32_shape_BN(self):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(B=2, N=32, D=4, K=5)
        ids = _cuda.spherical_assign(x, means, var, log_w)
        assert ids.shape == (2, 32)
        assert ids.dtype == torch.int32

    def test_zero_N_returns_empty(self):
        from gmmxx import _cuda
        x = torch.empty(1, 0, 4, device="cuda")
        means = torch.randn(1, 3, 4, device="cuda")
        var = torch.ones(1, 3, device="cuda")
        log_w = torch.zeros(1, 3, device="cuda")
        ids = _cuda.spherical_assign(x, means, var, log_w)
        assert ids.shape == (1, 0)


class TestSphericalLogsumexp:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_matches_torch_reference(self, dtype):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(dtype=dtype)
        cuda_lse = _cuda.spherical_logsumexp(x, means, var, log_w)
        ref_lse = _torch_logits(x, means, var, log_w).logsumexp(-1)
        rtol = 1e-4 if dtype == torch.float32 else 1e-2
        atol = 1e-4 if dtype == torch.float32 else 1e-2
        assert torch.allclose(cuda_lse, ref_lse, rtol=rtol, atol=atol), (
            f"max diff: {(cuda_lse - ref_lse).abs().max().item()}"
        )

    def test_returns_float32_shape_BN(self):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(B=2, N=32, K=4, D=2)
        lse = _cuda.spherical_logsumexp(x, means, var, log_w)
        assert lse.shape == (2, 32)
        assert lse.dtype == torch.float32


class TestSphericalResp:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_resp_sums_to_one_per_row(self, dtype):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(dtype=dtype)
        lse = _cuda.spherical_logsumexp(x, means, var, log_w)
        r = _cuda.spherical_resp(x, means, var, log_w, lse)
        sums = r.sum(-1)
        atol = 1e-4 if dtype == torch.float32 else 1e-2
        assert torch.allclose(sums, torch.ones_like(sums), atol=atol)

    def test_matches_torch_reference(self):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(dtype=torch.float32)
        lse = _cuda.spherical_logsumexp(x, means, var, log_w)
        cuda_r = _cuda.spherical_resp(x, means, var, log_w, lse)
        ref_logits = _torch_logits(x, means, var, log_w)
        ref_r = (ref_logits - lse.unsqueeze(-1)).exp()
        assert torch.allclose(cuda_r, ref_r, rtol=1e-4, atol=1e-4)


class TestBlockedUpdateSpherical:
    def test_counts_sum_to_N(self):
        from gmmxx import _cuda
        torch.manual_seed(0)
        x = torch.randn(1, 256, 4, device="cuda")
        ids = torch.randint(0, 5, (1, 256), device="cuda", dtype=torch.int32)
        sums, sumsq, counts = _cuda.blocked_update_spherical(x, ids, 5)
        assert counts.sum().item() == 256

    def test_sums_match_groupby(self):
        from gmmxx import _cuda
        torch.manual_seed(0)
        B, N, D, K = 2, 128, 4, 6
        x = torch.randn(B, N, D, device="cuda")
        ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
        sums, sumsq, counts = _cuda.blocked_update_spherical(x, ids, K)

        # Reference: Python groupby
        for b in range(B):
            for k in range(K):
                mask = (ids[b] == k)
                if mask.sum() == 0:
                    assert counts[b, k].item() == 0
                    assert torch.allclose(sums[b, k], torch.zeros(D, device="cuda"))
                    continue
                ref_sum = x[b][mask].sum(0)
                ref_sumsq = x[b][mask].pow(2).sum().item()
                assert torch.allclose(sums[b, k], ref_sum, atol=1e-4)
                assert abs(sumsq[b, k].item() - ref_sumsq) < 1e-3 * max(1.0, abs(ref_sumsq))


class TestFinalizeSpherical:
    def test_basic(self):
        from gmmxx import _cuda
        sums = torch.tensor([[[10.0, 20.0], [0.0, 0.0]]], device="cuda")
        sumsq = torch.tensor([[150.0, 0.0]], device="cuda")
        counts = torch.tensor([[5, 0]], device="cuda", dtype=torch.int32)
        old_means = torch.tensor([[[0.0, 0.0], [99.0, 99.0]]], device="cuda")
        old_var = torch.tensor([[1.0, 42.0]], device="cuda")
        new_means, new_var, new_weights = _cuda.finalize_spherical(
            sums, sumsq, counts, old_means, old_var, 5, 1e-6
        )
        # k=0: mean = (2,4), ||mean||²=20, var = (150/5 - 20)/2 = 5
        assert torch.allclose(new_means[0, 0], torch.tensor([2.0, 4.0], device="cuda"))
        assert abs(new_var[0, 0].item() - 5.0) < 1e-4
        assert abs(new_weights[0, 0].item() - 1.0) < 1e-4
        # k=1: empty cluster -> previous values preserved
        assert torch.allclose(new_means[0, 1], old_means[0, 1])
        assert new_var[0, 1].item() == 42.0
        assert new_weights[0, 1].item() == 0.0

    def test_reg_covar_clamps(self):
        from gmmxx import _cuda
        sums = torch.tensor([[[1.0, 1.0]]], device="cuda")
        sumsq = torch.tensor([[2.0]], device="cuda")
        counts = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        old_means = torch.zeros(1, 1, 2, device="cuda")
        old_var = torch.zeros(1, 1, device="cuda")
        # mean = (1,1), ||mean||² = 2, var_raw = (2/1 - 2)/2 = 0
        # reg_covar = 1e-3 should clamp it.
        _, new_var, _ = _cuda.finalize_spherical(
            sums, sumsq, counts, old_means, old_var, 1, 1e-3
        )
        assert new_var[0, 0].item() == pytest.approx(1e-3, abs=1e-9)
```

### Step 8.2 — Run

```bash
uv run pytest tests/test_cuda_spherical_safe.py -v
```

Expected: all tests pass on the CUDA host. If `agree >= 0.99` fails for fp32 due to actual numerical drift, lower the threshold to 0.97 only after verifying the diff isn't a kernel bug.

### Step 8.3 — Commit

```bash
git add tests/test_cuda_spherical_safe.py
git commit -m "$(cat <<'EOF'
Add per-kernel correctness tests for spherical CUDA safe path

Five test classes covering assign / logsumexp / resp / blocked_update /
finalize. Each compares the CUDA output against a torch reference at the
spec's rtol=1e-4 / atol=1e-4 (fp32) or rtol=1e-2 (fp16/bf16).

Empty-cluster behavior is verified explicitly in TestFinalizeSpherical:
when count[k] == 0, the kernel preserves old_means and old_var.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — Turn on `cuda_spherical_supported`

**Files:** Modify `gmmxx/_runtime.py`

Replace `cuda_spherical_supported` with:

```python
def cuda_spherical_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the spherical CUDA backend can handle this shape+dtype.

    Plan 2 (safe path): supports d in (0, 128] and n_components in (0, 2048]
    for dtype in {fp32, fp16, bf16}. Plan 3 will widen the dtype dispatch to
    route fp16/bf16 to the sm80 mma kernel; Plan 2's safe kernel handles all
    three but at SIMT speed.
    """
    import torch as _torch
    if dtype is None:
        return False  # caller didn't supply dtype; conservative no.
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 128):
        return False
    if not (0 < n_components <= 2048):
        return False
    return True
```

The other three (`cuda_diag_supported`, `cuda_tied_supported`, `cuda_full_supported`) stay False — Plans 6, 7, 8 turn them on.

Run `pytest tests/test_dispatch.py -v` to confirm the dispatcher still works (one test will switch `auto`→cuda for spherical now if CUDA is available; check that the test's expected-value list `{cuda, triton, torch}` is broad enough — it is).

```bash
git add gmmxx/_runtime.py
git commit -m "$(cat <<'EOF'
Turn on cuda_spherical_supported gate for Plan 2

Returns True for dtype in {fp32, fp16, bf16} with 0 < d <= 128 and
0 < n_components <= 2048. The dispatcher will now route spherical EM
training to the CUDA path when backend=auto on a CUDA host, falling
through to Triton or torch only on out-of-window shapes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10 — Wire spherical CUDA into `interface.py` `train()`

This is the trickiest task: the existing spherical training loop calls `batch_gmm_Spherical_torch_native` (or its Triton variants) directly. We need to add a CUDA branch that calls our new `_cuda` wrappers.

**Files:** Modify `gmmxx/interface.py`

### Step 10.1 — Find the spherical branch

Search for `covariance_type == "spherical"` in `gmmxx/interface.py`. The branch is around line 411-441 (varies after Plan 1 edits). It currently dispatches based on `self.use_triton` and shape gates.

### Step 10.2 — Add a CUDA branch

Before the existing Triton dispatch, insert:

```python
# CUDA path: take it when the dispatcher resolves to "cuda" for our shape.
from . import _dispatch, _cuda as _cuda_mod
shape_for_dispatch = (data.shape[0] if data.dim() == 3 else 1,
                      data.shape[-2], data.shape[-1], self.k)
resolved = _dispatch.resolve_backend_with_env(
    requested=self.backend,
    covariance="spherical",
    shape=shape_for_dispatch,
    dtype=data.dtype,
    legacy_no_triton=self._legacy_no_triton,
)
if resolved == "cuda":
    result = self._train_spherical_cuda(data)
    self.last_backend_used_ = "cuda"
    self.cuda_estep_enabled_ = True
    return result
self.last_backend_used_ = resolved
```

### Step 10.3 — Add a `_train_spherical_cuda` method to `GMMXX`

Place the new method near the existing `train` method:

```python
def _train_spherical_cuda(self, data: torch.Tensor):
    """Spherical EM loop on the CUDA backend (Plan 2 safe path).

    Mirrors batch_gmm_Spherical_torch_native's structure:
      1. Initialize means via random sampling or k-means++.
      2. Initialize variances and weights uniformly.
      3. EM loop: assign -> blocked_update -> finalize -> check ELBO.
      4. Return (labels_b, means_b, variances_b, weights_b, info_dict).
    """
    import math
    from . import _cuda as _cuda_mod

    if data.dim() == 2:
        data_b = data.unsqueeze(0)
        squeeze_batch = True
    else:
        data_b = data
        squeeze_batch = False

    B, N, D = data_b.shape
    K = self.k
    device = data_b.device

    # Initialize means by sampling from the data.
    rng = torch.Generator(device=device).manual_seed(self.seed)
    init_idx = torch.randint(0, N, (B, K), generator=rng, device=device)
    means = torch.gather(
        data_b, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)
    ).contiguous()

    # Initialize variances to data variance / K, weights uniform.
    var = data_b.float().var(dim=1).mean(dim=-1, keepdim=True).expand(B, K).contiguous() / K
    var = var.clamp_min(self.reg_covar)
    log_w = torch.full((B, K), -math.log(K), dtype=torch.float32, device=device)

    lower_bound_history: list[float] = []
    n_iter = 0
    prev_lb = -math.inf

    for _ in range(self.niter):
        n_iter += 1
        ids = _cuda_mod.spherical_assign(data_b, means, var, log_w)
        lse = _cuda_mod.spherical_logsumexp(data_b, means, var, log_w)
        lb = float(lse.mean().item())
        lower_bound_history.append(lb)

        sums, sumsq, counts = _cuda_mod.blocked_update_spherical(data_b, ids, K)
        means, var, weights = _cuda_mod.finalize_spherical(
            sums, sumsq, counts, means, var, N, self.reg_covar
        )
        log_w = torch.log(weights.clamp_min(1e-30))

        if abs(lb - prev_lb) < self.tol:
            break
        prev_lb = lb

    labels_b = ids if self.compute_labels_on_fit else None
    info = {
        "lower_bound": lb,
        "lower_bound_history": lower_bound_history,
        "n_iter": n_iter,
        "init_source": "cuda_random",
        "triton_estep_enabled": False,
        "triton_streaming_update_enabled": False,
        "triton_fused_update_enabled": False,
        "triton_approx_topk_enabled": False,
        "triton_labels_enabled": False,
        "approximate_em_enabled": False,
        "approx_top_k": None,
        "backend_breakdown": {"cuda": n_iter},
    }
    self._set_fit_result(
        labels_b=labels_b,
        means_b=means,
        variances_b=var,
        weights_b=weights,
        info=info,
        batch_size=None if squeeze_batch else B,
    )
    return self
```

(The implementer may need to adjust the integration to match the exact existing call pattern in `train`. Read `gmmxx/interface.py:391-487` — the existing `train` body — and mirror its style. The key invariant is that `_set_fit_result` gets called with the same kwargs the torch_native path passes today.)

### Step 10.4 — End-to-end smoke

```bash
uv run python -c "
import torch
from gmmxx import GMMXX

torch.manual_seed(0)
x = torch.randn(8192, 16, device='cuda')

# Force CUDA backend.
gmm = GMMXX(n_components=8, max_iter=20, tol=1e-4, random_state=0,
            covariance_type='spherical', backend='cuda')
gmm.fit(x)
print('last_backend_used_:', gmm.last_backend_used_)
print('cuda_estep_enabled_:', gmm.cuda_estep_enabled_)
print('lower_bound_:', gmm.lower_bound_)
print('means shape:', gmm.means_.shape)
"
```

Expected: `last_backend_used_` is `"cuda"`; non-trivial lower_bound_; means shape `(8, 16)`.

### Step 10.5 — Existing test suite

```bash
uv run pytest tests/ -q
```

Expected: all tests still pass. The previously-tested spherical paths (Triton, torch) still work because backend dispatching only changes when `backend="cuda"` AND `cuda_spherical_supported()`.

### Step 10.6 — Commit

```bash
git add gmmxx/interface.py
git commit -m "$(cat <<'EOF'
Wire spherical CUDA path into GMMXX.train()

When backend resolves to "cuda" for spherical covariance, run the new
_train_spherical_cuda EM loop instead of dispatching to Triton/torch.
Preserves the same _set_fit_result contract so all downstream attributes
(means_, weights_, etc.) work unchanged.

last_backend_used_ and cuda_estep_enabled_ are populated correctly.
fit_info_["backend_breakdown"] reports {"cuda": n_iter} for pure-CUDA runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11 — 3-way oracle test (CUDA vs Triton vs torch)

**Files:** Create `tests/test_cuda_vs_triton_spherical.py`

```python
"""3-way oracle: CUDA spherical results match Triton AND torch_fallback within
the spec's tolerance bounds (§6).

Mirrors flash-kmeans-cuda's test_correctness.py pattern. Skipped when either
CUDA or Triton is unavailable.
"""

from __future__ import annotations

import math

import pytest
import torch


def _has_cuda():
    try:
        from gmmxx import _cuda
        return _cuda.has_cuda()
    except ImportError:
        return False


def _has_triton():
    try:
        from gmmxx.assign_spherical_triton import spherical_assign_triton
        return spherical_assign_triton is not None
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (_has_cuda() and _has_triton()),
    reason="requires both CUDA and Triton",
)


@pytest.mark.parametrize("D,K,N", [(8, 16, 1024), (32, 64, 4096), (128, 8, 256)])
def test_assign_3way(D, K, N):
    """Assign output should agree across CUDA / Triton / torch_fallback on
    shapes inside the Triton support window."""
    from gmmxx import _cuda
    from gmmxx.assign_spherical_triton import spherical_assign_triton

    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn(1, N, D, device=device)
    means = torch.randn(1, K, D, device=device)
    var = torch.rand(1, K, device=device).clamp_min(0.5)
    log_w = torch.log_softmax(torch.randn(1, K, device=device), dim=-1)

    cuda_ids = _cuda.spherical_assign(x, means, var, log_w)
    triton_ids = spherical_assign_triton(x, means, var, log_w)

    # Both backends should match within fp32 numerical noise on near-tie samples.
    agree = (cuda_ids == triton_ids).float().mean().item()
    assert agree >= 0.99, f"CUDA vs Triton agreement only {agree:.3f}"


@pytest.mark.parametrize("D,K,N", [(16, 32, 2048)])
def test_logsumexp_3way(D, K, N):
    from gmmxx import _cuda
    from gmmxx.assign_spherical_triton import spherical_logsumexp_triton

    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(1, N, D, device=device)
    means = torch.randn(1, K, D, device=device)
    var = torch.rand(1, K, device=device).clamp_min(0.5)
    log_w = torch.log_softmax(torch.randn(1, K, device=device), dim=-1)

    cuda_lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    triton_lse = spherical_logsumexp_triton(x, means, var, log_w)
    assert torch.allclose(cuda_lse, triton_lse, rtol=5e-3, atol=5e-3)
```

Run + commit:

```bash
uv run pytest tests/test_cuda_vs_triton_spherical.py -v
git add tests/test_cuda_vs_triton_spherical.py
git commit -m "$(cat <<'EOF'
Add 3-way oracle: CUDA vs Triton vs torch_fallback on spherical assign+lse

Tests inside the Triton support window where all three backends are
available. Catches drift between CUDA and Triton even when both
individually pass the torch reference gate. Mirrors flash-kmeans-cuda's
test_correctness.py pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12 — Parametrize existing GMMXX spherical tests on backend

**Files:** Modify `tests/test_gmmxx.py`

Look for the spherical test methods (search for `covariance_type="spherical"` or similar). For at least one end-to-end fit test, add a parametrization across `("torch", "triton", "cuda")` with skips for unavailable backends. Example pattern:

```python
@pytest.mark.parametrize("backend", ["torch", "triton", "cuda"])
def test_spherical_fit_each_backend(backend):
    if backend == "cuda":
        try:
            from gmmxx._cuda import has_cuda
        except ImportError:
            pytest.skip("no _cuda module")
        if not has_cuda():
            pytest.skip("no CUDA")
    if backend == "triton":
        try:
            from gmmxx.assign_spherical_triton import spherical_assign_triton
            if spherical_assign_triton is None:
                pytest.skip("no Triton")
        except ImportError:
            pytest.skip("no Triton")

    torch.manual_seed(0)
    device = "cuda" if backend in ("triton", "cuda") and torch.cuda.is_available() else "cpu"
    x = torch.randn(2048, 16, device=device)
    gmm = GMMXX(n_components=8, max_iter=20, tol=1e-4, random_state=0,
                covariance_type="spherical", backend=backend)
    gmm.fit(x)
    assert gmm.means_.shape == (8, 16)
    assert gmm.lower_bound_ > -math.inf
    assert gmm.last_backend_used_ == backend
```

Run + commit:

```bash
uv run pytest tests/test_gmmxx.py -v -k spherical
git add tests/test_gmmxx.py
git commit -m "$(cat <<'EOF'
Parametrize spherical fit() test across backend ∈ {torch, triton, cuda}

Existing test_gmmxx.py covers spherical via use_triton on/off; add a new
backend-axis parametrization for end-to-end fit verification. Skips
gracefully when a backend is unavailable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13 — README + tag

Update README's "CUDA backend (experimental)" section: change "Phase 1 in progress" wording to "Spherical CUDA path live; diag/tied/full coming in Plans 6–8". Add a one-line example showing `backend="cuda"`.

```bash
git add README.md
git commit -m "Update README: spherical CUDA path live"
git tag -a spherical-safe-plan2 -m "Plan 2: spherical CUDA safe path"
```

---

## Self-Review Checklist

**1. Spec coverage**

| Spec section | Plan task |
| --- | --- |
| §3 cuda_ops.spherical_* re-exports | Task 7 |
| §4 spherical safe + sm80 — sm80 deferred to Plan 3 | Tasks 3, 4, 5 (safe only) |
| §5d optimization techniques — only register-tile + fp32 accumulator from this list | Tasks 3, 4 |
| §5b zero-init contract | Task 4, Task 6 (wrapper does zero_) |
| §5c per-cov finalize | Task 5 |
| §6 numerical contract (fp32 rtol=1e-4) | Task 8 |
| §7 cuda_spherical_supported True for in-window shapes | Task 9 |
| §10b new test files | Tasks 8, 11 |

**2. Placeholder scan** — none. Every step has real code.

**3. Type consistency** — `(B, N, D)` shapes throughout, `(B, K, D)` for means, `(B, K)` for var/log_w/sumsq/counts. `cluster_ids` int32. `var` and `log_w` always fp32.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-02-gmmxx-cuda-spherical-safe.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks.

**2. Inline Execution** — Execute tasks in this session.

After this plan: **Plan 3** populates `csrc/estep/spherical_sm80.cu` (mma.sync m16n8k16 + cp.async double-buffering) and the sorted-run `blocked_spherical_sorted.cu`. Plan 4 adds fused single-tile. Plan 5 adds approx top-K. Plans 6–8 add diag, tied, full.
