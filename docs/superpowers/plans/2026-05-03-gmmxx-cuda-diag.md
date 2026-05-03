# GMMXX CUDA Backend — Plan 6: Diagonal CUDA Covariance

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the diagonal covariance type to the CUDA backend. After this plan, `GMMXX(covariance_type="diag", backend="cuda")` runs full EM training and inference on hand-written CUDA kernels for shapes inside the diagonal support window. Mirrors Plan 2 (spherical safe path) structure but with per-feature variance accumulation. Spec §4 lists diagonal as a Phase 1 deliverable.

**Architecture:** Two new C++ TUs (`csrc/estep/diag_safe.cu` + `csrc/mstep/blocked_diag.cu` + `csrc/mstep/finalize_diag.cu`); a host dispatcher analogous to `spherical_dispatch.cu` (route fp32 to safe; fp16/bf16+sm_80+ stub to safe initially, mma variant deferred). Python wrappers in `_cuda.py`, `cuda_ops.py`. `_runtime.cuda_diag_supported` flips to True for `0 < D ≤ 64, 0 < K ≤ 512`. `interface.py` adds a `_train_diag_cuda` method analogous to `_train_spherical_cuda`, plus diag CUDA inference branches in `predict()`/`predict_proba()`/`score_samples()`.

**Tech Stack:** CUDA 12.8+, sm_80+ optional. PowerShell + uv on the host.

**Spec sections covered:** §4 diag E/M coverage, §5 host-side launcher contract, §6 numerical tolerance, §7 dispatch.

**Out of scope (deferred):**
- sm_80 mma optimized diag E-step → Plan 7+
- Sorted-run M-step for diag → Plan 7+
- Fused single-tile diag E/M → Plan 7+
- Approx top-K (spherical-only) — already deferred
- Tied / full covariance → Plans 8–9
- `large_n.py` integration → Plan 10

**Foundation assumed:** Plans 1–5 complete (`spherical-fused-plan5` tag). Spherical is feature-complete on CUDA. The dispatcher truth-table, `_cuda` wrapper layer, and `cuda_ops` re-export pattern are well established.

---

## Numerical formulas — diagonal covariance

For a diagonal Gaussian mixture, component `k` with mean `μ_k ∈ R^D`, per-feature variance `σ_k² ∈ R^D` (a vector), weight `π_k`:

- **Per-component log-likelihood**:
  `log p_k(x) = log π_k − 0.5 · Σ_d log(2π σ_{k,d}²) − 0.5 · Σ_d (x_d − μ_{k,d})²/σ_{k,d}²`

  Compared to spherical (`σ_k² ∈ R` scalar): the log-determinant is a sum over D, and the Mahalanobis distance has per-feature normalization.

- **Stable logsumexp / responsibility**: same as spherical (subtract per-row max).

- **M-step sufficient stats** (soft EM):
  - `n_k = Σ_n r_{n,k}` — soft count (B, K).
  - `sum_x_k = Σ_n r_{n,k} · x_n` — (B, K, D).
  - `sum_xx_k = Σ_n r_{n,k} · x_n²` — (B, K, D), **per-feature** squared sums (not scalar).

- **Finalize**:
  - `μ_k = sum_x_k / n_k` — (B, K, D).
  - `σ_{k,d}² = max(sum_xx_{k,d} / n_k − μ_{k,d}², reg_covar)` — (B, K, D), per-feature clamp.
  - `π_k = n_k / N` — (B, K).

The KEY difference vs spherical: `var` and `sumsq` are `(B, K, D)` instead of `(B, K)`. The M-step accumulates `sum_xx_{k,d} = Σ_n r_{n,k} · x_{n,d}²` per feature, not a scalar `Σ_n r_{n,k} · ||x_n||²`.

---

## File Structure

### Created

| Path | Responsibility |
| --- | --- |
| `gmmxx/csrc/estep/diag.h` | Public dispatcher + safe-path declarations. |
| `gmmxx/csrc/estep/diag_safe.cu` | Three SIMT kernels (assign, logsumexp, resp) with per-feature dist normalization. |
| `gmmxx/csrc/estep/diag_dispatch.cu` | Host router; for now identity-routes to safe (sm80 mma variant is a follow-up). |
| `gmmxx/csrc/mstep/diag.h` | M-step declarations (blocked_update + finalize). |
| `gmmxx/csrc/mstep/blocked_diag.cu` | Per-token atomicAdd into (B, K, D) sums + (B, K, D) sumsq + (B, K) counts. |
| `gmmxx/csrc/mstep/finalize_diag.cu` | Per-cluster, per-feature divide + clamp; preserves old means/var on empty cluster. |
| `tests/test_cuda_diag_safe.py` | Per-kernel correctness vs torch_fallback at fp32 rtol=1e-4. |
| `tests/test_cuda_diag_inference.py` | predict/predict_proba/score_samples/score under backend='cuda' for diag. |

### Modified

| Path | Change |
| --- | --- |
| `gmmxx/csrc/bindings.cpp` | Expose 5 new ops: `diag_assign`, `diag_logsumexp`, `diag_resp`, `blocked_update_diag`, `finalize_diag`. |
| `setup.py` | Add the three new `.cu` files to sources. |
| `gmmxx/_cuda.py` | Add 5 wrappers (similar to spherical pattern). |
| `gmmxx/cuda_ops.py` | Re-export the 5 callables. |
| `gmmxx/_runtime.py` | `cuda_diag_supported(d, k, dtype) → True` for `0 < D ≤ 64, 0 < K ≤ 512, dtype ∈ {fp32, fp16, bf16}`. |
| `gmmxx/interface.py` | Add `_train_diag_cuda` method; route diag-cov training through it when `backend="cuda"` and shape is supported. Add CUDA branches in predict/predict_proba/score_samples for diag covariance. |
| `gmmxx/_dispatch.py` | Extend `_TRITON_OPS_BY_NAME` map with diag entries. |
| `tests/test_gmmxx.py` | Add `test_diag_full_pipeline_each_backend` parametrized test. |
| `README.md` | Update CUDA section to note diag now on CUDA. |

---

## Numerical contract (same as spherical Plan 2)

| Output | fp32 inputs | fp16/bf16 inputs |
| --- | --- | --- |
| `means_`, `weights_` | rtol=1e-4, atol=1e-4 | rtol=5e-3, atol=5e-3 |
| `covariances_` (diag — per-feature variance) | rtol=1e-4, atol=1e-4 | rtol=5e-3, atol=5e-3 |
| `lower_bound_`, `score_samples` | rtol=1e-4 | rtol=1e-2 |
| `labels_` | ≥ 99% agreement on separable data; ≥ 95% on near-degenerate | same |

---

## Conventions

- Working directory: `C:\Users\HEQ\Projects\flashGMM2`. Branch: `GMMXX-cuda` (post-`spherical-fused-plan5`).
- Dev rebuild: `$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .` after each `.cu`/`.cpp`/setup.py change.
- Reference: existing `gmmxx/assign_diag_triton.py` (Triton kernels) and `gmmxx/torch_fallback._batch_gmm_Diagonal_torch_native_inner` for the math.

---

## Task 1 — Diagonal E-step kernels (assign/logsumexp/resp safe)

**Files:** Create `gmmxx/csrc/estep/diag.h`; create `gmmxx/csrc/estep/diag_safe.cu`; create `gmmxx/csrc/estep/diag_dispatch.cu`; modify `gmmxx/csrc/bindings.cpp`; modify `setup.py`.

### Step 1.1 — Write `gmmxx/csrc/estep/diag.h`

```cpp
#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace estep { namespace diag {

// Public dispatchers.
//
// x: (B, N, D) fp32 / fp16 / bf16 contiguous CUDA.
// means: (B, K, D) same dtype as x.
// var: (B, K, D) fp32 — per-feature variance.
// log_w: (B, K) fp32 — per-component log mixture weight.
at::Tensor assign(const at::Tensor& x, const at::Tensor& means,
                  const at::Tensor& var, const at::Tensor& log_w,
                  c10::optional<at::Tensor> out);

at::Tensor logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     c10::optional<at::Tensor> out);

at::Tensor resp(const at::Tensor& x, const at::Tensor& means,
                const at::Tensor& var, const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out);

// Safe path implementations.
at::Tensor assign_safe(const at::Tensor& x, const at::Tensor& means,
                       const at::Tensor& var, const at::Tensor& log_w,
                       c10::optional<at::Tensor> out);
at::Tensor logsumexp_safe(const at::Tensor& x, const at::Tensor& means,
                          const at::Tensor& var, const at::Tensor& log_w,
                          c10::optional<at::Tensor> out);
at::Tensor resp_safe(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     const at::Tensor& log_norm,
                     c10::optional<at::Tensor> out);

}}}  // namespace gmmxx::estep::diag
```

### Step 1.2 — Write `gmmxx/csrc/estep/diag_safe.cu`

The kernels mirror `spherical_safe.cu` exactly except for the logit formula. Reference the existing `gmmxx/csrc/estep/spherical_safe.cu` for the templated kernel structure (one thread per (b, n) for assign; one CTA per (b, n) with K threads for logsumexp; one thread per (b, n, k) for resp).

The diagonal logit:

```cpp
// Per (m, k):
//   log p_k(x_m) = log_w[k] - 0.5 * Σ_d log(2π σ_{k,d}²) - 0.5 * Σ_d (x_m,d - μ_{k,d})²/σ_{k,d}²
const T* x_n = x_b + n * D;
const T* mu_k = means_b + k * D;
const float* var_k = var_b + k * D;
float log_det = 0.0f;
float dist = 0.0f;
const float TWO_PI = 6.283185307179586f;
for (int d = 0; d < D; ++d) {
    float v = var_k[d];
    log_det += logf(TWO_PI * v);
    float dx = static_cast<float>(x_n[d]) - static_cast<float>(mu_k[d]);
    dist += dx * dx / v;
}
float logit = log_w_b[k] - 0.5f * log_det - 0.5f * dist;
```

Compare to spherical:
- Spherical: scalar `var[k]`, `log_det = D * log(2π·var[k])`, `dist /= var[k]`.
- Diagonal: vector `var[k][d]`, log_det summed over d, dist normalized per d.

The kernel structure is otherwise identical. Implement the three kernels (assign / logsumexp / resp) following the spherical_safe.cu template.

Notes:
- `var_b = var + (size_t)b * K * D` (3D indexing).
- All other shapes/inputs are identical to spherical.
- Output shapes: `assign → (B, N) int32`, `logsumexp → (B, N) fp32`, `resp → (B, N, K) fp32`.

### Step 1.3 — Write `gmmxx/csrc/estep/diag_dispatch.cu`

```cpp
#include "diag.h"

namespace gmmxx { namespace estep { namespace diag {

// Plan 6: route everything to safe. A future task adds an sm80 mma variant.
at::Tensor assign(const at::Tensor& x, const at::Tensor& means,
                  const at::Tensor& var, const at::Tensor& log_w,
                  c10::optional<at::Tensor> out) {
    return assign_safe(x, means, var, log_w, std::move(out));
}

at::Tensor logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     c10::optional<at::Tensor> out) {
    return logsumexp_safe(x, means, var, log_w, std::move(out));
}

at::Tensor resp(const at::Tensor& x, const at::Tensor& means,
                const at::Tensor& var, const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out) {
    return resp_safe(x, means, var, log_w, log_norm, std::move(out));
}

}}}
```

### Step 1.4 — Update `setup.py`

Add to `sources`:
```python
str(CSRC / "estep" / "diag_safe.cu"),
str(CSRC / "estep" / "diag_dispatch.cu"),
```

### Step 1.5 — Wire `bindings.cpp`

Add include:
```cpp
#include "estep/diag.h"
```

Add three `m.def` entries inside NB_MODULE:
```cpp
m.def("diag_assign", &gmmxx::estep::diag::assign,
      nb::arg("x"), nb::arg("means"), nb::arg("var"), nb::arg("log_w"),
      nb::arg("out") = nb::none(),
      "Diagonal E-step assign. Returns int32 (B, N).");

m.def("diag_logsumexp", &gmmxx::estep::diag::logsumexp,
      nb::arg("x"), nb::arg("means"), nb::arg("var"), nb::arg("log_w"),
      nb::arg("out") = nb::none(),
      "Diagonal E-step stable logsumexp. Returns fp32 (B, N).");

m.def("diag_resp", &gmmxx::estep::diag::resp,
      nb::arg("x"), nb::arg("means"), nb::arg("var"), nb::arg("log_w"),
      nb::arg("log_norm"), nb::arg("out") = nb::none(),
      "Diagonal E-step responsibilities. Returns fp32 (B, N, K).");
```

### Step 1.6 — Build + smoke

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .
```

```bash
uv run python -c "
import torch, math
from gmmxx import _C
B, N, D, K = 1, 64, 8, 4
torch.manual_seed(0)
x = torch.randn(B, N, D, device='cuda')
means = torch.randn(B, K, D, device='cuda')
var = torch.rand(B, K, D, device='cuda').clamp_min(0.5)  # per-feature
log_w = torch.zeros(B, K, device='cuda')

ids = _C.diag_assign(x, means, var, log_w)
print('assign shape:', ids.shape, 'dtype:', ids.dtype)

lse = _C.diag_logsumexp(x, means, var, log_w)
print('lse shape:', lse.shape)

r = _C.diag_resp(x, means, var, log_w, lse)
print('resp shape:', r.shape, 'sum per row mean:', r.sum(-1).mean().item())
"
```

Expected: shapes correct; resp rows sum to ~1.0.

### Step 1.7 — Commit

```bash
git add gmmxx/csrc/estep/diag.h gmmxx/csrc/estep/diag_safe.cu gmmxx/csrc/estep/diag_dispatch.cu gmmxx/csrc/bindings.cpp setup.py
git commit -m "$(cat <<'EOF'
Add diagonal CUDA E-step safe-path kernels (assign / logsumexp / resp)

Three template kernels (fp32/fp16/bf16) computing log p_k(x) for the
diagonal Gaussian formula with per-feature variance. Mirrors
spherical_safe.cu structure; only the logit computation differs:
log_det summed over D, dist normalized per feature.

Public dispatcher in diag_dispatch.cu currently identity-routes to
safe; a follow-up task may add an sm80 mma variant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Diagonal M-step (blocked_update + finalize)

**Files:** Create `gmmxx/csrc/mstep/diag.h`; create `gmmxx/csrc/mstep/blocked_diag.cu`; create `gmmxx/csrc/mstep/finalize_diag.cu`; modify `gmmxx/csrc/bindings.cpp`; modify `setup.py`.

### Header

```cpp
#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace mstep { namespace diag {

// Hard-assignment M-step accumulator. Caller MUST zero sums_out, sumsq_out,
// counts_out before calling.
//
// x: (B, N, D)
// cluster_ids: (B, N) int32 — per-point hard assignment
// sums_out: (B, K, D) fp32 — Σ x_n by cluster
// sumsq_out: (B, K, D) fp32 — Σ x_n² PER FEATURE by cluster (not scalar)
// counts_out: (B, K) int32
void blocked_update(const at::Tensor& x,
                    const at::Tensor& cluster_ids,
                    at::Tensor& sums_out,
                    at::Tensor& sumsq_out,
                    at::Tensor& counts_out);

// Finalize: divide sums/counts, clamp per-feature variance to reg_covar.
//
// old_means: (B, K, D) — preserved when count[k] == 0.
// old_var: (B, K, D) — per-feature, preserved when count[k] == 0.
// Returns:
//   means: (B, K, D)
//   var: (B, K, D) — per-feature variance
//   weights: (B, K) — sum to 1 per batch
std::tuple<at::Tensor, at::Tensor, at::Tensor> finalize(
    const at::Tensor& sums,
    const at::Tensor& sumsq,
    const at::Tensor& counts,
    const at::Tensor& old_means,
    const at::Tensor& old_var,
    int64_t total_n,
    double reg_covar);

}}}
```

### `blocked_diag.cu`

Mirrors `blocked_spherical.cu` (per-token atomicAdd) but `sumsq` is per-feature:

```cpp
template <typename T>
__global__ void __launch_bounds__(128)
blocked_update_diag_kernel(
    const T* __restrict__ x,
    const int32_t* __restrict__ ids,
    float* __restrict__ sums,    // (B, K, D)
    float* __restrict__ sumsq,   // (B, K, D)  PER-FEATURE
    int32_t* __restrict__ counts,
    int B, int N, int K, int D
) {
    int b = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || n >= N) return;

    const T* x_n = x + ((size_t)b * N + n) * D;
    int k = ids[(size_t)b * N + n];
    if (k < 0 || k >= K) return;

    for (int d = 0; d < D; ++d) {
        float v = static_cast<float>(x_n[d]);
        atomicAdd(sums + ((size_t)b * K + k) * D + d, v);
        atomicAdd(sumsq + ((size_t)b * K + k) * D + d, v * v);  // per-feature
    }
    atomicAdd(counts + (size_t)b * K + k, 1);
}
```

The host launcher mirrors `blocked_spherical`'s; just the `sumsq_out` shape check changes from (B, K) to (B, K, D).

### `finalize_diag.cu`

Mirrors `finalize_spherical` but computes per-feature variance:

```cpp
template <typename T>
__global__ void
finalize_diag_kernel(
    const float* __restrict__ sums,    // (B, K, D)
    const float* __restrict__ sumsq,   // (B, K, D)
    const int32_t* __restrict__ counts,// (B, K)
    const T* __restrict__ old_means,   // (B, K, D)
    const float* __restrict__ old_var, // (B, K, D)
    T* __restrict__ new_means,
    float* __restrict__ new_var,       // (B, K, D)
    float* __restrict__ new_weights,   // (B, K)
    int B, int K, int D, int total_n,
    float reg_covar
) {
    int b = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || k >= K) return;

    int32_t cnt = counts[(size_t)b * K + k];
    if (cnt <= 0) {
        // Empty: preserve old_means and old_var.
        for (int d = 0; d < D; ++d) {
            new_means[((size_t)b * K + k) * D + d] = old_means[((size_t)b * K + k) * D + d];
            new_var[((size_t)b * K + k) * D + d] = old_var[((size_t)b * K + k) * D + d];
        }
        new_weights[(size_t)b * K + k] = 0.0f;
        return;
    }

    float n_inv = 1.0f / (float)cnt;
    for (int d = 0; d < D; ++d) {
        size_t idx = ((size_t)b * K + k) * D + d;
        float mu_d = sums[idx] * n_inv;
        new_means[idx] = static_cast<T>(mu_d);
        float var_d = sumsq[idx] * n_inv - mu_d * mu_d;
        new_var[idx] = fmaxf(var_d, reg_covar);
    }
    new_weights[(size_t)b * K + k] = (float)cnt / (float)total_n;
}
```

### Bindings + setup

Add to `setup.py` sources:
```python
str(CSRC / "mstep" / "blocked_diag.cu"),
str(CSRC / "mstep" / "finalize_diag.cu"),
```

In `bindings.cpp`, add include `#include "mstep/diag.h"` and two m.def entries:
```cpp
m.def("blocked_update_diag", &gmmxx::mstep::diag::blocked_update,
      nb::arg("x"), nb::arg("cluster_ids"),
      nb::arg("sums_out"), nb::arg("sumsq_out"), nb::arg("counts_out"),
      "Diagonal M-step accumulator (per-token atomicAdd). Caller MUST zero "
      "sums_out (B,K,D), sumsq_out (B,K,D), counts_out (B,K) before calling.");

m.def("finalize_diag", &gmmxx::mstep::diag::finalize,
      nb::arg("sums"), nb::arg("sumsq"), nb::arg("counts"),
      nb::arg("old_means"), nb::arg("old_var"),
      nb::arg("total_n"), nb::arg("reg_covar"),
      "Finalize diagonal M-step. Returns (means (B,K,D), var (B,K,D), weights (B,K)).");
```

Build + smoke + commit.

---

## Task 3 — Python wrappers in `_cuda.py`

Append to `gmmxx/_cuda.py` after the spherical block:

```python
# ---------------------------------------------------------------------------
# Diagonal kernels (Plan 6 — safe path)
# ---------------------------------------------------------------------------


def diag_assign(x, means, var, log_w, out=None):
    """Diagonal E-step assign. Returns int32 (B, N)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    try:
        return _C.diag_assign(x, means, var, log_w, out)
    except RuntimeError as exc:
        if _no_fallback(): raise
        raise CudaRuntimeFallback(f"diag_assign failed: {exc}") from exc


def diag_logsumexp(x, means, var, log_w, out=None):
    """Diagonal E-step logsumexp. Returns fp32 (B, N)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    try:
        return _C.diag_logsumexp(x, means, var, log_w, out)
    except RuntimeError as exc:
        if _no_fallback(): raise
        raise CudaRuntimeFallback(f"diag_logsumexp failed: {exc}") from exc


def diag_resp(x, means, var, log_w, log_norm, out=None):
    """Diagonal E-step responsibilities. Returns fp32 (B, N, K)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    log_norm = _check_input(log_norm, "log_norm", dtype=torch.float32)
    try:
        return _C.diag_resp(x, means, var, log_w, log_norm, out)
    except RuntimeError as exc:
        if _no_fallback(): raise
        raise CudaRuntimeFallback(f"diag_resp failed: {exc}") from exc


def blocked_update_diag(x, cluster_ids, n_components):
    """Diagonal M-step accumulator. Allocates and zero-initializes
    sums (B,K,D), sumsq (B,K,D), counts (B,K)."""
    require_cuda()
    x = _check_input(x, "x")
    cluster_ids = _check_input(cluster_ids, "cluster_ids", dtype=torch.int32)
    B, N, D = x.shape
    K = int(n_components)
    sums = torch.zeros((B, K, D), dtype=torch.float32, device=x.device)
    sumsq = torch.zeros((B, K, D), dtype=torch.float32, device=x.device)
    counts = torch.zeros((B, K), dtype=torch.int32, device=x.device)
    try:
        _C.blocked_update_diag(x, cluster_ids, sums, sumsq, counts)
    except RuntimeError as exc:
        if _no_fallback(): raise
        raise CudaRuntimeFallback(f"blocked_update_diag failed: {exc}") from exc
    return sums, sumsq, counts


def finalize_diag(sums, sumsq, counts, old_means, old_var, total_n, reg_covar):
    """Diagonal M-step finalize. Returns (means (B,K,D), var (B,K,D), weights (B,K))."""
    require_cuda()
    sums = _check_input(sums, "sums", dtype=torch.float32)
    sumsq = _check_input(sumsq, "sumsq", dtype=torch.float32)
    counts = _check_input(counts, "counts", dtype=torch.int32)
    old_means = _check_input(old_means, "old_means")
    old_var = _check_input(old_var, "old_var", dtype=torch.float32)
    try:
        return _C.finalize_diag(sums, sumsq, counts, old_means, old_var,
                                 int(total_n), float(reg_covar))
    except RuntimeError as exc:
        if _no_fallback(): raise
        raise CudaRuntimeFallback(f"finalize_diag failed: {exc}") from exc
```

Smoke + commit.

---

## Task 4 — Re-export through `cuda_ops.py`

Add 5 entries to `gmmxx/cuda_ops.py` after the spherical block:

```python
# Diagonal (Plan 6).
diag_assign = _cuda.diag_assign
diag_logsumexp = _cuda.diag_logsumexp
diag_resp = _cuda.diag_resp
blocked_update_diag = _cuda.blocked_update_diag
finalize_diag = _cuda.finalize_diag
```

Add the names to `__all__`. Smoke + commit.

---

## Task 5 — Update `_runtime.cuda_diag_supported`

Replace the stub:

```python
def cuda_diag_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the diagonal CUDA backend can handle this shape+dtype.

    Plan 6 (safe path): supports d in (0, 64] and n_components in (0, 512]
    for dtype in {fp32, fp16, bf16}.
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 64):
        return False
    if not (0 < n_components <= 512):
        return False
    return True
```

Verify + commit.

---

## Task 6 — Per-kernel correctness tests

Create `tests/test_cuda_diag_safe.py` mirroring `test_cuda_spherical_safe.py` but for diagonal. Five test classes (TestDiagAssign / Logsumexp / Resp / BlockedUpdate / Finalize). Reference test_cuda_spherical_safe.py for structure; only the formula and shape differences matter.

The torch reference for diag logits:
```python
def _torch_diag_logits(x, means, var, log_w):
    B, N, D = x.shape
    K = means.shape[1]
    x_f = x.float().unsqueeze(2)         # (B, N, 1, D)
    means_f = means.float().unsqueeze(1) # (B, 1, K, D)
    var_f = var.float().unsqueeze(1)     # (B, 1, K, D)
    log_det = torch.log(2 * math.pi * var_f).sum(-1)         # (B, 1, K)
    dist = ((x_f - means_f).pow(2) / var_f).sum(-1)          # (B, N, K)
    return log_w.unsqueeze(1) - 0.5 * log_det - 0.5 * dist
```

Build correctness tests at fp32 rtol=1e-4, fp16/bf16 rtol=5e-3 for assign argmax agreement (≥0.95 fp32 / ≥0.95 fp16), logsumexp matching, resp sums-to-one, blocked_update vs Python groupby, finalize empty-cluster preservation.

Run + commit.

---

## Task 7 — Wire `_train_diag_cuda` into `interface.py`

Read the existing `_train_spherical_cuda` and create an analog `_train_diag_cuda` method. Differences:
- `var` is `(B, K, D)` instead of `(B, K)` — initialize as data variance (per-feature).
- `_cuda_mod.blocked_update_diag` returns `(sums, sumsq, counts)` where `sumsq` is `(B, K, D)`.
- `_cuda_mod.finalize_diag` returns `(means, var, weights)` where `var` is `(B, K, D)`.

In `train()`, find the diag covariance branch. Add a CUDA dispatch check (analogous to spherical's):

```python
elif self.covariance_type == "diag":
    if backend == "cuda":  # but routed through _dispatch.resolve_backend
        # Check cuda_diag_supported, dispatch to _train_diag_cuda
        ...
    # Otherwise fall through to existing torch_native / Triton path.
```

Run smoke + full pytest. Commit.

---

## Task 8 — Wire diag CUDA inference in predict / predict_proba / score_samples / score

Mirror Plan 3 Task 6 (spherical inference rewire). For each of the four inference methods, find the diag branch and add a CUDA early-return that calls the dispatcher with covariance="diag" and routes through `_dispatch.dispatch_kernel`.

Add diag entries to `_TRITON_OPS_BY_NAME` in `_dispatch.py`:
```python
"diag_assign":    "gmmxx.assign_diag_triton.diag_assign_triton",
"diag_logsumexp": "gmmxx.assign_diag_triton.diag_logsumexp_triton",
"diag_resp":      "gmmxx.assign_diag_triton.diag_resp_triton",
```

Run + commit.

---

## Task 9 — Inference correctness tests

Create `tests/test_cuda_diag_inference.py` mirroring `test_cuda_inference_spherical.py`. Verifies fit→predict→predict_proba→score_samples→score under `backend="cuda"` for diag covariance.

Run + commit.

---

## Task 10 — Parametrize `test_gmmxx.py` diag pipeline

Append `test_diag_full_pipeline_each_backend` analogous to the spherical version. Runs full pipeline under torch / triton / cuda.

Run + commit.

---

## Task 11 — README + tag

Update README CUDA section: replace "Diagonal, tied, and full are still on Triton/PyTorch" with "Diagonal is now on CUDA (Plan 6); tied and full coming in Plans 7-8."

Tag `diag-cuda-plan6` and commit.

---

## Self-Review Checklist

**1. Spec coverage**

| Spec section | Plan task |
| --- | --- |
| §4 diag E/M kernel inventory | Tasks 1, 2 |
| §5 host-side launcher contract | Task 1 (CUDAGuard, getCurrentCUDAStream throughout) |
| §6 numerical tolerance | Task 6 |
| §7 dispatch + dispatch_kernel diag entries | Tasks 5, 8 |

**2. Placeholder scan** — Tasks 1 and 2 reference the spherical kernels as templates; the implementer adapts the formula but the structure is verbatim. No vague pseudocode.

**3. Type consistency** — `var` and `sumsq` are `(B, K, D)` consistently across header, kernel, host launcher, Python wrapper, and tests. `_cuda.diag_*` signatures match `_C.diag_*` 1:1.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-03-gmmxx-cuda-diag.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Tasks 1, 2 use sonnet (well-templated kernels). Tasks 3-11 use sonnet. No opus required since there's no novel mma engineering — Plan 7+ adds optimizations.

**2. Inline Execution** — Same session.

After Plan 6: **Plan 7** = tied + full + sm80 mma diag + sorted-run diag M-step. Or split into two plans: **Plan 7** = tied; **Plan 8** = full. Then **Plan 9** = sm80 + sorted-run optimizations across diag/tied/full. Then **Plan 10** = `large_n.py` integration. Then **Plan 11** = approx top-K spherical (deferred from earlier; nice-to-have).
