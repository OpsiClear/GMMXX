# GMMXX CUDA Backend — Plan 5: Fused Single-Tile E/M for Spherical

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fused single-tile spherical E/M kernel that combines logit computation, softmax responsibilities, and per-cluster sufficient-statistic accumulation in a single CTA pass. Inside the support window `D ≤ 64, K ≤ 128`, the fused kernel replaces the separate (assign → blocked_update → finalize) sequence with a single launch that keeps all per-row state in registers/SMEM and writes only the (B, K, D+1+1) accumulator tile to global memory.

This is the headline perf feature for medium shapes — eliminates two global-memory round-trips per EM iteration (responsibility tensor `(B,N,K)` and sorted permutation buffer) and amortizes the centroid SMEM load across both E and M steps.

**Architecture:** One new CUDA TU (`csrc/fused/spherical_fused.cu`); one new dispatch gate in `_runtime.py` (`cuda_spherical_fused_supported`); one new Python wrapper in `_cuda.py` (`fused_spherical`); one branch in `_train_spherical_cuda` that prefers the fused path when supported. Outside the (D≤64, K≤128) window, the existing assign+blocked_update+finalize pipeline still runs (no behavior change for out-of-window shapes).

**Tech Stack:** CUDA 12.8+, sm_80+ for the mma path; the safe SIMT fp32 path also gets a fused variant. Reference: existing `gmmxx/fused_update_triton.py` Triton kernel.

**Spec sections covered:** §4 fused single-tile E/M (`spherical D≤64, K≤128`), §5d "Register-tiled fused min-over-K accumulator" (here repurposed for fused E/M), §10 perf gate.

**Out of scope (deferred to Plan 6+):**
- Persistent E-step kernels.
- Multi-stream `cudaEvent_t` plumbing.
- Fused single-tile for diag / tied (Plans 7–8 add their own fused variants).
- Approx top-K (Plan 7).

**Foundation assumed:** Plans 1–4 complete (`spherical-fast-plan4` tag). Spherical assign / logsumexp / resp / blocked_update_sorted / finalize all live; the dispatcher gates them by dtype and compute capability. Plan 5 adds a fifth path that subsumes assign+blocked_update+finalize for in-window shapes.

---

## Why fused?

After Plan 4, a single EM iteration on the CUDA path does:

```
ids = spherical_assign(x, means, var, log_w)         # (B,N) int32 — round-trip 1
sums, sumsq, counts = blocked_update_spherical(x, ids, K)  # 3 atomics tensors — round-trip 2
means, var, weights = finalize_spherical(...)         # post-process
```

For D≤64 and K≤128, a CTA can hold:
- One `(BLOCK_N, D)` x_tile in SMEM.
- All `(K, D)` centroids in SMEM (since K≤128 and D≤64, that's at most 128×64×2 = 16 KiB for fp16).
- Per-thread fragment register tile for the cross-product (BLOCK_N × K).
- Per-cluster partial accumulators `(K, D+1+1)` for sums + sumsq + count, in SMEM.

In one pass:
1. Load x_tile + all centroids. (One global → SMEM round-trip per CTA — vs. K-chunked double-buffering in the unfused path.)
2. mma logits across all K (no chunking needed — whole K fits).
3. Stable softmax over K → responsibilities (per row, in registers).
4. Atomic-accumulate `r[m,k] * x[m]` into the per-cluster partial in SMEM (no global atomics for the inner accumulation; cross-CTA merge happens via a final per-CTA flush to global).
5. Per-CTA flush: one `atomicAdd` per (cluster, feature) tuple per CTA — comparable to the sorted-run M-step's CTA-level coalescing.

Net: one global round-trip per EM iteration (for x), instead of three. SMEM bandwidth replaces global atomic-issue pressure for the K-dimension of the M-step accumulator.

For D > 64 or K > 128, the centroid + accumulator working set doesn't fit; the unfused path is correct and stays in service.

---

## File Structure

### Created

| Path | Responsibility |
| --- | --- |
| `gmmxx/csrc/fused/spherical_fused.cu` | Fused E/M kernel + host launcher. Two variants: safe (SIMT, fp32 inputs) and sm80 (mma, fp16/bf16 inputs). Each launches a single CTA per (B, BLOCK_N) tile, writes per-cluster partials directly to the caller's accumulators. |
| `gmmxx/csrc/fused/spherical.h` | Host signatures: `fused(...)` (public dispatcher) + `fused_safe(...)` + `fused_sm80(...)`. |
| `tests/test_cuda_spherical_fused.py` | Per-kernel correctness vs the unfused (assign + blocked_update + finalize) pipeline. |

### Modified

| Path | Change |
| --- | --- |
| `gmmxx/csrc/bindings.cpp` | Expose `spherical_fused` as a public op. |
| `gmmxx/_cuda.py` | Add `fused_spherical(x, init_means, init_var, init_log_w, n_components, total_n, reg_covar)` Python wrapper that returns `(means, var, weights, lse_per_sample, ids)`. |
| `gmmxx/cuda_ops.py` | Re-export `fused_spherical`. |
| `gmmxx/_runtime.py` | Add `cuda_spherical_fused_supported(d, k, dtype)` returning True for `0 < d ≤ 64, 0 < k ≤ 128, dtype ∈ {fp32, fp16, bf16}`. |
| `gmmxx/interface.py` | `_train_spherical_cuda` calls the fused wrapper when `cuda_spherical_fused_supported` is True; otherwise falls back to the existing assign+blocked+finalize path. Sets `cuda_fused_update_enabled_=True` on success. |
| `setup.py` | Add `spherical_fused.cu` and the new include dir. |
| `benchmarks/benchmark_cuda_vs_triton_spherical.py` | Add a fused-vs-unfused row to the timing table on supported shapes. |
| `README.md` | Update CUDA section to reflect Plan 5. |

---

## Numerical contract

Same as Plan 2/3/4: `rtol=1e-4` fp32, `rtol=5e-3` fp16/bf16 for `means`/`var`/`weights`. The fused kernel's per-CTA partials use fp32 accumulators throughout; the only difference vs the unfused pipeline is reduction order (per-CTA → global vs per-token global), which is bounded by atomic ULP drift.

`labels` (cluster IDs, recovered as `argmax(resp[m, :])` in the fused epilogue): ≥99% agreement with the unfused argmax-of-logits — same threshold as Plan 2.

---

## Conventions

- Working directory: `C:\Users\HEQ\Projects\flashGMM2`. Branch: `GMMXX-cuda` (post-`spherical-fast-plan4`).
- Dev rebuild: `$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .` after each `.cu`/`.cpp`/setup.py change.
- Test: `uv run pytest tests/<file> -v` per task; `uv run pytest tests/ -q` for full.

---

## Task 1 — Fused kernel (safe path)

**Files:** Create `gmmxx/csrc/fused/spherical_fused.cu`; create `gmmxx/csrc/fused/spherical.h`; modify `setup.py`.

The safe-path fused kernel is one thread per row × scalar FMA over K and D. Simpler than mma; ships first as a correctness baseline. Plan 5 Task 4 adds the sm80 mma variant.

### Header

```cpp
// gmmxx/csrc/fused/spherical.h
#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace fused { namespace spherical {

// Fused E + M single-tile launch.
//
// Inputs:
//   x: (B, N, D) fp32 / fp16 / bf16, contiguous, CUDA.
//   means: (B, K, D) same dtype as x. Current iterate.
//   var: (B, K) fp32. Current iterate.
//   log_w: (B, K) fp32. Current iterate.
//   reg_covar: clamp threshold for variance.
//
// Outputs (returned as a tuple):
//   new_means: (B, K, D) same dtype as x.
//   new_var: (B, K) fp32.
//   new_weights: (B, K) fp32.
//   lse_per_sample: (B, N) fp32 — log-likelihood used for ELBO.
//   labels: (B, N) int32 — argmax of responsibilities; matches Plan 2 assign output.
//
// Constraints: D <= 64, K <= 128. Caller verifies via cuda_spherical_fused_supported.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused(const at::Tensor& x,
      const at::Tensor& means,
      const at::Tensor& var,
      const at::Tensor& log_w,
      double reg_covar);

// Safe SIMT path (fp32 inputs always; fp16/bf16 fallback if sm80 path can't be used).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_safe(const at::Tensor& x,
           const at::Tensor& means,
           const at::Tensor& var,
           const at::Tensor& log_w,
           double reg_covar);

// sm80+ mma path (fp16/bf16 inputs only). Plan 5 Task 4 adds this.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_sm80(const at::Tensor& x,
           const at::Tensor& means,
           const at::Tensor& var,
           const at::Tensor& log_w,
           double reg_covar);

}}}
```

### Safe kernel pseudocode

The safe path uses one thread per row of x. Each thread:

1. Loads its row x[m] into registers (D ≤ 64 fp32 registers per thread is fine).
2. Loads all K centroids into SMEM once at CTA start (one cooperative load).
3. Per K cluster k (loop over K):
   - Compute `dist = Σ_d (x[d] − μ_k[d])²`.
   - Compute `logit_k = log_w[k] − 0.5·D·log(2π·var[k]) − 0.5·dist/var[k]`.
   - Track running `(max, sumexp)` for stable softmax.
4. After K loop: per-thread `log_norm = max + log(sumexp)`; write to `lse_per_sample[m]`.
5. Per K cluster k (second loop over K):
   - Recompute logit_k (no per-thread state to hold all K logits if K=128 — that's 128 fp32 = 512B, OK actually).
   - `r_k = exp(logit_k − log_norm)`.
   - Atomic-add to per-CTA SMEM accumulator `partials[k]` += `(r_k * x, r_k * ||x||², r_k)`.
6. After all rows in the CTA processed: warp 0 flushes `partials[k]` to global `sums[b,k]`, `sumsq[b,k]`, `counts[b,k]` via one atomicAdd per (cluster, feature) tuple.
7. Argmax of resp = argmax of logit_k (idempotent under monotone exp); track during step 5 and write to `labels[m]`.

After the fused kernel runs across all CTAs (covering N), a small `finalize_spherical` finalizer kernel divides sums/counts and clamps variance — the existing Plan 2 finalize kernel works as-is.

Actually — the fused kernel can do the finalize inline if we add a second launch (one CTA per cluster) or a synchronization pattern. But in Plan 5 we keep the design simple: fused returns the partials, then call existing `finalize_spherical`. The Python wrapper handles this.

### Implementation

Read the existing `gmmxx/csrc/estep/spherical_safe.cu` (Plan 2's safe assign/logsumexp/resp). The fused kernel is a structural fusion of those three plus the M-step accumulator.

Key constants:
```cpp
static constexpr int FUSED_BLOCK_N = 128;  // rows per CTA
static constexpr int FUSED_THREADS = 128;  // one thread per row
static constexpr int FUSED_MAX_K = 128;
static constexpr int FUSED_MAX_D = 64;
```

The per-CTA SMEM layout:
- Centroids: `means_smem[K * D]` → up to 128*64 = 8192 elements = 32 KiB fp32 (or 16 KiB fp16). Fits in shared mem on sm_80+ (164 KiB per SM).
- Partials: `partial_sums[K * D]` (fp32) + `partial_sumsq[K]` (fp32) + `partial_counts[K]` (int32) → up to 32 KiB + 512 B + 512 B.
- Total: ~64 KiB SMEM per CTA at the upper bound. Fits in sm_89's 100 KiB / SM dynamic SMEM with cudaFuncSetAttribute.

For the safe (fp32) path the centroids are loaded straight into SMEM and stay resident; the K loop reads from SMEM. For fp16/bf16 inputs (still routed to safe in Task 1 — sm80 path is Task 4), we cast on the fly in registers.

### Step 1.1 — Write the safe path

Create `gmmxx/csrc/fused/spherical_fused.cu` containing:
- The templated `spherical_fused_safe_kernel<T>` (per-thread loop over K twice).
- `fused_safe(...)` host launcher: validates inputs (D ≤ 64, K ≤ 128, etc.), allocates outputs, calls finalize after the per-CTA partials are accumulated, returns the tuple.
- A stub `fused_sm80(...)` that calls `fused_safe` (Task 4 replaces).
- `fused(...)` dispatcher: route by dtype + compute capability. fp32 → safe; fp16/bf16 + sm_80+ → sm80 (currently stub→safe); fp16/bf16 + older arch → safe.

Reference the existing `assign_safe`, `blocked_update_spherical`, and `finalize_spherical` for the math; the formulas don't change.

### Step 1.2 — Update setup.py

Add to `sources`:
```python
str(CSRC / "fused" / "spherical_fused.cu"),
```
Add to `include_dirs`:
```python
str(CSRC / "fused"),
```

### Step 1.3 — Wire bindings.cpp

Add the include:
```cpp
#include "fused/spherical.h"
```

And in `NB_MODULE`:
```cpp
m.def(
    "spherical_fused",
    &gmmxx::fused::spherical::fused,
    nb::arg("x"),
    nb::arg("means"),
    nb::arg("var"),
    nb::arg("log_w"),
    nb::arg("reg_covar"),
    "Fused single-tile spherical E/M. Returns (means, var, weights, "
    "lse_per_sample, labels). Requires D <= 64, K <= 128.");
```

### Step 1.4 — Build + smoke test

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .
```

```bash
uv run python -c "
import torch
from gmmxx import _C
torch.manual_seed(0)
B, N, D, K = 1, 256, 16, 8
x = torch.randn(B, N, D, device='cuda')
means = torch.randn(B, K, D, device='cuda')
var = torch.ones(B, K, device='cuda')
log_w = torch.zeros(B, K, device='cuda')
new_means, new_var, new_weights, lse, labels = _C.spherical_fused(x, means, var, log_w, 1e-6)
print('new_means shape:', new_means.shape, 'new_weights sum:', new_weights.sum().item())
print('weights ≈ 1:', abs(new_weights.sum().item() - 1.0) < 1e-3)
"
```

Expected: `new_means` shape `(1, 8, 16)`, weights summing to ~1.0, labels populated.

### Step 1.5 — Commit

```bash
git add gmmxx/csrc/fused/ gmmxx/csrc/bindings.cpp setup.py
git commit -m "$(cat <<'EOF'
Add fused single-tile spherical E/M kernel (safe path)

spherical_fused: one CTA processes BLOCK_N=128 rows × all K clusters
in a single kernel pass. Computes logits + softmax responsibilities
+ per-cluster partial sums/sumsq/counts in registers/SMEM, then
flushes partials to global atomically. Eliminates two global-memory
round-trips per EM iteration vs the assign+blocked+finalize sequence.

Constraints: D <= 64, K <= 128 (centroid + partial accumulator working
set must fit in SMEM). Outside this window, the unfused pipeline is
still correct and stays in service.

This task adds the safe SIMT path; Task 4 will add the sm80 mma path
for fp16/bf16. Task 5 wires the dispatcher gate; Task 6 adds the
correctness tests; Tasks 7-9 wire the train loop and ship.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Python wrapper in `_cuda.py`

**Files:** Modify `gmmxx/_cuda.py`; modify `gmmxx/cuda_ops.py`.

```python
def fused_spherical(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    reg_covar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused E/M single-tile spherical kernel.

    Returns (means, var, weights, lse_per_sample, labels). Caller is
    responsible for checking the shape window (D <= 64, K <= 128) via
    cuda_spherical_fused_supported.
    """
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    try:
        return _C.spherical_fused(x, means, var, log_w, float(reg_covar))
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"fused_spherical failed: {exc}") from exc
```

Re-export through `cuda_ops.py`:
```python
fused_spherical = _cuda.fused_spherical
```
And add to `__all__`.

Smoke + commit.

---

## Task 3 — Add `cuda_spherical_fused_supported` gate

**Files:** Modify `gmmxx/_runtime.py`.

```python
def cuda_spherical_fused_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the fused single-tile spherical kernel can handle this shape+dtype.

    Plan 5 (safe + sm80): supports d in (0, 64], n_components in (0, 128] for
    dtype in {fp32, fp16, bf16}. Outside this window, the unfused path is
    used (correct but ~2-3x slower for medium shapes).
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 64):
        return False
    if not (0 < n_components <= 128):
        return False
    return True
```

Note: this is a separate gate from `cuda_spherical_supported`. The fused gate is a STRICT SUBSET — it implies the regular spherical gate is also True.

Quick test in `tests/test_dispatch.py` to verify the new gate works.

Commit.

---

## Task 4 — sm80 mma variant of the fused kernel

**Files:** Modify `gmmxx/csrc/fused/spherical_fused.cu`.

Replace the stub `fused_sm80(...)` with a real implementation that uses mma.sync to compute the cross-product `cross[m,k] = Σ_d x[m,d] * μ_k[d]` for the (BLOCK_N, K) tile in one mma loop, then computes logits + responsibilities + partial accumulation in fp32 register/SMEM.

Reference: the existing `assign_sm80_kernel` for the mma loop; the `logsumexp_sm80_kernel` for the running (max, sumexp) pattern; this kernel's safe variant for the M-step partial accumulation pattern.

**Scope-down permission**: if implementing the full sm80 variant is too much in one task, leave the stub in place for now and report status BLOCKED with the specific blocking reason. The safe-path fused kernel (Task 1) ships value on its own — fp32 fused works without the mma variant. The sm80 variant adds another ~2× perf for fp16/bf16 in the (D≤64, K≤128) window.

Smoke + commit.

---

## Task 5 — Per-kernel correctness tests

**Files:** Create `tests/test_cuda_spherical_fused.py`.

Compares fused output to:
- The unfused (assign + blocked_update_spherical + finalize_spherical) pipeline at fp32 rtol=1e-4.
- A torch reference at fp16 rtol=5e-3.

```python
"""Correctness tests for the fused single-tile spherical E/M kernel."""

from __future__ import annotations
import math
import pytest
import torch


def _has_cuda():
    try:
        from gmmxx._cuda import has_cuda
        return has_cuda()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")


def _setup(B=1, N=256, D=32, K=16, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    means = torch.randn(B, K, D, device="cuda", dtype=dtype)
    var = torch.rand(B, K, device="cuda").clamp_min(0.5).float()
    log_w = torch.log_softmax(torch.randn(B, K, device="cuda"), dim=-1).float()
    return x, means, var, log_w


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D,K", [(8, 16), (16, 32), (32, 64), (64, 128)])
def test_fused_matches_unfused_pipeline(dtype, D, K):
    """Fused output should match the unfused assign+blocked+finalize sequence."""
    from gmmxx import _cuda
    x, means, var, log_w = _setup(N=256, D=D, K=K, dtype=dtype)
    reg_covar = 1e-6

    # Unfused path
    ids_u = _cuda.spherical_assign(x, means, var, log_w)
    sums_u, sumsq_u, counts_u = _cuda.blocked_update_spherical(x, ids_u, K)
    new_means_u, new_var_u, new_weights_u = _cuda.finalize_spherical(
        sums_u, sumsq_u, counts_u, means, var, x.shape[1], reg_covar
    )
    lse_u = _cuda.spherical_logsumexp(x, means, var, log_w)

    # Fused path
    new_means_f, new_var_f, new_weights_f, lse_f, ids_f = _cuda.fused_spherical(
        x, means, var, log_w, reg_covar
    )

    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (5e-3, 5e-3)
    assert torch.allclose(new_means_f, new_means_u, rtol=rtol, atol=atol)
    assert torch.allclose(new_var_f, new_var_u, rtol=rtol, atol=atol)
    assert torch.allclose(new_weights_f, new_weights_u, rtol=rtol, atol=atol)
    assert torch.allclose(lse_f, lse_u, rtol=rtol, atol=atol)
    # Labels should agree on argmax (ties may flip).
    agree = (ids_f == ids_u).float().mean().item()
    assert agree >= 0.99, f"label agreement {agree:.3f}"


def test_fused_zero_N():
    from gmmxx import _cuda
    x = torch.empty(1, 0, 16, device="cuda")
    means = torch.randn(1, 8, 16, device="cuda")
    var = torch.ones(1, 8, device="cuda")
    log_w = torch.zeros(1, 8, device="cuda")
    nm, nv, nw, lse, ids = _cuda.fused_spherical(x, means, var, log_w, 1e-6)
    assert nm.shape == (1, 8, 16)
    # Empty input → unchanged means and zero counts; weights[k]=0 for all k.
    assert torch.allclose(nm, means)


@pytest.mark.parametrize("D,K", [(8, 16), (32, 64), (64, 128)])
def test_fused_train_loop_converges(D, K):
    """A few iterations of fused should converge to a finite ELBO."""
    from gmmxx import _cuda
    torch.manual_seed(0)
    x = torch.randn(1, 4096, D, device="cuda")
    means = torch.randn(1, K, D, device="cuda")
    var = torch.ones(1, K, device="cuda")
    log_w = torch.full((1, K), -math.log(K), device="cuda")

    last_lse = None
    for _ in range(5):
        means, var, weights, lse, ids = _cuda.fused_spherical(
            x, means, var, log_w, 1e-6
        )
        log_w = torch.log(weights.clamp_min(1e-30))
        elbo = lse.mean().item()
        assert math.isfinite(elbo)
        if last_lse is not None:
            assert elbo >= last_lse - 1e-3, "ELBO should not decrease materially"
        last_lse = elbo
```

Run + commit.

---

## Task 6 — Wire fused into `_train_spherical_cuda`

**Files:** Modify `gmmxx/interface.py`.

Replace the (assign → blocked_update → finalize) block in `_train_spherical_cuda` with a dispatch:

```python
from . import _runtime
if _runtime.cuda_spherical_fused_supported(D, K, dtype):
    means, var, weights, lse, ids = _cuda_mod.fused_spherical(
        data_b, means, var, log_w, self.reg_covar
    )
    self.cuda_fused_update_enabled_ = True
    log_w = torch.log(weights.clamp_min(1e-30))
    lower_bound_history.append(float(lse.mean().item()))
    if abs(lower_bound_history[-1] - prev_lb) < self.tol:
        break
    prev_lb = lower_bound_history[-1]
    n_iter += 1
    continue  # skip the unfused branch
# Existing unfused path (assign + blocked_update + finalize) follows here.
ids = _cuda_mod.spherical_assign(data_b, means, var, log_w)
# ... existing code ...
```

The `cuda_fused_update_enabled_` attribute is already declared (Plan 1) — wire the True case here.

Run full suite to check for regressions:
```bash
uv run pytest tests/ -q
```

Commit.

---

## Task 7 — Perf benchmark: fused vs unfused

**Files:** Modify `benchmarks/benchmark_cuda_vs_triton_spherical.py`.

Add a `--fused-only` flag and a side-by-side comparison row. Verify the fused path is ≥1.0× the unfused path on supported shapes (it should be 1.5–2.5× faster).

If the fused path is slower (unlikely but possible for very small N where launch overhead dominates), add an N threshold to the dispatcher gate.

Commit with the timing table in the message.

---

## Task 8 — README + tag

Update CUDA section to note Plan 5 closure:
> "Spherical CUDA is fully feature-complete: assign/logsumexp/resp on sm80 mma, sorted-run M-step atomic coalescing, AND fused single-tile E/M for D≤64, K≤128 shapes (Plans 2–5)."

Tag `spherical-fused-plan5` and commit.

---

## Self-Review Checklist

**1. Spec coverage**

| Spec section | Plan task |
| --- | --- |
| §4 fused single-tile spherical D≤64, K≤128 | Tasks 1, 4 |
| §5d register-tiled per-cluster accumulator | Task 1 |
| §6 fp32/fp16/bf16 tolerance | Task 5 |
| §7 dispatch | Tasks 3, 6 |

Unaddressed (Plan 6+):
- Persistent E-step
- Multi-stream events
- Diag/tied/full fused paths

**2. Placeholder scan** — Task 1's kernel pseudocode is structural; the implementer must fill in the actual arithmetic. Task 4's sm80 variant references the existing assign_sm80 / logsumexp_sm80 as templates — no novel mma layout needed.

**3. Type consistency** — `at::Tensor` throughout; tuple of (means, var, weights, lse, labels) consistent across safe/sm80/dispatcher; `_runtime.cuda_spherical_fused_supported` 3-arg signature matching the existing pattern.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-03-gmmxx-cuda-spherical-fused.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Tasks 1, 4 benefit from opus given the kernel-engineering complexity. Tasks 2, 3, 5, 6, 7, 8 use sonnet.

**2. Inline Execution** — Same session.

After Plan 5: **Plan 6** = persistent E-step kernels + multi-stream events + approx top-K spherical. Then **Plans 7–9** for diag, tied, full. Then **Plan 10** = `large_n.py` integration.
