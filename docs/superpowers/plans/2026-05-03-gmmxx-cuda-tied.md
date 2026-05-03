# GMMXX CUDA Backend — Plan 7: Tied CUDA Covariance

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the tied covariance type to the CUDA backend. After this plan, `GMMXX(covariance_type="tied", backend="cuda")` runs full EM training and inference on CUDA. Tied covariance has a single shared `D×D` covariance matrix (stored as a lower-triangular Cholesky factor `L` such that `Σ = L Lᵀ`).

**Key insight: tied = spherical on projected coordinates.** If we project `y_n = L⁻¹·x_n` and `ν_k = L⁻¹·μ_k`, then `||L⁻¹(x_n − μ_k)||² = ||y_n − ν_k||²`. The Mahalanobis distance reduces to Euclidean distance in projected space. The full tied log-likelihood becomes:

```
log p_k(x_n) = log w_k − 0.5·D·log(2π) − log|L| − 0.5·||y_n − ν_k||²
```

This is exactly the spherical formula with `var=1` and a constant log-determinant offset. **The existing `spherical_assign` / `spherical_logsumexp` / `spherical_resp` kernels work as-is on projected coordinates.** No new E-step kernels needed.

**Architecture:** No new `.cu` files for the E-step (reuse spherical kernels via projection). One new kernel for the M-step (`blocked_tied.cu`) that accumulates per-cluster sums + a global `(B, D, D)` outer-product accumulator. Tied finalize runs in Python via `torch.linalg.cholesky` (cheap, one Cholesky per fit iteration). New Python wrappers in `_cuda.py` orchestrate projection, kernel calls, and Cholesky factorization. `interface.py` adds `_train_tied_cuda` analogous to spherical.

**Tech Stack:** CUDA 12.8+, sm_80+ optional (existing spherical_sm80 mma path is reused for free on projected fp16/bf16 coordinates). PowerShell + uv on the host.

**Spec sections covered:** §4 tied E/M coverage, §5c per-covariance finalize semantics (tied uses `+ reg_covar·I` then Cholesky), §7 dispatch.

**Out of scope (deferred):**
- Fused single-tile tied E/M → Plan 9 perf optimizations
- sm_80 mma + sorted-run M-step optimizations for tied → Plan 9
- Full covariance → Plan 8
- `large_n.py` integration → Plan 10

**Foundation assumed:** Plans 1–6 complete (`diag-cuda-plan6` tag). Spherical CUDA is feature-complete and used as the projected-coordinate E-step. Diagonal CUDA exists for reference patterns.

---

## Numerical formulas — tied covariance

For tied Gaussian mixture, all components share a single covariance matrix `Σ` with Cholesky factor `L` (lower-triangular, `Σ = L Lᵀ`):

- **log-likelihood**: `log p_k(x) = log w_k − 0.5·D·log(2π) − log|L| − 0.5·||L⁻¹(x − μ_k)||²`
- **log|L|** = `Σ_d log L[d,d]` — sum of log of L's diagonal entries; constant across `k` and across `n`.

In **projected coordinates**:
- `y_n = L⁻¹·x_n` — solve triangular system once per fit iteration.
- `ν_k = L⁻¹·μ_k` — solve once per fit iteration.
- `||L⁻¹(x_n − μ_k)||² = ||y_n − ν_k||²` — Euclidean in projected space.
- The constant `log|L|` term is folded into a per-component constant: `log_w_adj_k = log w_k − log|L|`.

So the E-step on projected coordinates is exactly: spherical assign/logsumexp/resp with `var=1`, `means=ν`, `log_w=log_w_adj`. The existing CUDA spherical kernels work as-is.

**M-step**:
- `μ_k_new = (Σ_n r_{n,k} x_n) / n_k` — per-cluster sum / count, same as spherical/diag (sums computed in original `x` space, NOT projected).
- `Σ_new = (1/N) · Σ_k Σ_n r_{n,k} (x_n − μ_k_new)(x_n − μ_k_new)ᵀ`

Algebraic simplification using `Σ_k r_{n,k} = 1`:
- `Σ_n Σ_k r_{n,k} x_n x_nᵀ = Σ_n x_n x_nᵀ = XᵀX` (single global accumulator!)
- `Σ_n Σ_k r_{n,k} x_n μ_kᵀ = Σ_k (n_k μ_k_new) μ_k_newᵀ = Σ_k n_k μ_k μ_kᵀ` (after using μ_k_new = Σ_n r_{n,k} x_n / n_k)

So: `Σ_new = (1/N) · [XᵀX − Σ_k n_k μ_k_new μ_k_newᵀ]`

That's:
- One global `XᵀX` accumulator: `(B, D, D)` — computed via a single torch matmul on GPU. Or accumulated in the M-step kernel.
- After means are computed, subtract `Σ_k n_k μ_k μ_kᵀ` (cheap: K outer products of D-vectors, O(K·D²)).
- Add `reg_covar · I`.
- Cholesky factor → new `L`.

**Implementation choice**: do `XᵀX` via `torch.matmul(x.transpose(-1, -2), x)` on host, NOT in a CUDA kernel. This is one cuBLAS GEMM call per fit iteration — fast and saves us writing an outer-product kernel. The per-cluster sums (B, K, D) reuse `blocked_update_spherical`'s `sums` output (we ignore its `sumsq` — wasted compute but small).

This means **Plan 7 needs no new CUDA kernels**. Just Python orchestration.

Wait — the per-cluster `sums` is what `blocked_update_spherical` produces. That works for tied too (same formula `Σ_n r_{n,k} x_n` per cluster). So we reuse it.

---

## File Structure

### Created

| Path | Responsibility |
| --- | --- |
| `tests/test_cuda_tied.py` | Per-method correctness for tied CUDA (fit, predict, predict_proba, score_samples). |

### Modified

| Path | Change |
| --- | --- |
| `gmmxx/_cuda.py` | Add 5 host-side helpers: `tied_project(x, L)`, `tied_assign(x, means, L, log_w_adj)`, `tied_logsumexp(...)`, `tied_resp(...)`, `tied_finalize(sums, xx_total, counts, total_n, reg_covar)`. The first 4 are thin compositions of `solve_triangular` + the existing spherical wrappers. `tied_finalize` does the algebraic Σ_new = (XᵀX − Σ_k n_k μ_k μ_kᵀ) / N + reg_covar·I, then Cholesky. |
| `gmmxx/cuda_ops.py` | Re-export the 5 new tied helpers. |
| `gmmxx/_runtime.py` | `cuda_tied_supported(d, k, dtype) → True` for `0 < D ≤ 64, 0 < K ≤ 512, dtype ∈ {fp32, fp16, bf16}`. |
| `gmmxx/interface.py` | Add `_train_tied_cuda` method analogous to `_train_diag_cuda`. Add CUDA branches in `predict()`/`predict_proba()`/`score_samples()`/`score()` for tied. The fit() returns `means`, `chol` (B, D, D — Cholesky factor); `covariances_` should expose `chol @ chol.T` as the symmetric covariance for sklearn compat. |
| `gmmxx/_dispatch.py` | Add tied entries to `_TRITON_OPS_BY_NAME`. |
| `tests/test_gmmxx.py` | Add `test_tied_full_pipeline_each_backend` parametrized test. |
| `README.md` | Update CUDA section to note tied now on CUDA. |

---

## Numerical contract

| Output | fp32 | fp16/bf16 |
| --- | --- | --- |
| `means_` | rtol=1e-4, atol=1e-4 | rtol=5e-3 |
| `covariances_` (full D×D matrix from chol @ chol.T) | rtol=1e-3, atol=1e-3 | rtol=1e-2 |
| `lower_bound_` / `score_samples` | rtol=1e-4 | rtol=1e-2 |
| `labels_` | ≥ 99% agreement on separable; ≥ 95% on near-degenerate | same |

Tolerances slightly looser on covariances because Cholesky factorization compounds rounding.

---

## Conventions

- Working directory: `C:\Users\HEQ\Projects\flashGMM2`. Branch: `GMMXX-cuda` (post-`diag-cuda-plan6`).
- Reference: `gmmxx/torch_fallback._batch_gmm_Tied_torch_native_inner` for the math; `gmmxx/fused_update_triton._fused_single_tile_tied_native_kernel` for the Triton tied implementation.

---

## Task 1 — Tied helpers in `gmmxx/_cuda.py`

Append to `gmmxx/_cuda.py` after the diag block:

```python
# ---------------------------------------------------------------------------
# Tied kernels (Plan 7 — projected-coords approach)
#
# The tied E-step reuses the spherical CUDA kernels on projected coordinates:
#   y = L⁻¹ x; ν_k = L⁻¹ μ_k; ||L⁻¹(x − μ_k)||² = ||y − ν_k||²
# The shared log|L| term is folded into a per-component log_w offset.
# ---------------------------------------------------------------------------


def tied_project(x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Project x via L⁻¹ where L is lower-triangular. Returns same shape as x.

    For (B, N, D) x and (B, D, D) L:
      y[b, n, :] = solve_triangular(L[b], x[b, n, :].T, upper=False).T
    Implemented in batch as a single solve_triangular call.
    """
    require_cuda()
    x = _check_input(x, "x")
    L = _check_input(L, "L")
    # solve_triangular expects (B, D, D) @ (B, D, N) for batched. Reshape:
    # x: (B, N, D); we want to solve L @ Y^T = X^T → Y^T has shape (B, D, N).
    # Then transpose back to (B, N, D).
    x_t = x.transpose(-1, -2).contiguous()  # (B, D, N)
    y_t = torch.linalg.solve_triangular(L, x_t, upper=False)  # (B, D, N)
    return y_t.transpose(-1, -2).contiguous()  # (B, N, D)


def tied_log_det(L: torch.Tensor) -> torch.Tensor:
    """log|L| = Σ_d log L[b, d, d]. Returns (B,) fp32."""
    diag = torch.diagonal(L, dim1=-2, dim2=-1)  # (B, D)
    return diag.abs().log().sum(-1)


def tied_assign(
    x: torch.Tensor,
    means: torch.Tensor,
    L: torch.Tensor,
    log_w: torch.Tensor,
) -> torch.Tensor:
    """Tied E-step assign via spherical kernel on projected coordinates.

    Returns int32 (B, N).
    """
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    # Project both x and means.
    y = tied_project(x, L)
    nu = tied_project(means, L)
    # Spherical with var=1; the constant -log|L| - 0.5*D*log(2π) cancels under
    # argmax across k, so we can pass var=1 and the unmodified log_w.
    B, K, D = nu.shape
    var = torch.ones(B, K, dtype=torch.float32, device=x.device)
    return spherical_assign(y, nu, var, log_w)


def tied_logsumexp(
    x: torch.Tensor,
    means: torch.Tensor,
    L: torch.Tensor,
    log_w: torch.Tensor,
) -> torch.Tensor:
    """Tied E-step logsumexp. Returns (B, N) fp32 — true log-likelihood per sample."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    y = tied_project(x, L)
    nu = tied_project(means, L)
    B, K, D = nu.shape
    var = torch.ones(B, K, dtype=torch.float32, device=x.device)
    # spherical_logsumexp with var=1 returns log Σ_k exp(log_w_k − 0.5·D·log(2π) − 0.5·||y−ν_k||²)
    # We want log Σ_k exp(log_w_k − log|L| − 0.5·D·log(2π) − 0.5·||y−ν_k||²)
    # = spherical_lse(y, ν, 1, log_w) − log|L|.
    spherical_lse = spherical_logsumexp(y, nu, var, log_w)
    log_det_L = tied_log_det(L).unsqueeze(-1)  # (B, 1)
    return spherical_lse - log_det_L


def tied_resp(
    x: torch.Tensor,
    means: torch.Tensor,
    L: torch.Tensor,
    log_w: torch.Tensor,
    log_norm: torch.Tensor,
) -> torch.Tensor:
    """Tied E-step responsibilities. Returns fp32 (B, N, K).

    log_norm: must be the TRUE tied log-norm (= tied_logsumexp output, not the
    spherical projection's logsumexp).
    """
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    log_norm = _check_input(log_norm, "log_norm", dtype=torch.float32)
    y = tied_project(x, L)
    nu = tied_project(means, L)
    B, K, D = nu.shape
    var = torch.ones(B, K, dtype=torch.float32, device=x.device)
    # We need exp(logit_k − log_norm). Spherical resp computes exp(logit'_k − log_norm')
    # where logit' is the spherical (no log|L| term) and log_norm' is the spherical
    # logsumexp. To get tied resp, we shift: pass (log_norm + log|L|) as the
    # spherical log_norm so the difference works out.
    log_det_L = tied_log_det(L).unsqueeze(-1)  # (B, 1)
    log_norm_for_spherical = log_norm + log_det_L  # (B, N)
    return spherical_resp(y, nu, var, log_w, log_norm_for_spherical)


def tied_finalize(
    sums: torch.Tensor,        # (B, K, D) — Σ_n r_{n,k} x_n  from spherical blocked_update
    xx_total: torch.Tensor,    # (B, D, D) — Σ_n x_n x_nᵀ (compute on host via x.T @ x)
    counts: torch.Tensor,      # (B, K) int32 — Σ_n 1 (hard-assign counts) OR fp32 if soft
    total_n: int,
    reg_covar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tied finalize: divide sums, compute new Σ from XᵀX − Σ_k n_k μ_k μ_kᵀ,
    add reg_covar·I, Cholesky factor.

    Returns (new_means (B, K, D), new_chol (B, D, D), new_weights (B, K)).
    """
    require_cuda()
    sums = _check_input(sums, "sums", dtype=torch.float32)
    xx_total = _check_input(xx_total, "xx_total", dtype=torch.float32)
    counts = _check_input(counts, "counts")  # accept int32 or fp32

    B, K, D = sums.shape
    # New means: (B, K, D) = sums / counts. Empty clusters: keep at zero (rare).
    counts_f = counts.float()
    n_k = counts_f.clamp_min(1e-30)
    means_new = sums / n_k.unsqueeze(-1)  # (B, K, D)

    # Empty-cluster: weights are 0, means stay at 0 (or could be re-initialized
    # by the caller). Since this is the simple case, leave as-is.
    weights_new = counts_f / float(total_n)

    # Σ_k n_k μ_k μ_k^T summed over k:
    # weighted_outer = Σ_k n_k μ_k μ_kᵀ  has shape (B, D, D).
    # Compute via einsum: for each batch b, sum_k counts[b,k] * means[b,k,:] @ means[b,k,:].T
    # Equivalent to: (means * counts.unsqueeze(-1)).transpose(-1,-2) @ means
    weighted_means = means_new * counts_f.unsqueeze(-1)  # (B, K, D)
    sigma_k_sum = weighted_means.transpose(-1, -2) @ means_new  # (B, D, D)

    # Σ_new = (1/N) (xx_total − Σ_k n_k μ_k μ_k^T) + reg_covar I
    sigma = (xx_total - sigma_k_sum) / float(total_n)
    eye = torch.eye(D, device=sigma.device, dtype=sigma.dtype).unsqueeze(0)
    sigma = sigma + reg_covar * eye
    # Symmetrize to handle fp accumulation drift.
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))

    # Cholesky.
    chol_new = torch.linalg.cholesky(sigma)  # (B, D, D)

    return means_new, chol_new, weights_new
```

Smoke + commit. The smoke test should verify projection round-trip and a single tied EM iteration produces sane outputs.

```bash
uv run python -c "
import torch, math
from gmmxx import _cuda
torch.manual_seed(0)
B, N, D, K = 1, 256, 8, 4

# Random ground-truth tied params.
L_gt = torch.tril(torch.randn(B, D, D, device='cuda'))
L_gt += torch.eye(D, device='cuda').unsqueeze(0) * 2  # ensure positive definite
means_gt = torch.randn(B, K, D, device='cuda')

x = torch.randn(B, N, D, device='cuda')

# Verify projection round-trip
y = _cuda.tied_project(x, L_gt)
x_recon = (L_gt @ y.transpose(-1, -2)).transpose(-1, -2)
print('projection round-trip diff:', (x - x_recon).abs().max().item())

# Single tied iteration
log_w = torch.full((B, K), -math.log(K), device='cuda')
ids = _cuda.tied_assign(x, means_gt, L_gt, log_w)
lse = _cuda.tied_logsumexp(x, means_gt, L_gt, log_w)
print('lse mean:', lse.mean().item())

r = _cuda.tied_resp(x, means_gt, L_gt, log_w, lse)
print('resp sums:', r.sum(-1).mean().item())  # ~1.0
"
```

Commit.

---

## Task 2 — Re-export through `gmmxx/cuda_ops.py`

Add the 5 tied helpers to the public surface. Add to `__all__`.

Smoke + commit.

---

## Task 3 — `_runtime.cuda_tied_supported`

Replace the False stub:

```python
def cuda_tied_supported(d: int, n_components: int, dtype) -> bool:
    """Plan 7: D ≤ 64, K ≤ 512, dtype ∈ {fp32, fp16, bf16}."""
    import torch as _torch
    if dtype is None or dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 64): return False
    if not (0 < n_components <= 512): return False
    return True
```

Smoke + commit.

---

## Task 4 — `_train_tied_cuda` in `interface.py`

Add a method analogous to `_train_diag_cuda` but using the tied helpers:

```python
def _train_tied_cuda(self, x_b, batch_size):
    import math
    from . import _cuda as _cuda_mod
    B, N, D = x_b.shape
    K = self.k
    device = x_b.device

    # Init: random means; L = Cholesky of identity scaled by data variance.
    rng = torch.Generator(device=device).manual_seed(self.seed)
    init_idx = torch.randint(0, N, (B, K), generator=rng, device=device)
    means = torch.gather(x_b, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)).contiguous()
    data_var = x_b.float().var(dim=1).mean(-1, keepdim=True)  # (B, 1)
    L = (data_var.sqrt().unsqueeze(-1) * torch.eye(D, device=device).unsqueeze(0)).clone()
    log_w = torch.full((B, K), -math.log(K), dtype=torch.float32, device=device)
    weights = torch.full((B, K), 1.0 / K, dtype=torch.float32, device=device)

    # Precompute X^T X for the M-step (constant across iterations).
    xx_total = x_b.float().transpose(-1, -2) @ x_b.float()  # (B, D, D)

    lower_bound_history = []
    n_iter = 0
    prev_lb = -math.inf

    for _ in range(self.niter):
        n_iter += 1
        ids = _cuda_mod.tied_assign(x_b, means, L, log_w)
        lse = _cuda_mod.tied_logsumexp(x_b, means, L, log_w)
        lb = float(lse.mean().item())
        lower_bound_history.append(lb)

        # M-step: blocked_update_spherical reuses for sums (ignore sumsq).
        sums, _, counts = _cuda_mod.blocked_update_spherical(x_b, ids, K)
        means, L, weights = _cuda_mod.tied_finalize(
            sums, xx_total, counts, N, self.reg_covar
        )
        log_w = torch.log(weights.clamp_min(1e-30))

        if abs(lb - prev_lb) < self.tol:
            break
        prev_lb = lb

    # GMMXX stores covariances_ as the full D×D matrix (sklearn convention).
    cov_b = L @ L.transpose(-1, -2)  # (B, D, D)

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
        variances_b=cov_b,  # (B, D, D) — symmetric covariance matrix
        weights_b=weights,
        info=info,
        batch_size=batch_size,
    )
```

In `train()`, find the tied branch and add a CUDA early-return analogous to spherical/diag.

Smoke test: fit a small tied model, check `last_backend_used_=="cuda"` and `covariances_.shape == (D, D)`.

Run full suite + commit.

---

## Task 5 — Tied CUDA inference in `predict()`/`predict_proba()`/`score_samples()`/`score()`

Mirror the spherical/diag pattern. The methods need access to the Cholesky factor L. Where to store it?

Option A: extract from `self.covariances_b` via `torch.linalg.cholesky` at call time (cheap if D is small, one chol per call).

Option B: cache `self._L_b` set in `_train_tied_cuda` and reuse in inference paths. Faster but introduces state.

Plan 7 picks **Option A** for simplicity. The chol cost is O(D³) per inference call; for D ≤ 64 that's a few microseconds.

```python
elif self.covariance_type == "tied":
    from . import _dispatch
    shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
    resolved = _dispatch.resolve_backend_with_env(
        requested=self.backend, covariance="tied", shape=shape_for_dispatch,
        dtype=x_b.dtype, legacy_no_triton=self._legacy_no_triton,
    )
    if resolved == "cuda":
        # Recompute L from covariances_b.
        L = torch.linalg.cholesky(self.covariances_b)
        log_w = torch.log(self.weights_b.clamp_min(1e-30))
        if hasattr(self, '_method_name_predict'):  # adjust per method
            ids_b = _dispatch.dispatch_kernel(
                "tied_assign", "cuda", x_b, self.means_b, L, log_w
            )
            self.last_backend_used_ = "cuda"
            return self._squeeze_if_unbatched(ids_b).long()
    self.last_backend_used_ = resolved
    # Fall through to existing tied branch.
```

Add tied entries to `_TRITON_OPS_BY_NAME` in `_dispatch.py` if Triton wraps tied_assign etc. (Check `gmmxx/assign_*.py` — there may not be a separate `assign_tied_triton.py`; tied uses `_fused_single_tile_tied_native_kernel`. If no dedicated Triton op exists for these names, leave the map alone — `dispatch_kernel("tied_assign", "cuda", ...)` resolves only the cuda branch via `_cuda.tied_assign`.)

Run + commit.

---

## Task 6 — Tests

Create `tests/test_cuda_tied.py` covering:
- `tied_project` round-trip identity.
- `tied_assign` matches torch reference (manually compute via `linalg.solve_triangular`).
- `tied_logsumexp` matches torch reference.
- `tied_resp` rows sum to 1.
- `tied_finalize` produces a positive-definite Σ; Cholesky succeeds.
- End-to-end fit + predict + score_samples for `covariance_type="tied"`, `backend="cuda"`.

Run + commit.

---

## Task 7 — Parametrize `test_gmmxx.py` tied pipeline

Append `test_tied_full_pipeline_each_backend` analogous to spherical and diag. Note: `covariances_` shape for tied is `(D, D)` not `(K, D)`.

Run + commit.

---

## Task 8 — README + tag

Update CUDA section: "Tied is now on CUDA (Plan 7)…". Tag `tied-cuda-plan7`.

---

## Self-Review

**Spec coverage:** §4 tied E/M (covered via projection reuse), §5c tied finalize (`+ reg_covar·I` then Cholesky), §7 dispatch.

**No new CUDA kernels** — Plan 7 is pure Python orchestration. The performance is bounded by:
1. `solve_triangular` on the projection (O(N·D²))
2. Spherical kernels on projected coords (already optimized)
3. Single matmul `XᵀX` per iteration (O(N·D²))
4. Cholesky on (D, D) (O(D³))

This is the optimal O(N·D²) per iteration for tied EM. No room to do better without a fused projected-and-assign kernel — that's a Plan 9 perf optimization.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-03-gmmxx-cuda-tied.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — All tasks use sonnet (no novel CUDA needed; Python orchestration).
2. **Inline Execution** — Same session.

After Plan 7: **Plan 8** = full covariance CUDA (per-cluster covariance matrix + per-cluster Cholesky); **Plan 9** = sm80/sorted-run/fused optimizations across diag/tied/full; **Plan 10** = `large_n.py` integration; **Plan 11** = approx top-K.
