# GMMXX CUDA Backend — Plan 8: Full CUDA Covariance

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add full covariance to the CUDA backend. After this plan, `GMMXX(covariance_type="full", backend="cuda")` runs full EM training and inference on CUDA for `D ≤ 16, K ≤ 32`. Each component has its own `D×D` covariance matrix `Σ_k`, stored as a Cholesky factor `L_k` (per-cluster).

**Key insight:** Like Plan 7 (tied), Plan 8 can be **pure-torch orchestration** — `torch.linalg.solve_triangular` handles per-cluster projection in batch, and `torch.scatter_add_` handles per-cluster outer-product accumulation. **No new CUDA kernels needed.** This stays under spec §4's "fit/update uses Triton only for profitable D ≤ 8 shapes; inference supports D ≤ 16" — we'll target D ≤ 16, K ≤ 32 (slightly wider than Triton's D ≤ 8) since the torch primitives parallelize well even at D=16.

**Architecture:** Five Python helpers in `_cuda.py` orchestrating `solve_triangular` + `scatter_add_` + `cholesky_ex`. `_train_full_cuda` method in `interface.py`. Full CUDA inference branches in predict/predict_proba/score_samples. The shape window is more conservative than spherical/diag/tied because each cluster has an O(D²) precision representation.

**Tech Stack:** No new CUDA. Heavy use of `torch.linalg.solve_triangular`, `torch.linalg.cholesky_ex`, `torch.scatter_add_`. Reference: existing `gmmxx/torch_fallback._batch_gmm_Full_torch_native_inner` for the math.

**Spec sections covered:** §4 full E/M coverage, §5c full finalize semantics (`+ reg_covar·I`, per-cluster Cholesky), §7 dispatch.

**Out of scope (deferred):**
- A dedicated full CUDA kernel for D > 16 → Plan 9 perf optimizations
- sm_80 mma fused for full → Plan 9
- `large_n.py` integration → Plan 10
- Approx top-K (spherical-only) → Plan 11

**Foundation assumed:** Plans 1–7 complete (`tied-cuda-plan7` tag). Spherical, diag, tied all on CUDA. Plan 8 is the final per-covariance-type plan.

---

## Math reference

For full covariance, each component `k` has its own mean `μ_k`, Cholesky factor `L_k` (lower-triangular, `Σ_k = L_k L_kᵀ`), and weight `π_k`:

```
log p_k(x) = log π_k − 0.5·D·log(2π) − log|L_k| − 0.5·||L_k⁻¹(x − μ_k)||²
```

The Mahalanobis distance `||L_k⁻¹(x − μ_k)||²` requires a triangular solve per (n, k) pair. We vectorize via `torch.linalg.solve_triangular`:

```python
# x: (B, N, D); means: (B, K, D); L: (B, K, D, D)
diff = x.unsqueeze(2) - means.unsqueeze(1)         # (B, N, K, D)
diff_t = diff.permute(0, 2, 3, 1).contiguous()      # (B, K, D, N)
z = torch.linalg.solve_triangular(L, diff_t, upper=False)  # (B, K, D, N)
dist_sq = z.pow(2).sum(2).permute(0, 2, 1)          # (B, N, K)
log_det_L = L.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)  # (B, K)
log_norm_const = 0.5 * D * math.log(2 * math.pi)
logits = log_w.unsqueeze(1) - log_norm_const - log_det_L.unsqueeze(1) - 0.5 * dist_sq
```

This is a single batched `solve_triangular` call covering all `(B, K, D, N)` solves at once.

For the M-step (hard-assign):
```python
# Per-cluster sums + per-cluster outer-product accumulator.
# scatter_add_ on (B, K, D) for sums and (B, K, D, D) for outer.
sums = torch.zeros(B, K, D)
sums.scatter_add_(1, ids.unsqueeze(-1).expand(-1, -1, D), x.float())

xx_per_point = x.float().unsqueeze(-1) * x.float().unsqueeze(-2)  # (B, N, D, D)
outer_sums = torch.zeros(B, K, D, D)
outer_sums.scatter_add_(1, ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D, D), xx_per_point)

counts = torch.zeros(B, K, dtype=torch.int32)
counts.scatter_add_(1, ids, torch.ones_like(ids))
```

For finalize:
```python
# Σ_k = (outer_sums_k / n_k) - μ_k μ_k^T   (Plan 7-style algebra)
n_k = counts.float().clamp_min(1e-30)
means_new = sums / n_k.unsqueeze(-1)
Σ_k = outer_sums / n_k.unsqueeze(-1).unsqueeze(-1) - means_new.unsqueeze(-1) * means_new.unsqueeze(-2)
Σ_k = Σ_k + reg_covar * eye(D)
Σ_k = 0.5 * (Σ_k + Σ_k.transpose(-1, -2))  # symmetrize
L_k = torch.linalg.cholesky_ex(Σ_k).L  # per-cluster Cholesky; cholesky_ex handles non-PD gracefully
```

`cholesky_ex` returns a tuple `(L, info)`; `info != 0` indicates the matrix wasn't positive definite. Safer than `cholesky` which raises.

---

## File Structure

### Created

| Path | Responsibility |
| --- | --- |
| `tests/test_cuda_full.py` | Per-method correctness vs torch reference; end-to-end fit + inference. |

### Modified

| Path | Change |
| --- | --- |
| `gmmxx/_cuda.py` | Add 5 helpers: `full_assign / full_logsumexp / full_resp / full_blocked_update / full_finalize`. All pure-torch (no _C kernel calls). |
| `gmmxx/cuda_ops.py` | Re-export the 5 helpers. |
| `gmmxx/_runtime.py` | `cuda_full_supported(d, k, dtype) → True` for `0 < D ≤ 16, 0 < K ≤ 32`, dtype `∈ {fp32, fp16, bf16}`. |
| `gmmxx/interface.py` | Add `_train_full_cuda` method; route full-cov training through it; add full CUDA branches in predict/predict_proba/score_samples. |
| `tests/test_gmmxx.py` | Add `test_full_full_pipeline_each_backend` parametrized test. |
| `README.md` | Update CUDA section: "All four covariance types now on CUDA." |

---

## Numerical contract

Per spec §6 (with adjustments for full's outer-product cancellation):

| Output | fp32 | fp16/bf16 |
| --- | --- | --- |
| `means_` | rtol=1e-4, atol=1e-3 | rtol=5e-3 |
| `covariances_` (B, K, D, D) | rtol=1e-3, atol=1e-5 (outer-product cancellation) | rtol=1e-2 |
| `lower_bound_` / `score_samples` | rtol=1e-4 | rtol=1e-2 |
| `labels_` | ≥ 99% agreement separable; ≥ 95% near-degenerate | same |

---

## Conventions

- Working directory: `C:\Users\HEQ\Projects\flashGMM2`. Branch: `GMMXX-cuda` (post-`tied-cuda-plan7`).
- No rebuild needed for any task (pure-Python edits).
- Reference: `gmmxx/torch_fallback._batch_gmm_Full_torch_native_inner` for the math; `gmmxx/assign_full_triton.py` for the existing Triton pattern.

---

## Task 1 — Full helpers in `gmmxx/_cuda.py`

Append after the tied block:

```python
# ---------------------------------------------------------------------------
# Full kernels (Plan 8 — pure-torch orchestration for D <= 16, K <= 32)
# ---------------------------------------------------------------------------


def _full_logits(
    x: torch.Tensor, means: torch.Tensor, L: torch.Tensor, log_w: torch.Tensor
) -> torch.Tensor:
    """Compute (B, N, K) logits via batched solve_triangular.

    x: (B, N, D); means: (B, K, D); L: (B, K, D, D); log_w: (B, K).
    Returns fp32 logits.
    """
    import math
    B, N, D = x.shape
    K = means.shape[1]
    x_f = x.float()
    means_f = means.float()
    L_f = L.float()
    diff = x_f.unsqueeze(2) - means_f.unsqueeze(1)  # (B, N, K, D)
    diff_t = diff.permute(0, 2, 3, 1).contiguous()   # (B, K, D, N)
    z = torch.linalg.solve_triangular(L_f, diff_t, upper=False)  # (B, K, D, N)
    dist_sq = z.pow(2).sum(2).permute(0, 2, 1)       # (B, N, K)
    log_det_L = L_f.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)  # (B, K)
    log_norm_const = 0.5 * D * math.log(2 * math.pi)
    return (
        log_w.float().unsqueeze(1)
        - log_norm_const
        - log_det_L.unsqueeze(1)
        - 0.5 * dist_sq
    )


def full_assign(
    x: torch.Tensor, means: torch.Tensor, L: torch.Tensor, log_w: torch.Tensor
) -> torch.Tensor:
    """Full covariance E-step assign. Returns int32 (B, N)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    L = _check_input(L, "L")
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    logits = _full_logits(x, means, L, log_w)
    return logits.argmax(-1).to(torch.int32)


def full_logsumexp(
    x: torch.Tensor, means: torch.Tensor, L: torch.Tensor, log_w: torch.Tensor
) -> torch.Tensor:
    """Full covariance E-step logsumexp. Returns fp32 (B, N) — per-sample log-likelihood."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    L = _check_input(L, "L")
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    return _full_logits(x, means, L, log_w).logsumexp(-1)


def full_resp(
    x: torch.Tensor,
    means: torch.Tensor,
    L: torch.Tensor,
    log_w: torch.Tensor,
    log_norm: torch.Tensor,
) -> torch.Tensor:
    """Full covariance E-step responsibilities. Returns fp32 (B, N, K)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    L = _check_input(L, "L")
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    log_norm = _check_input(log_norm, "log_norm", dtype=torch.float32)
    logits = _full_logits(x, means, L, log_w)
    return (logits - log_norm.unsqueeze(-1)).exp()


def full_blocked_update(
    x: torch.Tensor, cluster_ids: torch.Tensor, n_components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full covariance hard-assign M-step accumulator.

    Returns (sums (B,K,D), outer_sums (B,K,D,D), counts (B,K) int32).
    """
    require_cuda()
    x = _check_input(x, "x")
    cluster_ids = _check_input(cluster_ids, "cluster_ids", dtype=torch.int32)
    B, N, D = x.shape
    K = int(n_components)
    device = x.device
    x_f = x.float()
    ids_long = cluster_ids.long()

    # Sums via scatter_add_ along K dim.
    sums = torch.zeros(B, K, D, dtype=torch.float32, device=device)
    sums.scatter_add_(
        dim=1,
        index=ids_long.unsqueeze(-1).expand(-1, -1, D),
        src=x_f,
    )

    # Outer-product accumulator.
    xx_per_point = x_f.unsqueeze(-1) * x_f.unsqueeze(-2)  # (B, N, D, D)
    outer_sums = torch.zeros(B, K, D, D, dtype=torch.float32, device=device)
    outer_sums.scatter_add_(
        dim=1,
        index=ids_long.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D, D),
        src=xx_per_point,
    )

    # Counts via scatter_add of ones.
    ones_int32 = torch.ones_like(cluster_ids, dtype=torch.int32)
    counts = torch.zeros(B, K, dtype=torch.int32, device=device)
    counts.scatter_add_(dim=1, index=ids_long, src=ones_int32)

    return sums, outer_sums, counts


def full_finalize(
    sums: torch.Tensor,
    outer_sums: torch.Tensor,
    counts: torch.Tensor,
    old_means: torch.Tensor,
    old_L: torch.Tensor,
    total_n: int,
    reg_covar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full M-step finalize.

    Returns (means_new (B,K,D), L_new (B,K,D,D), weights_new (B,K)).

    For empty clusters (count==0), preserves old_means and old_L.
    """
    require_cuda()
    sums = _check_input(sums, "sums", dtype=torch.float32)
    outer_sums = _check_input(outer_sums, "outer_sums", dtype=torch.float32)
    counts = _check_input(counts, "counts")
    old_means = _check_input(old_means, "old_means")
    old_L = _check_input(old_L, "old_L")

    B, K, D = sums.shape
    device = sums.device
    counts_f = counts.float()
    n_k = counts_f.clamp_min(1e-30)

    means_new = sums / n_k.unsqueeze(-1)  # (B, K, D)
    weights_new = counts_f / float(total_n)  # (B, K)

    # Per-cluster Σ_k = outer_sums_k / n_k - μ_k μ_k^T
    sigma = outer_sums / n_k.unsqueeze(-1).unsqueeze(-1) - (
        means_new.unsqueeze(-1) * means_new.unsqueeze(-2)
    )
    # Add reg_covar * I per cluster.
    eye = torch.eye(D, device=device, dtype=sigma.dtype).view(1, 1, D, D)
    sigma = sigma + reg_covar * eye
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))  # symmetrize

    # Per-cluster Cholesky (handles non-PD via cholesky_ex info field).
    L_new, info = torch.linalg.cholesky_ex(sigma)
    # For clusters where Cholesky failed (info != 0) OR count was zero,
    # preserve old_L.
    counts_int = counts.to(torch.int32) if counts.dtype != torch.int32 else counts
    failed = (info != 0) | (counts_int <= 0)  # (B, K) bool
    if failed.any():
        # Broadcast failed mask to (B, K, D, D) and (B, K, D) for means.
        means_new = torch.where(
            failed.unsqueeze(-1).expand_as(means_new),
            old_means.float(),
            means_new,
        ).to(old_means.dtype) if old_means.dtype != torch.float32 else torch.where(
            failed.unsqueeze(-1).expand_as(means_new),
            old_means,
            means_new,
        )
        L_new = torch.where(
            failed.unsqueeze(-1).unsqueeze(-1).expand_as(L_new),
            old_L.float() if old_L.dtype != torch.float32 else old_L,
            L_new,
        )
        weights_new = torch.where(failed, torch.zeros_like(weights_new), weights_new)

    return means_new.to(old_means.dtype) if old_means.dtype != torch.float32 else means_new, L_new, weights_new
```

The empty-cluster / failed-Cholesky handling is the trickiest part. Test it explicitly.

Smoke test:

```bash
uv run python -c "
import torch, math
from gmmxx import _cuda
torch.manual_seed(0)
B, N, D, K = 1, 256, 8, 4

# Random ground-truth params.
means_gt = torch.randn(B, K, D, device='cuda')
L_gt = torch.tril(torch.randn(B, K, D, D, device='cuda'))
diag_idx = torch.arange(D, device='cuda')
L_gt[:, :, diag_idx, diag_idx] = L_gt[:, :, diag_idx, diag_idx].abs() + 1.0
log_w = torch.full((B, K), -math.log(K), device='cuda')

x = torch.randn(B, N, D, device='cuda')

ids = _cuda.full_assign(x, means_gt, L_gt, log_w)
print('assign shape:', ids.shape, 'dtype:', ids.dtype)

lse = _cuda.full_logsumexp(x, means_gt, L_gt, log_w)
print('lse shape:', lse.shape, 'mean:', lse.mean().item())

r = _cuda.full_resp(x, means_gt, L_gt, log_w, lse)
print('resp sum mean:', r.sum(-1).mean().item())  # ~1.0

sums, outer_sums, counts = _cuda.full_blocked_update(x, ids, K)
print('sums shape:', sums.shape, 'outer_sums shape:', outer_sums.shape)
print('counts sum:', counts.sum().item(), '(expected', N, ')')

new_means, new_L, new_w = _cuda.full_finalize(
    sums, outer_sums, counts, means_gt, L_gt, N, 1e-6
)
print('new_means shape:', new_means.shape)
print('new_L shape:', new_L.shape)
print('new_L is lower triangular per cluster:', (new_L.triu(1) == 0).all().item())
print('new_w sum:', new_w.sum().item())
"
```

Expected: shapes correct, resp sums ~1.0, new_L lower triangular per cluster, weights sum to 1.

Commit.

---

## Task 2 — Re-export through `gmmxx/cuda_ops.py`

Add 5 entries (`full_assign`, `full_logsumexp`, `full_resp`, `full_blocked_update`, `full_finalize`) after the tied block. Add to `__all__`. Smoke + commit.

---

## Task 3 — Flip `cuda_full_supported` gate

```python
def cuda_full_supported(d: int, n_components: int, dtype) -> bool:
    """Plan 8: D ≤ 16, K ≤ 32, dtype ∈ {fp32, fp16, bf16}.

    Tighter window than spherical/diag/tied because each cluster has an
    O(D²) precision representation. Spec §4 says D ≤ 16 for inference;
    Plan 8 supports it for fit too via pure-torch orchestration.
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 16):
        return False
    if not (0 < n_components <= 32):
        return False
    return True
```

Verify + commit.

---

## Task 4 — `_train_full_cuda` in `interface.py`

Add a method analogous to `_train_tied_cuda`:

```python
def _train_full_cuda(self, x_b: torch.Tensor, batch_size: Optional[int]) -> None:
    """Full-covariance EM loop on the CUDA backend (D ≤ 16, K ≤ 32)."""
    import math
    from . import _cuda as _cuda_mod

    B, N, D = x_b.shape
    K = self.k
    device = x_b.device

    # Init: random means; per-cluster L_k = sqrt(data_var) * I.
    rng = torch.Generator(device=device).manual_seed(self.seed)
    init_idx = torch.randint(0, N, (B, K), generator=rng, device=device)
    means = torch.gather(
        x_b, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)
    ).contiguous()
    data_var_scalar = x_b.float().var(dim=1).mean(-1, keepdim=True)  # (B, 1)
    sqrt_var = data_var_scalar.clamp_min(self.reg_covar).sqrt().view(B, 1, 1, 1)
    eye = torch.eye(D, device=device, dtype=x_b.dtype).view(1, 1, D, D)
    L = (sqrt_var.to(x_b.dtype) * eye.expand(B, K, D, D)).contiguous()  # (B, K, D, D)
    log_w = torch.full((B, K), -math.log(K), dtype=torch.float32, device=device)
    weights = torch.full((B, K), 1.0 / K, dtype=torch.float32, device=device)

    lower_bound_history = []
    n_iter = 0
    prev_lb = -math.inf

    for _ in range(self.niter):
        n_iter += 1
        ids = _cuda_mod.full_assign(x_b, means, L, log_w)
        lse = _cuda_mod.full_logsumexp(x_b, means, L, log_w)
        lb = float(lse.mean().item())
        lower_bound_history.append(lb)

        sums, outer_sums, counts = _cuda_mod.full_blocked_update(x_b, ids, K)
        means, L, weights = _cuda_mod.full_finalize(
            sums, outer_sums, counts, means, L, N, self.reg_covar
        )
        log_w = torch.log(weights.clamp_min(1e-30))

        if abs(lb - prev_lb) < self.tol:
            break
        prev_lb = lb

    # GMMXX exposes covariances_ as the (B, K, D, D) covariance matrices.
    cov_b = L @ L.transpose(-1, -2)  # (B, K, D, D)

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
        variances_b=cov_b,
        weights_b=weights,
        info=info,
        batch_size=batch_size,
    )
```

In `train()`, find the full branch and add the CUDA dispatch:

```python
elif self.covariance_type == "full":
    from . import _dispatch
    shape = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
    resolved = _dispatch.resolve_backend_with_env(
        requested=self.backend, covariance="full", shape=shape,
        dtype=x_b.dtype, legacy_no_triton=self._legacy_no_triton,
    )
    if resolved == "cuda" and self.approx_top_k is None:
        self._train_full_cuda(x_b, batch_size)
        self.last_backend_used_ = "cuda"
        self.cuda_estep_enabled_ = True
        return
    self.last_backend_used_ = resolved
    # Existing full Triton/torch logic.
    ...
```

Smoke + run full suite + commit.

---

## Task 5 — Full CUDA inference paths

Mirror the tied (Plan 7) pattern. For each method (predict / predict_proba / score_samples), insert a CUDA branch:

```python
elif self.covariance_type == "full":
    from . import _dispatch
    shape = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
    resolved = _dispatch.resolve_backend_with_env(
        requested=self.backend, covariance="full", shape=shape,
        dtype=x_b.dtype, legacy_no_triton=self._legacy_no_triton,
    )
    if resolved == "cuda":
        from . import _cuda as _cuda_mod
        if self.dtype is not None and x_b.dtype != self.dtype:
            x_b_compute = x_b.to(self.dtype)
            means_b = self.means_b.to(self.dtype)
        else:
            x_b_compute = x_b
            means_b = self.means_b
        log_w = torch.log(self.weights_b.clamp_min(1e-30))
        # Cholesky factor per cluster from stored covariances.
        L, _ = torch.linalg.cholesky_ex(self.covariances_b)
        # Method-specific:
        # ... predict: full_assign
        # ... predict_proba: full_logsumexp + full_resp
        # ... score_samples: full_logsumexp directly
        ...
        self.last_backend_used_ = "cuda"
        return self._squeeze_if_unbatched(...)
    self.last_backend_used_ = resolved
    # Existing full inference.
    ...
```

Run + commit.

---

## Task 6 — Tests

Create `tests/test_cuda_full.py` covering:
- `_full_logits` consistency vs torch reference (manual `solve_triangular`).
- `full_assign` argmax agreement.
- `full_logsumexp` matches reference.
- `full_resp` rows sum to 1.
- `full_blocked_update` per-cluster sums and outer_sums match groupby reference.
- `full_finalize` produces lower-triangular L per cluster; weights sum to 1; empty cluster preservation; non-PD fallback to old_L.
- End-to-end `GMMXX(covariance_type="full", backend="cuda")` fit + predict + score.

Run + commit.

---

## Task 7 — Parametrize full pipeline test

Append `test_full_full_pipeline_each_backend` to `tests/test_gmmxx.py`. Note: full's `covariances_.shape == (K, D, D)`. Use D=8 to stay well inside everyone's window.

Run + commit.

---

## Task 8 — README + tag

Update CUDA section: replace "Full is still on Triton/PyTorch — coming in Plan 8." with:

> "Full is now on CUDA (Plan 8) — pure-torch orchestration via batched `solve_triangular` + per-cluster `cholesky_ex` for `D ≤ 16, K ≤ 32`. **All four covariance types are now on CUDA.**"

Tag `full-cuda-plan8`. Commit.

---

## Self-Review

**Spec coverage:** §4 full E/M (covered for D ≤ 16, K ≤ 32 — slightly wider than spec's D ≤ 8 because pure-torch primitives parallelize well), §5c per-cluster Cholesky finalize, §7 dispatch.

**No new CUDA kernels** — pure Python/torch orchestration. Performance dominated by `solve_triangular` (the torch impl uses cuSOLVER under the hood — fast).

**Empty-cluster handling** is the trickiest detail. `cholesky_ex` returns an `info` field per matrix; we use it together with `counts <= 0` to fall back to old_L for those clusters.

---

## Execution Handoff

Saved to `docs/superpowers/plans/2026-05-03-gmmxx-cuda-full.md`. Subagent-driven (sonnet for all tasks; no novel CUDA).

After Plan 8: **all four covariance types are on CUDA**. Remaining roadmap:
- **Plan 9**: perf optimizations (sm_80 mma + sorted-run + fused) for diag/tied/full
- **Plan 10**: `large_n.py` integration
- **Plan 11**: approx top-K spherical
