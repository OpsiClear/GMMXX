"""Internal Python wrappers around gmmxx._C (the compiled CUDA extension).

This module:
  * Imports `_C` lazily and tolerates ImportError so `import gmmxx` succeeds
    on hosts without a CUDA build.
  * Validates inputs (contiguous, dtype, device) before crossing the FFI
    boundary.
  * Wraps every FFI call in try/except so runtime CUDA errors (OOM, illegal
    instruction on a specific GPU/driver combo, mma.sync regressions) don't
    take down the whole process — they raise a custom exception that
    `_dispatch.resolve_backend` catches and falls through to the next backend.

End users do not import this module directly; they use the `GMMXX` class or
the `gmmxx.cuda_ops` re-export.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    from . import _C  # noqa: F401  -- compiled extension
    _HAS_CUDA = True
    _IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:
    _C = None
    _HAS_CUDA = False
    _IMPORT_ERROR = exc

_SORT_THRESHOLD_NK = 2 ** 21  # ~2M; below this, sort cost dominates


class CudaBackendUnavailable(RuntimeError):
    """Raised when gmmxx._C was not built (e.g. GMMXX_SKIP_CUDA=1)."""


class CudaRuntimeFallback(RuntimeError):
    """Raised when a CUDA kernel fails at runtime; the dispatcher catches
    this and falls through to Triton or torch."""


def has_cuda() -> bool:
    """True iff gmmxx._C imported successfully AND torch.cuda is available."""
    return _HAS_CUDA and torch.cuda.is_available()


def _no_fallback() -> bool:
    """If GMMXX_CUDA_NO_FALLBACK=1, runtime errors propagate instead of being
    caught. Used in CI to make CUDA bugs loud."""
    return os.environ.get("GMMXX_CUDA_NO_FALLBACK", "").lower() in {"1", "true", "yes"}


def require_cuda() -> None:
    """Raise CudaBackendUnavailable if the extension wasn't built. Used by
    the dispatcher when the user explicitly requests backend='cuda'."""
    if _C is None:
        raise CudaBackendUnavailable(
            "gmmxx._C extension not built; reinstall without GMMXX_SKIP_CUDA "
            f"(original ImportError: {_IMPORT_ERROR!r})"
        )


def _check_input(t: torch.Tensor, name: str, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not t.is_cuda:
        raise ValueError(f"{name}: must be on a CUDA device, got {t.device}")
    if not t.is_contiguous():
        t = t.contiguous()
    if dtype is not None and t.dtype != dtype:
        raise ValueError(f"{name}: dtype must be {dtype}, got {t.dtype}")
    return t


def canary_add_offset(input: torch.Tensor, offset: int) -> torch.Tensor:
    """Smoke-test wrapper. Calls the canary kernel with proper validation
    and runtime-error fallback semantics.

    Returns input + offset; raises CudaRuntimeFallback on kernel failure
    (unless GMMXX_CUDA_NO_FALLBACK=1, in which case the raw RuntimeError
    propagates).
    """
    require_cuda()
    input = _check_input(input, "canary input", dtype=torch.int32)
    try:
        return _C.canary_add_offset(input, offset)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"canary kernel failed: {exc}") from exc


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
    *,
    force_sort: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M-step accumulator. Picks sorted-run vs per-token by N*K heuristic.

    force_sort: True forces sorted path; False forces per-token; None auto.

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

    use_sort = force_sort if force_sort is not None else (N * K >= _SORT_THRESHOLD_NK)

    try:
        if use_sort:
            sorted_ids, perm = cluster_ids.sort(dim=1)
            x_sorted = torch.gather(x, 1, perm.unsqueeze(-1).expand(-1, -1, D))
            _C.blocked_update_spherical_sorted(
                x_sorted.contiguous(), sorted_ids.int().contiguous(),
                sums, sumsq, counts,
            )
        else:
            _C.blocked_update_spherical(x, cluster_ids, sums, sumsq, counts)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(
            f"blocked_update_spherical (use_sort={use_sort}) failed: {exc}"
        ) from exc
    return sums, sumsq, counts


def blocked_update_spherical_sorted(
    x_sorted: torch.Tensor,
    sorted_ids: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct sorted-run wrapper. Caller is responsible for sorting cluster_ids
    and gathering x to match. For most uses, prefer blocked_update_spherical
    which handles the sort + heuristic."""
    require_cuda()
    x_sorted = _check_input(x_sorted, "x_sorted")
    sorted_ids = _check_input(sorted_ids, "sorted_ids", dtype=torch.int32)
    B, N, D = x_sorted.shape
    K = int(n_components)
    sums = torch.zeros((B, K, D), dtype=torch.float32, device=x_sorted.device)
    sumsq = torch.zeros((B, K), dtype=torch.float32, device=x_sorted.device)
    counts = torch.zeros((B, K), dtype=torch.int32, device=x_sorted.device)
    try:
        _C.blocked_update_spherical_sorted(x_sorted, sorted_ids, sums, sumsq, counts)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(
            f"blocked_update_spherical_sorted failed: {exc}"
        ) from exc
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


# ---------------------------------------------------------------------------
# Diagonal kernels (Plan 6 — safe path)
# ---------------------------------------------------------------------------


def diag_assign(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Diagonal E-step assign. Returns int32 (B, N)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    try:
        return _C.diag_assign(x, means, var, log_w, out)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"diag_assign failed: {exc}") from exc


def diag_logsumexp(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Diagonal E-step logsumexp. Returns fp32 (B, N)."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    try:
        return _C.diag_logsumexp(x, means, var, log_w, out)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"diag_logsumexp failed: {exc}") from exc


def diag_resp(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    log_norm: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"diag_resp failed: {exc}") from exc


def blocked_update_diag(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"blocked_update_diag failed: {exc}") from exc
    return sums, sumsq, counts


def finalize_diag(
    sums: torch.Tensor,
    sumsq: torch.Tensor,
    counts: torch.Tensor,
    old_means: torch.Tensor,
    old_var: torch.Tensor,
    total_n: int,
    reg_covar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagonal M-step finalize. Returns (means (B,K,D), var (B,K,D), weights (B,K))."""
    require_cuda()
    sums = _check_input(sums, "sums", dtype=torch.float32)
    sumsq = _check_input(sumsq, "sumsq", dtype=torch.float32)
    counts = _check_input(counts, "counts", dtype=torch.int32)
    old_means = _check_input(old_means, "old_means")
    old_var = _check_input(old_var, "old_var", dtype=torch.float32)
    try:
        return _C.finalize_diag(
            sums, sumsq, counts, old_means, old_var,
            int(total_n), float(reg_covar),
        )
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"finalize_diag failed: {exc}") from exc


def fused_spherical(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    reg_covar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused E/M single-tile spherical kernel.

    Combines logit computation, softmax responsibilities, and per-cluster
    sufficient-statistic accumulation in a single CTA pass. Constraints:
    D <= 64, K <= 128. Caller is responsible for checking the support
    window via gmmxx._runtime.cuda_spherical_fused_supported.

    Returns (new_means, new_var, new_weights, lse_per_sample, labels)
    where:
      new_means: (B, K, D) same dtype as input means.
      new_var: (B, K) fp32 (clamped to >= reg_covar).
      new_weights: (B, K) fp32 (sum to 1 per batch when N > 0).
      lse_per_sample: (B, N) fp32 — per-sample log-likelihood; mean is the ELBO.
      labels: (B, N) int32 — argmax of responsibilities (= argmax of logits).
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


# ---------------------------------------------------------------------------
# Tied kernels (Plan 7 — projected-coords approach)
#
# The tied E-step reuses the spherical CUDA kernels on projected coordinates:
#   y = L⁻¹ x; ν_k = L⁻¹ μ_k; ||L⁻¹(x − μ_k)||² = ||y − ν_k||²
# The shared log|L| term shifts the per-component log_w by a constant.
# ---------------------------------------------------------------------------


def tied_project(x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Project x via L⁻¹ where L is lower-triangular. Returns same shape as x.

    For (B, N, D) x and (B, D, D) L:
      y[b, n, :] = L[b]⁻¹ x[b, n, :]
    Implemented via batched solve_triangular: L @ Y^T = X^T → Y = (L⁻¹ X^T)^T.
    """
    require_cuda()
    x = _check_input(x, "x")
    L = _check_input(L, "L")
    # solve_triangular wants RHS shape compatible with L's last two dims.
    # x is (B, N, D); transpose to (B, D, N); solve L @ ?  = X^T → ? = L⁻¹ X^T.
    x_t = x.transpose(-1, -2).contiguous()  # (B, D, N)
    # Promote dtypes: solve_triangular requires both inputs to share dtype.
    if x_t.dtype != L.dtype:
        x_t = x_t.to(L.dtype)
    y_t = torch.linalg.solve_triangular(L, x_t, upper=False)  # (B, D, N)
    return y_t.transpose(-1, -2).contiguous()


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
    y = tied_project(x, L)
    nu = tied_project(means, L)
    B, K, D = nu.shape
    var = torch.ones(B, K, dtype=torch.float32, device=x.device)
    # The constant -log|L| - 0.5*D*log(2π*1) cancels under argmax across k,
    # so we pass the unmodified log_w and var=1.
    return spherical_assign(y, nu, var, log_w)


def tied_logsumexp(
    x: torch.Tensor,
    means: torch.Tensor,
    L: torch.Tensor,
    log_w: torch.Tensor,
) -> torch.Tensor:
    """Tied E-step logsumexp. Returns (B, N) fp32 — true tied log-likelihood per sample."""
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    y = tied_project(x, L)
    nu = tied_project(means, L)
    B, K, D = nu.shape
    var = torch.ones(B, K, dtype=torch.float32, device=x.device)
    # spherical_lse(y, ν, 1, log_w) = log Σ_k exp(log_w_k − 0.5·D·log(2π) − 0.5·||y−ν_k||²)
    # tied_lse = log|L| extra subtracted because tied logit has -log|L| inside the exp:
    # tied_lse = log Σ_k exp(log_w_k − log|L| − 0.5·D·log(2π) − 0.5·||y−ν_k||²)
    #          = spherical_lse − log|L|
    spherical_lse = spherical_logsumexp(y, nu, var, log_w)
    log_det_L = tied_log_det(L).unsqueeze(-1)  # (B, 1)
    return spherical_lse - log_det_L


def tied_resp(
    x: torch.Tensor,
    means: torch.Tensor,
    L: torch.Tensor,
    log_w: torch.Tensor,
    log_norm: torch.Tensor,  # tied logsumexp output (with -log|L| applied)
) -> torch.Tensor:
    """Tied E-step responsibilities. Returns fp32 (B, N, K).

    log_norm: pass tied_logsumexp output (which already accounts for log|L|).
    Internally we shift back by +log|L| to use spherical_resp.
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
    # spherical_resp computes exp(spherical_logit_k − spherical_log_norm).
    # tied responsibilities: r_k = exp(tied_logit_k − tied_log_norm).
    # Both tied_logit and tied_log_norm have -log|L| relative to spherical;
    # the difference cancels. So we pass log_norm + log|L| as the spherical
    # log_norm.
    log_det_L = tied_log_det(L).unsqueeze(-1)  # (B, 1)
    log_norm_for_spherical = log_norm + log_det_L
    return spherical_resp(y, nu, var, log_w, log_norm_for_spherical)


def tied_finalize(
    sums: torch.Tensor,        # (B, K, D) — Σ_n r_{n,k} x_n  from blocked_update
    xx_total: torch.Tensor,    # (B, D, D) — Σ_n x_n x_n^T (caller-supplied)
    counts: torch.Tensor,      # (B, K) int32 OR fp32 (soft counts)
    total_n: int,
    reg_covar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tied finalize: divide sums, compute new Σ from X^T X − Σ_k n_k μ_k μ_k^T,
    add reg_covar·I, Cholesky factor.

    Returns (means_new (B, K, D), L_new (B, D, D), weights_new (B, K)).
    """
    require_cuda()
    sums = _check_input(sums, "sums", dtype=torch.float32)
    xx_total = _check_input(xx_total, "xx_total", dtype=torch.float32)
    counts = _check_input(counts, "counts")  # accept int32 or fp32

    B, K, D = sums.shape
    counts_f = counts.float()
    n_k = counts_f.clamp_min(1e-30)
    means_new = sums / n_k.unsqueeze(-1)  # (B, K, D)
    weights_new = counts_f / float(total_n)

    # Σ_k n_k μ_k μ_k^T  =  (means * counts.unsqueeze(-1))^T @ means
    weighted_means = means_new * counts_f.unsqueeze(-1)  # (B, K, D)
    sigma_k_sum = weighted_means.transpose(-1, -2) @ means_new  # (B, D, D)

    # Σ_new = (1/N) (X^T X − Σ_k n_k μ_k μ_k^T) + reg_covar I
    sigma = (xx_total - sigma_k_sum) / float(total_n)
    eye = torch.eye(D, device=sigma.device, dtype=sigma.dtype).unsqueeze(0)
    sigma = sigma + reg_covar * eye
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))  # symmetrize

    chol_new = torch.linalg.cholesky(sigma)  # (B, D, D)
    return means_new, chol_new, weights_new
