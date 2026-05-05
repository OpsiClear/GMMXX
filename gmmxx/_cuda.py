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


def _spherical_logits_torch(x: torch.Tensor, means: torch.Tensor,
                             var: torch.Tensor, log_w: torch.Tensor) -> torch.Tensor:
    """Compute (B, N, K) spherical logits via cuBLAS GEMM. fp32 cuBLAS uses TF32
    on Ampere+ — same arithmetic class as Triton's tl.dot(input_precision='tf32x3').
    """
    import math
    B, N, D = x.shape
    K = means.shape[1]
    x_f = x.float()
    means_f = means.float()
    cross = torch.matmul(x_f, means_f.transpose(-1, -2))               # (B, N, K)
    x_sq = (x_f * x_f).sum(-1, keepdim=True)                            # (B, N, 1)
    c_sq = (means_f * means_f).sum(-1).unsqueeze(1)                     # (B, 1, K)
    dist = x_sq + c_sq - 2.0 * cross                                    # (B, N, K)
    log_norm_const = 0.5 * float(D) * torch.log(2 * math.pi * var).unsqueeze(1)  # (B, 1, K)
    return log_w.unsqueeze(1) - log_norm_const - 0.5 * dist / var.unsqueeze(1)


def _use_torch_fastpath_spherical(x: torch.Tensor) -> bool:
    """True when the cuBLAS GEMM path is expected to beat the safe SIMT kernel.

    Currently: fp32 inputs (no mma) for any D >= 16. fp16/bf16 already use the
    sm80 mma kernel which is fast.
    """
    return x.dtype == torch.float32 and x.is_cuda and x.dim() == 3 and x.shape[-1] >= 16


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
    if _use_torch_fastpath_spherical(x):
        ids = _spherical_logits_torch(x, means, var, log_w).argmax(-1).to(torch.int32)
        if out is not None:
            out.copy_(ids)
            return out
        return ids
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
    if _use_torch_fastpath_spherical(x):
        lse = _spherical_logits_torch(x, means, var, log_w).logsumexp(-1)
        if out is not None:
            out.copy_(lse)
            return out
        return lse
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
    if _use_torch_fastpath_spherical(x):
        r = (_spherical_logits_torch(x, means, var, log_w) - log_norm.unsqueeze(-1)).exp()
        if out is not None:
            out.copy_(r)
            return out
        return r
    try:
        return _C.spherical_resp(x, means, var, log_w, log_norm, out)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"spherical_resp failed: {exc}") from exc


def soft_update_spherical(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    reg_covar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact spherical soft-EM update using CUDA E-step kernels + torch reductions.

    This is the fallback for shapes where the single-tile fused kernel is
    correct but slower. Returns (means, var, weights, lse_per_sample, labels).
    """
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    try:
        B, N, D = x.shape
        lse = spherical_logsumexp(x, means, var, log_w)
        resp = spherical_resp(x, means, var, log_w, lse)
        ids = resp.argmax(dim=-1).to(torch.int32)
        x_f = x.float()
        nk = resp.sum(dim=1)
        nk_safe = nk.clamp_min(1e-8)
        sum_x = torch.bmm(resp.transpose(1, 2), x_f)
        x_sq = x_f.square().sum(dim=-1)
        sum_x_sq = (resp * x_sq.unsqueeze(-1)).sum(dim=1)
        active_mask = nk > 1e-8
        means_new = (sum_x / nk_safe.unsqueeze(-1)).to(x.dtype)
        means_new = torch.where(active_mask.unsqueeze(-1), means_new, means)
        mean_sq = means_new.float().square().sum(dim=-1)
        var_new = (sum_x_sq - nk * mean_sq).clamp_min(0.0) / (
            nk_safe * float(D)
        )
        var_new = var_new.clamp_min(float(reg_covar))
        var_new = torch.where(active_mask, var_new, var)
        weights = (nk / float(N)).clamp_min(1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return means_new, var_new, weights, lse, ids
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"soft_update_spherical failed: {exc}") from exc


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


def approx_topk_update_spherical(
    x: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    log_w: torch.Tensor,
    *,
    top_k: int,
    chunk_size_K: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate spherical soft-EM update over each row's top-k components.

    This is a torch-on-CUDA implementation of the existing approximate EM
    contract. It stays on CUDA tensors, streams over K to avoid materializing
    the full (B, N, K) logits tensor, and accumulates fp32 sufficient stats.

    Returns (nk, sum_x, sum_x_sq, log_likelihood_sum).
    """
    require_cuda()
    x = _check_input(x, "x")
    means = _check_input(means, "means")
    var = _check_input(var, "var", dtype=torch.float32)
    log_w = _check_input(log_w, "log_w", dtype=torch.float32)
    try:
        import math

        if x.ndim != 3:
            raise ValueError("x must have shape (B, N, D)")
        if means.ndim != 3:
            raise ValueError("means must have shape (B, K, D)")
        B, N, D = x.shape
        Bm, K, Dm = means.shape
        if (Bm, Dm) != (B, D):
            raise ValueError("means shape mismatch")
        if var.shape != (B, K):
            raise ValueError("var shape mismatch")
        if log_w.shape != (B, K):
            raise ValueError("log_w shape mismatch")
        top_k = int(top_k)
        if top_k <= 0 or top_k >= K:
            raise ValueError("top_k must be in [1, K - 1]")
        chunk_size_K = int(chunk_size_K)
        if chunk_size_K <= 0:
            raise ValueError("chunk_size_K must be positive")

        device = x.device
        x_f = x.float()
        means_f = means.float()
        var_f = var.float().clamp_min(1e-30)
        log_w_f = log_w.float()

        x_sq = x_f.square().sum(dim=-1)  # (B, N)
        best_logits = torch.full(
            (B, N, top_k),
            -torch.inf,
            device=device,
            dtype=torch.float32,
        )
        best_indices = torch.zeros(
            (B, N, top_k),
            device=device,
            dtype=torch.long,
        )
        log_2pi = math.log(2.0 * math.pi)

        for k_start in range(0, K, chunk_size_K):
            k_end = min(k_start + chunk_size_K, K)
            means_tile = means_f[:, k_start:k_end, :]  # (B, T, D)
            var_tile = var_f[:, k_start:k_end]
            log_w_tile = log_w_f[:, k_start:k_end]
            means_sq = means_tile.square().sum(dim=-1)
            cross = torch.bmm(x_f, means_tile.transpose(1, 2))
            dist = (x_sq.unsqueeze(-1) + means_sq.unsqueeze(1) - 2.0 * cross).clamp_min(0.0)
            logits = log_w_tile.unsqueeze(1) - 0.5 * (
                dist / var_tile.unsqueeze(1)
                + float(D) * (log_2pi + var_tile.log()).unsqueeze(1)
            )
            tile_k = k_end - k_start
            tile_idx = torch.arange(k_start, k_end, device=device, dtype=torch.long)
            tile_idx = tile_idx.view(1, 1, tile_k).expand(B, N, tile_k)
            candidates = torch.cat((best_logits, logits), dim=-1)
            candidate_idx = torch.cat((best_indices, tile_idx), dim=-1)
            best_logits, positions = candidates.topk(top_k, dim=-1)
            best_indices = candidate_idx.gather(-1, positions)

        log_norm = best_logits.logsumexp(dim=-1)
        resp = (best_logits - log_norm.unsqueeze(-1)).exp()

        nk = torch.zeros((B, K), dtype=torch.float32, device=device)
        sum_x = torch.zeros((B, K, D), dtype=torch.float32, device=device)
        sum_x_sq = torch.zeros((B, K), dtype=torch.float32, device=device)

        for local_idx in range(top_k):
            idx = best_indices[:, :, local_idx]
            r = resp[:, :, local_idx]
            nk.scatter_add_(1, idx, r)
            sum_x.scatter_add_(
                1,
                idx.unsqueeze(-1).expand(-1, -1, D),
                r.unsqueeze(-1) * x_f,
            )
            sum_x_sq.scatter_add_(1, idx, r * x_sq)

        return nk, sum_x, sum_x_sq, log_norm.sum()
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"approx_topk_update_spherical failed: {exc}") from exc


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
    *,
    force_sort: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagonal M-step accumulator.

    Picks sorted-run vs per-token by the same N*K heuristic used by spherical.
    force_sort: True forces sorted path; False forces per-token; None auto.
    """
    require_cuda()
    x = _check_input(x, "x")
    cluster_ids = _check_input(cluster_ids, "cluster_ids", dtype=torch.int32)
    B, N, D = x.shape
    K = int(n_components)
    sums = torch.zeros((B, K, D), dtype=torch.float32, device=x.device)
    sumsq = torch.zeros((B, K, D), dtype=torch.float32, device=x.device)
    counts = torch.zeros((B, K), dtype=torch.int32, device=x.device)
    use_sort = force_sort if force_sort is not None else (N * K >= _SORT_THRESHOLD_NK)
    try:
        if use_sort:
            sorted_ids, perm = cluster_ids.sort(dim=1)
            x_sorted = torch.gather(x, 1, perm.unsqueeze(-1).expand(-1, -1, D))
            _C.blocked_update_diag_sorted(
                x_sorted.contiguous(), sorted_ids.int().contiguous(),
                sums, sumsq, counts,
            )
        else:
            _C.blocked_update_diag(x, cluster_ids, sums, sumsq, counts)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(
            f"blocked_update_diag (use_sort={use_sort}) failed: {exc}"
        ) from exc
    return sums, sumsq, counts


def blocked_update_diag_sorted(
    x_sorted: torch.Tensor,
    sorted_ids: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct sorted-run diagonal wrapper.

    Caller is responsible for sorting cluster_ids and gathering x to match.
    Most users should call blocked_update_diag(..., force_sort=...).
    """
    require_cuda()
    x_sorted = _check_input(x_sorted, "x_sorted")
    sorted_ids = _check_input(sorted_ids, "sorted_ids", dtype=torch.int32)
    B, N, D = x_sorted.shape
    K = int(n_components)
    sums = torch.zeros((B, K, D), dtype=torch.float32, device=x_sorted.device)
    sumsq = torch.zeros((B, K, D), dtype=torch.float32, device=x_sorted.device)
    counts = torch.zeros((B, K), dtype=torch.int32, device=x_sorted.device)
    try:
        _C.blocked_update_diag_sorted(x_sorted, sorted_ids, sums, sumsq, counts)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"blocked_update_diag_sorted failed: {exc}") from exc
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
    """Full covariance E-step logsumexp. Returns fp32 (B, N)."""
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

    sums = torch.zeros(B, K, D, dtype=torch.float32, device=device)
    sums.scatter_add_(
        dim=1,
        index=ids_long.unsqueeze(-1).expand(-1, -1, D),
        src=x_f,
    )

    xx_per_point = x_f.unsqueeze(-1) * x_f.unsqueeze(-2)  # (B, N, D, D)
    outer_sums = torch.zeros(B, K, D, D, dtype=torch.float32, device=device)
    outer_sums.scatter_add_(
        dim=1,
        index=ids_long.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D, D),
        src=xx_per_point,
    )

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

    For empty clusters or non-PD Σ, preserves old_means and old_L.
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
    weights_new = counts_f / float(total_n)

    sigma = outer_sums / n_k.unsqueeze(-1).unsqueeze(-1) - (
        means_new.unsqueeze(-1) * means_new.unsqueeze(-2)
    )
    eye = torch.eye(D, device=device, dtype=sigma.dtype).view(1, 1, D, D)
    sigma = sigma + reg_covar * eye
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))

    L_new, info = torch.linalg.cholesky_ex(sigma)

    failed = (info != 0) | (counts_f <= 0.0)  # (B, K) bool
    if failed.any():
        # Cast old_means / old_L to fp32 for arithmetic, then back at the end.
        old_means_f = old_means.float()
        old_L_f = old_L.float()
        means_new = torch.where(
            failed.unsqueeze(-1).expand_as(means_new),
            old_means_f,
            means_new,
        )
        L_new = torch.where(
            failed.unsqueeze(-1).unsqueeze(-1).expand_as(L_new),
            old_L_f,
            L_new,
        )
        weights_new = torch.where(failed, torch.zeros_like(weights_new), weights_new)

    if old_means.dtype != torch.float32:
        means_new = means_new.to(old_means.dtype)
    if old_L.dtype != torch.float32:
        L_new = L_new.to(old_L.dtype)

    return means_new, L_new, weights_new
