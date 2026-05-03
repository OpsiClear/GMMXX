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
