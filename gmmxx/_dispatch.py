"""Backend dispatch for gmmxx.

Decides whether to route a kernel call through the CUDA backend (compiled
gmmxx._C), the Triton backend (existing JIT modules), or the PyTorch
fallback. Encapsulates per-shape gates so the GMMXX orchestrator never
hardcodes backend choices.

The high-level public API exposed via this module is:

    resolve_backend(requested, covariance, shape, dtype, legacy_no_triton=False)
        -> "cuda" | "triton" | "torch"
    resolve_backend_with_env(requested, ...)  # same, but consults GMMXX_BACKEND
                                                 when requested == "auto".

Plan 1 wires the truth table; Plan 2 onwards adds dispatch_kernel() that
actually routes calls into the right module per backend.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from . import _cuda
from ._runtime import (
    cuda_diag_supported,
    cuda_full_supported,
    cuda_spherical_supported,
    cuda_tied_supported,
    triton_spherical_supported,
)

_VALID_BACKENDS = {"auto", "cuda", "triton", "torch"}


def _shape_dk(shape: tuple) -> tuple[int, int]:
    """Extract (D, K) from a (B, N, D, K) or (B, N, D) shape tuple.

    Most callers pass (B, N, D, K) when they know K (i.e., during fit).
    Inference paths only have (B, N, D) and the orchestrator passes K
    separately; in that case this helper returns (D, 0) so the K-dependent
    triton/cuda gates accept any K.
    """
    if len(shape) == 4:
        return shape[2], shape[3]
    if len(shape) == 3:
        return shape[2], 0
    raise ValueError(f"shape must have 3 or 4 dims, got {shape!r}")


def _triton_supported(covariance: str, shape: tuple, dtype: Any) -> bool:
    """True iff the Triton path can handle this call."""
    d, k = _shape_dk(shape)
    if covariance == "spherical":
        # Existing policy.
        if k == 0:
            return 0 < d <= 128  # inference-only; K not known yet
        return triton_spherical_supported(d, k)
    if covariance == "diag":
        return 0 < d <= 64 and (k == 0 or k <= 512)
    if covariance == "tied":
        return 0 < d <= 64 and (k == 0 or k <= 512)
    if covariance == "full":
        return 0 < d <= 16  # full Triton path is conservative
    return False


def _cuda_supported(covariance: str, shape: tuple, dtype: Any) -> bool:
    """True iff the CUDA backend can handle this call.

    Plan 1: stubs in _runtime.py all return False, so this returns False
    everywhere. Subsequent plans will turn it on per covariance type.
    """
    if not _cuda.has_cuda():
        return False
    d, k = _shape_dk(shape)
    if covariance == "spherical":
        return cuda_spherical_supported(d, k, dtype)
    if covariance == "diag":
        return cuda_diag_supported(d, k, dtype)
    if covariance == "tied":
        return cuda_tied_supported(d, k, dtype)
    if covariance == "full":
        return cuda_full_supported(d, k, dtype)
    return False


def resolve_backend(
    requested: str,
    covariance: str,
    shape: tuple,
    dtype: Any,
    legacy_no_triton: bool = False,
) -> str:
    """Returns one of "cuda", "triton", "torch" given the user request and call shape.

    legacy_no_triton: True when called from a deprecated use_triton=False shim.
                      Filters Triton out of the resolution chain regardless of
                      requested.
    """
    if requested not in _VALID_BACKENDS:
        raise ValueError(f"unknown backend {requested!r}; expected one of {_VALID_BACKENDS}")
    if requested == "torch":
        return "torch"
    if requested == "triton":
        if legacy_no_triton:
            raise ValueError(
                "backend='triton' incompatible with use_triton=False; pass backend='auto' or remove use_triton."
            )
        return "triton" if _triton_supported(covariance, shape, dtype) else "torch"
    if requested == "cuda":
        if _cuda_supported(covariance, shape, dtype):
            return "cuda"
        # Explicit cuda but unsupported shape — only require_cuda() raises;
        # if the extension is built but the shape gate is False, fall through.
        if _cuda._HAS_CUDA:
            return "torch"
        # Extension unbuilt and user explicitly asked for cuda → loud error.
        _cuda.require_cuda()  # raises CudaBackendUnavailable
    # requested == "auto"
    if _cuda_supported(covariance, shape, dtype):
        return "cuda"
    if (not legacy_no_triton) and _triton_supported(covariance, shape, dtype):
        return "triton"
    return "torch"


def resolve_backend_with_env(
    requested: str,
    covariance: str,
    shape: tuple,
    dtype: Any,
    legacy_no_triton: bool = False,
) -> str:
    """Same as resolve_backend, but consults GMMXX_BACKEND when requested == 'auto'.

    The kwarg always wins when explicit. Invalid env-var values are ignored.
    """
    effective = requested
    if effective == "auto":
        env = os.environ.get("GMMXX_BACKEND")
        if env in _VALID_BACKENDS:
            effective = env
    return resolve_backend(effective, covariance, shape, dtype, legacy_no_triton=legacy_no_triton)
