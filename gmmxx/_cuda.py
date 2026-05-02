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
