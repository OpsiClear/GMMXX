"""Experimental public re-export of low-level CUDA kernel callables.

WARNING — Experimental: API may change before v1.0. The only API stability
guarantee in GMMXX is the ``GMMXX`` class itself. Internal kernel signatures
exposed here may evolve across minor versions as Phase 2 adds fp8, WGMMA,
multi-stream event plumbing, and similar features.

Plan 1 exposes only the canary kernel as a smoke test of the re-export
plumbing. Plans 2 onwards add the real spherical/diag/tied/full ops as their
kernels land.
"""

from __future__ import annotations

from . import _cuda

# Re-export with the documented public names.
canary_add_offset = _cuda.canary_add_offset

# Lifecycle / introspection helpers (also exposed at top level).
has_cuda = _cuda.has_cuda
require_cuda = _cuda.require_cuda
CudaBackendUnavailable = _cuda.CudaBackendUnavailable
CudaRuntimeFallback = _cuda.CudaRuntimeFallback


__all__ = [
    "canary_add_offset",
    "has_cuda",
    "require_cuda",
    "CudaBackendUnavailable",
    "CudaRuntimeFallback",
]
