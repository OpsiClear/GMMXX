"""Experimental public re-export of low-level CUDA kernel callables.

WARNING — Experimental: API may change before v1.0. The only API stability
guarantee in GMMXX is the ``GMMXX`` class itself.
"""

from __future__ import annotations

from . import _cuda

# Smoke-test (Plan 1).
canary_add_offset = _cuda.canary_add_offset

# Spherical (Plan 2).
spherical_assign = _cuda.spherical_assign
spherical_logsumexp = _cuda.spherical_logsumexp
spherical_resp = _cuda.spherical_resp
blocked_update_spherical = _cuda.blocked_update_spherical
blocked_update_spherical_sorted = _cuda.blocked_update_spherical_sorted
finalize_spherical = _cuda.finalize_spherical
fused_spherical = _cuda.fused_spherical

# Diagonal (Plan 6).
diag_assign = _cuda.diag_assign
diag_logsumexp = _cuda.diag_logsumexp
diag_resp = _cuda.diag_resp
blocked_update_diag = _cuda.blocked_update_diag
finalize_diag = _cuda.finalize_diag

# Tied (Plan 7).
tied_project = _cuda.tied_project
tied_log_det = _cuda.tied_log_det
tied_assign = _cuda.tied_assign
tied_logsumexp = _cuda.tied_logsumexp
tied_resp = _cuda.tied_resp
tied_finalize = _cuda.tied_finalize

# Full (Plan 8).
full_assign = _cuda.full_assign
full_logsumexp = _cuda.full_logsumexp
full_resp = _cuda.full_resp
full_blocked_update = _cuda.full_blocked_update
full_finalize = _cuda.full_finalize

# Lifecycle / introspection helpers.
has_cuda = _cuda.has_cuda
require_cuda = _cuda.require_cuda
CudaBackendUnavailable = _cuda.CudaBackendUnavailable
CudaRuntimeFallback = _cuda.CudaRuntimeFallback


__all__ = [
    "canary_add_offset",
    "spherical_assign",
    "spherical_logsumexp",
    "spherical_resp",
    "blocked_update_spherical",
    "blocked_update_spherical_sorted",
    "finalize_spherical",
    "fused_spherical",
    "diag_assign",
    "diag_logsumexp",
    "diag_resp",
    "blocked_update_diag",
    "finalize_diag",
    "tied_project",
    "tied_log_det",
    "tied_assign",
    "tied_logsumexp",
    "tied_resp",
    "tied_finalize",
    "full_assign",
    "full_logsumexp",
    "full_resp",
    "full_blocked_update",
    "full_finalize",
    "has_cuda",
    "require_cuda",
    "CudaBackendUnavailable",
    "CudaRuntimeFallback",
]
