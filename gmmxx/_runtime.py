from __future__ import annotations


TRITON_SPHERICAL_MAX_D = 128
TRITON_SPHERICAL_MAX_K = 2048


def triton_spherical_supported(d: int, n_components: int) -> bool:
    """Validated spherical Triton shape policy.

    Keep this intentionally small and explicit. Shapes outside this range use
    the PyTorch/cuBLAS implementation instead of carrying extra runtime
    branches through the production path.
    """
    return 0 < d <= TRITON_SPHERICAL_MAX_D and 0 < n_components <= TRITON_SPHERICAL_MAX_K
