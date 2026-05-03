from __future__ import annotations


# Triton shape policy (existing).
TRITON_SPHERICAL_MAX_D = 128
TRITON_SPHERICAL_MAX_K = 2048


def triton_spherical_supported(d: int, n_components: int) -> bool:
    """Validated spherical Triton shape policy.

    Keep this intentionally small and explicit. Shapes outside this range use
    the PyTorch/cuBLAS implementation instead of carrying extra runtime
    branches through the production path.
    """
    return 0 < d <= TRITON_SPHERICAL_MAX_D and 0 < n_components <= TRITON_SPHERICAL_MAX_K


# CUDA shape policy (Plan 1: all stubs return False; Plan 2 onwards populates
# them as kernels land).

def cuda_spherical_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the spherical CUDA backend can handle this shape+dtype.

    Plan 2 (safe path): supports d in (0, 128] and n_components in (0, 2048]
    for dtype in {fp32, fp16, bf16}. Plan 3 will widen the dtype dispatch to
    route fp16/bf16 to the sm80 mma kernel; Plan 2's safe kernel handles all
    three but at SIMT speed.
    """
    import torch as _torch
    if dtype is None:
        return False  # caller didn't supply dtype; conservative no.
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 128):
        return False
    if not (0 < n_components <= 2048):
        return False
    return True


def cuda_diag_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the CUDA backend can handle diagonal EM at this shape+dtype.

    Plan 1 stub: returns False until Plan 3.
    """
    del d, n_components, dtype
    return False


def cuda_tied_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the CUDA backend can handle tied EM at this shape+dtype.

    Plan 1 stub: returns False until Plan 4.
    """
    del d, n_components, dtype
    return False


def cuda_full_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the CUDA backend can handle full-covariance EM at this shape+dtype.

    Plan 1 stub: returns False until Plan 5. Note the spec's D <= 16 cap is
    enforced once the kernel exists; for now the stub simply refuses everything.
    """
    del d, n_components, dtype
    return False
