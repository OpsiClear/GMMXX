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
    """True iff the diagonal CUDA backend can handle this shape+dtype.

    Plan 6 (safe path): supports d in (0, 64] and n_components in (0, 512]
    for dtype in {fp32, fp16, bf16}. Outside this window, the dispatcher
    falls back to Triton or torch.
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 64):
        return False
    if not (0 < n_components <= 512):
        return False
    return True


def cuda_tied_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the tied CUDA backend can handle this shape+dtype.

    Plan 7: D <= 64 (Cholesky is O(D³); reasonable up to D=64), K <= 512,
    dtype in {fp32, fp16, bf16}.
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 64):
        return False
    if not (0 < n_components <= 512):
        return False
    return True


def cuda_full_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the full CUDA backend can handle this shape+dtype.

    Plan 8: D <= 16, K <= 32, dtype in {fp32, fp16, bf16}. Tighter window
    than spherical/diag/tied because each cluster has an O(D²) precision
    representation.
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


def cuda_spherical_fused_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the fused single-tile spherical E/M kernel can handle this shape+dtype.

    Plan 5 (safe + sm80): supports d in (0, 64], n_components in (0, 128] for
    dtype in {fp32, fp16, bf16}. Outside this window, the unfused
    (assign + blocked_update + finalize) pipeline runs (correct but slower
    for medium shapes).
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 64):
        return False
    if not (0 < n_components <= 128):
        return False
    # Benchmark on RTX 4090 showed the fused single-tile kernel regresses badly
    # at the max tile corner (D=64,K=128). Route that shape to the exact
    # soft-update CUDA path until a native fused fix lands.
    if d >= 64 and n_components >= 128:
        return False
    return True
