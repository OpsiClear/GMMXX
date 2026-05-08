from __future__ import annotations


# Triton shape policy (existing).
TRITON_SPHERICAL_MAX_D = 128
TRITON_SPHERICAL_MAX_K = 2048
CUDA_SPHERICAL_MAX_K = 8192
CUDA_SPHERICAL_APPROX_TOPK_MAX_K = 8192
CUDA_SPHERICAL_APPROX_TOPK_MAX_TOP_K = 16
CUDA_DIAG_STREAMED_MAX_D = 128
CUDA_DIAG_STREAMED_MAX_K = 8192
CUDA_TIED_STREAMED_MAX_D = 128
CUDA_TIED_STREAMED_MAX_K = 8192
CUDA_FULL_STREAMED_MAX_D = 128
CUDA_FULL_STREAMED_MAX_COV_ELEMENTS = 2_000_000


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

    Supports d in (0, 128] and n_components in (0, 8192] for dtype in
    {fp32, fp16, bf16}. Exact large-K support uses the same streamed
    assign/update/finalize pipeline as the K<=2048 path.
    """
    import torch as _torch
    if dtype is None:
        return False  # caller didn't supply dtype; conservative no.
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= 128):
        return False
    if not (0 < n_components <= CUDA_SPHERICAL_MAX_K):
        return False
    return True


def cuda_spherical_approx_topk_supported(
    d: int,
    n_components: int,
    dtype,
    top_k: int | None,
) -> bool:
    """True iff CUDA approximate spherical EM can handle this shape.

    Approximate top-k keeps only ``top_k`` candidates per sample and streams
    over K, so it can support the flash-kmeans-sized large-K cases without
    changing the exact EM support window.
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if top_k is None:
        return False
    top_k = int(top_k)
    if not (0 < d <= 128):
        return False
    if not (0 < n_components <= CUDA_SPHERICAL_APPROX_TOPK_MAX_K):
        return False
    if not (0 < top_k < n_components):
        return False
    if top_k > CUDA_SPHERICAL_APPROX_TOPK_MAX_TOP_K:
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


def cuda_diag_streamed_supported(d: int, n_components: int, dtype) -> bool:
    """True iff CUDA-tensor streamed diagonal EM can handle this shape+dtype.

    This is the large-shape in-memory path. It deliberately sits beside
    ``cuda_diag_supported`` because the native CUDA diag loop materializes
    ``(B, N, K)`` responsibilities and should stay in its smaller window.
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= CUDA_DIAG_STREAMED_MAX_D):
        return False
    if not (0 < n_components <= CUDA_DIAG_STREAMED_MAX_K):
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


def cuda_tied_streamed_supported(d: int, n_components: int, dtype) -> bool:
    """True iff CUDA-tensor streamed tied EM can handle this shape+dtype.

    Tied covariance has one ``(D, D)`` covariance matrix, so the large-K
    parameter state remains bounded unlike full covariance.
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= CUDA_TIED_STREAMED_MAX_D):
        return False
    if not (0 < n_components <= CUDA_TIED_STREAMED_MAX_K):
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


def cuda_full_streamed_supported(d: int, n_components: int, dtype) -> bool:
    """True iff CUDA-tensor streamed full EM can handle this shape+dtype.

    Full covariance stores ``K`` dense ``D x D`` matrices, so the large-shape
    gate is state-size based instead of matching the diag/tied K limit.
    """
    import torch as _torch
    if dtype is None:
        return False
    if dtype not in (_torch.float32, _torch.float16, _torch.bfloat16):
        return False
    if not (0 < d <= CUDA_FULL_STREAMED_MAX_D):
        return False
    if n_components <= 0:
        return False
    if int(n_components) * int(d) * int(d) > CUDA_FULL_STREAMED_MAX_COV_ELEMENTS:
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
