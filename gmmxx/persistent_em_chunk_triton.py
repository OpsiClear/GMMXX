"""Exp78: Persistent-CTA Triton kernel that processes the full N range
of a chunk in one launch, accumulating the (K, D+2) partial in
register/shared-memory across BLOCK_N row tiles. Eliminates the
per-row-tile baddbmm and the resp gmem round-trip.

Each CTA owns a contiguous N range. Inside, the kernel iterates over
its range in tiles of BLOCK_N, doing the full E-step (bf16 GEMM +
softmax) and the M-step partial (resp.T @ x_aug fp32) per tile, with
the partial accumulated locally across tiles. After the outer loop,
the CTA writes its per-CTA partial to a unique slot in a global
buffer; a final torch.sum reduces across CTAs.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _persistent_em_kernel(
    x_ptr,           # (N, D1) bf16
    means_t_ptr,     # (D1, K) bf16
    alpha_ptr,       # (K,) bf16
    x_aug_ptr,       # (N, D2) fp32
    partial_ptr,     # (NUM_CTAS, K, D2) fp32
    N,
    K: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_mt_d: tl.constexpr,
    stride_mt_k: tl.constexpr,
    stride_xa_n: tl.constexpr,
    stride_xa_d: tl.constexpr,
    stride_p_b: tl.constexpr,
    stride_p_k: tl.constexpr,
    stride_p_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    X_AUG_IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d2 = tl.program_id(1)

    # Each CTA processes N rows partitioned evenly across NUM_CTAS.
    rows_per_cta = (N + NUM_CTAS - 1) // NUM_CTAS
    n_start_cta = pid * rows_per_cta
    n_end_cta = tl.minimum(n_start_cta + rows_per_cta, N)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d2 = pid_d2 * BLOCK_D2 + tl.arange(0, BLOCK_D2)
    k_mask = offs_k < K
    d2_mask = offs_d2 < D2

    # Per-CTA partial accumulator (K, D2 chunk).
    partial = tl.zeros((BLOCK_K, BLOCK_D2), dtype=tl.float32)

    # Load alpha once into registers.
    alpha = tl.load(alpha_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)

    # Outer loop over the CTA's N rows in BLOCK_N tiles.
    n_tile = n_start_cta
    while n_tile < n_end_cta:
        offs_n = n_tile + tl.arange(0, BLOCK_N)
        n_mask = offs_n < n_end_cta

        # Stage 1: GEMM logits = x @ means_t (D1 K-tiled).
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for d_start in range(0, D1, BLOCK_D1):
            offs_d1 = d_start + tl.arange(0, BLOCK_D1)
            d1_mask = offs_d1 < D1
            x_t = tl.load(
                x_ptr + offs_n[:, None] * stride_x_n + offs_d1[None, :] * stride_x_d,
                mask=n_mask[:, None] & d1_mask[None, :],
                other=0.0,
            )
            mt_t = tl.load(
                means_t_ptr + offs_d1[:, None] * stride_mt_d + offs_k[None, :] * stride_mt_k,
                mask=d1_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(x_t, mt_t, acc=acc)
        logits = acc + alpha[None, :]
        logits = tl.where(k_mask[None, :], logits, -3.4e38)

        # Stage 2: softmax row-wise.
        rmax = tl.max(logits, axis=1)
        expl = tl.exp(logits - rmax[:, None])
        rsum = tl.sum(expl, axis=1)
        resp = expl / rsum[:, None]
        resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

        # Stage 3: partial += resp.T @ x_aug_tile.
        # Use bf16 x_aug + bf16 resp via tl.dot (HMMA, fp32 acc).
        xa = tl.load(
            x_aug_ptr + offs_n[:, None] * stride_xa_n + offs_d2[None, :] * stride_xa_d,
            mask=n_mask[:, None] & d2_mask[None, :],
            other=0.0,
        )
        if X_AUG_IS_BF16:
            partial += tl.dot(tl.trans(resp).to(tl.bfloat16), xa)
        else:
            partial += tl.dot(tl.trans(resp), xa)

        n_tile += BLOCK_N

    # Write per-CTA partial slice to global.
    tl.store(
        partial_ptr
        + pid * stride_p_b
        + offs_k[:, None] * stride_p_k
        + offs_d2[None, :] * stride_p_d,
        partial,
        mask=k_mask[:, None] & d2_mask[None, :],
    )


def persistent_em_iter(
    x_bf16: torch.Tensor,        # (N, D1) bf16
    means_aug_t_bf16: torch.Tensor, # (D1, K) bf16
    alpha_bf16: torch.Tensor,     # (K,) bf16
    x_aug: torch.Tensor,         # (N, D2) fp32
    *,
    NUM_CTAS: int = 128,
    BLOCK_N: int = 64,
    BLOCK_D1: int = 64,
    BLOCK_D2: int = 64,
    partial_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns sum_aug (K, D2) fp32 = sum_n softmax(addmm(alpha, x, means_t))[n,:].T * x_aug[n,:]."""
    N, D1 = x_bf16.shape
    D1_m, K = means_aug_t_bf16.shape
    D2 = x_aug.size(1)
    assert D1 == D1_m
    assert x_aug.size(0) == N

    BLOCK_K = max(16, triton.next_power_of_2(K))
    d2_blocks = triton.cdiv(D2, BLOCK_D2)

    if partial_buffer is None or partial_buffer.shape != (NUM_CTAS, K, D2):
        partial_buffer = torch.empty(
            (NUM_CTAS, K, D2), device=x_bf16.device, dtype=torch.float32
        )

    grid = (NUM_CTAS, d2_blocks)
    _persistent_em_kernel[grid](
        x_bf16, means_aug_t_bf16, alpha_bf16, x_aug, partial_buffer,
        N, K, D1, D2, NUM_CTAS,
        x_bf16.stride(0), x_bf16.stride(1),
        means_aug_t_bf16.stride(0), means_aug_t_bf16.stride(1),
        x_aug.stride(0), x_aug.stride(1),
        partial_buffer.stride(0), partial_buffer.stride(1), partial_buffer.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_D1=BLOCK_D1,
        BLOCK_D2=BLOCK_D2,
        X_AUG_IS_BF16=(x_aug.dtype == torch.bfloat16),
        num_warps=4,
        num_stages=2,
    )
    return partial_buffer.sum(dim=0)
