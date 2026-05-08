"""Triton kernel that fuses addmm bf16 + softmax for the spherical
chunked path. Skips the logits materialization that currently sits between
the cuBLAS bf16 GEMM and the torch softmax launch. Output is resp fp32
which feeds straight into torch.baddbmm.

Per-CTA processes BLOCK_N rows × all K cols. tl.dot uses bf16 HMMA when
inputs are bf16. After softmax, resp is written contiguous to gmem.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _addmm_softmax_kernel(
    x_ptr,            # (cn, D1) bf16
    means_t_ptr,      # (D1, K) bf16  -- already transposed
    alpha_ptr,        # (K,) bf16
    resp_ptr,         # (cn, K) fp32
    cn,
    K: tl.constexpr,
    D1: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_mt_d: tl.constexpr,
    stride_mt_k: tl.constexpr,
    stride_resp_n: tl.constexpr,
    stride_resp_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < cn
    k_mask = offs_k < K

    # GEMM: logits = x @ means_t  (with k-tiling along D1)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for d_start in range(0, D1, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D1
        x_tile = tl.load(
            x_ptr + offs_n[:, None] * stride_x_n + offs_d[None, :] * stride_x_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        mt_tile = tl.load(
            means_t_ptr + offs_d[:, None] * stride_mt_d + offs_k[None, :] * stride_mt_k,
            mask=d_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(x_tile, mt_tile, acc=acc)

    # Add alpha (bias) and mask invalid K columns to -inf for the softmax.
    alpha = tl.load(alpha_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
    logits = acc + alpha[None, :]
    logits = tl.where(k_mask[None, :], logits, -3.4e38)

    # Softmax in fp32.
    rmax = tl.max(logits, axis=1)
    exp_l = tl.exp(logits - rmax[:, None])
    rsum = tl.sum(exp_l, axis=1)
    resp = exp_l / rsum[:, None]
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    tl.store(
        resp_ptr + offs_n[:, None] * stride_resp_n + offs_k[None, :] * stride_resp_k,
        resp,
        mask=n_mask[:, None] & k_mask[None, :],
    )


def addmm_softmax_chunk(
    x_chunk_bf16: torch.Tensor,    # (cn, D1) bf16
    means_aug_t_bf16: torch.Tensor, # (D1, K) bf16 -- pre-transposed and contiguous
    alpha_bf16: torch.Tensor,       # (K,) bf16
    *,
    BLOCK_N: int = 64,
    BLOCK_D: int = 64,
) -> torch.Tensor:
    """Returns resp (1, cn, K) fp32 = softmax(x @ means_t + alpha)."""
    assert x_chunk_bf16.is_cuda and x_chunk_bf16.dtype == torch.bfloat16
    assert means_aug_t_bf16.is_cuda and means_aug_t_bf16.is_contiguous()
    assert means_aug_t_bf16.dtype == torch.bfloat16
    assert alpha_bf16.is_cuda and alpha_bf16.dtype == torch.bfloat16
    cn, D1 = x_chunk_bf16.shape
    D1_m, K = means_aug_t_bf16.shape
    assert D1 == D1_m
    assert alpha_bf16.shape == (K,)

    BLOCK_K = max(16, triton.next_power_of_2(K))
    resp = torch.empty((1, cn, K), device=x_chunk_bf16.device, dtype=torch.float32)
    resp_2d = resp.view(cn, K)

    n_blocks = triton.cdiv(cn, BLOCK_N)
    grid = (n_blocks,)
    _addmm_softmax_kernel[grid](
        x_chunk_bf16, means_aug_t_bf16, alpha_bf16, resp_2d,
        cn, K, D1,
        x_chunk_bf16.stride(0), x_chunk_bf16.stride(1),
        means_aug_t_bf16.stride(0), means_aug_t_bf16.stride(1),
        resp_2d.stride(0), resp_2d.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return resp
