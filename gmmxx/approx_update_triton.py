from __future__ import annotations

import torch
import triton
import triton.language as tl


def approx_topk_update_spherical_config(d: int, k: int, top_k: int) -> dict[str, int] | None:
    """Shape policy for approximate top-k spherical E/M update."""
    if d <= 0 or k <= 0 or top_k <= 0:
        return None
    if top_k >= k or top_k > 16:
        return None
    if d <= 16:
        return {"BLOCK_N": 64, "BLOCK_D": 16, "BLOCK_K": 64, "TOP_K": int(top_k)}
    if d <= 32:
        return {"BLOCK_N": 64, "BLOCK_D": 32, "BLOCK_K": 64, "TOP_K": int(top_k)}
    if d <= 64:
        return {"BLOCK_N": 64, "BLOCK_D": 64, "BLOCK_K": 64, "TOP_K": int(top_k)}
    if d <= 128 and top_k <= 8:
        return {"BLOCK_N": 32, "BLOCK_D": 128, "BLOCK_K": 64, "TOP_K": int(top_k)}
    return None


@triton.jit
def _approx_topk_spherical_update_kernel(
    x_ptr,
    means_ptr,
    x_sq_ptr,
    means_sq_ptr,
    variances_ptr,
    log_weights_ptr,
    nk_ptr,
    sum_x_ptr,
    sum_x_sq_ptr,
    partial_ll_ptr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_means_b: tl.constexpr,
    stride_means_k: tl.constexpr,
    stride_means_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_meanssq_b: tl.constexpr,
    stride_meanssq_k: tl.constexpr,
    stride_var_b: tl.constexpr,
    stride_var_k: tl.constexpr,
    stride_logw_b: tl.constexpr,
    stride_logw_k: tl.constexpr,
    stride_nk_b: tl.constexpr,
    stride_nk_k: tl.constexpr,
    stride_sumx_b: tl.constexpr,
    stride_sumx_k: tl.constexpr,
    stride_sumx_d: tl.constexpr,
    stride_sumxsq_b: tl.constexpr,
    stride_sumxsq_k: tl.constexpr,
    stride_pll_b: tl.constexpr,
    stride_pll_nb: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    offs_top = tl.arange(0, TOP_K).to(tl.int64)
    offs_k_tile = tl.arange(0, BLOCK_K).to(tl.int64)
    n_mask = offs_n < N
    d_mask = offs_d < D

    x_tile = tl.load(
        x_ptr
        + b * stride_x_b
        + offs_n[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    x_sq = tl.load(
        x_sq_ptr + b * stride_xsq_b + offs_n * stride_xsq_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    neg_inf = tl.full((1,), -3.4e38, tl.float32)
    best_vals = tl.full((BLOCK_N, TOP_K), -3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N, TOP_K), tl.int64)
    d_const = tl.full((1,), D, tl.float32)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = (k_start + offs_k_tile).to(tl.int64)
        k_mask = offs_k < K

        means_tile = tl.load(
            means_ptr
            + b * stride_means_b
            + offs_k[None, :] * stride_means_k
            + offs_d[:, None] * stride_means_d,
            mask=k_mask[None, :] & d_mask[:, None],
            other=0.0,
        )
        means_sq = tl.load(
            means_sq_ptr + b * stride_meanssq_b + offs_k * stride_meanssq_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        variances = tl.load(
            variances_ptr + b * stride_var_b + offs_k * stride_var_k,
            mask=k_mask,
            other=1.0,
        ).to(tl.float32)
        log_weights = tl.load(
            log_weights_ptr + b * stride_logw_b + offs_k * stride_logw_k,
            mask=k_mask,
            other=-3.4e38,
        ).to(tl.float32)

        cross = tl.dot(x_tile, means_tile, input_precision="tf32x3").to(tl.float32)
        dist = tl.maximum(x_sq[:, None] + means_sq[None, :] - 2.0 * cross, 0.0)
        log_det_term = d_const * (log_2pi + tl.log(variances))
        logits = log_weights[None, :] - 0.5 * (
            dist / variances[None, :] + log_det_term[None, :]
        )
        logits = tl.where(n_mask[:, None] & k_mask[None, :], logits, neg_inf)

        for _ in tl.static_range(0, TOP_K):
            curr_max = tl.max(logits, axis=1)
            curr_idx = tl.argmax(logits, axis=1).to(tl.int64)
            min_pos = tl.argmax(-best_vals, axis=1).to(tl.int64)
            min_best = tl.max(-best_vals, axis=1) * -1.0
            update = curr_max > min_best
            slot_mask = offs_top[None, :] == min_pos[:, None]
            best_vals = tl.where(update[:, None] & slot_mask, curr_max[:, None], best_vals)
            best_idx = tl.where(
                update[:, None] & slot_mask,
                (tl.full((BLOCK_N,), k_start, tl.int64) + curr_idx)[:, None],
                best_idx,
            )
            logits = tl.where(offs_k_tile[None, :] == curr_idx[:, None], neg_inf, logits)

    best_vals = tl.where(n_mask[:, None], best_vals, neg_inf)
    row_max = tl.max(best_vals, axis=1)
    exp_sums = tl.sum(tl.exp(best_vals - row_max[:, None]), axis=1)
    log_norm = row_max + tl.log(exp_sums)
    resp = tl.exp(best_vals - log_norm[:, None])
    resp = tl.where(n_mask[:, None], resp, 0.0)
    acc_ll = tl.sum(tl.where(n_mask, log_norm, 0.0), axis=0)

    for top_slot in tl.static_range(0, TOP_K):
        top_mask = offs_top[None, :] == top_slot
        k_idx = tl.sum(tl.where(top_mask, best_idx, 0), axis=1)
        r = tl.sum(tl.where(top_mask, resp, 0.0), axis=1)
        valid = n_mask & (r > 0.0)

        tl.atomic_add(
            nk_ptr + b * stride_nk_b + k_idx * stride_nk_k,
            r,
            sem="relaxed",
            mask=valid,
        )
        tl.atomic_add(
            sum_x_sq_ptr + b * stride_sumxsq_b + k_idx * stride_sumxsq_k,
            r * x_sq,
            sem="relaxed",
            mask=valid,
        )
        tl.atomic_add(
            sum_x_ptr
            + b * stride_sumx_b
            + k_idx[:, None] * stride_sumx_k
            + offs_d[None, :] * stride_sumx_d,
            r[:, None] * x_tile,
            sem="relaxed",
            mask=valid[:, None] & d_mask[None, :],
        )

    tl.store(partial_ll_ptr + b * stride_pll_b + pid_n * stride_pll_nb, acc_ll)


def _alloc_or_slice(
    value: torch.Tensor | None,
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.empty(shape, device=device, dtype=torch.float32)
    slices = tuple(slice(0, dim) for dim in shape)
    return value[slices]


def triton_approx_topk_update_spherical(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    top_k: int,
    x_sq: torch.Tensor | None = None,
    means_sq: torch.Tensor | None = None,
    log_weights: torch.Tensor | None = None,
    nk: torch.Tensor | None = None,
    sum_x: torch.Tensor | None = None,
    sum_x_sq: torch.Tensor | None = None,
    partial_log_likelihood: torch.Tensor | None = None,
    BLOCK_N: int = 64,
    BLOCK_D: int = 64,
    BLOCK_K: int = 64,
    TOP_K: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.is_cuda and means.is_cuda and variances.is_cuda and weights.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bm, K, Dm = means.shape
    assert (Bm, Dm) == (B, D), "means shape mismatch"
    assert variances.shape == (B, K), "variances shape mismatch"
    assert weights.shape == (B, K), "weights shape mismatch"
    top_k = int(top_k if TOP_K is None else TOP_K)
    if top_k <= 0 or top_k >= K or top_k > 16:
        raise ValueError("top_k must be in [1, min(K - 1, 16)]")
    if D > BLOCK_D:
        raise ValueError(f"approx top-k spherical update requires D <= BLOCK_D; got D={D}, BLOCK_D={BLOCK_D}")
    if x_sq is None:
        x_sq = x.to(torch.float32).square().sum(dim=-1)
    if means_sq is None:
        means_sq = means.to(torch.float32).square().sum(dim=-1)
    if log_weights is None:
        log_weights = torch.log(weights.to(torch.float32))
    if nk is None:
        nk = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    if sum_x is None:
        sum_x = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    if sum_x_sq is None:
        sum_x_sq = torch.zeros((B, K), device=x.device, dtype=torch.float32)

    n_blocks = triton.cdiv(N, BLOCK_N)
    partial_log_likelihood = _alloc_or_slice(
        partial_log_likelihood,
        (B, n_blocks),
        device=x.device,
    )

    grid = (n_blocks, B)
    _approx_topk_spherical_update_kernel[grid](
        x,
        means,
        x_sq.to(torch.float32),
        means_sq.to(torch.float32),
        variances.to(torch.float32),
        log_weights.to(torch.float32),
        nk,
        sum_x,
        sum_x_sq,
        partial_log_likelihood,
        x.stride(0), x.stride(1), x.stride(2),
        means.stride(0), means.stride(1), means.stride(2),
        x_sq.stride(0), x_sq.stride(1),
        means_sq.stride(0), means_sq.stride(1),
        variances.stride(0), variances.stride(1),
        log_weights.stride(0), log_weights.stride(1),
        nk.stride(0), nk.stride(1),
        sum_x.stride(0), sum_x.stride(1), sum_x.stride(2),
        sum_x_sq.stride(0), sum_x_sq.stride(1),
        partial_log_likelihood.stride(0), partial_log_likelihood.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        TOP_K=top_k,
        num_warps=4 if BLOCK_D <= 64 else 8,
        num_stages=1,
    )
    return nk, sum_x, sum_x_sq, partial_log_likelihood.sum()
