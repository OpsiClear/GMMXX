from __future__ import annotations

import torch
import triton
import triton.language as tl


def fused_single_tile_update_config(d: int, k: int) -> dict[str, int] | None:
    """Shape policy for exact one-K-tile fused E/M updates."""
    if k <= 0 or d <= 0:
        return None
    if d <= 16 and k <= 128:
        return {"BLOCK_N": 64, "BLOCK_D": 16, "BLOCK_K": 128}
    if d <= 32 and k <= 128:
        return {"BLOCK_N": 64, "BLOCK_D": 32, "BLOCK_K": 128}
    if d <= 64 and k <= 64:
        return {"BLOCK_N": 64, "BLOCK_D": 64, "BLOCK_K": 64}
    return None


@triton.jit
def _fused_single_tile_spherical_kernel(
    x_ptr,
    means_ptr,
    x_sq_ptr,
    means_sq_ptr,
    variances_ptr,
    log_weights_ptr,
    partial_nk_ptr,
    partial_sum_x_ptr,
    partial_sum_x_sq_ptr,
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
    stride_pnk_b: tl.constexpr,
    stride_pnk_nb: tl.constexpr,
    stride_pnk_k: tl.constexpr,
    stride_psumx_b: tl.constexpr,
    stride_psumx_nb: tl.constexpr,
    stride_psumx_k: tl.constexpr,
    stride_psumx_d: tl.constexpr,
    stride_psumxsq_b: tl.constexpr,
    stride_psumxsq_nb: tl.constexpr,
    stride_psumxsq_k: tl.constexpr,
    stride_pll_b: tl.constexpr,
    stride_pll_nb: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    n_mask = offs_n < N
    d_mask = offs_d < D
    k_mask = offs_k < K

    x_tile = tl.load(
        x_ptr
        + b * stride_x_b
        + offs_n[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    means_tile = tl.load(
        means_ptr
        + b * stride_means_b
        + offs_k[None, :] * stride_means_k
        + offs_d[:, None] * stride_means_d,
        mask=d_mask[:, None] & k_mask[None, :],
        other=0.0,
    )
    x_sq = tl.load(
        x_sq_ptr + b * stride_xsq_b + offs_n * stride_xsq_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
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
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    d_const = tl.full((1,), D, tl.float32)
    log_det_term = d_const * (log_2pi + tl.log(variances))
    logits = log_weights[None, :] - 0.5 * (
        dist / variances[None, :] + log_det_term[None, :]
    )
    logits = tl.where(k_mask[None, :], logits, -3.4e38)

    row_max = tl.max(logits, axis=1)
    exp_sums = tl.sum(tl.exp(logits - row_max[:, None]), axis=1)
    log_norm = row_max + tl.log(exp_sums)
    resp = tl.exp(logits - log_norm[:, None])
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    acc_nk = tl.sum(resp, axis=0)
    acc_sum_x = tl.dot(tl.trans(resp), x_tile, input_precision="tf32x3").to(tl.float32)
    acc_sum_x_sq = tl.sum(resp * x_sq[:, None], axis=0)
    acc_ll = tl.sum(tl.where(n_mask, log_norm, 0.0), axis=0)

    tl.store(
        partial_nk_ptr + b * stride_pnk_b + pid_n * stride_pnk_nb + offs_k * stride_pnk_k,
        acc_nk,
        mask=k_mask,
    )
    tl.store(
        partial_sum_x_sq_ptr
        + b * stride_psumxsq_b
        + pid_n * stride_psumxsq_nb
        + offs_k * stride_psumxsq_k,
        acc_sum_x_sq,
        mask=k_mask,
    )
    tl.store(
        partial_sum_x_ptr
        + b * stride_psumx_b
        + pid_n * stride_psumx_nb
        + offs_k[:, None] * stride_psumx_k
        + offs_d[None, :] * stride_psumx_d,
        acc_sum_x,
        mask=k_mask[:, None] & d_mask[None, :],
    )
    tl.store(partial_ll_ptr + b * stride_pll_b + pid_n * stride_pll_nb, acc_ll)


@triton.jit
def _fused_single_tile_diag_kernel(
    x_ptr,
    precision_ptr,
    weighted_means_ptr,
    mean_precision_mean_ptr,
    logdet_ptr,
    log_weights_ptr,
    partial_nk_ptr,
    partial_sum_x_ptr,
    partial_sum_x_sq_ptr,
    partial_ll_ptr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_precision_b: tl.constexpr,
    stride_precision_k: tl.constexpr,
    stride_precision_d: tl.constexpr,
    stride_wmeans_b: tl.constexpr,
    stride_wmeans_k: tl.constexpr,
    stride_wmeans_d: tl.constexpr,
    stride_mpm_b: tl.constexpr,
    stride_mpm_k: tl.constexpr,
    stride_logdet_b: tl.constexpr,
    stride_logdet_k: tl.constexpr,
    stride_logw_b: tl.constexpr,
    stride_logw_k: tl.constexpr,
    stride_pnk_b: tl.constexpr,
    stride_pnk_nb: tl.constexpr,
    stride_pnk_k: tl.constexpr,
    stride_psumx_b: tl.constexpr,
    stride_psumx_nb: tl.constexpr,
    stride_psumx_k: tl.constexpr,
    stride_psumx_d: tl.constexpr,
    stride_psumxsq_b: tl.constexpr,
    stride_psumxsq_nb: tl.constexpr,
    stride_psumxsq_k: tl.constexpr,
    stride_psumxsq_d: tl.constexpr,
    stride_pll_b: tl.constexpr,
    stride_pll_nb: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    n_mask = offs_n < N
    d_mask = offs_d < D
    k_mask = offs_k < K

    x_tile = tl.load(
        x_ptr
        + b * stride_x_b
        + offs_n[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x_sq_tile = x_tile * x_tile
    precision_tile = tl.load(
        precision_ptr
        + b * stride_precision_b
        + offs_k[None, :] * stride_precision_k
        + offs_d[:, None] * stride_precision_d,
        mask=k_mask[None, :] & d_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    weighted_means_tile = tl.load(
        weighted_means_ptr
        + b * stride_wmeans_b
        + offs_k[None, :] * stride_wmeans_k
        + offs_d[:, None] * stride_wmeans_d,
        mask=k_mask[None, :] & d_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    mean_precision_mean = tl.load(
        mean_precision_mean_ptr + b * stride_mpm_b + offs_k * stride_mpm_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    logdet = tl.load(
        logdet_ptr + b * stride_logdet_b + offs_k * stride_logdet_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    log_weights = tl.load(
        log_weights_ptr + b * stride_logw_b + offs_k * stride_logw_k,
        mask=k_mask,
        other=-3.4e38,
    ).to(tl.float32)

    x_precision_x = tl.dot(x_sq_tile, precision_tile, input_precision="tf32x3").to(tl.float32)
    cross = tl.dot(x_tile, weighted_means_tile, input_precision="tf32x3").to(tl.float32)
    quad = tl.maximum(x_precision_x - 2.0 * cross + mean_precision_mean[None, :], 0.0)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    logits = log_weights[None, :] - 0.5 * (
        quad + tl.full((1,), D, tl.float32) * log_2pi + logdet[None, :]
    )
    logits = tl.where(k_mask[None, :], logits, -3.4e38)

    row_max = tl.max(logits, axis=1)
    exp_sums = tl.sum(tl.exp(logits - row_max[:, None]), axis=1)
    log_norm = row_max + tl.log(exp_sums)
    resp = tl.exp(logits - log_norm[:, None])
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    acc_nk = tl.sum(resp, axis=0)
    acc_sum_x = tl.dot(tl.trans(resp), x_tile, input_precision="tf32x3").to(tl.float32)
    acc_sum_x_sq = tl.dot(tl.trans(resp), x_sq_tile, input_precision="tf32x3").to(tl.float32)
    acc_ll = tl.sum(tl.where(n_mask, log_norm, 0.0), axis=0)

    tl.store(
        partial_nk_ptr + b * stride_pnk_b + pid_n * stride_pnk_nb + offs_k * stride_pnk_k,
        acc_nk,
        mask=k_mask,
    )
    tl.store(
        partial_sum_x_ptr
        + b * stride_psumx_b
        + pid_n * stride_psumx_nb
        + offs_k[:, None] * stride_psumx_k
        + offs_d[None, :] * stride_psumx_d,
        acc_sum_x,
        mask=k_mask[:, None] & d_mask[None, :],
    )
    tl.store(
        partial_sum_x_sq_ptr
        + b * stride_psumxsq_b
        + pid_n * stride_psumxsq_nb
        + offs_k[:, None] * stride_psumxsq_k
        + offs_d[None, :] * stride_psumxsq_d,
        acc_sum_x_sq,
        mask=k_mask[:, None] & d_mask[None, :],
    )
    tl.store(partial_ll_ptr + b * stride_pll_b + pid_n * stride_pll_nb, acc_ll)


@triton.jit
def _fused_single_tile_tied_native_kernel(
    x_ptr,
    chol_precision_ptr,
    means_projected_ptr,
    means_projected_sq_ptr,
    logdet_ptr,
    log_weights_ptr,
    partial_nk_ptr,
    partial_sum_x_ptr,
    partial_ll_ptr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_chol_b: tl.constexpr,
    stride_chol_d0: tl.constexpr,
    stride_chol_d1: tl.constexpr,
    stride_means_b: tl.constexpr,
    stride_means_k: tl.constexpr,
    stride_means_d: tl.constexpr,
    stride_meanssq_b: tl.constexpr,
    stride_meanssq_k: tl.constexpr,
    stride_logdet_b: tl.constexpr,
    stride_logw_b: tl.constexpr,
    stride_logw_k: tl.constexpr,
    stride_pnk_b: tl.constexpr,
    stride_pnk_nb: tl.constexpr,
    stride_pnk_k: tl.constexpr,
    stride_psumx_b: tl.constexpr,
    stride_psumx_nb: tl.constexpr,
    stride_psumx_k: tl.constexpr,
    stride_psumx_d: tl.constexpr,
    stride_pll_b: tl.constexpr,
    stride_pll_nb: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    n_mask = offs_n < N
    d_mask = offs_d < D
    k_mask = offs_k < K

    x_tile = tl.load(
        x_ptr
        + b * stride_x_b
        + offs_n[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    chol = tl.load(
        chol_precision_ptr
        + b * stride_chol_b
        + offs_d[:, None] * stride_chol_d0
        + offs_d[None, :] * stride_chol_d1,
        mask=d_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x_projected = tl.dot(x_tile, chol, input_precision="tf32x3").to(tl.float32)
    x_projected_sq = tl.sum(x_projected * x_projected, axis=1)

    means_projected = tl.load(
        means_projected_ptr
        + b * stride_means_b
        + offs_k[None, :] * stride_means_k
        + offs_d[:, None] * stride_means_d,
        mask=k_mask[None, :] & d_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    means_projected_sq = tl.load(
        means_projected_sq_ptr + b * stride_meanssq_b + offs_k * stride_meanssq_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    log_weights = tl.load(
        log_weights_ptr + b * stride_logw_b + offs_k * stride_logw_k,
        mask=k_mask,
        other=-3.4e38,
    ).to(tl.float32)
    logdet = tl.load(logdet_ptr + b * stride_logdet_b).to(tl.float32)

    cross = tl.dot(x_projected, means_projected, input_precision="tf32x3").to(tl.float32)
    dist = tl.maximum(x_projected_sq[:, None] + means_projected_sq[None, :] - 2.0 * cross, 0.0)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    logits = log_weights[None, :] - 0.5 * (
        dist + tl.full((1,), D, tl.float32) * log_2pi
    )
    logits = tl.where(k_mask[None, :], logits, -3.4e38)

    row_max = tl.max(logits, axis=1)
    exp_sums = tl.sum(tl.exp(logits - row_max[:, None]), axis=1)
    projected_log_norm = row_max + tl.log(exp_sums)
    resp = tl.exp(logits - projected_log_norm[:, None])
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    acc_nk = tl.sum(resp, axis=0)
    acc_sum_x = tl.dot(tl.trans(resp), x_tile, input_precision="tf32x3").to(tl.float32)
    acc_ll = tl.sum(tl.where(n_mask, projected_log_norm - 0.5 * logdet, 0.0), axis=0)

    tl.store(
        partial_nk_ptr + b * stride_pnk_b + pid_n * stride_pnk_nb + offs_k * stride_pnk_k,
        acc_nk,
        mask=k_mask,
    )
    tl.store(
        partial_sum_x_ptr
        + b * stride_psumx_b
        + pid_n * stride_psumx_nb
        + offs_k[:, None] * stride_psumx_k
        + offs_d[None, :] * stride_psumx_d,
        acc_sum_x,
        mask=k_mask[:, None] & d_mask[None, :],
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


def triton_fused_single_tile_update_spherical(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    x_sq: torch.Tensor | None = None,
    means_sq: torch.Tensor | None = None,
    log_weights: torch.Tensor | None = None,
    partial_nk: torch.Tensor | None = None,
    partial_sum_x: torch.Tensor | None = None,
    partial_sum_x_sq: torch.Tensor | None = None,
    partial_log_likelihood: torch.Tensor | None = None,
    BLOCK_N: int = 128,
    BLOCK_D: int = 64,
    BLOCK_K: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.is_cuda and means.is_cuda and variances.is_cuda and weights.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bm, K, Dm = means.shape
    assert (Bm, Dm) == (B, D), "means shape mismatch"
    assert variances.shape == (B, K), "variances shape mismatch"
    assert weights.shape == (B, K), "weights shape mismatch"
    if K > BLOCK_K:
        raise ValueError(f"fused spherical update requires K <= BLOCK_K; got K={K}, BLOCK_K={BLOCK_K}")
    if D > BLOCK_D:
        raise ValueError(f"fused spherical update requires D <= BLOCK_D; got D={D}, BLOCK_D={BLOCK_D}")
    if x_sq is None:
        x_sq = x.to(torch.float32).square().sum(dim=-1)
    if means_sq is None:
        means_sq = means.to(torch.float32).square().sum(dim=-1)
    if log_weights is None:
        log_weights = torch.log(weights.to(torch.float32))

    n_blocks = triton.cdiv(N, BLOCK_N)
    partial_nk = _alloc_or_slice(partial_nk, (B, n_blocks, K), device=x.device)
    partial_sum_x = _alloc_or_slice(partial_sum_x, (B, n_blocks, K, D), device=x.device)
    partial_sum_x_sq = _alloc_or_slice(partial_sum_x_sq, (B, n_blocks, K), device=x.device)
    partial_log_likelihood = _alloc_or_slice(partial_log_likelihood, (B, n_blocks), device=x.device)

    grid = (n_blocks, B)
    _fused_single_tile_spherical_kernel[grid](
        x,
        means,
        x_sq.to(torch.float32),
        means_sq.to(torch.float32),
        variances.to(torch.float32),
        log_weights.to(torch.float32),
        partial_nk,
        partial_sum_x,
        partial_sum_x_sq,
        partial_log_likelihood,
        x.stride(0), x.stride(1), x.stride(2),
        means.stride(0), means.stride(1), means.stride(2),
        x_sq.stride(0), x_sq.stride(1),
        means_sq.stride(0), means_sq.stride(1),
        variances.stride(0), variances.stride(1),
        log_weights.stride(0), log_weights.stride(1),
        partial_nk.stride(0), partial_nk.stride(1), partial_nk.stride(2),
        partial_sum_x.stride(0), partial_sum_x.stride(1), partial_sum_x.stride(2), partial_sum_x.stride(3),
        partial_sum_x_sq.stride(0), partial_sum_x_sq.stride(1), partial_sum_x_sq.stride(2),
        partial_log_likelihood.stride(0), partial_log_likelihood.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=1,
    )
    return (
        partial_nk.sum(dim=1),
        partial_sum_x.sum(dim=1),
        partial_sum_x_sq.sum(dim=1),
        partial_log_likelihood.sum(),
    )


def triton_fused_single_tile_update_diag(
    x: torch.Tensor,
    precision: torch.Tensor,
    weighted_means: torch.Tensor,
    mean_precision_mean: torch.Tensor,
    logdet: torch.Tensor,
    log_weights: torch.Tensor,
    *,
    partial_nk: torch.Tensor | None = None,
    partial_sum_x: torch.Tensor | None = None,
    partial_sum_x_sq: torch.Tensor | None = None,
    partial_log_likelihood: torch.Tensor | None = None,
    BLOCK_N: int = 128,
    BLOCK_D: int = 64,
    BLOCK_K: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.is_cuda and precision.is_cuda and weighted_means.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bp, K, Dp = precision.shape
    assert (Bp, Dp) == (B, D), "precision shape mismatch"
    assert weighted_means.shape == (B, K, D), "weighted_means shape mismatch"
    assert mean_precision_mean.shape == (B, K), "mean_precision_mean shape mismatch"
    assert logdet.shape == (B, K), "logdet shape mismatch"
    assert log_weights.shape == (B, K), "log_weights shape mismatch"
    if K > BLOCK_K:
        raise ValueError(f"fused diagonal update requires K <= BLOCK_K; got K={K}, BLOCK_K={BLOCK_K}")
    if D > BLOCK_D:
        raise ValueError(f"fused diagonal update requires D <= BLOCK_D; got D={D}, BLOCK_D={BLOCK_D}")

    n_blocks = triton.cdiv(N, BLOCK_N)
    partial_nk = _alloc_or_slice(partial_nk, (B, n_blocks, K), device=x.device)
    partial_sum_x = _alloc_or_slice(partial_sum_x, (B, n_blocks, K, D), device=x.device)
    partial_sum_x_sq = _alloc_or_slice(partial_sum_x_sq, (B, n_blocks, K, D), device=x.device)
    partial_log_likelihood = _alloc_or_slice(partial_log_likelihood, (B, n_blocks), device=x.device)

    grid = (n_blocks, B)
    _fused_single_tile_diag_kernel[grid](
        x,
        precision.to(torch.float32),
        weighted_means.to(torch.float32),
        mean_precision_mean.to(torch.float32),
        logdet.to(torch.float32),
        log_weights.to(torch.float32),
        partial_nk,
        partial_sum_x,
        partial_sum_x_sq,
        partial_log_likelihood,
        x.stride(0), x.stride(1), x.stride(2),
        precision.stride(0), precision.stride(1), precision.stride(2),
        weighted_means.stride(0), weighted_means.stride(1), weighted_means.stride(2),
        mean_precision_mean.stride(0), mean_precision_mean.stride(1),
        logdet.stride(0), logdet.stride(1),
        log_weights.stride(0), log_weights.stride(1),
        partial_nk.stride(0), partial_nk.stride(1), partial_nk.stride(2),
        partial_sum_x.stride(0), partial_sum_x.stride(1), partial_sum_x.stride(2), partial_sum_x.stride(3),
        partial_sum_x_sq.stride(0), partial_sum_x_sq.stride(1), partial_sum_x_sq.stride(2), partial_sum_x_sq.stride(3),
        partial_log_likelihood.stride(0), partial_log_likelihood.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=1,
    )
    return (
        partial_nk.sum(dim=1),
        partial_sum_x.sum(dim=1),
        partial_sum_x_sq.sum(dim=1),
        partial_log_likelihood.sum(),
    )


def triton_fused_single_tile_update_tied_native(
    x: torch.Tensor,
    chol_precision: torch.Tensor,
    means_projected: torch.Tensor,
    means_projected_sq: torch.Tensor,
    logdet: torch.Tensor,
    log_weights: torch.Tensor,
    *,
    partial_nk: torch.Tensor | None = None,
    partial_sum_x: torch.Tensor | None = None,
    partial_log_likelihood: torch.Tensor | None = None,
    BLOCK_N: int = 128,
    BLOCK_D: int = 64,
    BLOCK_K: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.is_cuda and chol_precision.is_cuda and means_projected.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bc, D0, D1 = chol_precision.shape
    Bm, K, Dm = means_projected.shape
    assert (Bc, D0, D1) == (B, D, D), "chol_precision shape mismatch"
    assert (Bm, Dm) == (B, D), "means_projected shape mismatch"
    assert means_projected_sq.shape == (B, K), "means_projected_sq shape mismatch"
    assert logdet.shape == (B,), "logdet shape mismatch"
    assert log_weights.shape == (B, K), "log_weights shape mismatch"
    if K > BLOCK_K:
        raise ValueError(f"fused tied update requires K <= BLOCK_K; got K={K}, BLOCK_K={BLOCK_K}")
    if D > BLOCK_D:
        raise ValueError(f"fused tied update requires D <= BLOCK_D; got D={D}, BLOCK_D={BLOCK_D}")

    n_blocks = triton.cdiv(N, BLOCK_N)
    partial_nk = _alloc_or_slice(partial_nk, (B, n_blocks, K), device=x.device)
    partial_sum_x = _alloc_or_slice(partial_sum_x, (B, n_blocks, K, D), device=x.device)
    partial_log_likelihood = _alloc_or_slice(partial_log_likelihood, (B, n_blocks), device=x.device)

    grid = (n_blocks, B)
    _fused_single_tile_tied_native_kernel[grid](
        x,
        chol_precision.to(torch.float32),
        means_projected.to(torch.float32),
        means_projected_sq.to(torch.float32),
        logdet.to(torch.float32),
        log_weights.to(torch.float32),
        partial_nk,
        partial_sum_x,
        partial_log_likelihood,
        x.stride(0), x.stride(1), x.stride(2),
        chol_precision.stride(0), chol_precision.stride(1), chol_precision.stride(2),
        means_projected.stride(0), means_projected.stride(1), means_projected.stride(2),
        means_projected_sq.stride(0), means_projected_sq.stride(1),
        logdet.stride(0),
        log_weights.stride(0), log_weights.stride(1),
        partial_nk.stride(0), partial_nk.stride(1), partial_nk.stride(2),
        partial_sum_x.stride(0), partial_sum_x.stride(1), partial_sum_x.stride(2), partial_sum_x.stride(3),
        partial_log_likelihood.stride(0), partial_log_likelihood.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=1,
    )
    return partial_nk.sum(dim=1), partial_sum_x.sum(dim=1), partial_log_likelihood.sum()
