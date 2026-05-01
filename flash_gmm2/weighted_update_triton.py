from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _blocked_update_spherical_kernel(
    x_ptr,                  # *f16 / *f32 [B, N, D]
    means_ptr,              # *f16 / *f32 [B, K, D]
    x_sq_ptr,               # *f32        [B, N]
    means_sq_ptr,           # *f32        [B, K]
    variances_ptr,          # *f32        [B, K]
    log_weights_ptr,        # *f32        [B, K]
    log_norm_ptr,           # *f32        [B, N]
    partial_nk_ptr,         # *f32        [B, NB, K]
    partial_sum_x_ptr,      # *f32        [B, NB, K, D]
    partial_sum_x_sq_ptr,   # *f32        [B, NB, K]
    stride_x_b, stride_x_n, stride_x_d,
    stride_means_b, stride_means_k, stride_means_d,
    stride_xsq_b, stride_xsq_n,
    stride_meanssq_b, stride_meanssq_k,
    stride_var_b, stride_var_k,
    stride_logw_b, stride_logw_k,
    stride_lognorm_b, stride_lognorm_n,
    stride_pnk_b, stride_pnk_nb, stride_pnk_k,
    stride_psumx_b, stride_psumx_nb, stride_psumx_k, stride_psumx_d,
    stride_psumxsq_b, stride_psumxsq_nb, stride_psumxsq_k,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    b = tl.program_id(axis=2).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    n_mask = offs_n < N
    k_mask = offs_k < K
    d_mask = offs_d < D

    means_tile = tl.load(
        means_ptr
        + b * stride_means_b
        + offs_k[None, :] * stride_means_k
        + offs_d[:, None] * stride_means_d,
        mask=d_mask[:, None] & k_mask[None, :],
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
        other=-float("inf"),
    ).to(tl.float32)
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
    log_norm = tl.load(
        log_norm_ptr + b * stride_lognorm_b + offs_n * stride_lognorm_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    d_const = tl.full((1,), D, tl.float32)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    log_det_term = d_const * (log_2pi + tl.log(variances))
    cross = tl.dot(x_tile, means_tile, input_precision="tf32x3").to(tl.float32)
    dist = tl.maximum(x_sq[:, None] + means_sq[None, :] - 2.0 * cross, 0.0)
    logits = log_weights[None, :] - 0.5 * (
        dist / variances[None, :] + log_det_term[None, :]
    )
    resp = tl.exp(logits - log_norm[:, None])
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    acc_nk = tl.sum(resp, axis=0)
    acc_xsq = tl.sum(resp * x_sq[:, None], axis=0)
    acc_sum_x = tl.dot(tl.trans(resp), x_tile, input_precision="tf32x3").to(tl.float32)

    partial_nk_ptrs = (
        partial_nk_ptr
        + b * stride_pnk_b
        + pid_n * stride_pnk_nb
        + offs_k * stride_pnk_k
    )
    partial_sum_x_sq_ptrs = (
        partial_sum_x_sq_ptr
        + b * stride_psumxsq_b
        + pid_n * stride_psumxsq_nb
        + offs_k * stride_psumxsq_k
    )
    partial_sum_x_ptrs = (
        partial_sum_x_ptr
        + b * stride_psumx_b
        + pid_n * stride_psumx_nb
        + offs_k[:, None] * stride_psumx_k
        + offs_d[None, :] * stride_psumx_d
    )
    tl.store(partial_nk_ptrs, acc_nk, mask=k_mask)
    tl.store(partial_sum_x_sq_ptrs, acc_xsq, mask=k_mask)
    tl.store(partial_sum_x_ptrs, acc_sum_x, mask=k_mask[:, None] & d_mask[None, :])


@triton.jit
def _streaming_update_spherical_kernel(
    x_ptr,                  # *f16 / *f32 [B, N, D]
    means_ptr,              # *f16 / *f32 [B, K, D]
    x_sq_ptr,               # *f32        [B, N]
    means_sq_ptr,           # *f32        [B, K]
    variances_ptr,          # *f32        [B, K]
    log_weights_ptr,        # *f32        [B, K]
    log_norm_ptr,           # *f32        [B, N]
    nk_ptr,                 # *f32        [B, K]
    sum_x_ptr,              # *f32        [B, K, D]
    sum_x_sq_ptr,           # *f32        [B, K]
    stride_x_b, stride_x_n, stride_x_d,
    stride_means_b, stride_means_k, stride_means_d,
    stride_xsq_b, stride_xsq_n,
    stride_meanssq_b, stride_meanssq_k,
    stride_var_b, stride_var_k,
    stride_logw_b, stride_logw_k,
    stride_lognorm_b, stride_lognorm_n,
    stride_nk_b, stride_nk_k,
    stride_sumx_b, stride_sumx_k, stride_sumx_d,
    stride_sumxsq_b, stride_sumxsq_k,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    b = tl.program_id(axis=2).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    k_mask = offs_k < K
    d_mask = offs_d < D

    means_tile = tl.load(
        means_ptr
        + b * stride_means_b
        + offs_k[None, :] * stride_means_k
        + offs_d[:, None] * stride_means_d,
        mask=d_mask[:, None] & k_mask[None, :],
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
        other=-float("inf"),
    ).to(tl.float32)

    acc_nk = tl.zeros((BLOCK_K,), tl.float32)
    acc_xsq = tl.zeros((BLOCK_K,), tl.float32)
    acc_sum_x = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)

    d_const = tl.full((1,), D, tl.float32)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    log_det_term = d_const * (log_2pi + tl.log(variances))

    n_mask = offs_n < N
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
    log_norm = tl.load(
        log_norm_ptr + b * stride_lognorm_b + offs_n * stride_lognorm_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    cross = tl.dot(x_tile, means_tile, input_precision="tf32x3").to(tl.float32)
    dist = tl.maximum(x_sq[:, None] + means_sq[None, :] - 2.0 * cross, 0.0)
    logits = log_weights[None, :] - 0.5 * (
        dist / variances[None, :] + log_det_term[None, :]
    )
    resp = tl.exp(logits - log_norm[:, None])
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    acc_nk += tl.sum(resp, axis=0)
    acc_xsq += tl.sum(resp * x_sq[:, None], axis=0)
    acc_sum_x += tl.dot(tl.trans(resp), x_tile, input_precision="tf32x3").to(tl.float32)

    nk_ptrs = nk_ptr + b * stride_nk_b + offs_k * stride_nk_k
    sum_x_sq_ptrs = sum_x_sq_ptr + b * stride_sumxsq_b + offs_k * stride_sumxsq_k
    sum_x_ptrs = (
        sum_x_ptr
        + b * stride_sumx_b
        + offs_k[:, None] * stride_sumx_k
        + offs_d[None, :] * stride_sumx_d
    )
    tl.atomic_add(nk_ptrs, acc_nk, mask=k_mask)
    tl.atomic_add(sum_x_sq_ptrs, acc_xsq, mask=k_mask)
    tl.atomic_add(sum_x_ptrs, acc_sum_x, mask=k_mask[:, None] & d_mask[None, :])


@triton.jit
def _blocked_update_diag_kernel(
    x_ptr,                  # *f16 / *f32 [B, N, D]
    precision_ptr,          # *f32        [B, K, D]
    weighted_means_ptr,     # *f32        [B, K, D]
    mean_precision_mean_ptr,# *f32        [B, K]
    logdet_ptr,             # *f32        [B, K]
    log_weights_ptr,        # *f32        [B, K]
    log_norm_ptr,           # *f32        [B, N]
    partial_nk_ptr,         # *f32        [B, NB, K]
    partial_sum_x_ptr,      # *f32        [B, NB, K, D]
    partial_sum_x_sq_ptr,   # *f32        [B, NB, K, D]
    stride_x_b, stride_x_n, stride_x_d,
    stride_precision_b, stride_precision_k, stride_precision_d,
    stride_wmeans_b, stride_wmeans_k, stride_wmeans_d,
    stride_mpm_b, stride_mpm_k,
    stride_logdet_b, stride_logdet_k,
    stride_logw_b, stride_logw_k,
    stride_lognorm_b, stride_lognorm_n,
    stride_pnk_b, stride_pnk_nb, stride_pnk_k,
    stride_psumx_b, stride_psumx_nb, stride_psumx_k, stride_psumx_d,
    stride_psumxsq_b, stride_psumxsq_nb, stride_psumxsq_k, stride_psumxsq_d,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    b = tl.program_id(axis=2).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    n_mask = offs_n < N
    k_mask = offs_k < K
    d_mask = offs_d < D

    x_tile = tl.load(
        x_ptr
        + b * stride_x_b
        + offs_n[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    x_tile_f = x_tile.to(tl.float32)
    x_sq_tile = x_tile_f * x_tile_f
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
    log_norm = tl.load(
        log_norm_ptr + b * stride_lognorm_b + offs_n * stride_lognorm_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    d_log_2pi = tl.full((1,), D, tl.float32) * log_2pi
    x_precision_x = tl.dot(x_sq_tile, precision_tile, input_precision="tf32x3").to(tl.float32)
    cross = tl.dot(x_tile_f, weighted_means_tile, input_precision="tf32x3").to(tl.float32)
    quad = tl.maximum(x_precision_x - 2.0 * cross + mean_precision_mean[None, :], 0.0)
    logits = log_weights[None, :] - 0.5 * (quad + d_log_2pi + logdet[None, :])
    resp = tl.exp(logits - log_norm[:, None])
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    acc_nk = tl.sum(resp, axis=0)
    acc_sum_x = tl.dot(tl.trans(resp), x_tile_f, input_precision="tf32x3").to(tl.float32)
    acc_sum_x_sq = tl.dot(tl.trans(resp), x_sq_tile, input_precision="tf32x3").to(tl.float32)

    partial_nk_ptrs = (
        partial_nk_ptr
        + b * stride_pnk_b
        + pid_n * stride_pnk_nb
        + offs_k * stride_pnk_k
    )
    partial_sum_x_ptrs = (
        partial_sum_x_ptr
        + b * stride_psumx_b
        + pid_n * stride_psumx_nb
        + offs_k[:, None] * stride_psumx_k
        + offs_d[None, :] * stride_psumx_d
    )
    partial_sum_x_sq_ptrs = (
        partial_sum_x_sq_ptr
        + b * stride_psumxsq_b
        + pid_n * stride_psumxsq_nb
        + offs_k[:, None] * stride_psumxsq_k
        + offs_d[None, :] * stride_psumxsq_d
    )
    tl.store(partial_nk_ptrs, acc_nk, mask=k_mask)
    tl.store(partial_sum_x_ptrs, acc_sum_x, mask=k_mask[:, None] & d_mask[None, :])
    tl.store(partial_sum_x_sq_ptrs, acc_sum_x_sq, mask=k_mask[:, None] & d_mask[None, :])


@triton.jit
def _blocked_update_tied_projected_kernel(
    x_proj_ptr,             # *f32        [B, N, D]
    x_orig_ptr,             # *f16 / *f32 [B, N, D]
    means_proj_ptr,         # *f32        [B, K, D]
    x_proj_sq_ptr,          # *f32        [B, N]
    means_proj_sq_ptr,      # *f32        [B, K]
    log_weights_ptr,        # *f32        [B, K]
    log_norm_ptr,           # *f32        [B, N]
    partial_nk_ptr,         # *f32        [B, NB, K]
    partial_sum_x_ptr,      # *f32        [B, NB, K, D]
    stride_xproj_b, stride_xproj_n, stride_xproj_d,
    stride_xorig_b, stride_xorig_n, stride_xorig_d,
    stride_means_b, stride_means_k, stride_means_d,
    stride_xsq_b, stride_xsq_n,
    stride_meanssq_b, stride_meanssq_k,
    stride_logw_b, stride_logw_k,
    stride_lognorm_b, stride_lognorm_n,
    stride_pnk_b, stride_pnk_nb, stride_pnk_k,
    stride_psumx_b, stride_psumx_nb, stride_psumx_k, stride_psumx_d,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    b = tl.program_id(axis=2).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    n_mask = offs_n < N
    k_mask = offs_k < K
    d_mask = offs_d < D

    x_proj = tl.load(
        x_proj_ptr
        + b * stride_xproj_b
        + offs_n[:, None] * stride_xproj_n
        + offs_d[None, :] * stride_xproj_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x_orig = tl.load(
        x_orig_ptr
        + b * stride_xorig_b
        + offs_n[:, None] * stride_xorig_n
        + offs_d[None, :] * stride_xorig_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    means_proj = tl.load(
        means_proj_ptr
        + b * stride_means_b
        + offs_k[None, :] * stride_means_k
        + offs_d[:, None] * stride_means_d,
        mask=k_mask[None, :] & d_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    x_sq = tl.load(
        x_proj_sq_ptr + b * stride_xsq_b + offs_n * stride_xsq_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    means_sq = tl.load(
        means_proj_sq_ptr + b * stride_meanssq_b + offs_k * stride_meanssq_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    log_weights = tl.load(
        log_weights_ptr + b * stride_logw_b + offs_k * stride_logw_k,
        mask=k_mask,
        other=-3.4e38,
    ).to(tl.float32)
    log_norm = tl.load(
        log_norm_ptr + b * stride_lognorm_b + offs_n * stride_lognorm_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    cross = tl.dot(x_proj, means_proj, input_precision="tf32x3").to(tl.float32)
    dist = tl.maximum(x_sq[:, None] + means_sq[None, :] - 2.0 * cross, 0.0)
    logits = log_weights[None, :] - 0.5 * (dist + tl.full((1,), D, tl.float32) * log_2pi)
    resp = tl.exp(logits - log_norm[:, None])
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    acc_nk = tl.sum(resp, axis=0)
    acc_sum_x = tl.dot(tl.trans(resp), x_orig, input_precision="tf32x3").to(tl.float32)

    partial_nk_ptrs = (
        partial_nk_ptr
        + b * stride_pnk_b
        + pid_n * stride_pnk_nb
        + offs_k * stride_pnk_k
    )
    partial_sum_x_ptrs = (
        partial_sum_x_ptr
        + b * stride_psumx_b
        + pid_n * stride_psumx_nb
        + offs_k[:, None] * stride_psumx_k
        + offs_d[None, :] * stride_psumx_d
    )
    tl.store(partial_nk_ptrs, acc_nk, mask=k_mask)
    tl.store(partial_sum_x_ptrs, acc_sum_x, mask=k_mask[:, None] & d_mask[None, :])


@triton.jit
def _blocked_update_full_kernel(
    x_ptr,                     # *f16 / *f32 [B, N, D]
    precision_ptr,             # *f32        [B, K, D, D]
    precision_means_ptr,       # *f32        [B, K, D]
    mean_precision_mean_ptr,   # *f32        [B, K]
    logdet_ptr,                # *f32        [B, K]
    log_weights_ptr,           # *f32        [B, K]
    log_norm_ptr,              # *f32        [B, N]
    partial_nk_ptr,            # *f32        [B, NB, K]
    partial_sum_x_ptr,         # *f32        [B, NB, K, D]
    partial_sum_xx_ptr,        # *f32        [B, NB, K, D, D]
    stride_x_b, stride_x_n, stride_x_d,
    stride_precision_b, stride_precision_k, stride_precision_d0, stride_precision_d1,
    stride_pm_b, stride_pm_k, stride_pm_d,
    stride_mpm_b, stride_mpm_k,
    stride_logdet_b, stride_logdet_k,
    stride_logw_b, stride_logw_k,
    stride_lognorm_b, stride_lognorm_n,
    stride_pnk_b, stride_pnk_nb, stride_pnk_k,
    stride_psumx_b, stride_psumx_nb, stride_psumx_k, stride_psumx_d,
    stride_psumxx_b, stride_psumxx_nb, stride_psumxx_k, stride_psumxx_d0, stride_psumxx_d1,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    b = tl.program_id(axis=2).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    n_mask = offs_n < N
    k_mask = offs_k < K
    d_mask = offs_d < D

    x_tile = tl.load(
        x_ptr
        + b * stride_x_b
        + offs_n[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    precision_means = tl.load(
        precision_means_ptr
        + b * stride_pm_b
        + offs_k[None, :] * stride_pm_k
        + offs_d[:, None] * stride_pm_d,
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
    log_norm = tl.load(
        log_norm_ptr + b * stride_lognorm_b + offs_n * stride_lognorm_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    x_precision_x = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    for row in tl.static_range(0, BLOCK_D):
        if row < D:
            x_row = tl.load(
                x_ptr
                + b * stride_x_b
                + offs_n * stride_x_n
                + row * stride_x_d,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            x_precision_row = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
            for col in tl.static_range(0, BLOCK_D):
                if col < D:
                    x_col = tl.load(
                        x_ptr
                        + b * stride_x_b
                        + offs_n * stride_x_n
                        + col * stride_x_d,
                        mask=n_mask,
                        other=0.0,
                    ).to(tl.float32)
                    precision_col = tl.load(
                        precision_ptr
                        + b * stride_precision_b
                        + offs_k * stride_precision_k
                        + row * stride_precision_d0
                        + col * stride_precision_d1,
                        mask=k_mask,
                        other=0.0,
                    ).to(tl.float32)
                    x_precision_row += x_col[:, None] * precision_col[None, :]
            x_precision_x += x_row[:, None] * x_precision_row

    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    cross = tl.dot(x_tile, precision_means, input_precision="tf32x3").to(tl.float32)
    quad = tl.maximum(x_precision_x - 2.0 * cross + mean_precision_mean[None, :], 0.0)
    logits = log_weights[None, :] - 0.5 * (
        quad + tl.full((1,), D, tl.float32) * log_2pi + logdet[None, :]
    )
    resp = tl.exp(logits - log_norm[:, None])
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    acc_nk = tl.sum(resp, axis=0)
    acc_sum_x = tl.dot(tl.trans(resp), x_tile, input_precision="tf32x3").to(tl.float32)

    partial_nk_ptrs = (
        partial_nk_ptr
        + b * stride_pnk_b
        + pid_n * stride_pnk_nb
        + offs_k * stride_pnk_k
    )
    partial_sum_x_ptrs = (
        partial_sum_x_ptr
        + b * stride_psumx_b
        + pid_n * stride_psumx_nb
        + offs_k[:, None] * stride_psumx_k
        + offs_d[None, :] * stride_psumx_d
    )
    tl.store(partial_nk_ptrs, acc_nk, mask=k_mask)
    tl.store(partial_sum_x_ptrs, acc_sum_x, mask=k_mask[:, None] & d_mask[None, :])

    for row in tl.static_range(0, BLOCK_D):
        if row < D:
            x_row = tl.load(
                x_ptr
                + b * stride_x_b
                + offs_n * stride_x_n
                + row * stride_x_d,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            for col in tl.static_range(0, BLOCK_D):
                if col < D:
                    x_col = tl.load(
                        x_ptr
                        + b * stride_x_b
                        + offs_n * stride_x_n
                        + col * stride_x_d,
                        mask=n_mask,
                        other=0.0,
                    ).to(tl.float32)
                    acc_xx = tl.sum(resp * (x_row * x_col)[:, None], axis=0)
                    partial_sum_xx_ptrs = (
                        partial_sum_xx_ptr
                        + b * stride_psumxx_b
                        + pid_n * stride_psumxx_nb
                        + offs_k * stride_psumxx_k
                        + row * stride_psumxx_d0
                        + col * stride_psumxx_d1
                    )
                    tl.store(partial_sum_xx_ptrs, acc_xx, mask=k_mask)


@triton.jit
def _weighted_sum_x_block_kernel(
    x_ptr,                  # *f16 / *f32 [B, N, D]
    resp_ptr,               # *f32        [B, N, K]
    sum_x_ptr,              # *f32        [B, K, D]
    stride_x_b, stride_x_n, stride_x_d,
    stride_resp_b, stride_resp_n, stride_resp_k,
    stride_sumx_b, stride_sumx_k, stride_sumx_d,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    b = tl.program_id(axis=2).to(tl.int64)

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
    k_mask = offs_k < K
    d_mask = offs_d < D

    acc = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + offs_n
        n_mask = n_offsets < N
        resp = tl.load(
            resp_ptr
            + b * stride_resp_b
            + n_offsets[:, None] * stride_resp_n
            + offs_k[None, :] * stride_resp_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr
            + b * stride_x_b
            + n_offsets[:, None] * stride_x_n
            + offs_d[None, :] * stride_x_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(tl.trans(resp), x, input_precision="tf32x3").to(tl.float32)

    out_ptrs = (
        sum_x_ptr
        + b * stride_sumx_b
        + offs_k[:, None] * stride_sumx_k
        + offs_d[None, :] * stride_sumx_d
    )
    old = tl.load(out_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
    tl.store(out_ptrs, old + acc, mask=k_mask[:, None] & d_mask[None, :])


@triton.jit
def _weighted_nk_xsq_block_kernel(
    resp_ptr,               # *f32 [B, N, K]
    x_sq_ptr,               # *f32 [B, N]
    nk_ptr,                 # *f32 [B, K]
    sum_x_sq_ptr,           # *f32 [B, K]
    stride_resp_b, stride_resp_n, stride_resp_k,
    stride_xsq_b, stride_xsq_n,
    stride_nk_b, stride_nk_k,
    stride_sumxsq_b, stride_sumxsq_k,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    b = tl.program_id(axis=1).to(tl.int64)

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    k_mask = offs_k < K

    acc_nk = tl.zeros((BLOCK_K,), tl.float32)
    acc_xsq = tl.zeros((BLOCK_K,), tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + offs_n
        n_mask = n_offsets < N
        resp = tl.load(
            resp_ptr
            + b * stride_resp_b
            + n_offsets[:, None] * stride_resp_n
            + offs_k[None, :] * stride_resp_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x_sq = tl.load(
            x_sq_ptr + b * stride_xsq_b + n_offsets * stride_xsq_n,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        acc_nk += tl.sum(resp, axis=0)
        acc_xsq += tl.sum(resp * x_sq[:, None], axis=0)

    nk_ptrs = nk_ptr + b * stride_nk_b + offs_k * stride_nk_k
    sum_x_sq_ptrs = sum_x_sq_ptr + b * stride_sumxsq_b + offs_k * stride_sumxsq_k
    old_nk = tl.load(nk_ptrs, mask=k_mask, other=0.0)
    old_xsq = tl.load(sum_x_sq_ptrs, mask=k_mask, other=0.0)
    tl.store(nk_ptrs, old_nk + acc_nk, mask=k_mask)
    tl.store(sum_x_sq_ptrs, old_xsq + acc_xsq, mask=k_mask)


def triton_weighted_update_spherical(
    x: torch.Tensor,
    responsibilities: torch.Tensor,
    *,
    nk: torch.Tensor | None = None,
    sum_x: torch.Tensor | None = None,
    sum_x_sq: torch.Tensor | None = None,
    x_sq: torch.Tensor | None = None,
    BLOCK_N: int = 128,
    BLOCK_D: int = 64,
    BLOCK_K: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate spherical-GMM sufficient statistics on CUDA with Triton.

    Parameters
    ----------
    x : Tensor [B, N, D]
        Input features.
    responsibilities : Tensor [B, N, K]
        Soft assignments for the current chunk or full batch.
    x_sq : optional Tensor [B, N]
        Precomputed squared norms. Passing this avoids recomputing them when the E-step
        already has them.
    nk, sum_x, sum_x_sq : optional output buffers
        Preallocated accumulation buffers. If provided, they are updated in-place.
    """
    assert x.is_cuda and responsibilities.is_cuda, "Inputs must be on CUDA device"
    B, N, D = x.shape
    Br, Nr, K = responsibilities.shape
    assert (Br, Nr) == (B, N), "responsibilities shape mismatch"

    if nk is None:
        nk = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    else:
        assert nk.shape == (B, K)
        assert nk.dtype == torch.float32

    if sum_x is None:
        sum_x = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    else:
        assert sum_x.shape == (B, K, D)
        assert sum_x.dtype == torch.float32

    if sum_x_sq is None:
        sum_x_sq = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    else:
        assert sum_x_sq.shape == (B, K)
        assert sum_x_sq.dtype == torch.float32

    if x_sq is None:
        x_sq = (x.to(torch.float32) ** 2).sum(dim=-1)
    else:
        assert x_sq.shape == (B, N)
        assert x_sq.is_cuda

    grid_stats = (triton.cdiv(K, BLOCK_K), B)
    _weighted_nk_xsq_block_kernel[grid_stats](
        responsibilities.to(torch.float32),
        x_sq.to(torch.float32),
        nk,
        sum_x_sq,
        responsibilities.stride(0), responsibilities.stride(1), responsibilities.stride(2),
        x_sq.stride(0), x_sq.stride(1),
        nk.stride(0), nk.stride(1),
        sum_x_sq.stride(0), sum_x_sq.stride(1),
        B, N, K,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
num_warps=4,
        num_stages=1,
    )

    grid_sum_x = (triton.cdiv(K, BLOCK_K), triton.cdiv(D, BLOCK_D), B)
    _weighted_sum_x_block_kernel[grid_sum_x](
        x,
        responsibilities.to(torch.float32),
        sum_x,
        x.stride(0), x.stride(1), x.stride(2),
        responsibilities.stride(0), responsibilities.stride(1), responsibilities.stride(2),
        sum_x.stride(0), sum_x.stride(1), sum_x.stride(2),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
num_warps=4,
        num_stages=1,
    )
    return nk, sum_x, sum_x_sq


def triton_blocked_update_spherical(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    log_norm: torch.Tensor,
    *,
    x_sq: torch.Tensor | None = None,
    means_sq: torch.Tensor | None = None,
    log_weights: torch.Tensor | None = None,
    partial_nk: torch.Tensor | None = None,
    partial_sum_x: torch.Tensor | None = None,
    partial_sum_x_sq: torch.Tensor | None = None,
    BLOCK_N: int = 128,
    BLOCK_D: int = 64,
    BLOCK_K: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-stage spherical-GMM update that avoids global atomics."""
    assert x.is_cuda and means.is_cuda and variances.is_cuda and weights.is_cuda and log_norm.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bm, K, Dm = means.shape
    assert (Bm, Dm) == (B, D), "means shape mismatch"
    assert variances.shape == (B, K), "variances shape mismatch"
    assert weights.shape == (B, K), "weights shape mismatch"
    assert log_norm.shape == (B, N), "log_norm shape mismatch"
    if D > BLOCK_D:
        raise ValueError(f"blocked spherical update requires D <= BLOCK_D; got D={D}, BLOCK_D={BLOCK_D}")

    if x_sq is None:
        x_sq = (x.to(torch.float32) ** 2).sum(dim=-1)
    if means_sq is None:
        means_sq = (means.to(torch.float32) ** 2).sum(dim=-1)
    if log_weights is None:
        log_weights = torch.log(weights.to(torch.float32))

    n_blocks = triton.cdiv(N, BLOCK_N)
    if partial_nk is None:
        partial_nk = torch.empty((B, n_blocks, K), device=x.device, dtype=torch.float32)
    else:
        assert partial_nk.shape[0] == B and partial_nk.shape[1] >= n_blocks and partial_nk.shape[2] >= K
        assert partial_nk.dtype == torch.float32 and partial_nk.is_cuda
        partial_nk = partial_nk[:, :n_blocks, :K]
    if partial_sum_x is None:
        partial_sum_x = torch.empty((B, n_blocks, K, D), device=x.device, dtype=torch.float32)
    else:
        assert partial_sum_x.shape[0] == B and partial_sum_x.shape[1] >= n_blocks
        assert partial_sum_x.shape[2] >= K and partial_sum_x.shape[3] >= D
        assert partial_sum_x.dtype == torch.float32 and partial_sum_x.is_cuda
        partial_sum_x = partial_sum_x[:, :n_blocks, :K, :D]
    if partial_sum_x_sq is None:
        partial_sum_x_sq = torch.empty((B, n_blocks, K), device=x.device, dtype=torch.float32)
    else:
        assert partial_sum_x_sq.shape[0] == B and partial_sum_x_sq.shape[1] >= n_blocks and partial_sum_x_sq.shape[2] >= K
        assert partial_sum_x_sq.dtype == torch.float32 and partial_sum_x_sq.is_cuda
        partial_sum_x_sq = partial_sum_x_sq[:, :n_blocks, :K]

    grid = (triton.cdiv(K, BLOCK_K), n_blocks, B)
    _blocked_update_spherical_kernel[grid](
        x,
        means,
        x_sq.to(torch.float32),
        means_sq.to(torch.float32),
        variances.to(torch.float32),
        log_weights.to(torch.float32),
        log_norm.to(torch.float32),
        partial_nk,
        partial_sum_x,
        partial_sum_x_sq,
        x.stride(0), x.stride(1), x.stride(2),
        means.stride(0), means.stride(1), means.stride(2),
        x_sq.stride(0), x_sq.stride(1),
        means_sq.stride(0), means_sq.stride(1),
        variances.stride(0), variances.stride(1),
        log_weights.stride(0), log_weights.stride(1),
        log_norm.stride(0), log_norm.stride(1),
        partial_nk.stride(0), partial_nk.stride(1), partial_nk.stride(2),
        partial_sum_x.stride(0), partial_sum_x.stride(1), partial_sum_x.stride(2), partial_sum_x.stride(3),
        partial_sum_x_sq.stride(0), partial_sum_x_sq.stride(1), partial_sum_x_sq.stride(2),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
num_warps=4,
        num_stages=1,
    )
    return partial_nk.sum(dim=1), partial_sum_x.sum(dim=1), partial_sum_x_sq.sum(dim=1)


def triton_blocked_update_diag(
    x: torch.Tensor,
    precision: torch.Tensor,
    weighted_means: torch.Tensor,
    mean_precision_mean: torch.Tensor,
    logdet: torch.Tensor,
    log_weights: torch.Tensor,
    log_norm: torch.Tensor,
    *,
    partial_nk: torch.Tensor | None = None,
    partial_sum_x: torch.Tensor | None = None,
    partial_sum_x_sq: torch.Tensor | None = None,
    BLOCK_N: int = 64,
    BLOCK_D: int = 64,
    BLOCK_K: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-stage diagonal-GMM update that avoids materialized responsibilities."""
    assert x.is_cuda and precision.is_cuda and weighted_means.is_cuda and log_norm.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bp, K, Dp = precision.shape
    assert (Bp, Dp) == (B, D), "precision shape mismatch"
    assert weighted_means.shape == (B, K, D), "weighted_means shape mismatch"
    assert mean_precision_mean.shape == (B, K), "mean_precision_mean shape mismatch"
    assert logdet.shape == (B, K), "logdet shape mismatch"
    assert log_weights.shape == (B, K), "log_weights shape mismatch"
    assert log_norm.shape == (B, N), "log_norm shape mismatch"
    if D > BLOCK_D:
        raise ValueError(f"blocked diagonal update requires D <= BLOCK_D; got D={D}, BLOCK_D={BLOCK_D}")

    n_blocks = triton.cdiv(N, BLOCK_N)
    if partial_nk is None:
        partial_nk = torch.empty((B, n_blocks, K), device=x.device, dtype=torch.float32)
    else:
        assert partial_nk.shape[0] == B and partial_nk.shape[1] >= n_blocks and partial_nk.shape[2] >= K
        assert partial_nk.dtype == torch.float32 and partial_nk.is_cuda
        partial_nk = partial_nk[:, :n_blocks, :K]
    if partial_sum_x is None:
        partial_sum_x = torch.empty((B, n_blocks, K, D), device=x.device, dtype=torch.float32)
    else:
        assert partial_sum_x.shape[0] == B and partial_sum_x.shape[1] >= n_blocks
        assert partial_sum_x.shape[2] >= K and partial_sum_x.shape[3] >= D
        assert partial_sum_x.dtype == torch.float32 and partial_sum_x.is_cuda
        partial_sum_x = partial_sum_x[:, :n_blocks, :K, :D]
    if partial_sum_x_sq is None:
        partial_sum_x_sq = torch.empty((B, n_blocks, K, D), device=x.device, dtype=torch.float32)
    else:
        assert partial_sum_x_sq.shape[0] == B and partial_sum_x_sq.shape[1] >= n_blocks
        assert partial_sum_x_sq.shape[2] >= K and partial_sum_x_sq.shape[3] >= D
        assert partial_sum_x_sq.dtype == torch.float32 and partial_sum_x_sq.is_cuda
        partial_sum_x_sq = partial_sum_x_sq[:, :n_blocks, :K, :D]

    grid = (triton.cdiv(K, BLOCK_K), n_blocks, B)
    _blocked_update_diag_kernel[grid](
        x,
        precision.to(torch.float32),
        weighted_means.to(torch.float32),
        mean_precision_mean.to(torch.float32),
        logdet.to(torch.float32),
        log_weights.to(torch.float32),
        log_norm.to(torch.float32),
        partial_nk,
        partial_sum_x,
        partial_sum_x_sq,
        x.stride(0), x.stride(1), x.stride(2),
        precision.stride(0), precision.stride(1), precision.stride(2),
        weighted_means.stride(0), weighted_means.stride(1), weighted_means.stride(2),
        mean_precision_mean.stride(0), mean_precision_mean.stride(1),
        logdet.stride(0), logdet.stride(1),
        log_weights.stride(0), log_weights.stride(1),
        log_norm.stride(0), log_norm.stride(1),
        partial_nk.stride(0), partial_nk.stride(1), partial_nk.stride(2),
        partial_sum_x.stride(0), partial_sum_x.stride(1), partial_sum_x.stride(2), partial_sum_x.stride(3),
        partial_sum_x_sq.stride(0), partial_sum_x_sq.stride(1), partial_sum_x_sq.stride(2), partial_sum_x_sq.stride(3),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
num_warps=4,
        num_stages=1,
    )
    return partial_nk.sum(dim=1), partial_sum_x.sum(dim=1), partial_sum_x_sq.sum(dim=1)


def triton_blocked_update_tied_projected(
    x_projected: torch.Tensor,
    x_original: torch.Tensor,
    means_projected: torch.Tensor,
    log_weights: torch.Tensor,
    log_norm: torch.Tensor,
    *,
    x_projected_sq: torch.Tensor | None = None,
    means_projected_sq: torch.Tensor | None = None,
    partial_nk: torch.Tensor | None = None,
    partial_sum_x: torch.Tensor | None = None,
    BLOCK_N: int = 64,
    BLOCK_D: int = 64,
    BLOCK_K: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tied-covariance update from projected responsibilities.

    Responsibilities are computed in the shared-whitened space
    (`x_projected`, `means_projected`), while sufficient statistics are
    accumulated in the original feature space (`x_original`).
    """
    assert x_projected.is_cuda and x_original.is_cuda and means_projected.is_cuda, "All tensors must be on CUDA"
    B, N, D = x_projected.shape
    assert x_original.shape == (B, N, D), "x_original shape mismatch"
    Bm, K, Dm = means_projected.shape
    assert (Bm, Dm) == (B, D), "means_projected shape mismatch"
    assert log_weights.shape == (B, K), "log_weights shape mismatch"
    assert log_norm.shape == (B, N), "log_norm shape mismatch"
    if D > BLOCK_D:
        raise ValueError(f"projected tied update requires D <= BLOCK_D; got D={D}, BLOCK_D={BLOCK_D}")
    if x_projected_sq is None:
        x_projected_sq = x_projected.to(torch.float32).square().sum(dim=-1)
    if means_projected_sq is None:
        means_projected_sq = means_projected.to(torch.float32).square().sum(dim=-1)

    n_blocks = triton.cdiv(N, BLOCK_N)
    if partial_nk is None:
        partial_nk = torch.empty((B, n_blocks, K), device=x_projected.device, dtype=torch.float32)
    else:
        assert partial_nk.shape[0] == B and partial_nk.shape[1] >= n_blocks and partial_nk.shape[2] >= K
        assert partial_nk.dtype == torch.float32 and partial_nk.is_cuda
        partial_nk = partial_nk[:, :n_blocks, :K]
    if partial_sum_x is None:
        partial_sum_x = torch.empty((B, n_blocks, K, D), device=x_projected.device, dtype=torch.float32)
    else:
        assert partial_sum_x.shape[0] == B and partial_sum_x.shape[1] >= n_blocks
        assert partial_sum_x.shape[2] >= K and partial_sum_x.shape[3] >= D
        assert partial_sum_x.dtype == torch.float32 and partial_sum_x.is_cuda
        partial_sum_x = partial_sum_x[:, :n_blocks, :K, :D]

    grid = (triton.cdiv(K, BLOCK_K), n_blocks, B)
    _blocked_update_tied_projected_kernel[grid](
        x_projected.to(torch.float32),
        x_original,
        means_projected.to(torch.float32),
        x_projected_sq.to(torch.float32),
        means_projected_sq.to(torch.float32),
        log_weights.to(torch.float32),
        log_norm.to(torch.float32),
        partial_nk,
        partial_sum_x,
        x_projected.stride(0), x_projected.stride(1), x_projected.stride(2),
        x_original.stride(0), x_original.stride(1), x_original.stride(2),
        means_projected.stride(0), means_projected.stride(1), means_projected.stride(2),
        x_projected_sq.stride(0), x_projected_sq.stride(1),
        means_projected_sq.stride(0), means_projected_sq.stride(1),
        log_weights.stride(0), log_weights.stride(1),
        log_norm.stride(0), log_norm.stride(1),
        partial_nk.stride(0), partial_nk.stride(1), partial_nk.stride(2),
        partial_sum_x.stride(0), partial_sum_x.stride(1), partial_sum_x.stride(2), partial_sum_x.stride(3),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
num_warps=4,
        num_stages=1,
    )
    return partial_nk.sum(dim=1), partial_sum_x.sum(dim=1)


def triton_blocked_update_full(
    x: torch.Tensor,
    precision: torch.Tensor,
    precision_means: torch.Tensor,
    mean_precision_mean: torch.Tensor,
    logdet: torch.Tensor,
    log_weights: torch.Tensor,
    log_norm: torch.Tensor,
    *,
    partial_nk: torch.Tensor | None = None,
    partial_sum_x: torch.Tensor | None = None,
    partial_sum_xx: torch.Tensor | None = None,
    BLOCK_N: int = 32,
    BLOCK_D: int = 16,
    BLOCK_K: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-stage full-covariance update for small-D full GMMs.

    Full covariance is quadratic in feature dimension, so this kernel is
    intentionally constrained to small `D` and avoids materializing an
    `N x K` responsibility tensor.
    """
    assert x.is_cuda and precision.is_cuda and precision_means.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bp, K, D0, D1 = precision.shape
    assert (Bp, D0, D1) == (B, D, D), "precision shape mismatch"
    assert precision_means.shape == (B, K, D), "precision_means shape mismatch"
    assert mean_precision_mean.shape == (B, K), "mean_precision_mean shape mismatch"
    assert logdet.shape == (B, K), "logdet shape mismatch"
    assert log_weights.shape == (B, K), "log_weights shape mismatch"
    assert log_norm.shape == (B, N), "log_norm shape mismatch"
    if D > 8:
        raise ValueError(f"blocked full update supports D <= 8, got D={D}")
    if D > BLOCK_D:
        raise ValueError(f"blocked full update requires D <= BLOCK_D; got D={D}, BLOCK_D={BLOCK_D}")

    n_blocks = triton.cdiv(N, BLOCK_N)
    if partial_nk is None:
        partial_nk = torch.empty((B, n_blocks, K), device=x.device, dtype=torch.float32)
    else:
        assert partial_nk.shape[0] == B and partial_nk.shape[1] >= n_blocks and partial_nk.shape[2] >= K
        assert partial_nk.dtype == torch.float32 and partial_nk.is_cuda
        partial_nk = partial_nk[:, :n_blocks, :K]
    if partial_sum_x is None:
        partial_sum_x = torch.empty((B, n_blocks, K, D), device=x.device, dtype=torch.float32)
    else:
        assert partial_sum_x.shape[0] == B and partial_sum_x.shape[1] >= n_blocks
        assert partial_sum_x.shape[2] >= K and partial_sum_x.shape[3] >= D
        assert partial_sum_x.dtype == torch.float32 and partial_sum_x.is_cuda
        partial_sum_x = partial_sum_x[:, :n_blocks, :K, :D]
    if partial_sum_xx is None:
        partial_sum_xx = torch.empty((B, n_blocks, K, D, D), device=x.device, dtype=torch.float32)
    else:
        assert partial_sum_xx.shape[0] == B and partial_sum_xx.shape[1] >= n_blocks
        assert partial_sum_xx.shape[2] >= K and partial_sum_xx.shape[3] >= D and partial_sum_xx.shape[4] >= D
        assert partial_sum_xx.dtype == torch.float32 and partial_sum_xx.is_cuda
        partial_sum_xx = partial_sum_xx[:, :n_blocks, :K, :D, :D]

    grid = (triton.cdiv(K, BLOCK_K), n_blocks, B)
    _blocked_update_full_kernel[grid](
        x,
        precision.to(torch.float32),
        precision_means.to(torch.float32),
        mean_precision_mean.to(torch.float32),
        logdet.to(torch.float32),
        log_weights.to(torch.float32),
        log_norm.to(torch.float32),
        partial_nk,
        partial_sum_x,
        partial_sum_xx,
        x.stride(0), x.stride(1), x.stride(2),
        precision.stride(0), precision.stride(1), precision.stride(2), precision.stride(3),
        precision_means.stride(0), precision_means.stride(1), precision_means.stride(2),
        mean_precision_mean.stride(0), mean_precision_mean.stride(1),
        logdet.stride(0), logdet.stride(1),
        log_weights.stride(0), log_weights.stride(1),
        log_norm.stride(0), log_norm.stride(1),
        partial_nk.stride(0), partial_nk.stride(1), partial_nk.stride(2),
        partial_sum_x.stride(0), partial_sum_x.stride(1), partial_sum_x.stride(2), partial_sum_x.stride(3),
        partial_sum_xx.stride(0), partial_sum_xx.stride(1), partial_sum_xx.stride(2),
        partial_sum_xx.stride(3), partial_sum_xx.stride(4),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
num_warps=4,
        num_stages=1,
    )
    return partial_nk.sum(dim=1), partial_sum_x.sum(dim=1), partial_sum_xx.sum(dim=1)


def triton_streaming_update_spherical(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    log_norm: torch.Tensor,
    *,
    nk: torch.Tensor | None = None,
    sum_x: torch.Tensor | None = None,
    sum_x_sq: torch.Tensor | None = None,
    x_sq: torch.Tensor | None = None,
    means_sq: torch.Tensor | None = None,
    log_weights: torch.Tensor | None = None,
    BLOCK_N: int = 128,
    BLOCK_D: int = 64,
    BLOCK_K: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused streaming E/M update for spherical GMM sufficient statistics.

    This recomputes logits from model parameters and `log_norm`, then reduces
    directly into `Nk`, `sum_x`, and `sum_x_sq` without materializing a
    responsibility tensor.
    """
    assert x.is_cuda and means.is_cuda and variances.is_cuda and weights.is_cuda and log_norm.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bm, K, Dm = means.shape
    assert (Bm, Dm) == (B, D), "means shape mismatch"
    assert variances.shape == (B, K), "variances shape mismatch"
    assert weights.shape == (B, K), "weights shape mismatch"
    assert log_norm.shape == (B, N), "log_norm shape mismatch"
    if D > BLOCK_D:
        raise ValueError(
            f"triton_streaming_update_spherical requires D <= BLOCK_D; got D={D}, BLOCK_D={BLOCK_D}"
        )

    if x_sq is None:
        x_sq = (x.to(torch.float32) ** 2).sum(dim=-1)
    else:
        assert x_sq.shape == (B, N)
        assert x_sq.is_cuda
    if means_sq is None:
        means_sq = (means.to(torch.float32) ** 2).sum(dim=-1)
    else:
        assert means_sq.shape == (B, K)
        assert means_sq.is_cuda
    if log_weights is None:
        log_weights = torch.log(weights.to(torch.float32))
    else:
        assert log_weights.shape == (B, K)
        assert log_weights.is_cuda

    if nk is None:
        nk = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    else:
        assert nk.shape == (B, K)
        assert nk.dtype == torch.float32
        assert nk.is_cuda
    if sum_x is None:
        sum_x = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    else:
        assert sum_x.shape == (B, K, D)
        assert sum_x.dtype == torch.float32
        assert sum_x.is_cuda
    if sum_x_sq is None:
        sum_x_sq = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    else:
        assert sum_x_sq.shape == (B, K)
        assert sum_x_sq.dtype == torch.float32
        assert sum_x_sq.is_cuda

    grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N), B)
    _streaming_update_spherical_kernel[grid](
        x,
        means,
        x_sq.to(torch.float32),
        means_sq.to(torch.float32),
        variances.to(torch.float32),
        log_weights.to(torch.float32),
        log_norm.to(torch.float32),
        nk,
        sum_x,
        sum_x_sq,
        x.stride(0), x.stride(1), x.stride(2),
        means.stride(0), means.stride(1), means.stride(2),
        x_sq.stride(0), x_sq.stride(1),
        means_sq.stride(0), means_sq.stride(1),
        variances.stride(0), variances.stride(1),
        log_weights.stride(0), log_weights.stride(1),
        log_norm.stride(0), log_norm.stride(1),
        nk.stride(0), nk.stride(1),
        sum_x.stride(0), sum_x.stride(1), sum_x.stride(2),
        sum_x_sq.stride(0), sum_x_sq.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
num_warps=4,
        num_stages=1,
    )
    return nk, sum_x, sum_x_sq
