from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _full_logsumexp_kernel(
    x_ptr,
    precision_ptr,
    precision_means_ptr,
    mean_precision_mean_ptr,
    logdet_ptr,
    log_weights_ptr,
    out_ptr,
    sum_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_precision_b: tl.constexpr,
    stride_precision_k: tl.constexpr,
    stride_precision_d0: tl.constexpr,
    stride_precision_d1: tl.constexpr,
    stride_pm_b: tl.constexpr,
    stride_pm_k: tl.constexpr,
    stride_pm_d: tl.constexpr,
    stride_mpm_b: tl.constexpr,
    stride_mpm_k: tl.constexpr,
    stride_logdet_b: tl.constexpr,
    stride_logdet_k: tl.constexpr,
    stride_logw_b: tl.constexpr,
    stride_logw_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    HAS_SUM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    n_mask = offs_n < N
    d_mask = offs_d < D

    x_tile = tl.load(
        x_ptr
        + b * stride_x_b
        + offs_n[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    running_max = tl.full((BLOCK_N,), -3.4e38, tl.float32)
    exp_sums = tl.zeros((BLOCK_N,), tl.float32)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    d_log_2pi = tl.full((1,), D, tl.float32) * log_2pi

    for k_start in range(0, K, BLOCK_K):
        k_offsets = (k_start + offs_k).to(tl.int64)
        k_mask = k_offsets < K

        precision_means = tl.load(
            precision_means_ptr
            + b * stride_pm_b
            + k_offsets[None, :] * stride_pm_k
            + offs_d[:, None] * stride_pm_d,
            mask=k_mask[None, :] & d_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        mean_precision_mean = tl.load(
            mean_precision_mean_ptr + b * stride_mpm_b + k_offsets * stride_mpm_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        logdet = tl.load(
            logdet_ptr + b * stride_logdet_b + k_offsets * stride_logdet_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        log_weights = tl.load(
            log_weights_ptr + b * stride_logw_b + k_offsets * stride_logw_k,
            mask=k_mask,
            other=-3.4e38,
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
                            + k_offsets * stride_precision_k
                            + row * stride_precision_d0
                            + col * stride_precision_d1,
                            mask=k_mask,
                            other=0.0,
                        ).to(tl.float32)
                        x_precision_row += x_col[:, None] * precision_col[None, :]
                x_precision_x += x_row[:, None] * x_precision_row

        cross = tl.dot(x_tile, precision_means, input_precision="tf32x3").to(tl.float32)
        quad = tl.maximum(x_precision_x - 2.0 * cross + mean_precision_mean[None, :], 0.0)
        logits = log_weights[None, :] - 0.5 * (quad + d_log_2pi + logdet[None, :])
        logits = tl.where(k_mask[None, :], logits, -3.4e38)

        tile_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        exp_sums = exp_sums * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(logits - new_max[:, None]),
            axis=1,
        )
        running_max = new_max

    values = running_max + tl.log(exp_sums)
    tl.store(out_ptr + b * stride_out_b + offs_n * stride_out_n, values, mask=n_mask)
    if HAS_SUM:
        tl.atomic_add(sum_ptr, tl.sum(tl.where(n_mask, values, 0.0), axis=0))


def _full_config(n: int, k: int, d: int) -> dict[str, int]:
    return {"BLOCK_N": 128, "BLOCK_K": 16, "BLOCK_D": 16, "num_warps": 4, "num_stages": 1}


@triton.jit
def _full_assign_kernel(
    x_ptr,
    precision_ptr,
    precision_means_ptr,
    mean_precision_mean_ptr,
    logdet_ptr,
    log_weights_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_precision_b: tl.constexpr,
    stride_precision_k: tl.constexpr,
    stride_precision_d0: tl.constexpr,
    stride_precision_d1: tl.constexpr,
    stride_pm_b: tl.constexpr,
    stride_pm_k: tl.constexpr,
    stride_pm_d: tl.constexpr,
    stride_mpm_b: tl.constexpr,
    stride_mpm_k: tl.constexpr,
    stride_logdet_b: tl.constexpr,
    stride_logdet_k: tl.constexpr,
    stride_logw_b: tl.constexpr,
    stride_logw_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    n_mask = offs_n < N
    d_mask = offs_d < D

    x_tile = tl.load(
        x_ptr
        + b * stride_x_b
        + offs_n[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    best_logit = tl.full((BLOCK_N,), -3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    d_log_2pi = tl.full((1,), D, tl.float32) * log_2pi

    for k_start in range(0, K, BLOCK_K):
        k_offsets = (k_start + offs_k).to(tl.int64)
        k_mask = k_offsets < K

        precision_means = tl.load(
            precision_means_ptr
            + b * stride_pm_b
            + k_offsets[None, :] * stride_pm_k
            + offs_d[:, None] * stride_pm_d,
            mask=k_mask[None, :] & d_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        mean_precision_mean = tl.load(
            mean_precision_mean_ptr + b * stride_mpm_b + k_offsets * stride_mpm_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        logdet = tl.load(
            logdet_ptr + b * stride_logdet_b + k_offsets * stride_logdet_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        log_weights = tl.load(
            log_weights_ptr + b * stride_logw_b + k_offsets * stride_logw_k,
            mask=k_mask,
            other=-3.4e38,
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
                            + k_offsets * stride_precision_k
                            + row * stride_precision_d0
                            + col * stride_precision_d1,
                            mask=k_mask,
                            other=0.0,
                        ).to(tl.float32)
                        x_precision_row += x_col[:, None] * precision_col[None, :]
                x_precision_x += x_row[:, None] * x_precision_row

        cross = tl.dot(x_tile, precision_means, input_precision="tf32x3").to(tl.float32)
        quad = tl.maximum(x_precision_x - 2.0 * cross + mean_precision_mean[None, :], 0.0)
        logits = log_weights[None, :] - 0.5 * (quad + d_log_2pi + logdet[None, :])
        logits = tl.where(k_mask[None, :], logits, -3.4e38)

        tile_max = tl.max(logits, axis=1)
        tile_idx = tl.argmax(logits, axis=1)
        update = tile_max > best_logit
        best_logit = tl.where(update, tile_max, best_logit)
        best_idx = tl.where(update, k_start + tile_idx, best_idx)

    tl.store(out_ptr + b * stride_out_b + offs_n * stride_out_n, best_idx, mask=n_mask)


@triton.jit
def _full_resp_kernel(
    x_ptr,
    precision_ptr,
    precision_means_ptr,
    mean_precision_mean_ptr,
    logdet_ptr,
    log_weights_ptr,
    log_norm_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_precision_b: tl.constexpr,
    stride_precision_k: tl.constexpr,
    stride_precision_d0: tl.constexpr,
    stride_precision_d1: tl.constexpr,
    stride_pm_b: tl.constexpr,
    stride_pm_k: tl.constexpr,
    stride_pm_d: tl.constexpr,
    stride_mpm_b: tl.constexpr,
    stride_mpm_k: tl.constexpr,
    stride_logdet_b: tl.constexpr,
    stride_logdet_k: tl.constexpr,
    stride_logw_b: tl.constexpr,
    stride_logw_k: tl.constexpr,
    stride_lognorm_b: tl.constexpr,
    stride_lognorm_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    stride_out_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)

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

    out_ptrs = (
        out_ptr
        + b * stride_out_b
        + offs_n[:, None] * stride_out_n
        + offs_k[None, :] * stride_out_k
    )
    tl.store(out_ptrs, resp, mask=n_mask[:, None] & k_mask[None, :])


def full_logsumexp_triton(
    x: torch.Tensor,
    precision: torch.Tensor,
    precision_means: torch.Tensor,
    mean_precision_mean: torch.Tensor,
    logdet: torch.Tensor,
    log_weights: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    out_sum: torch.Tensor | None = None,
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    assert x.is_cuda and precision.is_cuda and precision_means.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bp, K, D0, D1 = precision.shape
    assert (Bp, D0, D1) == (B, D, D), "precision shape mismatch"
    assert precision_means.shape == (B, K, D), "precision_means shape mismatch"
    assert mean_precision_mean.shape == (B, K), "mean_precision_mean shape mismatch"
    assert logdet.shape == (B, K), "logdet shape mismatch"
    assert log_weights.shape == (B, K), "log_weights shape mismatch"
    if D > 16:
        raise ValueError(f"full_logsumexp_triton supports D <= 16, got D={D}")
    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.float32)
    if out_sum is not None:
        assert out_sum.is_cuda and out_sum.numel() == 1 and out_sum.dtype == torch.float32

    selected = _full_config(N, K, D) if config is None else config
    grid = (triton.cdiv(N, selected["BLOCK_N"]), B)
    _full_logsumexp_kernel[grid](
        x,
        precision.to(torch.float32),
        precision_means.to(torch.float32),
        mean_precision_mean.to(torch.float32),
        logdet.to(torch.float32),
        log_weights.to(torch.float32),
        out,
        out_sum if out_sum is not None else out,
        B,
        N,
        K,
        D,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        precision.stride(0),
        precision.stride(1),
        precision.stride(2),
        precision.stride(3),
        precision_means.stride(0),
        precision_means.stride(1),
        precision_means.stride(2),
        mean_precision_mean.stride(0),
        mean_precision_mean.stride(1),
        logdet.stride(0),
        logdet.stride(1),
        log_weights.stride(0),
        log_weights.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_SUM=out_sum is not None,
        BLOCK_N=selected["BLOCK_N"],
        BLOCK_K=selected["BLOCK_K"],
        BLOCK_D=selected["BLOCK_D"],
        num_warps=selected["num_warps"],
        num_stages=selected["num_stages"],
    )
    return out


def full_assign_triton(
    x: torch.Tensor,
    precision: torch.Tensor,
    precision_means: torch.Tensor,
    mean_precision_mean: torch.Tensor,
    logdet: torch.Tensor,
    log_weights: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    assert x.is_cuda and precision.is_cuda and precision_means.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bp, K, D0, D1 = precision.shape
    assert (Bp, D0, D1) == (B, D, D), "precision shape mismatch"
    assert precision_means.shape == (B, K, D), "precision_means shape mismatch"
    assert mean_precision_mean.shape == (B, K), "mean_precision_mean shape mismatch"
    assert logdet.shape == (B, K), "logdet shape mismatch"
    assert log_weights.shape == (B, K), "log_weights shape mismatch"
    if D > 16:
        raise ValueError(f"full_assign_triton supports D <= 16, got D={D}")
    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)

    selected = _full_config(N, K, D) if config is None else config
    grid = (triton.cdiv(N, selected["BLOCK_N"]), B)
    _full_assign_kernel[grid](
        x,
        precision.to(torch.float32),
        precision_means.to(torch.float32),
        mean_precision_mean.to(torch.float32),
        logdet.to(torch.float32),
        log_weights.to(torch.float32),
        out,
        B,
        N,
        K,
        D,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        precision.stride(0),
        precision.stride(1),
        precision.stride(2),
        precision.stride(3),
        precision_means.stride(0),
        precision_means.stride(1),
        precision_means.stride(2),
        mean_precision_mean.stride(0),
        mean_precision_mean.stride(1),
        logdet.stride(0),
        logdet.stride(1),
        log_weights.stride(0),
        log_weights.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=selected["BLOCK_N"],
        BLOCK_K=selected["BLOCK_K"],
        BLOCK_D=selected["BLOCK_D"],
        num_warps=selected["num_warps"],
        num_stages=selected["num_stages"],
    )
    return out


def full_resp_triton(
    x: torch.Tensor,
    precision: torch.Tensor,
    precision_means: torch.Tensor,
    mean_precision_mean: torch.Tensor,
    logdet: torch.Tensor,
    log_weights: torch.Tensor,
    log_norm: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    assert x.is_cuda and precision.is_cuda and precision_means.is_cuda and log_norm.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    Bp, K, D0, D1 = precision.shape
    assert (Bp, D0, D1) == (B, D, D), "precision shape mismatch"
    assert precision_means.shape == (B, K, D), "precision_means shape mismatch"
    assert mean_precision_mean.shape == (B, K), "mean_precision_mean shape mismatch"
    assert logdet.shape == (B, K), "logdet shape mismatch"
    assert log_weights.shape == (B, K), "log_weights shape mismatch"
    assert log_norm.shape == (B, N), "log_norm shape mismatch"
    if D > 16:
        raise ValueError(f"full_resp_triton supports D <= 16, got D={D}")
    if out is None:
        out = torch.empty((B, N, K), device=x.device, dtype=torch.float32)

    selected = _full_config(N, K, D) if config is None else config
    grid = (triton.cdiv(N, selected["BLOCK_N"]), triton.cdiv(K, selected["BLOCK_K"]), B)
    _full_resp_kernel[grid](
        x,
        precision.to(torch.float32),
        precision_means.to(torch.float32),
        mean_precision_mean.to(torch.float32),
        logdet.to(torch.float32),
        log_weights.to(torch.float32),
        log_norm.to(torch.float32),
        out,
        B,
        N,
        K,
        D,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        precision.stride(0),
        precision.stride(1),
        precision.stride(2),
        precision.stride(3),
        precision_means.stride(0),
        precision_means.stride(1),
        precision_means.stride(2),
        mean_precision_mean.stride(0),
        mean_precision_mean.stride(1),
        logdet.stride(0),
        logdet.stride(1),
        log_weights.stride(0),
        log_weights.stride(1),
        log_norm.stride(0),
        log_norm.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=selected["BLOCK_N"],
        BLOCK_K=selected["BLOCK_K"],
        BLOCK_D=selected["BLOCK_D"],
        num_warps=selected["num_warps"],
        num_stages=selected["num_stages"],
    )
    return out
