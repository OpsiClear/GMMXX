from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ._runtime import triton_spherical_supported
from .torch_fallback import (
    _accumulate_topk_stats,
    _compute_chunk_logits,
    _compute_diag_chunk_logits,
    _compute_full_chunk_logits,
    _compute_tied_chunk_logits,
    _diag_stream_log_normalizer,
    _eye_like_covariance,
    _initialize_diag_parameters,
    _initialize_full_parameters,
    _initialize_parameters,
    _initialize_tied_parameters,
    _matrix_stream_log_normalizer,
    _precision_and_logdet,
    _resolve_approx_top_k,
    _stream_log_normalizer,
    _symmetrize_matrix,
    _topk_logits_for_chunk,
    _triton_blocked_update_config,
    _triton_diag_update_config,
    _triton_full_update_config,
    _triton_tied_logsum_config,
    _triton_tied_update_config,
)

try:
    from .approx_update_triton import (
        approx_topk_update_spherical_config,
        triton_approx_topk_update_spherical,
    )
    from .assign_spherical_triton import (
        spherical_assign_triton,
        spherical_logsumexp_triton,
        spherical_resp_triton,
    )
    from .assign_diag_triton import (
        diag_assign_triton,
        diag_logsumexp_triton,
        diag_resp_triton,
    )
    from .assign_full_triton import (
        full_assign_triton,
        full_logsumexp_triton,
        full_resp_triton,
    )
    from .weighted_update_triton import (
        triton_blocked_update_diag,
        triton_blocked_update_full,
        triton_blocked_update_spherical,
        triton_blocked_update_tied_projected,
    )
    from .fused_update_triton import (
        fused_single_tile_update_config,
        triton_fused_single_tile_update_diag,
        triton_fused_single_tile_update_spherical,
        triton_fused_single_tile_update_tied_native,
    )

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _largen_spherical_cuda(
    x_cpu: torch.Tensor,
    n_clusters: int,
    *,
    max_iters: int = 100,
    tol: float = 1e-4,
    dtype=None,
    device=None,
    chunk_size_data_cpu: int = 1048576,
    seed: int = 0,
    reg_covar: float = 1e-6,
    verbose: bool = False,
    init_centroids=None,
    **_unused_kwargs,
):
    """Spherical large-N EM training on CUDA. Streams chunks of x_cpu to GPU
    and runs per-chunk E-step + M-step partial accumulation; aggregates
    partials at the end of each iteration; finalizes via finalize_spherical.

    Returns the same tuple as the existing torch_native large_n path:
    (cluster_ids, means, variances, weights, info_dict). Caller should
    match the shape contract of the entry point (e.g., (B=1,N) for ids).
    """
    from . import _cuda as _cuda_mod
    import math

    N, D = x_cpu.shape
    K = int(n_clusters)
    if device is None:
        device = torch.device("cuda:0")
    if dtype is None:
        dtype = torch.float32

    # Initialize means by sampling K random points from the input.
    rng = torch.Generator().manual_seed(int(seed))
    if init_centroids is not None:
        # init_centroids is expected (K, D) per the existing contract.
        means = init_centroids.to(device=device, dtype=dtype).unsqueeze(0).contiguous()
    else:
        init_idx = torch.randint(0, N, (K,), generator=rng)
        means = x_cpu[init_idx].to(device=device, dtype=dtype).unsqueeze(0).contiguous()

    # Initialize variance from a small random sample of the data.
    sample_size = min(N, 65536)
    sample_idx = torch.randperm(N, generator=rng)[:sample_size]
    sample = x_cpu[sample_idx].to(device=device, dtype=dtype)
    var_scalar = sample.float().var(dim=0).mean()
    var = var_scalar.clamp_min(reg_covar).expand(1, K).contiguous().float() / max(K, 1)
    var = var.clamp_min(reg_covar)

    log_w = torch.full((1, K), -math.log(K), dtype=torch.float32, device=device)
    weights = torch.full((1, K), 1.0 / K, dtype=torch.float32, device=device)

    lower_bound_history: list[float] = []
    n_iter = 0
    prev_lb = -math.inf

    for _ in range(int(max_iters)):
        n_iter += 1
        # Per-iteration aggregator buffers.
        sums_total = torch.zeros((1, K, D), dtype=torch.float32, device=device)
        sumsq_total = torch.zeros((1, K), dtype=torch.float32, device=device)
        counts_total = torch.zeros((1, K), dtype=torch.int32, device=device)
        lse_sum = 0.0
        lse_count = 0

        for start in range(0, N, chunk_size_data_cpu):
            end = min(start + chunk_size_data_cpu, N)
            chunk = x_cpu[start:end].to(device=device, dtype=dtype, non_blocking=True)
            chunk_b = chunk.unsqueeze(0).contiguous()

            ids = _cuda_mod.spherical_assign(chunk_b, means, var, log_w)
            lse = _cuda_mod.spherical_logsumexp(chunk_b, means, var, log_w)
            lse_sum += float(lse.sum().item())
            lse_count += chunk_b.shape[1]

            sums_c, sumsq_c, counts_c = _cuda_mod.blocked_update_spherical(chunk_b, ids, K)
            sums_total += sums_c
            sumsq_total += sumsq_c
            counts_total += counts_c

        means, var, weights = _cuda_mod.finalize_spherical(
            sums_total, sumsq_total, counts_total, means, var, N, reg_covar
        )
        log_w = torch.log(weights.clamp_min(1e-30))

        lb = lse_sum / max(lse_count, 1)
        lower_bound_history.append(lb)
        if abs(lb - prev_lb) < tol:
            break
        prev_lb = lb

    # Final pass: assign cluster_ids across all chunks.
    cluster_ids_chunks = []
    for start in range(0, N, chunk_size_data_cpu):
        end = min(start + chunk_size_data_cpu, N)
        chunk = x_cpu[start:end].to(device=device, dtype=dtype, non_blocking=True)
        chunk_b = chunk.unsqueeze(0).contiguous()
        ids = _cuda_mod.spherical_assign(chunk_b, means, var, log_w)
        cluster_ids_chunks.append(ids.squeeze(0).cpu())
    cluster_ids = torch.cat(cluster_ids_chunks, dim=0)

    info = {
        "lower_bound": lower_bound_history[-1] if lower_bound_history else float("nan"),
        "lower_bound_history": lower_bound_history,
        "n_iter": n_iter,
        "init_source": "cuda_random" if init_centroids is None else "cuda_provided",
        "triton_estep_enabled": False,
        "triton_streaming_update_enabled": False,
        "triton_fused_update_enabled": False,
        "triton_approx_topk_enabled": False,
        "triton_labels_enabled": False,
        "approximate_em_enabled": False,
        "approx_top_k": None,
        "large_n_streaming_enabled": True,
        "copy_stream_prefetch_enabled": False,
        "fallback_reason": None,
        "backend_breakdown": {"cuda": n_iter},
    }
    return cluster_ids.unsqueeze(0), means, var, weights, info


def _largen_covariance_cuda(
    x_cpu: torch.Tensor,
    n_components: int,
    *,
    covariance_type: str,
    max_iters: int = 100,
    tol: float = 1e-4,
    dtype: torch.dtype,
    device: torch.device,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
    init_params: str = "kmeans",
    reg_covar: float = 1e-6,
    kmeans_init_iters: int = 10,
    kmeans_init_tol: float = 1e-4,
    kmeans_use_triton: bool = True,
    verbose: bool = False,
    max_init_samples: int = 65536,
    min_weight: float = 1e-8,
    compute_labels: bool = True,
) -> Tuple[Optional[torch.LongTensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Large-N CUDA EM for diag/tied/full covariance.

    Spherical has a historical helper above. This helper mirrors the in-memory
    CUDA EM loops while streaming chunks from CPU and aggregating sufficient
    statistics across chunks before each finalize step.
    """
    from . import _cuda as _cuda_mod

    _validate_large_n_input(x_cpu, device)
    if covariance_type not in {"diag", "tied", "full"}:
        raise ValueError("covariance_type must be 'diag', 'tied', or 'full'")

    _, n, d = x_cpu.shape
    k = int(n_components)
    means, variances, weights, init_source, init_sample_size = _initialize_from_sample(
        x_cpu,
        k,
        covariance_type=covariance_type,
        device=device,
        dtype=dtype,
        init_params=init_params,
        reg_covar=reg_covar,
        kmeans_init_iters=kmeans_init_iters,
        kmeans_init_tol=kmeans_init_tol,
        kmeans_use_triton=kmeans_use_triton,
        verbose=verbose,
        max_init_samples=max_init_samples,
    )

    if covariance_type == "tied":
        L = torch.linalg.cholesky(variances.to(torch.float32)).contiguous()
        xx_total = torch.zeros((1, d, d), dtype=torch.float32, device=device)
        for _, _, x_chunk, _ in _iter_device_chunks(
            x_cpu, device=device, dtype=dtype, chunk_size_N=chunk_size_N
        ):
            x_f = x_chunk.float()
            xx_total += x_f.transpose(-1, -2) @ x_f
        var = None
    elif covariance_type == "full":
        L, info = torch.linalg.cholesky_ex(variances.to(torch.float32))
        if bool((info != 0).any().item()):
            raise RuntimeError("full covariance initialization is not positive definite")
        L = L.contiguous()
        var = None
    else:
        var = variances.to(torch.float32).contiguous()
        L = None
        xx_total = None

    weights = weights.to(device=device, dtype=torch.float32).contiguous()
    weights = weights.clamp_min(min_weight)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    log_w = torch.log(weights.clamp_min(1e-30))

    lower_bound_history: list[float] = []
    prev_lb: Optional[float] = None
    prefetch_used = False

    for iteration in range(int(max_iters)):
        sums_total = torch.zeros((1, k, d), dtype=torch.float32, device=device)
        counts_total = torch.zeros((1, k), dtype=torch.float32, device=device)
        if covariance_type == "diag":
            sumsq_total = torch.zeros((1, k, d), dtype=torch.float32, device=device)
        elif covariance_type == "full":
            outer_total = torch.zeros((1, k, d, d), dtype=torch.float32, device=device)

        lse_sum = 0.0
        lse_count = 0

        for _, _, x_chunk, chunk_prefetch_used in _iter_device_chunks(
            x_cpu,
            device=device,
            dtype=dtype,
            chunk_size_N=chunk_size_N,
        ):
            prefetch_used = prefetch_used or chunk_prefetch_used
            x_f = x_chunk.float()
            if covariance_type == "diag":
                lse = _cuda_mod.diag_logsumexp(x_chunk, means, var, log_w)
                resp = _cuda_mod.diag_resp(x_chunk, means, var, log_w, lse)
                counts = resp.sum(dim=1)
                sums = torch.bmm(resp.transpose(1, 2), x_f)
                sumsq = torch.bmm(resp.transpose(1, 2), x_f.square())
                sumsq_total += sumsq
            elif covariance_type == "tied":
                lse = _cuda_mod.tied_logsumexp(x_chunk, means, L, log_w)
                resp = _cuda_mod.tied_resp(x_chunk, means, L, log_w, lse)
                counts = resp.sum(dim=1)
                sums = torch.bmm(resp.transpose(1, 2), x_f)
            else:
                lse = _cuda_mod.full_logsumexp(x_chunk, means, L, log_w)
                resp = _cuda_mod.full_resp(x_chunk, means, L, log_w, lse)
                counts = resp.sum(dim=1)
                sums = torch.bmm(resp.transpose(1, 2), x_f)
                outer_sums = torch.einsum("bnk,bnd,bne->bkde", resp, x_f, x_f)
                outer_total += outer_sums

            sums_total += sums
            counts_total += counts
            lse_sum += float(lse.sum().item())
            lse_count += int(x_chunk.shape[1])

        lb = lse_sum / max(lse_count, 1)
        lower_bound_history.append(lb)

        if covariance_type == "diag":
            active = counts_total > min_weight
            counts_safe = counts_total.clamp_min(min_weight)
            means_new = (sums_total / counts_safe.unsqueeze(-1)).to(dtype)
            means = torch.where(active.unsqueeze(-1), means_new, means)
            second_moment = sumsq_total / counts_safe.unsqueeze(-1)
            var_new = (second_moment - means.float().square()).clamp_min(reg_covar)
            var = torch.where(active.unsqueeze(-1), var_new, var)
            weights = counts_total / float(n)
        elif covariance_type == "tied":
            means, L, weights = _finalize_tied_large_n_cuda(
                sums_total, xx_total, counts_total, means, L, n, reg_covar
            )
        else:
            means, L, weights = _finalize_full_large_n_cuda(
                sums_total, outer_total, counts_total, means, L, n, reg_covar, min_weight
            )

        weights = weights.clamp_min(min_weight)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        log_w = torch.log(weights.clamp_min(1e-30))

        if prev_lb is not None and abs(lb - prev_lb) < tol:
            break
        prev_lb = lb

    if covariance_type == "diag":
        variances_out = var
    elif covariance_type == "tied":
        variances_out = L @ L.transpose(-1, -2)
    else:
        variances_out = L @ L.transpose(-1, -2)

    labels = None
    if compute_labels:
        labels_chunks = []
        for _, _, x_chunk, _ in _iter_device_chunks(
            x_cpu,
            device=device,
            dtype=dtype,
            chunk_size_N=chunk_size_N,
        ):
            if covariance_type == "diag":
                ids = _cuda_mod.diag_assign(x_chunk, means, var, log_w)
            elif covariance_type == "tied":
                ids = _cuda_mod.tied_assign(x_chunk, means, L, log_w)
            else:
                ids = _cuda_mod.full_assign(x_chunk, means, L, log_w)
            labels_chunks.append(ids.squeeze(0).cpu())
        labels = torch.cat(labels_chunks, dim=0).unsqueeze(0)

    n_iter = len(lower_bound_history)
    info = {
        "n_iter": n_iter,
        "lower_bound": lower_bound_history[-1] if lower_bound_history else float("nan"),
        "lower_bound_history": lower_bound_history,
        "init_source": init_source,
        "init_sample_size": init_sample_size,
        "large_n_streaming_enabled": True,
        "copy_stream_prefetch_enabled": bool(prefetch_used),
        "triton_estep_enabled": False,
        "triton_fused_update_enabled": False,
        "triton_approx_topk_enabled": False,
        "triton_streaming_update_enabled": False,
        "triton_labels_enabled": False,
        "approximate_em_enabled": False,
        "approx_top_k": None,
        "fallback_reason": None,
        "backend_breakdown": {"cuda": n_iter},
    }
    return labels, means, variances_out, weights, info


def _finalize_tied_large_n_cuda(
    sums: torch.Tensor,
    xx_total: torch.Tensor,
    counts: torch.Tensor,
    old_means: torch.Tensor,
    old_L: torch.Tensor,
    total_n: int,
    reg_covar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts_f = counts.float()
    active = counts_f > 0
    n_k = counts_f.clamp_min(1e-30)
    means_new = sums / n_k.unsqueeze(-1)
    means_new = torch.where(active.unsqueeze(-1), means_new, old_means.float())
    weights_new = counts_f / float(total_n)

    weighted_means = means_new * counts_f.unsqueeze(-1)
    sigma_k_sum = weighted_means.transpose(-1, -2) @ means_new
    sigma = (xx_total - sigma_k_sum) / float(total_n)
    d = sigma.shape[-1]
    eye = torch.eye(d, device=sigma.device, dtype=sigma.dtype).unsqueeze(0)
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))

    L_new = None
    for jitter in (reg_covar, max(reg_covar * 10.0, 1e-5), 1e-4, 1e-3):
        candidate, info = torch.linalg.cholesky_ex(sigma + float(jitter) * eye)
        if not bool((info != 0).any().item()):
            L_new = candidate
            break
    if L_new is None:
        L_new = old_L.float()

    if old_means.dtype != torch.float32:
        means_new = means_new.to(old_means.dtype)
    if old_L.dtype != torch.float32:
        L_new = L_new.to(old_L.dtype)
    return means_new, L_new, weights_new


def _finalize_full_large_n_cuda(
    sums: torch.Tensor,
    outer_sums: torch.Tensor,
    counts: torch.Tensor,
    old_means: torch.Tensor,
    old_L: torch.Tensor,
    total_n: int,
    reg_covar: float,
    min_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts_f = counts.float()
    active = counts_f > min_weight
    n_k = counts_f.clamp_min(min_weight)
    means_new = sums / n_k.unsqueeze(-1)
    sigma = outer_sums / n_k.unsqueeze(-1).unsqueeze(-1) - (
        means_new.unsqueeze(-1) * means_new.unsqueeze(-2)
    )
    d = sigma.shape[-1]
    eye = torch.eye(d, device=sigma.device, dtype=sigma.dtype).view(1, 1, d, d)
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2)) + reg_covar * eye
    L_new, info = torch.linalg.cholesky_ex(sigma)
    failed = (info != 0) | (~active)

    if failed.any():
        old_means_f = old_means.float()
        old_L_f = old_L.float()
        means_new = torch.where(failed.unsqueeze(-1), old_means_f, means_new)
        L_new = torch.where(failed.unsqueeze(-1).unsqueeze(-1), old_L_f, L_new)

    weights_new = counts_f / float(total_n)
    if old_means.dtype != torch.float32:
        means_new = means_new.to(old_means.dtype)
    if old_L.dtype != torch.float32:
        L_new = L_new.to(old_L.dtype)
    return means_new, L_new, weights_new


def _validate_large_n_input(x_cpu: torch.Tensor, device: torch.device) -> None:
    if x_cpu.ndim != 3:
        raise ValueError("x must have shape (B, N, D)")
    if x_cpu.shape[0] != 1:
        raise NotImplementedError("Large-N CPU streaming currently supports only B=1")
    if x_cpu.device.type != "cpu":
        raise ValueError("Large-N streaming expects a CPU input tensor")
    if device.type != "cuda":
        raise ValueError("Large-N streaming expects a CUDA target device")
    if not x_cpu.dtype.is_floating_point:
        raise TypeError("GMM input must be a floating point tensor")


def _to_device_chunk(
    x_cpu: torch.Tensor,
    start: int,
    end: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    chunk = x_cpu[:, start:end, :]
    non_blocking = False
    if device.type == "cuda":
        if chunk.is_pinned():
            return chunk.to(device=device, dtype=dtype, non_blocking=True)
        try:
            chunk = chunk.pin_memory()
            non_blocking = True
        except RuntimeError:
            non_blocking = False
    return chunk.to(device=device, dtype=dtype, non_blocking=non_blocking)


class _CudaChunkPrefetcher:
    def __init__(
        self,
        x_cpu: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        chunk_size_N: int,
    ) -> None:
        self.x_cpu = x_cpu
        self.device = device
        self.dtype = dtype
        self.chunk_size_N = int(chunk_size_N)
        self.n = int(x_cpu.shape[1])
        self.d = int(x_cpu.shape[2])
        self.enabled = False
        self.direct_pinned_source = bool(x_cpu.is_pinned() and x_cpu.dtype == dtype)
        self._has_compute_done = [False, False]
        try:
            self.copy_stream = torch.cuda.Stream(device=device)
            self.copy_done = [torch.cuda.Event(), torch.cuda.Event()]
            self.compute_done = [torch.cuda.Event(), torch.cuda.Event()]
            self.host_buffers = (
                []
                if self.direct_pinned_source
                else [
                    torch.empty(
                        (1, self.chunk_size_N, self.d),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    for _ in range(2)
                ]
            )
            self.device_buffers = [
                torch.empty(
                    (1, self.chunk_size_N, self.d),
                    dtype=dtype,
                    device=device,
                )
                for _ in range(2)
            ]
            self.enabled = True
        except RuntimeError:
            self.enabled = False

    def schedule(self, buffer_idx: int, start: int) -> Optional[tuple[int, int, int]]:
        if start >= self.n:
            return None
        end = min(start + self.chunk_size_N, self.n)
        rows = end - start
        src = self.x_cpu[:, start:end, :]
        if not self.direct_pinned_source:
            host = self.host_buffers[buffer_idx][:, :rows, :]
            host.copy_(src)
            src = host

        with torch.cuda.stream(self.copy_stream):
            if self._has_compute_done[buffer_idx]:
                self.copy_stream.wait_event(self.compute_done[buffer_idx])
            self.device_buffers[buffer_idx][:, :rows, :].copy_(src, non_blocking=True)
            self.copy_done[buffer_idx].record(self.copy_stream)
        return buffer_idx, start, end

    def wait_chunk(self, buffer_idx: int, rows: int) -> torch.Tensor:
        torch.cuda.current_stream(self.device).wait_event(self.copy_done[buffer_idx])
        return self.device_buffers[buffer_idx][:, :rows, :]

    def mark_compute_done(self, buffer_idx: int) -> None:
        self.compute_done[buffer_idx].record(torch.cuda.current_stream(self.device))
        self._has_compute_done[buffer_idx] = True


def _iter_device_chunks(
    x_cpu: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size_N: int,
    prefetch: bool = True,
):
    _, n, _ = x_cpu.shape
    if not prefetch or device.type != "cuda":
        for n_start in range(0, n, chunk_size_N):
            n_end = min(n_start + chunk_size_N, n)
            yield n_start, n_end, _to_device_chunk(
                x_cpu,
                n_start,
                n_end,
                device=device,
                dtype=dtype,
            ), False
        return

    streamer = _CudaChunkPrefetcher(
        x_cpu,
        device=device,
        dtype=dtype,
        chunk_size_N=chunk_size_N,
    )
    if not streamer.enabled:
        for n_start in range(0, n, chunk_size_N):
            n_end = min(n_start + chunk_size_N, n)
            yield n_start, n_end, _to_device_chunk(
                x_cpu,
                n_start,
                n_end,
                device=device,
                dtype=dtype,
            ), False
        return

    current = streamer.schedule(0, 0)
    next_start = 0 if current is None else current[2]
    while current is not None:
        buffer_idx, n_start, n_end = current
        rows = n_end - n_start
        x_chunk = streamer.wait_chunk(buffer_idx, rows)
        next_current = None
        if next_start < n:
            next_current = streamer.schedule(1 - buffer_idx, next_start)
            next_start = n if next_current is None else next_current[2]
        yield n_start, n_end, x_chunk, True
        streamer.mark_compute_done(buffer_idx)
        current = next_current


def _select_init_sample(
    x_cpu: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_components: int,
    max_init_samples: int,
) -> Tuple[torch.Tensor, int]:
    n = x_cpu.shape[1]
    target = max(4096, n_components * 64, n_components)
    sample_n = min(n, max(n_components, min(max_init_samples, target)))
    if sample_n == n:
        sample_cpu = x_cpu
    else:
        indices = torch.randint(0, n, (sample_n,), device=x_cpu.device)
        sample_cpu = x_cpu.index_select(1, indices)
    return sample_cpu.to(device=device, dtype=dtype, copy=False), sample_n


def _initialize_from_sample(
    x_cpu: torch.Tensor,
    n_components: int,
    *,
    covariance_type: str,
    device: torch.device,
    dtype: torch.dtype,
    init_params: str,
    reg_covar: float,
    kmeans_init_iters: int,
    kmeans_init_tol: float,
    kmeans_use_triton: bool,
    verbose: bool,
    max_init_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
    sample, sample_n = _select_init_sample(
        x_cpu,
        device=device,
        dtype=dtype,
        n_components=n_components,
        max_init_samples=max_init_samples,
    )
    if covariance_type == "spherical":
        x_sq = sample.to(torch.float32).square().sum(dim=-1)
        means, variances, weights, init_source = _initialize_parameters(
            sample,
            n_components,
            init_means=None,
            init_variances=None,
            init_weights=None,
            x_sq=x_sq,
            init_params=init_params,
            reg_covar=reg_covar,
            kmeans_max_iters=kmeans_init_iters,
            kmeans_tol=kmeans_init_tol,
            kmeans_use_triton=kmeans_use_triton,
            verbose=verbose,
        )
    elif covariance_type == "diag":
        means, variances, weights, init_source = _initialize_diag_parameters(
            sample,
            n_components,
            init_means=None,
            init_variances=None,
            init_weights=None,
            init_params=init_params,
            reg_covar=reg_covar,
            kmeans_max_iters=kmeans_init_iters,
            kmeans_tol=kmeans_init_tol,
            kmeans_use_triton=kmeans_use_triton,
            verbose=verbose,
        )
    elif covariance_type == "full":
        means, variances, weights, init_source = _initialize_full_parameters(
            sample,
            n_components,
            init_means=None,
            init_covariances=None,
            init_weights=None,
            init_params=init_params,
            reg_covar=reg_covar,
            kmeans_max_iters=kmeans_init_iters,
            kmeans_tol=kmeans_init_tol,
            kmeans_use_triton=kmeans_use_triton,
            verbose=verbose,
        )
    elif covariance_type == "tied":
        means, variances, weights, init_source = _initialize_tied_parameters(
            sample,
            n_components,
            init_means=None,
            init_covariance=None,
            init_weights=None,
            init_params=init_params,
            reg_covar=reg_covar,
            kmeans_max_iters=kmeans_init_iters,
            kmeans_tol=kmeans_init_tol,
            kmeans_use_triton=kmeans_use_triton,
            verbose=verbose,
        )
    else:
        raise ValueError("covariance_type must be 'spherical', 'diag', 'tied', or 'full'")

    if sample_n < x_cpu.shape[1]:
        init_source = f"{init_source}_sampled"
    return means, variances, weights, init_source, sample_n


def _matrix_terms(
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    covariance_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    precision, logdet = _precision_and_logdet(covariances)
    means_f = means.to(torch.float32)
    if covariance_type == "full":
        precision_means = torch.einsum("bkde,bke->bkd", precision, means_f)
    else:
        precision_means = torch.bmm(means_f, precision.transpose(1, 2))
    mean_precision_mean = (means_f * precision_means).sum(dim=-1)
    log_weights = torch.log(weights.to(torch.float32))
    return precision, logdet, precision_means, mean_precision_mean, log_weights


def _diag_terms(
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    precision = variances.to(torch.float32).clamp_min(1e-30).reciprocal()
    logdet = torch.log(variances.to(torch.float32).clamp_min(1e-30)).sum(dim=-1)
    weighted_means = means.to(torch.float32) * precision
    mean_precision_mean = (means.to(torch.float32) * weighted_means).sum(dim=-1)
    log_weights = torch.log(weights.to(torch.float32))
    return precision, logdet, weighted_means, mean_precision_mean, log_weights


def _stream_total_xx_cpu(
    x_cpu: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size_N: int,
) -> torch.Tensor:
    _, _, d = x_cpu.shape
    total_xx = torch.zeros((1, d, d), device=device, dtype=torch.float32)
    for _, _, x_chunk, _ in _iter_device_chunks(
        x_cpu,
        device=device,
        dtype=dtype,
        chunk_size_N=chunk_size_N,
    ):
        x_f = x_chunk.to(torch.float32)
        total_xx += torch.bmm(x_f.transpose(1, 2), x_f)
    return total_xx


def _resolve_triton_large_n(value: bool | str, covariance_type: str, d: int, k: int) -> bool:
    if value not in {True, False, "auto"}:
        raise ValueError("gmm_use_triton must be a bool or 'auto'")
    if value is False or not _HAS_TRITON:
        return False
    if covariance_type == "spherical":
        supported = d >= 16 and triton_spherical_supported(d, k)
    elif covariance_type == "diag":
        supported = 16 <= d <= 64 and k <= 512
    elif covariance_type == "tied":
        supported = 16 <= d <= 64 and k <= 512
    elif covariance_type == "full":
        supported = d <= 8 and k <= 128
    else:
        supported = False
    return bool(supported)


def _make_partial_buffers(
    *,
    covariance_type: str,
    device: torch.device,
    max_chunk_n: int,
    d: int,
    k: int,
) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, ...]], tuple[int, int, int]]:
    if covariance_type == "spherical":
        block_n, block_d, block_k = _triton_blocked_update_config(d, k)
        max_n_blocks = (max_chunk_n + block_n - 1) // block_n
        return (
            torch.empty((1, max_chunk_n), device=device, dtype=torch.float32),
            (
                torch.empty((1, max_n_blocks, k), device=device, dtype=torch.float32),
                torch.empty((1, max_n_blocks, k, d), device=device, dtype=torch.float32),
                torch.empty((1, max_n_blocks, k), device=device, dtype=torch.float32),
                torch.empty((1, max_n_blocks), device=device, dtype=torch.float32),
            ),
            (block_n, block_d, block_k),
        )
    if covariance_type == "diag":
        block_n, block_d, block_k = _triton_diag_update_config(d, k)
        max_n_blocks = (max_chunk_n + block_n - 1) // block_n
        return (
            torch.empty((1, max_chunk_n), device=device, dtype=torch.float32),
            (
                torch.empty((1, max_n_blocks, k), device=device, dtype=torch.float32),
                torch.empty((1, max_n_blocks, k, d), device=device, dtype=torch.float32),
                torch.empty((1, max_n_blocks, k, d), device=device, dtype=torch.float32),
                torch.empty((1, max_n_blocks), device=device, dtype=torch.float32),
            ),
            (block_n, block_d, block_k),
        )
    if covariance_type == "full":
        block_n, block_d, block_k = _triton_full_update_config(d, k)
        max_n_blocks = (max_chunk_n + block_n - 1) // block_n
        return (
            torch.empty((1, max_chunk_n), device=device, dtype=torch.float32),
            (
                torch.empty((1, max_n_blocks, k), device=device, dtype=torch.float32),
                torch.empty((1, max_n_blocks, k, d), device=device, dtype=torch.float32),
                torch.empty((1, max_n_blocks, k, d, d), device=device, dtype=torch.float32),
                torch.empty((1, max_n_blocks), device=device, dtype=torch.float32),
            ),
            (block_n, block_d, block_k),
        )

    block_n, block_d, block_k = _triton_tied_update_config(d, k)
    max_n_blocks = (max_chunk_n + block_n - 1) // block_n
    return (
        torch.empty((1, max_chunk_n), device=device, dtype=torch.float32),
        (
            torch.empty((1, max_n_blocks, k), device=device, dtype=torch.float32),
            torch.empty((1, max_n_blocks, k, d), device=device, dtype=torch.float32),
            torch.empty((1, max_n_blocks), device=device, dtype=torch.float32),
        ),
        (block_n, block_d, block_k),
    )


def batch_gmm_largeN_cpu(
    x_cpu: torch.Tensor,
    n_components: int,
    *,
    covariance_type: str,
    device: torch.device,
    dtype: torch.dtype,
    max_iters: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
    init_params: str = "kmeans",
    reg_covar: float = 1e-6,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
    kmeans_init_iters: int = 10,
    kmeans_init_tol: float = 1e-4,
    kmeans_use_triton: bool = True,
    gmm_use_triton: bool | str = "auto",
    min_weight: float = 1e-8,
    compute_labels: bool = True,
    max_init_samples: int = 65536,
    approx_top_k: Optional[int] = None,
    backend: str = "auto",
    legacy_no_triton: bool = False,
    seed: int = 0,
) -> Tuple[Optional[torch.LongTensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    _validate_large_n_input(x_cpu, device)
    if covariance_type not in {"spherical", "diag", "tied", "full"}:
        raise ValueError("covariance_type must be 'spherical', 'diag', 'tied', or 'full'")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if chunk_size_N <= 0 or chunk_size_K <= 0:
        raise ValueError("chunk_size_N and chunk_size_K must be positive")
    if min_weight <= 0.0:
        raise ValueError("min_weight must be positive")
    effective_approx_top_k = _resolve_approx_top_k(approx_top_k, n_components)
    cuda_fallback_reason = None

    # Plan 10: backend dispatch.
    from . import _dispatch
    representative_shape = (1, 1024, x_cpu.shape[-1], int(n_components))
    resolved = _dispatch.resolve_backend_with_env(
        requested=backend,
        covariance=covariance_type,
        shape=representative_shape,
        dtype=dtype if dtype is not None else torch.float32,
        legacy_no_triton=legacy_no_triton,
    )
    if (
        resolved == "cuda"
        and covariance_type == "spherical"
        and effective_approx_top_k is None
    ):
        try:
            # x_cpu may be (B=1, N, D) — squeeze the batch dim for the helper.
            x_2d = x_cpu.squeeze(0) if x_cpu.ndim == 3 else x_cpu
            return _largen_spherical_cuda(
                x_2d, n_components,
                max_iters=max_iters,
                tol=tol,
                dtype=dtype,
                device=device,
                chunk_size_data_cpu=chunk_size_N,
                seed=seed,
                reg_covar=reg_covar,
                verbose=verbose,
            )
        except Exception as exc:
            if backend == "cuda":
                raise
            cuda_fallback_reason = (
                f"large-N spherical CUDA EM failed: {type(exc).__name__}: {exc}"
            )
    if (
        resolved == "cuda"
        and covariance_type in {"diag", "tied", "full"}
        and effective_approx_top_k is None
    ):
        try:
            return _largen_covariance_cuda(
                x_cpu,
                n_components,
                covariance_type=covariance_type,
                max_iters=max_iters,
                tol=tol,
                dtype=dtype,
                device=device,
                chunk_size_N=chunk_size_N,
                chunk_size_K=chunk_size_K,
                init_params=init_params,
                reg_covar=reg_covar,
                kmeans_init_iters=kmeans_init_iters,
                kmeans_init_tol=kmeans_init_tol,
                kmeans_use_triton=kmeans_use_triton,
                verbose=verbose,
                max_init_samples=max_init_samples,
                min_weight=min_weight,
                compute_labels=compute_labels,
            )
        except Exception as exc:
            if backend == "cuda":
                raise
            cuda_fallback_reason = (
                f"large-N {covariance_type} CUDA EM failed: {type(exc).__name__}: {exc}"
            )

    _, n, d = x_cpu.shape
    means, variances, weights, init_source, init_sample_size = _initialize_from_sample(
        x_cpu,
        n_components,
        covariance_type=covariance_type,
        device=device,
        dtype=dtype,
        init_params=init_params,
        reg_covar=reg_covar,
        kmeans_init_iters=kmeans_init_iters,
        kmeans_init_tol=kmeans_init_tol,
        kmeans_use_triton=kmeans_use_triton,
        verbose=verbose,
        max_init_samples=max_init_samples,
    )

    eye = _eye_like_covariance(d, device, torch.float32)
    use_triton = _resolve_triton_large_n(gmm_use_triton, covariance_type, d, n_components)
    labels_use_triton = use_triton
    approx_triton_config = None
    if (
        effective_approx_top_k is not None
        and covariance_type == "spherical"
        and use_triton
        and _HAS_TRITON
    ):
        approx_triton_config = approx_topk_update_spherical_config(
            d,
            n_components,
            effective_approx_top_k,
        )
    if effective_approx_top_k is not None:
        use_triton = False
    triton_failed = False
    triton_used = False
    triton_fused_used = False
    triton_approx_used = False
    triton_approx_failed = False
    prefetch_used = False
    fallback_reason = cuda_fallback_reason
    log_norm_buffer = None
    partial_buffers = None
    triton_blocks = (0, 0, 0)
    if use_triton:
        try:
            log_norm_buffer, partial_buffers, triton_blocks = _make_partial_buffers(
                covariance_type=covariance_type,
                device=device,
                max_chunk_n=min(chunk_size_N, n),
                d=d,
                k=n_components,
            )
        except Exception as exc:
            fallback_reason = f"large-N Triton buffer setup failed: {type(exc).__name__}: {exc}"
            use_triton = False
    approx_partial_ll_buffer = None
    if approx_triton_config is not None:
        approx_block_n = int(approx_triton_config["BLOCK_N"])
        max_n_blocks = (min(chunk_size_N, n) + approx_block_n - 1) // approx_block_n
        approx_partial_ll_buffer = torch.empty((1, max_n_blocks), device=device, dtype=torch.float32)

    tied_total_xx = None

    prev_lower_bound: Optional[float] = None
    lower_bound_history = []

    for iteration in range(max_iters):
        weights = weights.clamp_min(min_weight)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        log_weights = torch.log(weights.to(torch.float32))
        means_sq = None
        if covariance_type == "spherical" and (use_triton or approx_triton_config is not None):
            means_sq = means.to(torch.float32).square().sum(dim=-1)
        fused_update_config = (
            fused_single_tile_update_config(d, n_components, covariance_type)
            if use_triton
            else None
        )

        if covariance_type == "diag":
            precision, logdet, weighted_means, mean_precision_mean, log_weights = _diag_terms(
                means,
                variances,
                weights,
            )
        elif covariance_type in {"full", "tied"}:
            precision, logdet, precision_means, mean_precision_mean, log_weights = _matrix_terms(
                means,
                variances,
                weights,
                covariance_type,
            )
            if covariance_type == "tied" and use_triton:
                try:
                    chol_precision = torch.linalg.cholesky(precision)
                    tied_means_projected = torch.bmm(means.to(torch.float32), chol_precision)
                    tied_means_projected_sq = tied_means_projected.square().sum(dim=-1)
                    tied_unit_variances = torch.ones(
                        (1, n_components),
                        device=device,
                        dtype=torch.float32,
                    )
                except Exception as exc:
                    fallback_reason = f"large-N tied Triton projection setup failed: {type(exc).__name__}: {exc}"
                    triton_failed = True
                    use_triton = False

        nk = torch.zeros((1, n_components), device=device, dtype=torch.float32)
        sum_x = torch.zeros((1, n_components, d), device=device, dtype=torch.float32)
        total_log_likelihood = torch.zeros((), device=device, dtype=torch.float32)
        total_xx_accum = (
            torch.zeros((1, d, d), device=device, dtype=torch.float32)
            if covariance_type == "tied" and tied_total_xx is None
            else None
        )
        if covariance_type == "spherical":
            sum_x_sq = torch.zeros((1, n_components), device=device, dtype=torch.float32)
        elif covariance_type == "diag":
            sum_x_sq = torch.zeros((1, n_components, d), device=device, dtype=torch.float32)
        elif covariance_type == "full":
            sum_xx = torch.zeros((1, n_components, d, d), device=device, dtype=torch.float32)

        for n_start, n_end, x_chunk, chunk_prefetch_used in _iter_device_chunks(
            x_cpu,
            device=device,
            dtype=dtype,
            chunk_size_N=chunk_size_N,
        ):
            prefetch_used = prefetch_used or chunk_prefetch_used
            x_f = x_chunk.to(torch.float32)
            if total_xx_accum is not None:
                total_xx_accum += torch.bmm(x_f.transpose(1, 2), x_f)
            chunk_log_norm_out = (
                None
                if log_norm_buffer is None
                else log_norm_buffer[:, : n_end - n_start]
            )

            if use_triton:
                try:
                    block_n, block_d, block_k = triton_blocks
                    if covariance_type == "spherical":
                        x_sq = x_f.square().sum(dim=-1)
                        if fused_update_config is not None:
                            nk_tile, sum_x_tile, sum_x_sq_tile, ll_tile = (
                                triton_fused_single_tile_update_spherical(
                                    x_chunk,
                                    means,
                                    variances.to(torch.float32),
                                    weights.to(torch.float32),
                                    x_sq=x_sq,
                                    means_sq=means_sq,
                                    log_weights=log_weights,
                                    partial_nk=None if partial_buffers is None else partial_buffers[0],
                                    partial_sum_x=None if partial_buffers is None else partial_buffers[1],
                                    partial_sum_x_sq=None if partial_buffers is None else partial_buffers[2],
                                    partial_log_likelihood=None if partial_buffers is None else partial_buffers[3],
                                    **fused_update_config,
                                )
                            )
                            total_log_likelihood = total_log_likelihood + ll_tile
                            nk += nk_tile
                            sum_x += sum_x_tile
                            sum_x_sq += sum_x_sq_tile
                            triton_used = True
                            triton_fused_used = True
                            continue

                        log_norm = spherical_logsumexp_triton(
                            x_chunk,
                            means,
                            variances.to(torch.float32),
                            weights.to(torch.float32),
                            x_sq=x_sq,
                            out=chunk_log_norm_out,
                            means_sq=means_sq,
                            log_weights=log_weights,
                        )
                        nk_tile, sum_x_tile, sum_x_sq_tile = triton_blocked_update_spherical(
                            x_chunk,
                            means,
                            variances.to(torch.float32),
                            weights.to(torch.float32),
                            log_norm,
                            x_sq=x_sq,
                            means_sq=means_sq,
                            log_weights=log_weights,
                            partial_nk=None if partial_buffers is None else partial_buffers[0],
                            partial_sum_x=None if partial_buffers is None else partial_buffers[1],
                            partial_sum_x_sq=None if partial_buffers is None else partial_buffers[2],
                            BLOCK_N=block_n,
                            BLOCK_D=block_d,
                            BLOCK_K=block_k,
                        )
                        total_log_likelihood = total_log_likelihood + log_norm.sum()
                        nk += nk_tile
                        sum_x += sum_x_tile
                        sum_x_sq += sum_x_sq_tile
                        triton_used = True
                        continue

                    if covariance_type == "diag":
                        if fused_update_config is not None:
                            nk_tile, sum_x_tile, sum_x_sq_tile, ll_tile = (
                                triton_fused_single_tile_update_diag(
                                    x_chunk,
                                    precision,
                                    weighted_means,
                                    mean_precision_mean,
                                    logdet,
                                    log_weights,
                                    partial_nk=None if partial_buffers is None else partial_buffers[0],
                                    partial_sum_x=None if partial_buffers is None else partial_buffers[1],
                                    partial_sum_x_sq=None if partial_buffers is None else partial_buffers[2],
                                    partial_log_likelihood=None if partial_buffers is None else partial_buffers[3],
                                    **fused_update_config,
                                )
                            )
                            total_log_likelihood = total_log_likelihood + ll_tile
                            nk += nk_tile
                            sum_x += sum_x_tile
                            sum_x_sq += sum_x_sq_tile
                            triton_used = True
                            triton_fused_used = True
                            continue

                        log_norm = diag_logsumexp_triton(
                            x_chunk,
                            precision,
                            weighted_means,
                            mean_precision_mean,
                            logdet,
                            log_weights,
                            out=chunk_log_norm_out,
                        )
                        nk_tile, sum_x_tile, sum_x_sq_tile = triton_blocked_update_diag(
                            x_chunk,
                            precision,
                            weighted_means,
                            mean_precision_mean,
                            logdet,
                            log_weights,
                            log_norm,
                            partial_nk=None if partial_buffers is None else partial_buffers[0],
                            partial_sum_x=None if partial_buffers is None else partial_buffers[1],
                            partial_sum_x_sq=None if partial_buffers is None else partial_buffers[2],
                            BLOCK_N=block_n,
                            BLOCK_D=block_d,
                            BLOCK_K=block_k,
                        )
                        total_log_likelihood = total_log_likelihood + log_norm.sum()
                        nk += nk_tile
                        sum_x += sum_x_tile
                        sum_x_sq += sum_x_sq_tile
                        triton_used = True
                        continue

                    if covariance_type == "full":
                        log_norm = full_logsumexp_triton(
                            x_chunk,
                            precision,
                            precision_means,
                            mean_precision_mean,
                            logdet,
                            log_weights,
                            out=chunk_log_norm_out,
                        )
                        nk_tile, sum_x_tile, sum_xx_tile = triton_blocked_update_full(
                            x_chunk,
                            precision,
                            precision_means,
                            mean_precision_mean,
                            logdet,
                            log_weights,
                            log_norm,
                            partial_nk=None if partial_buffers is None else partial_buffers[0],
                            partial_sum_x=None if partial_buffers is None else partial_buffers[1],
                            partial_sum_xx=None if partial_buffers is None else partial_buffers[2],
                            BLOCK_N=block_n,
                            BLOCK_D=block_d,
                            BLOCK_K=block_k,
                        )
                        total_log_likelihood = total_log_likelihood + log_norm.sum()
                        nk += nk_tile
                        sum_x += sum_x_tile
                        sum_xx += sum_xx_tile
                        triton_used = True
                        continue

                    if covariance_type == "tied" and fused_update_config is not None:
                        nk_tile, sum_x_tile, ll_tile = triton_fused_single_tile_update_tied_native(
                            x_chunk,
                            chol_precision,
                            tied_means_projected,
                            tied_means_projected_sq,
                            logdet,
                            log_weights,
                            partial_nk=None if partial_buffers is None else partial_buffers[0],
                            partial_sum_x=None if partial_buffers is None else partial_buffers[1],
                            partial_log_likelihood=None if partial_buffers is None else partial_buffers[2],
                            **fused_update_config,
                        )
                        total_log_likelihood = total_log_likelihood + ll_tile
                        nk += nk_tile
                        sum_x += sum_x_tile
                        triton_used = True
                        triton_fused_used = True
                        continue

                    x_projected = torch.bmm(x_f, chol_precision)
                    x_projected_sq = x_projected.square().sum(dim=-1)
                    log_norm = spherical_logsumexp_triton(
                        x_projected,
                        tied_means_projected,
                        tied_unit_variances,
                        weights.to(torch.float32),
                        x_sq=x_projected_sq,
                        out=chunk_log_norm_out,
                        means_sq=tied_means_projected_sq,
                        log_weights=log_weights,
                        config=_triton_tied_logsum_config(d, n_components),
                        unit_variance=True,
                    )
                    nk_tile, sum_x_tile = triton_blocked_update_tied_projected(
                        x_projected,
                        x_chunk,
                        tied_means_projected,
                        log_weights,
                        log_norm,
                        x_projected_sq=x_projected_sq,
                        means_projected_sq=tied_means_projected_sq,
                        partial_nk=None if partial_buffers is None else partial_buffers[0],
                        partial_sum_x=None if partial_buffers is None else partial_buffers[1],
                        BLOCK_N=block_n,
                        BLOCK_D=block_d,
                        BLOCK_K=block_k,
                    )
                    total_log_likelihood = (
                        total_log_likelihood
                        + log_norm.sum()
                        - 0.5 * float(n_end - n_start) * logdet.sum()
                    )
                    nk += nk_tile
                    sum_x += sum_x_tile
                    triton_used = True
                    continue
                except Exception as exc:
                    fallback_reason = f"large-N {covariance_type} Triton EM chunk failed: {type(exc).__name__}: {exc}"
                    triton_failed = True
                    use_triton = False

            if effective_approx_top_k is not None:
                if covariance_type == "spherical":
                    x_sq = x_f.square().sum(dim=-1)
                    if approx_triton_config is not None:
                        try:
                            _, _, _, ll_tile = triton_approx_topk_update_spherical(
                                x_chunk,
                                means,
                                variances.to(torch.float32),
                                weights.to(torch.float32),
                                top_k=effective_approx_top_k,
                                x_sq=x_sq,
                                means_sq=means_sq,
                                log_weights=log_weights,
                                nk=nk,
                                sum_x=sum_x,
                                sum_x_sq=sum_x_sq,
                                partial_log_likelihood=approx_partial_ll_buffer,
                                **approx_triton_config,
                            )
                            total_log_likelihood = total_log_likelihood + ll_tile
                            triton_approx_used = True
                            continue
                        except Exception as exc:
                            fallback_reason = f"large-N spherical approximate Triton chunk failed: {type(exc).__name__}: {exc}"
                            triton_approx_failed = True
                            approx_triton_config = None
                    terms = (x_sq, log_weights)
                    sum_x_sq_target = sum_x_sq
                    sum_xx_target = None
                elif covariance_type == "diag":
                    terms = (
                        precision,
                        logdet,
                        weighted_means,
                        mean_precision_mean,
                        log_weights,
                    )
                    x_sq = None
                    sum_x_sq_target = sum_x_sq
                    sum_xx_target = None
                else:
                    terms = (
                        precision,
                        logdet,
                        precision_means,
                        mean_precision_mean,
                        log_weights,
                    )
                    x_sq = None
                    sum_x_sq_target = None
                    sum_xx_target = sum_xx if covariance_type == "full" else None
                topk_logits, topk_indices = _topk_logits_for_chunk(
                    x_chunk,
                    means,
                    variances,
                    covariance_type=covariance_type,
                    chunk_size_K=chunk_size_K,
                    top_k=effective_approx_top_k,
                    terms=terms,
                )
                total_log_likelihood = _accumulate_topk_stats(
                    x_chunk,
                    topk_logits,
                    topk_indices,
                    covariance_type=covariance_type,
                    nk=nk,
                    sum_x=sum_x,
                    x_sq=x_sq,
                    sum_x_sq=sum_x_sq_target,
                    sum_xx=sum_xx_target,
                    total_log_likelihood=total_log_likelihood,
                )
                continue

            if covariance_type == "spherical":
                x_sq = x_f.square().sum(dim=-1)
                log_norm = _stream_log_normalizer(
                    x_chunk,
                    x_sq,
                    means,
                    variances,
                    weights,
                    chunk_size_K=chunk_size_K,
                )
            elif covariance_type == "diag":
                log_norm = _diag_stream_log_normalizer(
                    x_chunk,
                    means,
                    variances,
                    weights,
                    chunk_size_K=chunk_size_K,
                    precision=precision,
                    logdet=logdet,
                    log_weights=log_weights,
                    weighted_means=weighted_means,
                    mean_precision_mean=mean_precision_mean,
                )
            else:
                log_norm = _matrix_stream_log_normalizer(
                    x_chunk,
                    means,
                    variances,
                    weights,
                    chunk_size_K=chunk_size_K,
                    covariance_type=covariance_type,
                    precision=precision,
                    logdet=logdet,
                    log_weights=log_weights,
                    precision_means=precision_means,
                    mean_precision_mean=mean_precision_mean,
                )
            total_log_likelihood = total_log_likelihood + log_norm.sum()

            for k_start in range(0, n_components, chunk_size_K):
                k_end = min(k_start + chunk_size_K, n_components)
                if covariance_type == "spherical":
                    logits = _compute_chunk_logits(
                        x_chunk,
                        x_sq,
                        means[:, k_start:k_end, :],
                        variances[:, k_start:k_end],
                        log_weights[:, k_start:k_end],
                    )
                elif covariance_type == "diag":
                    logits = _compute_diag_chunk_logits(
                        x_chunk,
                        means[:, k_start:k_end, :],
                        variances[:, k_start:k_end, :],
                        log_weights[:, k_start:k_end],
                        precision_chunk=precision[:, k_start:k_end, :],
                        logdet_chunk=logdet[:, k_start:k_end],
                        weighted_means_chunk=weighted_means[:, k_start:k_end, :],
                        mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
                    )
                elif covariance_type == "full":
                    logits = _compute_full_chunk_logits(
                        x_chunk,
                        means[:, k_start:k_end, :],
                        variances[:, k_start:k_end, :, :],
                        log_weights[:, k_start:k_end],
                        precision_chunk=precision[:, k_start:k_end, :, :],
                        logdet_chunk=logdet[:, k_start:k_end],
                        precision_means_chunk=precision_means[:, k_start:k_end, :],
                        mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
                    )
                else:
                    logits = _compute_tied_chunk_logits(
                        x_chunk,
                        means[:, k_start:k_end, :],
                        variances,
                        log_weights[:, k_start:k_end],
                        precision=precision,
                        logdet=logdet,
                        precision_means_chunk=precision_means[:, k_start:k_end, :],
                        mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
                    )

                resp = torch.exp(logits - log_norm.unsqueeze(-1))
                nk[:, k_start:k_end] += resp.sum(dim=1)
                sum_x[:, k_start:k_end, :] += torch.bmm(resp.transpose(1, 2), x_f)
                if covariance_type == "spherical":
                    sum_x_sq[:, k_start:k_end] += (resp * x_sq.unsqueeze(-1)).sum(dim=1)
                elif covariance_type == "diag":
                    sum_x_sq[:, k_start:k_end, :] += torch.bmm(resp.transpose(1, 2), x_f.square())
                elif covariance_type == "full":
                    sum_xx[:, k_start:k_end, :, :] += torch.einsum(
                        "bnk,bnd,bne->bkde",
                        resp,
                        x_f,
                        x_f,
                    )

        if total_xx_accum is not None:
            tied_total_xx = total_xx_accum

        active_mask = nk > min_weight
        nk_safe = nk.clamp_min(min_weight)
        means_new = (sum_x / nk_safe.unsqueeze(-1)).to(dtype)
        means_new = torch.where(active_mask.unsqueeze(-1), means_new, means)
        means_f = means_new.to(torch.float32)

        if covariance_type == "spherical":
            mean_sq = means_f.square().sum(dim=-1)
            variances_new = (sum_x_sq - nk * mean_sq).clamp_min(0.0) / (
                nk_safe * float(d)
            )
            variances_new = variances_new.clamp_min(reg_covar)
            variances_new = torch.where(active_mask, variances_new.to(dtype), variances)
        elif covariance_type == "diag":
            second_moment = sum_x_sq / nk_safe.unsqueeze(-1)
            variances_new = (second_moment - means_f.square()).clamp_min(reg_covar)
            variances_new = torch.where(active_mask.unsqueeze(-1), variances_new.to(dtype), variances)
        else:
            means_outer = means_f.unsqueeze(-1) * means_f.unsqueeze(-2)
            if covariance_type == "full":
                scatter = sum_xx - nk[..., None, None] * means_outer
                variances_new = scatter / nk_safe[..., None, None]
                variances_new = _symmetrize_matrix(variances_new) + reg_covar * eye
                variances_new = torch.where(
                    active_mask[..., None, None],
                    variances_new.to(dtype),
                    variances,
                )
            else:
                scatter = tied_total_xx - (nk[..., None, None] * means_outer).sum(dim=1)
                variances_new = scatter / float(n)
                variances_new = (_symmetrize_matrix(variances_new) + reg_covar * eye).to(dtype)

        weights_new = nk / float(n)
        weights_new = weights_new.clamp_min(min_weight)
        weights_new = weights_new / weights_new.sum(dim=-1, keepdim=True)
        weights_new = weights_new.to(dtype)

        lower_bound = float((total_log_likelihood / float(n)).item())
        lower_bound_history.append(lower_bound)

        if verbose:
            mean_shift = (means_new - means).norm(dim=-1).max().item()
            print(
                f"Iter {iteration}, lower_bound: {lower_bound:.6f}, "
                f"mean_shift: {mean_shift:.6f}"
            )

        means = means_new
        variances = variances_new
        weights = weights_new

        if prev_lower_bound is not None and abs(lower_bound - prev_lower_bound) < tol:
            break
        prev_lower_bound = lower_bound

    labels_triton_used = False
    if compute_labels:
        labels_result = large_n_predict_cpu(
            x_cpu,
            means,
            variances,
            weights,
            covariance_type=covariance_type,
            device=device,
            dtype=dtype,
            chunk_size_N=chunk_size_N,
            chunk_size_K=chunk_size_K,
            use_triton=bool(labels_use_triton and not triton_failed),
            return_triton_used=True,
            backend=backend,
            legacy_no_triton=legacy_no_triton,
        )
        labels, labels_triton_used = labels_result
    else:
        labels = None
    info: Dict[str, object] = {
        "n_iter": iteration + 1,
        "lower_bound": lower_bound_history[-1],
        "lower_bound_history": lower_bound_history,
        "init_source": init_source,
        "init_sample_size": init_sample_size,
        "large_n_streaming_enabled": True,
        "copy_stream_prefetch_enabled": bool(prefetch_used),
        "triton_estep_enabled": bool((triton_used and not triton_failed) or (triton_approx_used and not triton_approx_failed)),
        "triton_fused_update_enabled": bool(triton_fused_used and not triton_failed),
        "triton_approx_topk_enabled": bool(triton_approx_used and not triton_approx_failed),
        "triton_streaming_update_enabled": bool((triton_used and not triton_failed) or (triton_approx_used and not triton_approx_failed)),
        "triton_labels_enabled": bool(labels_triton_used),
        "approximate_em_enabled": bool(effective_approx_top_k is not None),
        "approx_top_k": effective_approx_top_k,
        "fallback_reason": fallback_reason,
    }
    return labels, means, variances, weights, info


def _stream_log_norm_for_params(
    x_chunk: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    covariance_type: str,
    chunk_size_K: int,
):
    if covariance_type == "spherical":
        x_sq = x_chunk.to(torch.float32).square().sum(dim=-1)
        return _stream_log_normalizer(
            x_chunk,
            x_sq,
            means,
            variances,
            weights,
            chunk_size_K=chunk_size_K,
        ), (x_sq, torch.log(weights.to(torch.float32)))
    if covariance_type == "diag":
        precision, logdet, weighted_means, mean_precision_mean, log_weights = _diag_terms(
            means,
            variances,
            weights,
        )
        return _diag_stream_log_normalizer(
            x_chunk,
            means,
            variances,
            weights,
            chunk_size_K=chunk_size_K,
            precision=precision,
            logdet=logdet,
            log_weights=log_weights,
            weighted_means=weighted_means,
            mean_precision_mean=mean_precision_mean,
        ), (precision, logdet, weighted_means, mean_precision_mean, log_weights)

    precision, logdet, precision_means, mean_precision_mean, log_weights = _matrix_terms(
        means,
        variances,
        weights,
        covariance_type,
    )
    return _matrix_stream_log_normalizer(
        x_chunk,
        means,
        variances,
        weights,
        chunk_size_K=chunk_size_K,
        covariance_type=covariance_type,
        precision=precision,
        logdet=logdet,
        log_weights=log_weights,
        precision_means=precision_means,
        mean_precision_mean=mean_precision_mean,
    ), (precision, logdet, precision_means, mean_precision_mean, log_weights)


def _terms_for_chunk_params(
    x_chunk: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    covariance_type: str,
):
    if covariance_type == "spherical":
        x_sq = x_chunk.to(torch.float32).square().sum(dim=-1)
        return x_sq, torch.log(weights.to(torch.float32))
    if covariance_type == "diag":
        return _diag_terms(means, variances, weights)
    return _matrix_terms(means, variances, weights, covariance_type)


def _logits_for_tile(
    x_chunk: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    *,
    covariance_type: str,
    k_start: int,
    k_end: int,
    terms,
) -> torch.Tensor:
    if covariance_type == "spherical":
        x_sq, log_weights = terms
        return _compute_chunk_logits(
            x_chunk,
            x_sq,
            means[:, k_start:k_end, :],
            variances[:, k_start:k_end],
            log_weights[:, k_start:k_end],
        )
    if covariance_type == "diag":
        precision, logdet, weighted_means, mean_precision_mean, log_weights = terms
        return _compute_diag_chunk_logits(
            x_chunk,
            means[:, k_start:k_end, :],
            variances[:, k_start:k_end, :],
            log_weights[:, k_start:k_end],
            precision_chunk=precision[:, k_start:k_end, :],
            logdet_chunk=logdet[:, k_start:k_end],
            weighted_means_chunk=weighted_means[:, k_start:k_end, :],
            mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
        )
    precision, logdet, precision_means, mean_precision_mean, log_weights = terms
    if covariance_type == "full":
        return _compute_full_chunk_logits(
            x_chunk,
            means[:, k_start:k_end, :],
            variances[:, k_start:k_end, :, :],
            log_weights[:, k_start:k_end],
            precision_chunk=precision[:, k_start:k_end, :, :],
            logdet_chunk=logdet[:, k_start:k_end],
            precision_means_chunk=precision_means[:, k_start:k_end, :],
            mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
        )
    return _compute_tied_chunk_logits(
        x_chunk,
        means[:, k_start:k_end, :],
        variances,
        log_weights[:, k_start:k_end],
        precision=precision,
        logdet=logdet,
        precision_means_chunk=precision_means[:, k_start:k_end, :],
        mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
    )


def _triton_inference_dtype_supported(dtype: torch.dtype) -> bool:
    return dtype in {torch.float16, torch.bfloat16, torch.float32}


def _large_n_triton_inference_supported(
    *,
    covariance_type: str,
    d: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    use_triton: bool,
    labels: bool = False,
) -> bool:
    if not (use_triton and _HAS_TRITON and device.type == "cuda"):
        return False
    if not _triton_inference_dtype_supported(dtype):
        return False
    if covariance_type == "spherical":
        return d >= 16 and triton_spherical_supported(d, k)
    if covariance_type == "diag":
        return 16 <= d <= 64 and k <= 512
    if covariance_type == "full":
        return d <= 16 and k <= 128
    if covariance_type == "tied":
        return (not labels) and d >= 16 and triton_spherical_supported(d, k)
    return False


def _tied_projected_terms(
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    precision, logdet, _, _, log_weights = _matrix_terms(
        means,
        variances,
        weights,
        "tied",
    )
    chol_precision = torch.linalg.cholesky(precision)
    means_projected = torch.bmm(means.to(torch.float32), chol_precision)
    means_projected_sq = means_projected.square().sum(dim=-1)
    unit_variances = torch.ones_like(weights.to(torch.float32))
    return (
        chol_precision,
        means_projected,
        means_projected_sq,
        unit_variances,
        logdet,
        log_weights,
    )


def _resolve_large_n_inference_backend(
    *,
    covariance_type: str,
    d: int,
    k: int,
    dtype: torch.dtype,
    backend: str,
    legacy_no_triton: bool,
    use_triton: bool,
) -> str:
    """Resolve the large-N inference backend for a representative chunk."""
    from . import _dispatch

    effective_legacy_no_triton = legacy_no_triton or (backend == "auto" and not use_triton)
    resolved = _dispatch.resolve_backend_with_env(
        requested=backend,
        covariance=covariance_type,
        shape=(1, 1, d, k),
        dtype=dtype,
        legacy_no_triton=effective_legacy_no_triton,
    )
    if resolved == "triton" and not use_triton:
        return "torch"
    return resolved


def _dominant_backend(counts: dict[str, int]) -> str:
    if not any(counts.values()):
        return "torch"
    return max(("cuda", "triton", "torch"), key=lambda name: (counts[name], name == "cuda", name == "triton"))


def _cuda_inference_terms(
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    covariance_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare CUDA inference parameters once before the chunk loop.

    Returns (means_on_device, variance_or_cholesky, log_weights).
    """
    means_d = means.to(device=device, dtype=dtype).contiguous()
    weights_f = weights.to(device=device, dtype=torch.float32).contiguous()
    log_w = torch.log(weights_f.clamp_min(1e-30))

    if covariance_type in {"spherical", "diag"}:
        var_or_L = variances.to(device=device, dtype=torch.float32).contiguous()
    elif covariance_type == "tied":
        cov = variances.to(device=device, dtype=torch.float32).contiguous()
        var_or_L = torch.linalg.cholesky(cov).contiguous()
    elif covariance_type == "full":
        cov = variances.to(device=device, dtype=torch.float32).contiguous()
        var_or_L, info = torch.linalg.cholesky_ex(cov)
        if bool((info != 0).any().item()):
            raise RuntimeError("full covariance Cholesky failed during CUDA large-N inference setup")
        var_or_L = var_or_L.contiguous()
    else:
        raise ValueError("covariance_type must be 'spherical', 'diag', 'tied', or 'full'")
    return means_d, var_or_L, log_w


def _cuda_assign_chunk(
    x_chunk: torch.Tensor,
    terms: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    covariance_type: str,
) -> torch.Tensor:
    from . import _cuda as _cuda_mod

    means, var_or_L, log_w = terms
    if covariance_type == "spherical":
        return _cuda_mod.spherical_assign(x_chunk, means, var_or_L, log_w)
    if covariance_type == "diag":
        return _cuda_mod.diag_assign(x_chunk, means, var_or_L, log_w)
    if covariance_type == "tied":
        return _cuda_mod.tied_assign(x_chunk, means, var_or_L, log_w)
    if covariance_type == "full":
        return _cuda_mod.full_assign(x_chunk, means, var_or_L, log_w)
    raise ValueError("covariance_type must be 'spherical', 'diag', 'tied', or 'full'")


def _cuda_logsumexp_chunk(
    x_chunk: torch.Tensor,
    terms: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    covariance_type: str,
) -> torch.Tensor:
    from . import _cuda as _cuda_mod

    means, var_or_L, log_w = terms
    if covariance_type == "spherical":
        return _cuda_mod.spherical_logsumexp(x_chunk, means, var_or_L, log_w)
    if covariance_type == "diag":
        return _cuda_mod.diag_logsumexp(x_chunk, means, var_or_L, log_w)
    if covariance_type == "tied":
        return _cuda_mod.tied_logsumexp(x_chunk, means, var_or_L, log_w)
    if covariance_type == "full":
        return _cuda_mod.full_logsumexp(x_chunk, means, var_or_L, log_w)
    raise ValueError("covariance_type must be 'spherical', 'diag', 'tied', or 'full'")


def _cuda_resp_chunk(
    x_chunk: torch.Tensor,
    terms: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    log_norm: torch.Tensor,
    *,
    covariance_type: str,
) -> torch.Tensor:
    from . import _cuda as _cuda_mod

    means, var_or_L, log_w = terms
    if covariance_type == "spherical":
        return _cuda_mod.spherical_resp(x_chunk, means, var_or_L, log_w, log_norm)
    if covariance_type == "diag":
        return _cuda_mod.diag_resp(x_chunk, means, var_or_L, log_w, log_norm)
    if covariance_type == "tied":
        return _cuda_mod.tied_resp(x_chunk, means, var_or_L, log_w, log_norm)
    if covariance_type == "full":
        return _cuda_mod.full_resp(x_chunk, means, var_or_L, log_w, log_norm)
    raise ValueError("covariance_type must be 'spherical', 'diag', 'tied', or 'full'")


def large_n_predict_cpu(
    x_cpu: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    covariance_type: str,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size_N: int,
    chunk_size_K: int,
    use_triton: bool = True,
    return_triton_used: bool = False,
    return_backend_used: bool = False,
    backend: str = "auto",
    legacy_no_triton: bool = False,
) -> torch.LongTensor | tuple[torch.LongTensor, bool] | tuple[torch.LongTensor, str]:
    _validate_large_n_input(x_cpu, device)
    _, n, d = x_cpu.shape
    k = means.shape[1]
    labels = torch.empty((1, n), dtype=torch.long, device=x_cpu.device)
    triton_used = False
    triton_failed = False
    backend_counts = {"cuda": 0, "triton": 0, "torch": 0}
    resolved_backend = _resolve_large_n_inference_backend(
        covariance_type=covariance_type,
        d=d,
        k=k,
        dtype=dtype,
        backend=backend,
        legacy_no_triton=legacy_no_triton,
        use_triton=use_triton,
    )
    cuda_enabled = resolved_backend == "cuda"
    cuda_terms = None
    cuda_mod = None
    if cuda_enabled:
        from . import _cuda as cuda_mod
        try:
            cuda_terms = _cuda_inference_terms(
                means,
                variances,
                weights,
                covariance_type=covariance_type,
                device=device,
                dtype=dtype,
            )
        except Exception:
            cuda_enabled = False
    triton_enabled = _large_n_triton_inference_supported(
        covariance_type=covariance_type,
        d=d,
        k=k,
        dtype=dtype,
        device=device,
        use_triton=use_triton,
        labels=True,
    ) and resolved_backend == "triton"
    triton_terms = None
    if triton_enabled:
        try:
            if covariance_type == "spherical":
                triton_terms = (
                    variances.to(torch.float32),
                    weights.to(torch.float32),
                    means.to(torch.float32).square().sum(dim=-1),
                    torch.log(weights.to(torch.float32)),
                )
            elif covariance_type == "diag":
                triton_terms = _diag_terms(means, variances, weights)
            elif covariance_type == "full":
                triton_terms = _matrix_terms(means, variances, weights, covariance_type)
            else:
                triton_enabled = False
        except Exception:
            triton_enabled = False

    for n_start, n_end, x_chunk, _ in _iter_device_chunks(
        x_cpu,
        device=device,
        dtype=dtype,
        chunk_size_N=chunk_size_N,
    ):
        if cuda_enabled:
            try:
                labels_chunk = _cuda_assign_chunk(
                    x_chunk,
                    cuda_terms,
                    covariance_type=covariance_type,
                )
                labels[:, n_start:n_end] = labels_chunk.to(torch.long).cpu()
                backend_counts["cuda"] += 1
                continue
            except cuda_mod.CudaRuntimeFallback:
                cuda_enabled = False

        if triton_enabled:
            try:
                if covariance_type == "spherical":
                    variances_f, weights_f, means_sq, log_weights = triton_terms
                    x_sq = x_chunk.to(torch.float32).square().sum(dim=-1)
                    labels_chunk = spherical_assign_triton(
                        x_chunk,
                        means,
                        variances_f,
                        weights_f,
                        x_sq=x_sq,
                        means_sq=means_sq,
                        log_weights=log_weights,
                    )
                elif covariance_type == "diag":
                    precision, logdet, weighted_means, mean_precision_mean, log_weights = triton_terms
                    labels_chunk = diag_assign_triton(
                        x_chunk,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                    )
                else:
                    precision, logdet, precision_means, mean_precision_mean, log_weights = triton_terms
                    labels_chunk = full_assign_triton(
                        x_chunk,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                )
                labels[:, n_start:n_end] = labels_chunk.to(torch.long).cpu()
                triton_used = True
                backend_counts["triton"] += 1
                continue
            except Exception:
                triton_failed = True
                triton_enabled = False

        best_logits = torch.full((1, n_end - n_start), -torch.inf, device=device, dtype=torch.float32)
        best_labels = torch.zeros((1, n_end - n_start), dtype=torch.long, device=device)
        terms = _terms_for_chunk_params(
            x_chunk,
            means,
            variances,
            weights,
            covariance_type=covariance_type,
        )
        for k_start in range(0, k, chunk_size_K):
            k_end = min(k_start + chunk_size_K, k)
            logits = _logits_for_tile(
                x_chunk,
                means,
                variances,
                covariance_type=covariance_type,
                k_start=k_start,
                k_end=k_end,
                terms=terms,
            )
            tile_logits, tile_labels = logits.max(dim=-1)
            update_mask = tile_logits > best_logits
            best_logits = torch.where(update_mask, tile_logits, best_logits)
            best_labels = torch.where(update_mask, tile_labels + k_start, best_labels)
        labels[:, n_start:n_end] = best_labels.cpu()
        backend_counts["torch"] += 1
    if return_backend_used:
        return labels, _dominant_backend(backend_counts)
    if return_triton_used:
        return labels, bool(triton_used and not triton_failed)
    return labels


def large_n_score_samples_cpu(
    x_cpu: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    covariance_type: str,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size_N: int,
    chunk_size_K: int,
    use_triton: bool = True,
    backend: str = "auto",
    legacy_no_triton: bool = False,
    return_backend_used: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, str]:
    _validate_large_n_input(x_cpu, device)
    _, n, d = x_cpu.shape
    k = means.shape[1]
    scores = torch.empty((1, n), dtype=torch.float32, device=x_cpu.device)
    backend_counts = {"cuda": 0, "triton": 0, "torch": 0}
    resolved_backend = _resolve_large_n_inference_backend(
        covariance_type=covariance_type,
        d=d,
        k=k,
        dtype=dtype,
        backend=backend,
        legacy_no_triton=legacy_no_triton,
        use_triton=use_triton,
    )
    cuda_enabled = resolved_backend == "cuda"
    cuda_terms = None
    cuda_mod = None
    if cuda_enabled:
        from . import _cuda as cuda_mod
        try:
            cuda_terms = _cuda_inference_terms(
                means,
                variances,
                weights,
                covariance_type=covariance_type,
                device=device,
                dtype=dtype,
            )
        except Exception:
            cuda_enabled = False
    triton_enabled = _large_n_triton_inference_supported(
        covariance_type=covariance_type,
        d=d,
        k=k,
        dtype=dtype,
        device=device,
        use_triton=use_triton,
    ) and resolved_backend == "triton"
    triton_terms = None
    log_norm_buffer = None
    if triton_enabled:
        try:
            if covariance_type == "spherical":
                triton_terms = (
                    variances.to(torch.float32),
                    weights.to(torch.float32),
                    means.to(torch.float32).square().sum(dim=-1),
                    torch.log(weights.to(torch.float32)),
                )
            elif covariance_type == "diag":
                triton_terms = _diag_terms(means, variances, weights)
            elif covariance_type == "full":
                triton_terms = _matrix_terms(means, variances, weights, covariance_type)
            else:
                triton_terms = _tied_projected_terms(means, variances, weights)
            log_norm_buffer = torch.empty(
                (1, min(chunk_size_N, n)),
                device=device,
                dtype=torch.float32,
            )
        except Exception:
            triton_enabled = False

    for n_start, n_end, x_chunk, _ in _iter_device_chunks(
        x_cpu,
        device=device,
        dtype=dtype,
        chunk_size_N=chunk_size_N,
    ):
        if cuda_enabled:
            try:
                log_norm = _cuda_logsumexp_chunk(
                    x_chunk,
                    cuda_terms,
                    covariance_type=covariance_type,
                )
                scores[:, n_start:n_end] = log_norm.cpu()
                backend_counts["cuda"] += 1
                continue
            except cuda_mod.CudaRuntimeFallback:
                cuda_enabled = False

        if triton_enabled:
            try:
                out = log_norm_buffer[:, : n_end - n_start]
                if covariance_type == "spherical":
                    variances_f, weights_f, means_sq, log_weights = triton_terms
                    x_sq = x_chunk.to(torch.float32).square().sum(dim=-1)
                    log_norm = spherical_logsumexp_triton(
                        x_chunk,
                        means,
                        variances_f,
                        weights_f,
                        x_sq=x_sq,
                        out=out,
                        means_sq=means_sq,
                        log_weights=log_weights,
                    )
                elif covariance_type == "diag":
                    precision, logdet, weighted_means, mean_precision_mean, log_weights = triton_terms
                    log_norm = diag_logsumexp_triton(
                        x_chunk,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        out=out,
                    )
                elif covariance_type == "full":
                    precision, logdet, precision_means, mean_precision_mean, log_weights = triton_terms
                    log_norm = full_logsumexp_triton(
                        x_chunk,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        out=out,
                    )
                else:
                    (
                        chol_precision,
                        means_projected,
                        means_projected_sq,
                        unit_variances,
                        logdet,
                        log_weights,
                    ) = triton_terms
                    x_projected = torch.bmm(x_chunk.to(torch.float32), chol_precision)
                    x_projected_sq = x_projected.square().sum(dim=-1)
                    log_norm = spherical_logsumexp_triton(
                        x_projected,
                        means_projected,
                        unit_variances,
                        weights.to(torch.float32),
                        x_sq=x_projected_sq,
                        out=out,
                        means_sq=means_projected_sq,
                        log_weights=log_weights,
                        config=_triton_tied_logsum_config(d, k),
                        unit_variance=True,
                    )
                    log_norm = log_norm - 0.5 * logdet.unsqueeze(-1)
                scores[:, n_start:n_end] = log_norm.cpu()
                backend_counts["triton"] += 1
                continue
            except Exception:
                triton_enabled = False

        log_norm, _ = _stream_log_norm_for_params(
            x_chunk,
            means,
            variances,
            weights,
            covariance_type=covariance_type,
            chunk_size_K=chunk_size_K,
        )
        scores[:, n_start:n_end] = log_norm.cpu()
        backend_counts["torch"] += 1
    if return_backend_used:
        return scores, _dominant_backend(backend_counts)
    return scores


def large_n_predict_proba_cpu(
    x_cpu: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    covariance_type: str,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size_N: int,
    chunk_size_K: int,
    use_triton: bool = True,
    backend: str = "auto",
    legacy_no_triton: bool = False,
    return_backend_used: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, str]:
    _validate_large_n_input(x_cpu, device)
    _, n, d = x_cpu.shape
    k = means.shape[1]
    probs = torch.empty((1, n, k), dtype=torch.float32, device=x_cpu.device)
    backend_counts = {"cuda": 0, "triton": 0, "torch": 0}
    resolved_backend = _resolve_large_n_inference_backend(
        covariance_type=covariance_type,
        d=d,
        k=k,
        dtype=dtype,
        backend=backend,
        legacy_no_triton=legacy_no_triton,
        use_triton=use_triton,
    )
    cuda_enabled = resolved_backend == "cuda"
    cuda_terms = None
    cuda_mod = None
    if cuda_enabled:
        from . import _cuda as cuda_mod
        try:
            cuda_terms = _cuda_inference_terms(
                means,
                variances,
                weights,
                covariance_type=covariance_type,
                device=device,
                dtype=dtype,
            )
        except Exception:
            cuda_enabled = False
    triton_enabled = _large_n_triton_inference_supported(
        covariance_type=covariance_type,
        d=d,
        k=k,
        dtype=dtype,
        device=device,
        use_triton=use_triton,
    ) and resolved_backend == "triton"
    if covariance_type == "tied":
        triton_enabled = False
    triton_terms = None
    log_norm_buffer = None
    probs_buffer = None
    if triton_enabled:
        try:
            if covariance_type == "spherical":
                triton_terms = (
                    variances.to(torch.float32),
                    weights.to(torch.float32),
                    means.to(torch.float32).square().sum(dim=-1),
                    torch.log(weights.to(torch.float32)),
                )
            elif covariance_type == "diag":
                triton_terms = _diag_terms(means, variances, weights)
            elif covariance_type == "full":
                triton_terms = _matrix_terms(means, variances, weights, covariance_type)
            else:
                triton_terms = _tied_projected_terms(means, variances, weights)
            max_chunk_n = min(chunk_size_N, n)
            log_norm_buffer = torch.empty((1, max_chunk_n), device=device, dtype=torch.float32)
            probs_buffer = torch.empty((1, max_chunk_n, k), device=device, dtype=torch.float32)
        except Exception:
            triton_enabled = False

    for n_start, n_end, x_chunk, _ in _iter_device_chunks(
        x_cpu,
        device=device,
        dtype=dtype,
        chunk_size_N=chunk_size_N,
    ):
        if cuda_enabled:
            try:
                log_norm = _cuda_logsumexp_chunk(
                    x_chunk,
                    cuda_terms,
                    covariance_type=covariance_type,
                )
                resp = _cuda_resp_chunk(
                    x_chunk,
                    cuda_terms,
                    log_norm,
                    covariance_type=covariance_type,
                )
                probs[:, n_start:n_end, :] = resp.cpu()
                backend_counts["cuda"] += 1
                continue
            except cuda_mod.CudaRuntimeFallback:
                cuda_enabled = False

        if triton_enabled:
            try:
                rows = n_end - n_start
                log_out = log_norm_buffer[:, :rows]
                prob_out = probs_buffer[:, :rows, :]
                if covariance_type == "spherical":
                    variances_f, weights_f, means_sq, log_weights = triton_terms
                    x_sq = x_chunk.to(torch.float32).square().sum(dim=-1)
                    log_norm = spherical_logsumexp_triton(
                        x_chunk,
                        means,
                        variances_f,
                        weights_f,
                        x_sq=x_sq,
                        out=log_out,
                        means_sq=means_sq,
                        log_weights=log_weights,
                    )
                    resp = spherical_resp_triton(
                        x_chunk,
                        means,
                        variances_f,
                        weights_f,
                        log_norm,
                        x_sq=x_sq,
                        out=prob_out,
                        means_sq=means_sq,
                        log_weights=log_weights,
                    )
                elif covariance_type == "diag":
                    precision, logdet, weighted_means, mean_precision_mean, log_weights = triton_terms
                    log_norm = diag_logsumexp_triton(
                        x_chunk,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        out=log_out,
                    )
                    resp = diag_resp_triton(
                        x_chunk,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        log_norm,
                        out=prob_out,
                    )
                elif covariance_type == "full":
                    precision, logdet, precision_means, mean_precision_mean, log_weights = triton_terms
                    log_norm = full_logsumexp_triton(
                        x_chunk,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        out=log_out,
                    )
                    resp = full_resp_triton(
                        x_chunk,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        log_norm,
                        out=prob_out,
                    )
                else:
                    (
                        chol_precision,
                        means_projected,
                        means_projected_sq,
                        unit_variances,
                        _,
                        log_weights,
                    ) = triton_terms
                    x_projected = torch.bmm(x_chunk.to(torch.float32), chol_precision)
                    x_projected_sq = x_projected.square().sum(dim=-1)
                    tied_config = _triton_tied_logsum_config(d, k)
                    log_norm = spherical_logsumexp_triton(
                        x_projected,
                        means_projected,
                        unit_variances,
                        weights.to(torch.float32),
                        x_sq=x_projected_sq,
                        out=log_out,
                        means_sq=means_projected_sq,
                        log_weights=log_weights,
                        config=tied_config,
                        unit_variance=True,
                    )
                    resp = spherical_resp_triton(
                        x_projected,
                        means_projected,
                        unit_variances,
                        weights.to(torch.float32),
                        log_norm,
                        x_sq=x_projected_sq,
                        out=prob_out,
                        means_sq=means_projected_sq,
                        log_weights=log_weights,
                        config=tied_config,
                    )
                probs[:, n_start:n_end, :] = resp.cpu()
                backend_counts["triton"] += 1
                continue
            except Exception:
                triton_enabled = False

        log_norm, terms = _stream_log_norm_for_params(
            x_chunk,
            means,
            variances,
            weights,
            covariance_type=covariance_type,
            chunk_size_K=chunk_size_K,
        )
        for k_start in range(0, k, chunk_size_K):
            k_end = min(k_start + chunk_size_K, k)
            logits = _logits_for_tile(
                x_chunk,
                means,
                variances,
                covariance_type=covariance_type,
                k_start=k_start,
                k_end=k_end,
                terms=terms,
            )
            probs[:, n_start:n_end, k_start:k_end] = torch.exp(
                logits - log_norm.unsqueeze(-1)
            ).cpu()
        backend_counts["torch"] += 1
    if return_backend_used:
        return probs, _dominant_backend(backend_counts)
    return probs
