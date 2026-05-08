from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from ._runtime import triton_spherical_supported
from ._flash_kmeans import load_flash_kmeans

try:
    from .assign_spherical_triton import (
        spherical_assign_triton,
        spherical_logsumexp_triton,
        spherical_resp_triton,
    )
    _HAS_TRITON_ASSIGN = True
except Exception:
    _HAS_TRITON_ASSIGN = False

try:
    from .assign_diag_triton import diag_logsumexp_triton
    _HAS_TRITON_DIAG_ASSIGN = True
except Exception:
    _HAS_TRITON_DIAG_ASSIGN = False

try:
    from .assign_full_triton import full_logsumexp_triton
    _HAS_TRITON_FULL_ASSIGN = True
except Exception:
    _HAS_TRITON_FULL_ASSIGN = False

try:
    from .weighted_update_triton import (
        triton_blocked_update_diag,
        triton_blocked_update_full,
        triton_blocked_update_spherical,
        triton_blocked_update_tied_projected,
    )
    _HAS_TRITON_WEIGHTED_UPDATE = True
except Exception:
    _HAS_TRITON_WEIGHTED_UPDATE = False

try:
    from .fused_update_triton import (
        fused_single_tile_update_config,
        triton_fused_single_tile_update_diag,
        triton_fused_single_tile_update_spherical,
        triton_fused_single_tile_update_tied_native,
    )
    _HAS_TRITON_FUSED_UPDATE = True
except Exception:
    _HAS_TRITON_FUSED_UPDATE = False

try:
    from .approx_update_triton import (
        approx_topk_update_spherical_config,
        triton_approx_topk_update_spherical,
    )
    _HAS_TRITON_APPROX_UPDATE = True
except Exception:
    _HAS_TRITON_APPROX_UPDATE = False

LOG_2PI = math.log(2.0 * math.pi)
KMEANS_PLUS_PLUS_MAX_COMPONENTS = 256
KMEANS_PLUS_PLUS_MAX_SAMPLES = 65536
KMEANS_PLUS_PLUS_MIN_SAMPLES = 4096
KMEANS_PLUS_PLUS_SAMPLES_PER_COMPONENT = 64
KMEANS_PLUS_PLUS_VECTORIZE_MAX_ELEMENTS = 64 * 1024 * 1024


def _auto_use_triton_estep(x: torch.Tensor, n_components: int) -> bool:
    if not (_HAS_TRITON_ASSIGN and x.is_cuda):
        return False
    n = x.shape[1]
    d = x.shape[-1]
    if not triton_spherical_supported(d, n_components):
        return False
    return bool(d <= 64 or (d <= 128 and (n_components >= 128 or n >= 131072)))


def _auto_use_triton_labels(x: torch.Tensor, n_components: int) -> bool:
    if not (_HAS_TRITON_ASSIGN and x.is_cuda):
        return False
    d = x.shape[-1]
    return triton_spherical_supported(d, n_components)


def _auto_use_triton_streaming_update(x: torch.Tensor, n_components: int) -> bool:
    if not (_HAS_TRITON_WEIGHTED_UPDATE and x.is_cuda):
        return False
    n = x.shape[1]
    d = x.shape[-1]
    if not triton_spherical_supported(d, n_components):
        return False
    return bool((d <= 64 and n >= 131072) or (d <= 128 and n >= 131072))


def _auto_triton_streaming_chunk_size(n: int, d: int, requested_chunk_size: int) -> int:
    """Use larger chunks for flash-style spherical EM to reduce launch overhead.

    The PyTorch fallback keeps the requested chunk size. This is only applied to
    the Triton streaming path where memory pressure is bounded by log-normalizers
    and sufficient statistics instead of a materialized N x K responsibility
    tensor.
    """
    if d > 64 or n < 131072:
        return requested_chunk_size
    if n >= 2097152:
        return max(requested_chunk_size, min(n, 2097152))
    if n >= 1048576:
        return max(requested_chunk_size, min(n, 1048576))
    return max(requested_chunk_size, min(n, 524288))


def _auto_torch_spherical_chunk_size(
    n: int,
    d: int,
    n_components: int,
    requested_chunk_size: int,
) -> int:
    """Narrow high-D PyTorch fit heuristic from local CUDA benchmarks."""
    if requested_chunk_size != 32768:
        return requested_chunk_size
    if d == 256 and n_components <= 64 and n >= 1048576:
        return 65536
    return requested_chunk_size


def _auto_tied_chunk_size(
    n: int,
    n_components: int,
    requested_chunk_size: int,
) -> int:
    """Use larger CUDA chunks for tied covariance when memory stays bounded."""
    if requested_chunk_size != 32768:
        return requested_chunk_size
    target_resp_elements = 8 * 1024 * 1024
    target_chunk = target_resp_elements // max(n_components, 1)
    if target_chunk < requested_chunk_size:
        return requested_chunk_size
    target_chunk = min(target_chunk, 131072)
    return max(requested_chunk_size, min(n, target_chunk))


def _auto_use_triton_diag(x: torch.Tensor, n_components: int) -> bool:
    if not (_HAS_TRITON_DIAG_ASSIGN and _HAS_TRITON_WEIGHTED_UPDATE and x.is_cuda):
        return False
    n = x.shape[1]
    return bool(_triton_diag_supported(x, n_components) and n >= 65536)


def _triton_diag_supported(x: torch.Tensor, n_components: int) -> bool:
    if not (_HAS_TRITON_DIAG_ASSIGN and _HAS_TRITON_WEIGHTED_UPDATE and x.is_cuda):
        return False
    d = x.shape[-1]
    return bool(d <= 64 and n_components <= 512)


def _auto_use_triton_tied(x: torch.Tensor, n_components: int) -> bool:
    if not (_HAS_TRITON_ASSIGN and _HAS_TRITON_WEIGHTED_UPDATE and x.is_cuda):
        return False
    n = x.shape[1]
    return bool(_triton_tied_supported(x, n_components) and n >= 65536)


def _triton_tied_supported(x: torch.Tensor, n_components: int) -> bool:
    if not (_HAS_TRITON_ASSIGN and _HAS_TRITON_WEIGHTED_UPDATE and x.is_cuda):
        return False
    d = x.shape[-1]
    return bool(d <= 64 and n_components <= 512)


def _auto_use_triton_full(x: torch.Tensor, n_components: int) -> bool:
    if not (_HAS_TRITON_FULL_ASSIGN and _HAS_TRITON_WEIGHTED_UPDATE and x.is_cuda):
        return False
    n = x.shape[1]
    d = x.shape[-1]
    return bool(d <= 8 and _triton_full_supported(x, n_components) and n >= 131072 and n_components >= 32)


def _triton_full_supported(x: torch.Tensor, n_components: int) -> bool:
    if not (_HAS_TRITON_FULL_ASSIGN and _HAS_TRITON_WEIGHTED_UPDATE and x.is_cuda):
        return False
    d = x.shape[-1]
    return bool(d <= 8 and n_components <= 128)


def _triton_diag_update_config(d: int, n_components: int) -> tuple[int, int, int]:
    if d <= 16:
        return 64, 16, 64
    if d <= 32:
        return 64, 32, 64
    return 32, 64, 32


def _triton_tied_update_config(d: int, n_components: int) -> tuple[int, int, int]:
    if d <= 32:
        return 64, 32, 64
    return 64, 64, 32


def _triton_tied_logsum_config(d: int, n_components: int) -> dict[str, int] | None:
    if d > 32 and d <= 64 and 128 <= n_components <= 256:
        return {"BLOCK_N": 32, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
    if d <= 64 and n_components <= 256:
        return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
    return None


def _triton_full_update_config(d: int, n_components: int) -> tuple[int, int, int]:
    return 128, 16, 32


def _resolve_triton_option(value: bool | str, auto_value: bool, supported: bool, name: str) -> bool:
    if value == "auto":
        return bool(auto_value)
    if isinstance(value, bool):
        return bool(value and supported)
    raise ValueError(f"{name} must be a bool or 'auto'")


def _triton_blocked_update_config(d: int, n_components: int) -> tuple[int, int, int]:
    if d > 128:
        return 16, 256, 16
    if d > 64:
        block_n = 64 if n_components >= 2048 else 32
        return block_n, 128, 32
    if d <= 32:
        return 64, 32, 64
    return 64, 64, 32


def _check_batched_inputs(x: torch.Tensor, means: torch.Tensor, variances: torch.Tensor, weights: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError("x must have shape (B, N, D)")
    if means.ndim != 3:
        raise ValueError("means must have shape (B, K, D)")
    if variances.ndim != 2:
        raise ValueError("variances must have shape (B, K)")
    if weights.ndim != 2:
        raise ValueError("weights must have shape (B, K)")
    if x.shape[0] != means.shape[0] or x.shape[0] != variances.shape[0] or x.shape[0] != weights.shape[0]:
        raise ValueError("batch size mismatch between x and GMM parameters")
    if x.shape[2] != means.shape[2]:
        raise ValueError("feature dimension mismatch between x and means")
    if means.shape[1] != variances.shape[1] or means.shape[1] != weights.shape[1]:
        raise ValueError("component count mismatch between means, variances, and weights")


def _check_batched_diag_inputs(x: torch.Tensor, means: torch.Tensor, variances: torch.Tensor, weights: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError("x must have shape (B, N, D)")
    if means.ndim != 3:
        raise ValueError("means must have shape (B, K, D)")
    if variances.ndim != 3:
        raise ValueError("diag variances must have shape (B, K, D)")
    if weights.ndim != 2:
        raise ValueError("weights must have shape (B, K)")
    if x.shape[0] != means.shape[0] or x.shape[0] != variances.shape[0] or x.shape[0] != weights.shape[0]:
        raise ValueError("batch size mismatch between x and GMM parameters")
    if x.shape[2] != means.shape[2] or x.shape[2] != variances.shape[2]:
        raise ValueError("feature dimension mismatch between x, means, and variances")
    if means.shape[1] != variances.shape[1] or means.shape[1] != weights.shape[1]:
        raise ValueError("component count mismatch between means, variances, and weights")


def _check_batched_full_inputs(x: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError("x must have shape (B, N, D)")
    if means.ndim != 3:
        raise ValueError("means must have shape (B, K, D)")
    if covariances.ndim != 4:
        raise ValueError("full covariances must have shape (B, K, D, D)")
    if weights.ndim != 2:
        raise ValueError("weights must have shape (B, K)")
    if x.shape[0] != means.shape[0] or x.shape[0] != covariances.shape[0] or x.shape[0] != weights.shape[0]:
        raise ValueError("batch size mismatch between x and GMM parameters")
    if x.shape[2] != means.shape[2] or x.shape[2] != covariances.shape[2] or x.shape[2] != covariances.shape[3]:
        raise ValueError("feature dimension mismatch between x, means, and covariances")
    if means.shape[1] != covariances.shape[1] or means.shape[1] != weights.shape[1]:
        raise ValueError("component count mismatch between means, covariances, and weights")


def _check_batched_tied_inputs(x: torch.Tensor, means: torch.Tensor, covariance: torch.Tensor, weights: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError("x must have shape (B, N, D)")
    if means.ndim != 3:
        raise ValueError("means must have shape (B, K, D)")
    if covariance.ndim != 3:
        raise ValueError("tied covariance must have shape (B, D, D)")
    if weights.ndim != 2:
        raise ValueError("weights must have shape (B, K)")
    if x.shape[0] != means.shape[0] or x.shape[0] != covariance.shape[0] or x.shape[0] != weights.shape[0]:
        raise ValueError("batch size mismatch between x and GMM parameters")
    if x.shape[2] != means.shape[2] or x.shape[2] != covariance.shape[1] or x.shape[2] != covariance.shape[2]:
        raise ValueError("feature dimension mismatch between x, means, and covariance")
    if means.shape[1] != weights.shape[1]:
        raise ValueError("component count mismatch between means and weights")


def _eye_like_covariance(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(d, device=device, dtype=dtype)


def _symmetrize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _regularize_matrix(matrix: torch.Tensor, reg_covar: float) -> torch.Tensor:
    d = matrix.shape[-1]
    eye = _eye_like_covariance(d, matrix.device, matrix.dtype)
    return _symmetrize_matrix(matrix) + reg_covar * eye


def _precision_and_logdet(covariances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cov = _symmetrize_matrix(covariances.to(torch.float32))
    d = cov.shape[-1]
    try:
        chol = torch.linalg.cholesky(cov)
    except RuntimeError:
        eye = _eye_like_covariance(d, cov.device, cov.dtype)
        jitter = 1e-6
        chol = None
        for _ in range(6):
            chol, info = torch.linalg.cholesky_ex(cov + jitter * eye)
            if bool((info == 0).all().item()):
                break
            jitter *= 10.0
        if chol is None:
            chol = torch.linalg.cholesky(cov + jitter * eye)
    precision = torch.cholesky_inverse(chol)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    return precision, logdet


def _compute_chunk_logits(
    x_chunk: torch.Tensor,
    x_sq_chunk: torch.Tensor,
    means_chunk: torch.Tensor,
    variances_chunk: torch.Tensor,
    log_weights_chunk: torch.Tensor,
) -> torch.Tensor:
    _, _, d = x_chunk.shape
    means_sq = means_chunk.square().sum(dim=-1)
    dist_sq = (
        x_sq_chunk.unsqueeze(-1)
        - 2.0 * torch.bmm(x_chunk, means_chunk.transpose(1, 2))
        + means_sq.unsqueeze(-2)
    ).clamp_min_(0.0)
    log_norm = d * (LOG_2PI + torch.log(variances_chunk)).unsqueeze(-2)
    return log_weights_chunk.unsqueeze(-2) - 0.5 * (
        dist_sq / variances_chunk.unsqueeze(-2) + log_norm
    )


def _compute_diag_chunk_logits(
    x_chunk: torch.Tensor,
    means_chunk: torch.Tensor,
    variances_chunk: torch.Tensor,
    log_weights_chunk: torch.Tensor,
    *,
    precision_chunk: Optional[torch.Tensor] = None,
    logdet_chunk: Optional[torch.Tensor] = None,
    weighted_means_chunk: Optional[torch.Tensor] = None,
    mean_precision_mean_chunk: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _, _, d = x_chunk.shape
    x_f = x_chunk.to(torch.float32)
    means_f = means_chunk.to(torch.float32)
    precision = (
        precision_chunk
        if precision_chunk is not None
        else variances_chunk.to(torch.float32).clamp_min(1e-30).reciprocal()
    )
    weighted_means = (
        weighted_means_chunk
        if weighted_means_chunk is not None
        else means_f * precision
    )
    mean_precision_mean = (
        mean_precision_mean_chunk
        if mean_precision_mean_chunk is not None
        else (means_f * weighted_means).sum(dim=-1)
    )
    dist_sq = (
        torch.bmm(x_f.square(), precision.transpose(1, 2))
        - 2.0 * torch.bmm(x_f, weighted_means.transpose(1, 2))
        + mean_precision_mean.unsqueeze(1)
    ).clamp_min_(0.0)
    logdet = (
        logdet_chunk
        if logdet_chunk is not None
        else torch.log(variances_chunk.to(torch.float32).clamp_min(1e-30)).sum(dim=-1)
    )
    log_norm = d * LOG_2PI + logdet
    return log_weights_chunk.unsqueeze(1) - 0.5 * (dist_sq + log_norm.unsqueeze(1))


def _compute_full_chunk_logits(
    x_chunk: torch.Tensor,
    means_chunk: torch.Tensor,
    covariances_chunk: torch.Tensor,
    log_weights_chunk: torch.Tensor,
    *,
    precision_chunk: Optional[torch.Tensor] = None,
    logdet_chunk: Optional[torch.Tensor] = None,
    precision_means_chunk: Optional[torch.Tensor] = None,
    mean_precision_mean_chunk: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _, _, d = x_chunk.shape
    x_f = x_chunk.to(torch.float32)
    means_f = means_chunk.to(torch.float32)
    if precision_chunk is None or logdet_chunk is None:
        precision, logdet = _precision_and_logdet(covariances_chunk)
    else:
        precision, logdet = precision_chunk, logdet_chunk
    x_p_x = torch.einsum("bnd,bkde,bne->bnk", x_f, precision, x_f)
    precision_means = (
        precision_means_chunk
        if precision_means_chunk is not None
        else torch.einsum("bkde,bke->bkd", precision, means_f)
    )
    cross = torch.bmm(x_f, precision_means.transpose(1, 2))
    mean_p_mean = (
        mean_precision_mean_chunk
        if mean_precision_mean_chunk is not None
        else (means_f * precision_means).sum(dim=-1)
    )
    quad = (x_p_x - 2.0 * cross + mean_p_mean.unsqueeze(1)).clamp_min_(0.0)
    log_norm = d * LOG_2PI + logdet
    return log_weights_chunk.unsqueeze(1) - 0.5 * (quad + log_norm.unsqueeze(1))


def _compute_tied_chunk_logits(
    x_chunk: torch.Tensor,
    means_chunk: torch.Tensor,
    covariance: torch.Tensor,
    log_weights_chunk: torch.Tensor,
    *,
    precision: Optional[torch.Tensor] = None,
    logdet: Optional[torch.Tensor] = None,
    precision_means_chunk: Optional[torch.Tensor] = None,
    mean_precision_mean_chunk: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _, _, d = x_chunk.shape
    x_f = x_chunk.to(torch.float32)
    means_f = means_chunk.to(torch.float32)
    if precision is None or logdet is None:
        precision, logdet = _precision_and_logdet(covariance)
    x_p_x = torch.einsum("bnd,bde,bne->bn", x_f, precision, x_f)
    precision_means = (
        precision_means_chunk
        if precision_means_chunk is not None
        else torch.bmm(means_f, precision.transpose(1, 2))
    )
    cross = torch.bmm(x_f, precision_means.transpose(1, 2))
    mean_p_mean = (
        mean_precision_mean_chunk
        if mean_precision_mean_chunk is not None
        else (means_f * precision_means).sum(dim=-1)
    )
    quad = (x_p_x.unsqueeze(-1) - 2.0 * cross + mean_p_mean.unsqueeze(1)).clamp_min_(0.0)
    log_norm = d * LOG_2PI + logdet
    return log_weights_chunk.unsqueeze(1) - 0.5 * (quad + log_norm[:, None, None])


def _stream_log_normalizer(
    x_chunk: torch.Tensor,
    x_sq_chunk: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_K: int,
) -> torch.Tensor:
    bsz, n_chunk, _ = x_chunk.shape
    k = means.shape[1]
    max_logits = torch.full((bsz, n_chunk), -torch.inf, device=x_chunk.device, dtype=x_chunk.dtype)
    exp_sums = torch.zeros((bsz, n_chunk), device=x_chunk.device, dtype=x_chunk.dtype)
    log_weights = torch.log(weights)

    for k_start in range(0, k, chunk_size_K):
        k_end = min(k_start + chunk_size_K, k)
        logits = _compute_chunk_logits(
            x_chunk,
            x_sq_chunk,
            means[:, k_start:k_end, :],
            variances[:, k_start:k_end],
            log_weights[:, k_start:k_end],
        )
        tile_max = logits.max(dim=-1).values
        new_max = torch.maximum(max_logits, tile_max)
        exp_sums = exp_sums * torch.exp(max_logits - new_max) + torch.exp(
            logits - new_max.unsqueeze(-1)
        ).sum(dim=-1)
        max_logits = new_max

    return max_logits + torch.log(exp_sums)


def _matrix_stream_log_normalizer(
    x_chunk: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_K: int,
    covariance_type: str,
    precision: Optional[torch.Tensor] = None,
    logdet: Optional[torch.Tensor] = None,
    log_weights: Optional[torch.Tensor] = None,
    precision_means: Optional[torch.Tensor] = None,
    mean_precision_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz, n_chunk, _ = x_chunk.shape
    k = means.shape[1]
    max_logits = torch.full((bsz, n_chunk), -torch.inf, device=x_chunk.device, dtype=torch.float32)
    exp_sums = torch.zeros((bsz, n_chunk), device=x_chunk.device, dtype=torch.float32)
    log_weights = torch.log(weights.to(torch.float32)) if log_weights is None else log_weights

    for k_start in range(0, k, chunk_size_K):
        k_end = min(k_start + chunk_size_K, k)
        if covariance_type == "full":
            logits = _compute_full_chunk_logits(
                x_chunk,
                means[:, k_start:k_end, :],
                covariances[:, k_start:k_end, :, :],
                log_weights[:, k_start:k_end],
                precision_chunk=None if precision is None else precision[:, k_start:k_end, :, :],
                logdet_chunk=None if logdet is None else logdet[:, k_start:k_end],
                precision_means_chunk=None if precision_means is None else precision_means[:, k_start:k_end, :],
                mean_precision_mean_chunk=None if mean_precision_mean is None else mean_precision_mean[:, k_start:k_end],
            )
        elif covariance_type == "tied":
            logits = _compute_tied_chunk_logits(
                x_chunk,
                means[:, k_start:k_end, :],
                covariances,
                log_weights[:, k_start:k_end],
                precision=precision,
                logdet=logdet,
                precision_means_chunk=None if precision_means is None else precision_means[:, k_start:k_end, :],
                mean_precision_mean_chunk=None if mean_precision_mean is None else mean_precision_mean[:, k_start:k_end],
            )
        else:
            raise ValueError("covariance_type must be 'full' or 'tied'")
        tile_max = logits.max(dim=-1).values
        new_max = torch.maximum(max_logits, tile_max)
        exp_sums = exp_sums * torch.exp(max_logits - new_max) + torch.exp(
            logits - new_max.unsqueeze(-1)
        ).sum(dim=-1)
        max_logits = new_max

    return max_logits + torch.log(exp_sums)


def _diag_stream_log_normalizer(
    x_chunk: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_K: int,
    precision: Optional[torch.Tensor] = None,
    logdet: Optional[torch.Tensor] = None,
    log_weights: Optional[torch.Tensor] = None,
    weighted_means: Optional[torch.Tensor] = None,
    mean_precision_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz, n_chunk, _ = x_chunk.shape
    k = means.shape[1]
    max_logits = torch.full((bsz, n_chunk), -torch.inf, device=x_chunk.device, dtype=torch.float32)
    exp_sums = torch.zeros((bsz, n_chunk), device=x_chunk.device, dtype=torch.float32)
    log_weights = torch.log(weights.to(torch.float32)) if log_weights is None else log_weights

    for k_start in range(0, k, chunk_size_K):
        k_end = min(k_start + chunk_size_K, k)
        logits = _compute_diag_chunk_logits(
            x_chunk,
            means[:, k_start:k_end, :],
            variances[:, k_start:k_end, :],
            log_weights[:, k_start:k_end],
            precision_chunk=None if precision is None else precision[:, k_start:k_end, :],
            logdet_chunk=None if logdet is None else logdet[:, k_start:k_end],
            weighted_means_chunk=None if weighted_means is None else weighted_means[:, k_start:k_end, :],
            mean_precision_mean_chunk=None if mean_precision_mean is None else mean_precision_mean[:, k_start:k_end],
        )
        tile_max = logits.max(dim=-1).values
        new_max = torch.maximum(max_logits, tile_max)
        exp_sums = exp_sums * torch.exp(max_logits - new_max) + torch.exp(
            logits - new_max.unsqueeze(-1)
        ).sum(dim=-1)
        max_logits = new_max

    return max_logits + torch.log(exp_sums)


def _resolve_approx_top_k(approx_top_k: Optional[int], n_components: int) -> Optional[int]:
    if approx_top_k is None:
        return None
    top_k = int(approx_top_k)
    if top_k <= 0:
        raise ValueError("approx_top_k must be positive or None")
    if top_k >= n_components:
        return None
    return top_k


def _compute_approx_tile_logits(
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
        ).to(torch.float32)
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
        ).to(torch.float32)
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
        ).to(torch.float32)
    if covariance_type == "tied":
        return _compute_tied_chunk_logits(
            x_chunk,
            means[:, k_start:k_end, :],
            variances,
            log_weights[:, k_start:k_end],
            precision=precision,
            logdet=logdet,
            precision_means_chunk=precision_means[:, k_start:k_end, :],
            mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
        ).to(torch.float32)
    raise ValueError("covariance_type must be 'spherical', 'diag', 'tied', or 'full'")


def _topk_logits_for_chunk(
    x_chunk: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    *,
    covariance_type: str,
    chunk_size_K: int,
    top_k: int,
    terms,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    bsz, n_chunk, _ = x_chunk.shape
    n_components = means.shape[1]
    top_k = min(int(top_k), n_components)
    best_logits = torch.full(
        (bsz, n_chunk, top_k),
        -torch.inf,
        device=x_chunk.device,
        dtype=torch.float32,
    )
    best_indices = torch.zeros(
        (bsz, n_chunk, top_k),
        device=x_chunk.device,
        dtype=torch.long,
    )

    for k_start in range(0, n_components, chunk_size_K):
        k_end = min(k_start + chunk_size_K, n_components)
        logits = _compute_approx_tile_logits(
            x_chunk,
            means,
            variances,
            covariance_type=covariance_type,
            k_start=k_start,
            k_end=k_end,
            terms=terms,
        )
        tile_k = k_end - k_start
        tile_indices = torch.arange(
            k_start,
            k_end,
            device=x_chunk.device,
            dtype=torch.long,
        ).view(1, 1, tile_k)
        tile_indices = tile_indices.expand(bsz, n_chunk, tile_k)
        candidate_logits = torch.cat((best_logits, logits), dim=-1)
        candidate_indices = torch.cat((best_indices, tile_indices), dim=-1)
        best_logits, positions = candidate_logits.topk(top_k, dim=-1)
        best_indices = candidate_indices.gather(-1, positions)

    return best_logits, best_indices


def _accumulate_topk_stats(
    x_chunk: torch.Tensor,
    topk_logits: torch.Tensor,
    topk_indices: torch.LongTensor,
    *,
    covariance_type: str,
    nk: torch.Tensor,
    sum_x: torch.Tensor,
    total_log_likelihood: torch.Tensor,
    x_sq: Optional[torch.Tensor] = None,
    sum_x_sq: Optional[torch.Tensor] = None,
    sum_xx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    log_norm = torch.logsumexp(topk_logits, dim=-1)
    resp = torch.exp(topk_logits - log_norm.unsqueeze(-1))
    total_log_likelihood = total_log_likelihood + log_norm.sum()
    x_f = x_chunk.to(torch.float32)
    x_f_square = None
    outer = None
    if covariance_type == "diag":
        if sum_x_sq is None:
            raise ValueError("sum_x_sq is required for diagonal approximate EM")
        x_f_square = x_f.square()
    elif covariance_type == "spherical":
        if x_sq is None or sum_x_sq is None:
            raise ValueError("x_sq and sum_x_sq are required for spherical approximate EM")
    elif covariance_type == "full":
        if sum_xx is None:
            raise ValueError("sum_xx is required for full approximate EM")
        outer = x_f.unsqueeze(-1) * x_f.unsqueeze(-2)
    elif covariance_type != "tied":
        raise ValueError("covariance_type must be 'spherical', 'diag', 'tied', or 'full'")

    d = x_f.shape[-1]
    for local_idx in range(topk_logits.shape[-1]):
        idx = topk_indices[:, :, local_idx]
        r = resp[:, :, local_idx]
        nk.scatter_add_(1, idx, r)
        sum_x.scatter_add_(
            1,
            idx.unsqueeze(-1).expand(-1, -1, d),
            r.unsqueeze(-1) * x_f,
        )
        if covariance_type == "spherical":
            sum_x_sq.scatter_add_(1, idx, r * x_sq)
        elif covariance_type == "diag":
            sum_x_sq.scatter_add_(
                1,
                idx.unsqueeze(-1).expand(-1, -1, d),
                r.unsqueeze(-1) * x_f_square,
            )
        elif covariance_type == "full":
            sum_xx.scatter_add_(
                1,
                idx[:, :, None, None].expand(-1, -1, d, d),
                r[:, :, None, None] * outer,
            )

    return total_log_likelihood


def spherical_assign_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.LongTensor:
    _check_batched_inputs(x, means, variances, weights)
    bsz, n, _ = x.shape
    k = means.shape[1]
    labels = torch.empty((bsz, n), dtype=torch.long, device=x.device)
    x_sq = x.square().sum(dim=-1)
    log_weights = torch.log(weights)

    for n_start in range(0, n, chunk_size_N):
        n_end = min(n_start + chunk_size_N, n)
        x_chunk = x[:, n_start:n_end, :]
        x_sq_chunk = x_sq[:, n_start:n_end]
        best_logits = torch.full(
            (bsz, n_end - n_start), -torch.inf, device=x.device, dtype=x.dtype
        )
        best_labels = torch.zeros((bsz, n_end - n_start), dtype=torch.long, device=x.device)

        for k_start in range(0, k, chunk_size_K):
            k_end = min(k_start + chunk_size_K, k)
            logits = _compute_chunk_logits(
                x_chunk,
                x_sq_chunk,
                means[:, k_start:k_end, :],
                variances[:, k_start:k_end],
                log_weights[:, k_start:k_end],
            )
            tile_logits, tile_labels = logits.max(dim=-1)
            update_mask = tile_logits > best_logits
            best_logits = torch.where(update_mask, tile_logits, best_logits)
            best_labels = torch.where(update_mask, tile_labels + k_start, best_labels)

        labels[:, n_start:n_end] = best_labels

    return labels


def diagonal_assign_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.LongTensor:
    _check_batched_diag_inputs(x, means, variances, weights)
    bsz, n, _ = x.shape
    k = means.shape[1]
    labels = torch.empty((bsz, n), dtype=torch.long, device=x.device)
    log_weights = torch.log(weights.to(torch.float32))
    precision = variances.to(torch.float32).clamp_min(1e-30).reciprocal()
    logdet = torch.log(variances.to(torch.float32).clamp_min(1e-30)).sum(dim=-1)
    weighted_means = means.to(torch.float32) * precision
    mean_precision_mean = (means.to(torch.float32) * weighted_means).sum(dim=-1)

    for n_start in range(0, n, chunk_size_N):
        n_end = min(n_start + chunk_size_N, n)
        x_chunk = x[:, n_start:n_end, :]
        best_logits = torch.full(
            (bsz, n_end - n_start), -torch.inf, device=x.device, dtype=torch.float32
        )
        best_labels = torch.zeros((bsz, n_end - n_start), dtype=torch.long, device=x.device)

        for k_start in range(0, k, chunk_size_K):
            k_end = min(k_start + chunk_size_K, k)
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
            tile_logits, tile_labels = logits.max(dim=-1)
            update_mask = tile_logits > best_logits
            best_logits = torch.where(update_mask, tile_logits, best_logits)
            best_labels = torch.where(update_mask, tile_labels + k_start, best_labels)

        labels[:, n_start:n_end] = best_labels

    return labels


def full_assign_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.LongTensor:
    _check_batched_full_inputs(x, means, covariances, weights)
    return _matrix_assign_torch_native_chunked(
        x,
        means,
        covariances,
        weights,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
        covariance_type="full",
    )


def tied_assign_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    covariance: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.LongTensor:
    _check_batched_tied_inputs(x, means, covariance, weights)
    effective_chunk_size_N = (
        _auto_tied_chunk_size(x.shape[1], means.shape[1], chunk_size_N)
        if x.is_cuda
        else chunk_size_N
    )
    return _matrix_assign_torch_native_chunked(
        x,
        means,
        covariance,
        weights,
        chunk_size_N=effective_chunk_size_N,
        chunk_size_K=chunk_size_K,
        covariance_type="tied",
    )


def _matrix_assign_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int,
    chunk_size_K: int,
    covariance_type: str,
) -> torch.LongTensor:
    bsz, n, _ = x.shape
    k = means.shape[1]
    labels = torch.empty((bsz, n), dtype=torch.long, device=x.device)
    log_weights = torch.log(weights.to(torch.float32))
    precision, logdet = _precision_and_logdet(covariances)
    if covariance_type == "full":
        precision_means = torch.einsum("bkde,bke->bkd", precision, means.to(torch.float32))
    else:
        precision_means = torch.bmm(means.to(torch.float32), precision.transpose(1, 2))
    mean_precision_mean = (means.to(torch.float32) * precision_means).sum(dim=-1)

    for n_start in range(0, n, chunk_size_N):
        n_end = min(n_start + chunk_size_N, n)
        x_chunk = x[:, n_start:n_end, :]
        best_logits = torch.full(
            (bsz, n_end - n_start), -torch.inf, device=x.device, dtype=torch.float32
        )
        best_labels = torch.zeros((bsz, n_end - n_start), dtype=torch.long, device=x.device)

        for k_start in range(0, k, chunk_size_K):
            k_end = min(k_start + chunk_size_K, k)
            if covariance_type == "full":
                logits = _compute_full_chunk_logits(
                    x_chunk,
                    means[:, k_start:k_end, :],
                    covariances[:, k_start:k_end, :, :],
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
                    covariances,
                    log_weights[:, k_start:k_end],
                    precision=precision,
                    logdet=logdet,
                    precision_means_chunk=precision_means[:, k_start:k_end, :],
                    mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
                )
            tile_logits, tile_labels = logits.max(dim=-1)
            update_mask = tile_logits > best_logits
            best_logits = torch.where(update_mask, tile_logits, best_logits)
            best_labels = torch.where(update_mask, tile_labels + k_start, best_labels)

        labels[:, n_start:n_end] = best_labels

    return labels


def spherical_score_samples_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    _check_batched_inputs(x, means, variances, weights)
    _, n, _ = x.shape
    scores = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
    x_sq = x.square().sum(dim=-1)

    for n_start in range(0, n, chunk_size_N):
        n_end = min(n_start + chunk_size_N, n)
        scores[:, n_start:n_end] = _stream_log_normalizer(
            x[:, n_start:n_end, :],
            x_sq[:, n_start:n_end],
            means,
            variances,
            weights,
            chunk_size_K=chunk_size_K,
        )

    return scores


def diagonal_score_samples_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    _check_batched_diag_inputs(x, means, variances, weights)
    _, n, _ = x.shape
    scores = torch.empty((x.shape[0], n), device=x.device, dtype=torch.float32)
    precision = variances.to(torch.float32).clamp_min(1e-30).reciprocal()
    logdet = torch.log(variances.to(torch.float32).clamp_min(1e-30)).sum(dim=-1)
    log_weights = torch.log(weights.to(torch.float32))
    weighted_means = means.to(torch.float32) * precision
    mean_precision_mean = (means.to(torch.float32) * weighted_means).sum(dim=-1)

    for n_start in range(0, n, chunk_size_N):
        n_end = min(n_start + chunk_size_N, n)
        scores[:, n_start:n_end] = _diag_stream_log_normalizer(
            x[:, n_start:n_end, :],
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

    return scores.to(x.dtype) if x.dtype in (torch.float64, torch.float32) else scores


def full_score_samples_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    _check_batched_full_inputs(x, means, covariances, weights)
    return _matrix_score_samples_torch_native_chunked(
        x,
        means,
        covariances,
        weights,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
        covariance_type="full",
    )


def tied_score_samples_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    covariance: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    _check_batched_tied_inputs(x, means, covariance, weights)
    effective_chunk_size_N = (
        _auto_tied_chunk_size(x.shape[1], means.shape[1], chunk_size_N)
        if x.is_cuda
        else chunk_size_N
    )
    return _matrix_score_samples_torch_native_chunked(
        x,
        means,
        covariance,
        weights,
        chunk_size_N=effective_chunk_size_N,
        chunk_size_K=chunk_size_K,
        covariance_type="tied",
    )


def _matrix_score_samples_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int,
    chunk_size_K: int,
    covariance_type: str,
) -> torch.Tensor:
    _, n, _ = x.shape
    scores = torch.empty((x.shape[0], n), device=x.device, dtype=torch.float32)
    precision, logdet = _precision_and_logdet(covariances)
    log_weights = torch.log(weights.to(torch.float32))
    if covariance_type == "full":
        precision_means = torch.einsum("bkde,bke->bkd", precision, means.to(torch.float32))
    else:
        precision_means = torch.bmm(means.to(torch.float32), precision.transpose(1, 2))
    mean_precision_mean = (means.to(torch.float32) * precision_means).sum(dim=-1)

    for n_start in range(0, n, chunk_size_N):
        n_end = min(n_start + chunk_size_N, n)
        scores[:, n_start:n_end] = _matrix_stream_log_normalizer(
            x[:, n_start:n_end, :],
            means,
            covariances,
            weights,
            chunk_size_K=chunk_size_K,
            covariance_type=covariance_type,
            precision=precision,
            logdet=logdet,
            log_weights=log_weights,
            precision_means=precision_means,
            mean_precision_mean=mean_precision_mean,
        )

    return scores.to(x.dtype) if x.dtype in (torch.float64, torch.float32) else scores


def spherical_predict_proba_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    _check_batched_inputs(x, means, variances, weights)
    bsz, n, _ = x.shape
    k = means.shape[1]
    probs = torch.empty((bsz, n, k), device=x.device, dtype=x.dtype)
    x_sq = x.square().sum(dim=-1)
    log_weights = torch.log(weights)

    for n_start in range(0, n, chunk_size_N):
        n_end = min(n_start + chunk_size_N, n)
        x_chunk = x[:, n_start:n_end, :]
        x_sq_chunk = x_sq[:, n_start:n_end]
        log_norm = _stream_log_normalizer(
            x_chunk,
            x_sq_chunk,
            means,
            variances,
            weights,
            chunk_size_K=chunk_size_K,
        )

        for k_start in range(0, k, chunk_size_K):
            k_end = min(k_start + chunk_size_K, k)
            logits = _compute_chunk_logits(
                x_chunk,
                x_sq_chunk,
                means[:, k_start:k_end, :],
                variances[:, k_start:k_end],
                log_weights[:, k_start:k_end],
            )
            probs[:, n_start:n_end, k_start:k_end] = torch.exp(
                logits - log_norm.unsqueeze(-1)
            )

    return probs


def diagonal_predict_proba_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    _check_batched_diag_inputs(x, means, variances, weights)
    bsz, n, _ = x.shape
    k = means.shape[1]
    probs = torch.empty((bsz, n, k), device=x.device, dtype=torch.float32)
    log_weights = torch.log(weights.to(torch.float32))
    precision = variances.to(torch.float32).clamp_min(1e-30).reciprocal()
    logdet = torch.log(variances.to(torch.float32).clamp_min(1e-30)).sum(dim=-1)
    weighted_means = means.to(torch.float32) * precision
    mean_precision_mean = (means.to(torch.float32) * weighted_means).sum(dim=-1)

    for n_start in range(0, n, chunk_size_N):
        n_end = min(n_start + chunk_size_N, n)
        x_chunk = x[:, n_start:n_end, :]
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

        for k_start in range(0, k, chunk_size_K):
            k_end = min(k_start + chunk_size_K, k)
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
            probs[:, n_start:n_end, k_start:k_end] = torch.exp(
                logits - log_norm.unsqueeze(-1)
            )

    return probs.to(x.dtype) if x.dtype in (torch.float64, torch.float32) else probs


def full_predict_proba_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    _check_batched_full_inputs(x, means, covariances, weights)
    return _matrix_predict_proba_torch_native_chunked(
        x,
        means,
        covariances,
        weights,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
        covariance_type="full",
    )


def tied_predict_proba_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    covariance: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    _check_batched_tied_inputs(x, means, covariance, weights)
    effective_chunk_size_N = (
        _auto_tied_chunk_size(x.shape[1], means.shape[1], chunk_size_N)
        if x.is_cuda
        else chunk_size_N
    )
    return _matrix_predict_proba_torch_native_chunked(
        x,
        means,
        covariance,
        weights,
        chunk_size_N=effective_chunk_size_N,
        chunk_size_K=chunk_size_K,
        covariance_type="tied",
    )


def _matrix_predict_proba_torch_native_chunked(
    x: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size_N: int,
    chunk_size_K: int,
    covariance_type: str,
) -> torch.Tensor:
    bsz, n, _ = x.shape
    k = means.shape[1]
    probs = torch.empty((bsz, n, k), device=x.device, dtype=torch.float32)
    log_weights = torch.log(weights.to(torch.float32))
    precision, logdet = _precision_and_logdet(covariances)
    if covariance_type == "full":
        precision_means = torch.einsum("bkde,bke->bkd", precision, means.to(torch.float32))
    else:
        precision_means = torch.bmm(means.to(torch.float32), precision.transpose(1, 2))
    mean_precision_mean = (means.to(torch.float32) * precision_means).sum(dim=-1)

    for n_start in range(0, n, chunk_size_N):
        n_end = min(n_start + chunk_size_N, n)
        x_chunk = x[:, n_start:n_end, :]
        log_norm = _matrix_stream_log_normalizer(
            x_chunk,
            means,
            covariances,
            weights,
            chunk_size_K=chunk_size_K,
            covariance_type=covariance_type,
            precision=precision,
            logdet=logdet,
            log_weights=log_weights,
            precision_means=precision_means,
            mean_precision_mean=mean_precision_mean,
        )

        for k_start in range(0, k, chunk_size_K):
            k_end = min(k_start + chunk_size_K, k)
            if covariance_type == "full":
                logits = _compute_full_chunk_logits(
                    x_chunk,
                    means[:, k_start:k_end, :],
                    covariances[:, k_start:k_end, :, :],
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
                    covariances,
                    log_weights[:, k_start:k_end],
                    precision=precision,
                    logdet=logdet,
                    precision_means_chunk=precision_means[:, k_start:k_end, :],
                    mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
                )
            probs[:, n_start:n_end, k_start:k_end] = torch.exp(
                logits - log_norm.unsqueeze(-1)
            )

    return probs.to(x.dtype) if x.dtype in (torch.float64, torch.float32) else probs


def _global_spherical_variance(
    x: torch.Tensor,
    n_components: int,
    reg_covar: float,
    x_sq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _, _, d = x.shape
    x_f = x.to(torch.float32)
    mean = x_f.mean(dim=1)
    if x_sq is None:
        second_moment = x_f.square().sum(dim=-1).mean(dim=1)
    else:
        second_moment = x_sq.to(torch.float32).mean(dim=1)
    mean_sq = mean.square().sum(dim=-1)
    global_var = (second_moment - mean_sq).clamp_min(0.0) / float(d)
    return global_var.clamp_min(reg_covar).unsqueeze(-1).expand(-1, n_components).clone().to(x.dtype)


def _global_diag_variance(x: torch.Tensor, n_components: int, reg_covar: float) -> torch.Tensor:
    centered = x.to(torch.float32) - x.to(torch.float32).mean(dim=1, keepdim=True)
    global_var = centered.square().mean(dim=1).clamp_min(reg_covar)
    return global_var.unsqueeze(1).expand(-1, n_components, -1).clone().to(x.dtype)


def _global_tied_covariance(x: torch.Tensor, reg_covar: float) -> torch.Tensor:
    _, n, d = x.shape
    centered = x.to(torch.float32) - x.to(torch.float32).mean(dim=1, keepdim=True)
    covariance = torch.bmm(centered.transpose(1, 2), centered) / float(n)
    return _regularize_matrix(covariance, reg_covar).to(x.dtype)


def _global_full_covariance(x: torch.Tensor, n_components: int, reg_covar: float) -> torch.Tensor:
    covariance = _global_tied_covariance(x, reg_covar)
    return covariance.unsqueeze(1).expand(-1, n_components, -1, -1).clone()


def _random_init_means(x: torch.Tensor, n_components: int) -> torch.Tensor:
    bsz, n, d = x.shape
    indices = torch.randint(0, n, (bsz, n_components), device=x.device)
    return torch.gather(x, 1, indices[..., None].expand(-1, -1, d)).clone()


def _kmeans_plus_plus_init_means(x: torch.Tensor, n_components: int) -> Optional[torch.Tensor]:
    bsz, n, d = x.shape
    target_sample_n = max(
        KMEANS_PLUS_PLUS_MIN_SAMPLES,
        KMEANS_PLUS_PLUS_SAMPLES_PER_COMPONENT * n_components,
    )
    sample_n = min(n, KMEANS_PLUS_PLUS_MAX_SAMPLES, target_sample_n)
    if n_components <= 0 or n_components > sample_n or n_components > KMEANS_PLUS_PLUS_MAX_COMPONENTS:
        return None

    if sample_n == n:
        sample = x
    else:
        sample_idx = torch.randperm(n, device=x.device)[:sample_n]
        sample = x.index_select(1, sample_idx)

    sample_f = sample.to(torch.float32)
    centers = torch.empty((bsz, n_components, d), device=x.device, dtype=torch.float32)
    batch_idx = torch.arange(bsz, device=x.device)

    first_idx = torch.randint(0, sample_n, (bsz,), device=x.device)
    centers[:, 0, :] = sample_f[batch_idx, first_idx]
    closest_dist = (sample_f - centers[:, 0:1, :]).square().sum(dim=-1)

    local_trials = 2 + int(math.log(float(n_components)))
    for center_idx in range(1, n_components):
        dist_sum = closest_dist.sum(dim=1)
        cumulative = closest_dist.cumsum(dim=1)
        draws = torch.rand((bsz, local_trials), device=x.device, dtype=torch.float32) * dist_sum.unsqueeze(1)
        candidate_idx = torch.searchsorted(cumulative, draws).clamp_max(sample_n - 1)
        candidate_idx = torch.where(dist_sum.unsqueeze(1) > 0, candidate_idx, torch.zeros_like(candidate_idx))

        vectorized_elements = bsz * sample_n * local_trials * d
        if vectorized_elements <= KMEANS_PLUS_PLUS_VECTORIZE_MAX_ELEMENTS:
            candidates = sample_f[
                batch_idx[:, None],
                candidate_idx,
            ]
            new_dist = (sample_f.unsqueeze(2) - candidates.unsqueeze(1)).square().sum(dim=-1)
            trial_dist = torch.minimum(closest_dist.unsqueeze(-1), new_dist)
            best_trial = trial_dist.sum(dim=1).argmin(dim=1)
            best_idx = candidate_idx[batch_idx, best_trial]
            best_dist = trial_dist[batch_idx, :, best_trial]
        else:
            best_idx = candidate_idx[:, 0]
            best_dist = closest_dist
            best_potential = torch.full((bsz,), float("inf"), device=x.device, dtype=torch.float32)
            for trial_idx in range(local_trials):
                idx = candidate_idx[:, trial_idx]
                candidate = sample_f[batch_idx, idx].unsqueeze(1)
                new_dist = (sample_f - candidate).square().sum(dim=-1)
                trial_dist = torch.minimum(closest_dist, new_dist)
                potential = trial_dist.sum(dim=1)
                keep = potential < best_potential
                best_potential = torch.where(keep, potential, best_potential)
                best_idx = torch.where(keep, idx, best_idx)
                best_dist = torch.where(keep.unsqueeze(1), trial_dist, best_dist)

        centers[:, center_idx, :] = sample_f[batch_idx, best_idx]
        closest_dist = best_dist

    return centers.to(dtype=x.dtype)


def _hard_label_init_stats(
    x: torch.Tensor,
    means: torch.Tensor,
    labels: torch.Tensor,
    *,
    reg_covar: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, n, d = x.shape
    k = means.shape[1]
    counts = torch.zeros((bsz, k), device=x.device, dtype=x.dtype)
    sq_sums = torch.zeros((bsz, k), device=x.device, dtype=x.dtype)
    global_var = _global_spherical_variance(x, k, reg_covar)

    for b_idx in range(bsz):
        label_idx = labels[b_idx].to(torch.long)
        ones = torch.ones((n,), device=x.device, dtype=x.dtype)
        counts[b_idx].index_add_(0, label_idx, ones)
        diffs = x[b_idx] - means[b_idx].index_select(0, label_idx)
        dists = diffs.square().sum(dim=-1)
        sq_sums[b_idx].index_add_(0, label_idx, dists)

    weights = counts.clamp_min(1.0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    variances = (sq_sums / counts.clamp_min(1.0)) / float(d)
    variances = torch.where(counts > 0, variances, global_var)
    variances = variances.clamp_min(reg_covar)
    return weights, variances


def _hard_label_init_stats_diag(
    x: torch.Tensor,
    means: torch.Tensor,
    labels: torch.Tensor,
    *,
    reg_covar: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, n, d = x.shape
    k = means.shape[1]
    counts = torch.zeros((bsz, k), device=x.device, dtype=torch.float32)
    sq_sums = torch.zeros((bsz, k, d), device=x.device, dtype=torch.float32)
    global_var = _global_diag_variance(x, k, reg_covar).to(torch.float32)

    for b_idx in range(bsz):
        label_idx = labels[b_idx].to(torch.long)
        ones = torch.ones((n,), device=x.device, dtype=torch.float32)
        counts[b_idx].index_add_(0, label_idx, ones)
        diffs = x[b_idx].to(torch.float32) - means[b_idx].to(torch.float32).index_select(0, label_idx)
        sq_sums[b_idx].index_add_(0, label_idx, diffs.square())

    weights = counts.clamp_min(1.0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    variances = sq_sums / counts.clamp_min(1.0).unsqueeze(-1)
    variances = torch.where(counts.unsqueeze(-1) > 0, variances, global_var)
    variances = variances.clamp_min(reg_covar)
    return weights.to(x.dtype), variances.to(x.dtype)


def _hard_label_init_stats_full(
    x: torch.Tensor,
    means: torch.Tensor,
    labels: torch.Tensor,
    *,
    reg_covar: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, n, d = x.shape
    k = means.shape[1]
    x_f = x.to(torch.float32)
    means_f = means.to(torch.float32)
    labels_long = labels.to(torch.long)
    assignments = torch.nn.functional.one_hot(labels_long, num_classes=k).to(torch.float32)
    counts = assignments.sum(dim=1)
    sum_x = torch.bmm(assignments.transpose(1, 2), x_f)
    sum_xx = torch.einsum("bnk,bnd,bne->bkde", assignments, x_f, x_f)
    global_cov = _global_full_covariance(x, k, reg_covar).to(torch.float32)
    eye = _eye_like_covariance(d, x.device, torch.float32)
    counts_safe = counts.clamp_min(1.0)
    means_col = means_f.unsqueeze(-1)
    means_row = means_f.unsqueeze(-2)
    scatter = (
        sum_xx
        - means_col * sum_x.unsqueeze(-2)
        - sum_x.unsqueeze(-1) * means_row
        + counts[..., None, None] * (means_col * means_row)
    )
    covariances = scatter / counts_safe[..., None, None] + reg_covar * eye
    covariances = torch.where(counts[..., None, None] > 0, covariances, global_cov)

    weights = counts.clamp_min(1.0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(x.dtype), _symmetrize_matrix(covariances).to(x.dtype)


def _hard_label_init_stats_tied(
    x: torch.Tensor,
    means: torch.Tensor,
    labels: torch.Tensor,
    *,
    reg_covar: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, n, d = x.shape
    k = means.shape[1]
    counts = torch.zeros((bsz, k), device=x.device, dtype=torch.float32)
    covariances = torch.zeros((bsz, d, d), device=x.device, dtype=torch.float32)
    eye = _eye_like_covariance(d, x.device, torch.float32)

    for b_idx in range(bsz):
        label_idx = labels[b_idx].to(torch.long)
        ones = torch.ones((n,), device=x.device, dtype=torch.float32)
        counts[b_idx].index_add_(0, label_idx, ones)
        diffs = x[b_idx].to(torch.float32) - means[b_idx].to(torch.float32).index_select(0, label_idx)
        covariances[b_idx] = torch.mm(diffs.transpose(0, 1), diffs) / float(n) + reg_covar * eye

    weights = counts.clamp_min(1.0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(x.dtype), _symmetrize_matrix(covariances).to(x.dtype)


def _kmeans_initialize(
    x: torch.Tensor,
    n_components: int,
    *,
    max_iters: int,
    tol: float,
    use_triton: bool,
    verbose: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    flash_kmeans = load_flash_kmeans()
    if flash_kmeans is None:
        return None

    try:
        init_centroids = _kmeans_plus_plus_init_means(x, n_components)
    except Exception:
        return None

    if use_triton:
        try:
            labels, means, _ = flash_kmeans.batch_kmeans_Euclid(
                x,
                n_components,
                max_iters=max_iters,
                tol=tol,
                init_centroids=init_centroids,
                verbose=verbose,
            )
        except Exception:
            pass
        else:
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
            return labels.to(torch.long), means.to(dtype=x.dtype, device=x.device)

    try:
        from flash_kmeans.torch_fallback import batch_kmeans_Euclid_torch_native

        labels, means, _ = batch_kmeans_Euclid_torch_native(
            x,
            n_components,
            max_iters=max_iters,
            tol=tol,
            init_centroids=init_centroids,
            verbose=verbose,
        )
    except Exception:
        return None

    if labels.ndim == 1:
        labels = labels.unsqueeze(0)

    return labels.to(torch.long), means.to(dtype=x.dtype, device=x.device)


def _validate_init_means(init_means: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    means = init_means.clone()
    if means.ndim != 3:
        raise ValueError("init_means must have shape (B, K, D)")
    if means.shape[0] != x.shape[0] or means.shape[2] != x.shape[2]:
        raise ValueError("init_means must have shape (B, K, D) matching x")
    return means.to(device=x.device, dtype=x.dtype)


def _validate_init_weights(init_weights: Optional[torch.Tensor], x: torch.Tensor, k: int) -> torch.Tensor:
    if init_weights is None:
        weights = torch.full((x.shape[0], k), 1.0 / float(k), device=x.device, dtype=x.dtype)
    else:
        weights = init_weights.clone().to(device=x.device, dtype=x.dtype)
    if weights.shape != (x.shape[0], k):
        raise ValueError("init_weights must have shape (B, K)")
    weights = weights.clamp_min(1e-8)
    return weights / weights.sum(dim=-1, keepdim=True)


def _initialize_parameters(
    x: torch.Tensor,
    n_components: int,
    *,
    init_means: Optional[torch.Tensor],
    init_variances: Optional[torch.Tensor],
    init_weights: Optional[torch.Tensor],
    x_sq: Optional[torch.Tensor] = None,
    init_params: str,
    reg_covar: float,
    kmeans_max_iters: int,
    kmeans_tol: float,
    kmeans_use_triton: bool,
    verbose: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if init_means is not None:
        means = init_means.clone()
        if means.ndim != 3:
            raise ValueError("init_means must have shape (B, K, D)")
        k = means.shape[1]
        variances = (
            init_variances.clone()
            if init_variances is not None
            else _global_spherical_variance(x, k, reg_covar, x_sq=x_sq)
        )
        weights = (
            init_weights.clone()
            if init_weights is not None
            else torch.full((x.shape[0], k), 1.0 / float(k), device=x.device, dtype=x.dtype)
        )
        if variances.shape != (x.shape[0], k):
            raise ValueError("init_variances must have shape (B, K)")
        if weights.shape != (x.shape[0], k):
            raise ValueError("init_weights must have shape (B, K)")
        weights = weights.clamp_min(1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return means, variances.clamp_min(reg_covar), weights, "user"

    if init_params not in {"kmeans", "random"}:
        raise ValueError("init_params must be 'kmeans' or 'random'")

    if init_params == "kmeans":
        kmeans_init = _kmeans_initialize(
            x,
            n_components,
            max_iters=kmeans_max_iters,
            tol=kmeans_tol,
            use_triton=kmeans_use_triton,
            verbose=verbose,
        )
        if kmeans_init is not None:
            labels, means = kmeans_init
            weights, variances = _hard_label_init_stats(
                x,
                means,
                labels,
                reg_covar=reg_covar,
            )
            return means, variances, weights, "kmeans"

    means = _random_init_means(x, n_components)
    variances = _global_spherical_variance(x, n_components, reg_covar, x_sq=x_sq)
    weights = torch.full(
        (x.shape[0], n_components),
        1.0 / float(n_components),
        device=x.device,
        dtype=x.dtype,
    )
    return means, variances, weights, "random"


def _initialize_diag_parameters(
    x: torch.Tensor,
    n_components: int,
    *,
    init_means: Optional[torch.Tensor],
    init_variances: Optional[torch.Tensor],
    init_weights: Optional[torch.Tensor],
    init_params: str,
    reg_covar: float,
    kmeans_max_iters: int,
    kmeans_tol: float,
    kmeans_use_triton: bool,
    verbose: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if init_means is not None:
        means = _validate_init_means(init_means, x)
        k = means.shape[1]
        variances = (
            init_variances.clone().to(device=x.device, dtype=x.dtype)
            if init_variances is not None
            else _global_diag_variance(x, k, reg_covar)
        )
        if variances.shape != (x.shape[0], k, x.shape[2]):
            raise ValueError("init_variances must have shape (B, K, D) for diag covariance")
        weights = _validate_init_weights(init_weights, x, k)
        return means, variances.clamp_min(reg_covar), weights, "user"

    if init_params not in {"kmeans", "random"}:
        raise ValueError("init_params must be 'kmeans' or 'random'")

    if init_params == "kmeans":
        kmeans_init = _kmeans_initialize(
            x,
            n_components,
            max_iters=kmeans_max_iters,
            tol=kmeans_tol,
            use_triton=kmeans_use_triton,
            verbose=verbose,
        )
        if kmeans_init is not None:
            labels, means = kmeans_init
            weights, variances = _hard_label_init_stats_diag(
                x,
                means,
                labels,
                reg_covar=reg_covar,
            )
            return means, variances, weights, "kmeans"

    means = _random_init_means(x, n_components)
    variances = _global_diag_variance(x, n_components, reg_covar)
    weights = torch.full(
        (x.shape[0], n_components),
        1.0 / float(n_components),
        device=x.device,
        dtype=x.dtype,
    )
    return means, variances, weights, "random"


def _initialize_full_parameters(
    x: torch.Tensor,
    n_components: int,
    *,
    init_means: Optional[torch.Tensor],
    init_covariances: Optional[torch.Tensor],
    init_weights: Optional[torch.Tensor],
    init_params: str,
    reg_covar: float,
    kmeans_max_iters: int,
    kmeans_tol: float,
    kmeans_use_triton: bool,
    verbose: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if init_means is not None:
        means = _validate_init_means(init_means, x)
        k = means.shape[1]
        covariances = (
            init_covariances.clone().to(device=x.device, dtype=x.dtype)
            if init_covariances is not None
            else _global_full_covariance(x, k, reg_covar)
        )
        if covariances.shape != (x.shape[0], k, x.shape[2], x.shape[2]):
            raise ValueError("init_variances must have shape (B, K, D, D) for full covariance")
        weights = _validate_init_weights(init_weights, x, k)
        return means, _symmetrize_matrix(covariances).to(x.dtype), weights, "user"

    if init_params not in {"kmeans", "random"}:
        raise ValueError("init_params must be 'kmeans' or 'random'")

    if init_params == "kmeans":
        kmeans_init = _kmeans_initialize(
            x,
            n_components,
            max_iters=kmeans_max_iters,
            tol=kmeans_tol,
            use_triton=kmeans_use_triton,
            verbose=verbose,
        )
        if kmeans_init is not None:
            labels, means = kmeans_init
            weights, covariances = _hard_label_init_stats_full(
                x,
                means,
                labels,
                reg_covar=reg_covar,
            )
            return means, covariances, weights, "kmeans"

    means = _random_init_means(x, n_components)
    covariances = _global_full_covariance(x, n_components, reg_covar)
    weights = torch.full(
        (x.shape[0], n_components),
        1.0 / float(n_components),
        device=x.device,
        dtype=x.dtype,
    )
    return means, covariances, weights, "random"


def _initialize_tied_parameters(
    x: torch.Tensor,
    n_components: int,
    *,
    init_means: Optional[torch.Tensor],
    init_covariance: Optional[torch.Tensor],
    init_weights: Optional[torch.Tensor],
    init_params: str,
    reg_covar: float,
    kmeans_max_iters: int,
    kmeans_tol: float,
    kmeans_use_triton: bool,
    verbose: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if init_means is not None:
        means = _validate_init_means(init_means, x)
        k = means.shape[1]
        covariance = (
            init_covariance.clone().to(device=x.device, dtype=x.dtype)
            if init_covariance is not None
            else _global_tied_covariance(x, reg_covar)
        )
        if covariance.shape != (x.shape[0], x.shape[2], x.shape[2]):
            raise ValueError("init_variances must have shape (B, D, D) for tied covariance")
        weights = _validate_init_weights(init_weights, x, k)
        return means, _symmetrize_matrix(covariance).to(x.dtype), weights, "user"

    if init_params not in {"kmeans", "random"}:
        raise ValueError("init_params must be 'kmeans' or 'random'")

    if init_params == "kmeans":
        kmeans_init = _kmeans_initialize(
            x,
            n_components,
            max_iters=kmeans_max_iters,
            tol=kmeans_tol,
            use_triton=kmeans_use_triton,
            verbose=verbose,
        )
        if kmeans_init is not None:
            labels, means = kmeans_init
            weights, covariance = _hard_label_init_stats_tied(
                x,
                means,
                labels,
                reg_covar=reg_covar,
            )
            return means, covariance, weights, "kmeans"

    means = _random_init_means(x, n_components)
    covariance = _global_tied_covariance(x, reg_covar)
    weights = torch.full(
        (x.shape[0], n_components),
        1.0 / float(n_components),
        device=x.device,
        dtype=x.dtype,
    )
    return means, covariance, weights, "random"


def batch_gmm_Spherical_torch_native(
    x: torch.Tensor,
    n_components: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_means: Optional[torch.Tensor] = None,
    init_variances: Optional[torch.Tensor] = None,
    init_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    init_params: str = "kmeans",
    reg_covar: float = 1e-6,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
    kmeans_init_iters: int = 10,
    kmeans_init_tol: float = 1e-4,
    kmeans_use_triton: bool = True,
    gmm_use_triton_estep: bool | str = "auto",
    gmm_use_triton_streaming_update: bool | str = "auto",
    min_weight: float = 1e-8,
    compute_labels: bool = True,
    approx_top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    if x.ndim != 3:
        raise ValueError("x must have shape (B, N, D)")

    bsz, n, d = x.shape
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n == 0:
        raise ValueError("x must contain at least one sample")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if chunk_size_N <= 0 or chunk_size_K <= 0:
        raise ValueError("chunk_size_N and chunk_size_K must be positive")
    if min_weight <= 0.0:
        raise ValueError("min_weight must be positive")
    effective_approx_top_k = _resolve_approx_top_k(approx_top_k, n_components)
    if gmm_use_triton_estep == "auto":
        gmm_use_triton_estep_enabled = _auto_use_triton_estep(x, n_components)
    elif isinstance(gmm_use_triton_estep, bool):
        gmm_use_triton_estep_enabled = bool(
            gmm_use_triton_estep
            and _HAS_TRITON_ASSIGN
            and x.is_cuda
            and triton_spherical_supported(d, n_components)
        )
    else:
        raise ValueError("gmm_use_triton_estep must be a bool or 'auto'")
    if gmm_use_triton_estep == "auto":
        gmm_use_triton_labels_enabled = _auto_use_triton_labels(x, n_components)
    else:
        gmm_use_triton_labels_enabled = bool(
            gmm_use_triton_estep_enabled and _auto_use_triton_labels(x, n_components)
        )
    if gmm_use_triton_streaming_update == "auto":
        gmm_use_triton_streaming_update_enabled = _auto_use_triton_streaming_update(x, n_components)
    elif isinstance(gmm_use_triton_streaming_update, bool):
        gmm_use_triton_streaming_update_enabled = (
            gmm_use_triton_streaming_update
            and _HAS_TRITON_WEIGHTED_UPDATE
            and x.is_cuda
            and triton_spherical_supported(d, n_components)
        )
    else:
        raise ValueError("gmm_use_triton_streaming_update must be a bool or 'auto'")
    approx_triton_config = None
    if (
        effective_approx_top_k is not None
        and _HAS_TRITON_APPROX_UPDATE
        and x.is_cuda
        and (gmm_use_triton_estep != False or gmm_use_triton_streaming_update != False)
    ):
        approx_triton_config = approx_topk_update_spherical_config(
            d,
            n_components,
            effective_approx_top_k,
        )
    if effective_approx_top_k is not None:
        gmm_use_triton_estep_enabled = False
        gmm_use_triton_streaming_update_enabled = False
    spherical_fused_config = (
        fused_single_tile_update_config(d, n_components, "spherical")
        if (
            _HAS_TRITON_FUSED_UPDATE
            and x.is_cuda
            and (gmm_use_triton_estep != False or gmm_use_triton_streaming_update != False)
            and effective_approx_top_k is None
        )
        else None
    )
    x_sq_all = (x.to(torch.float32) ** 2).sum(dim=-1)
    means, variances, weights, init_source = _initialize_parameters(
        x,
        n_components,
        init_means=init_means,
        init_variances=init_variances,
        init_weights=init_weights,
        x_sq=x_sq_all,
        init_params=init_params,
        reg_covar=reg_covar,
        kmeans_max_iters=kmeans_init_iters,
        kmeans_tol=kmeans_init_tol,
        kmeans_use_triton=kmeans_use_triton,
        verbose=verbose,
    )

    prev_lower_bound_tensor = None
    lower_bound_history = []
    effective_chunk_size_N = chunk_size_N
    if (
        gmm_use_triton_streaming_update_enabled
        and gmm_use_triton_streaming_update == "auto"
    ):
        effective_chunk_size_N = _auto_triton_streaming_chunk_size(n, d, chunk_size_N)
    elif not gmm_use_triton_streaming_update_enabled:
        effective_chunk_size_N = _auto_torch_spherical_chunk_size(
            n,
            d,
            n_components,
            chunk_size_N,
        )
    triton_estep_failed = False
    triton_streaming_update_failed = False
    triton_fused_update_used = False
    triton_approx_update_failed = False
    triton_approx_update_used = False
    triton_labels_used = False
    log_norm_buffer = None
    if gmm_use_triton_estep_enabled and _HAS_TRITON_ASSIGN and x.is_cuda:
        log_norm_buffer = torch.empty(
            (bsz, min(effective_chunk_size_N, n)),
            device=x.device,
            dtype=torch.float32,
        )
    nk_buffer = torch.empty((bsz, n_components), device=x.device, dtype=torch.float32)
    sum_x_buffer = torch.empty((bsz, n_components, d), device=x.device, dtype=torch.float32)
    sum_x_sq_buffer = torch.empty((bsz, n_components), device=x.device, dtype=torch.float32)
    approx_partial_ll_buffer = None
    if approx_triton_config is not None:
        approx_block_n = int(approx_triton_config["BLOCK_N"])
        max_chunk_n = min(effective_chunk_size_N, n)
        max_n_blocks = (max_chunk_n + approx_block_n - 1) // approx_block_n
        approx_partial_ll_buffer = torch.empty((bsz, max_n_blocks), device=x.device, dtype=torch.float32)
    blocked_partial_buffers = None
    if (
        (gmm_use_triton_streaming_update_enabled or spherical_fused_config is not None)
        and triton_spherical_supported(d, n_components)
        and x.is_cuda
    ):
        if spherical_fused_config is None:
            blocked_block_n, _, _ = _triton_blocked_update_config(d, n_components)
        else:
            blocked_block_n = int(spherical_fused_config["BLOCK_N"])
        max_chunk_n = min(effective_chunk_size_N, n)
        max_n_blocks = (max_chunk_n + blocked_block_n - 1) // blocked_block_n
        blocked_partial_buffers = (
            torch.empty((bsz, max_n_blocks, n_components), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks, n_components, d), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks, n_components), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks), device=x.device, dtype=torch.float32),
        )

    for iteration in range(max_iters):
        collect_lower_bound = bool(tol > 0.0 or verbose or iteration + 1 == max_iters)
        variances = variances.clamp_min(reg_covar)
        weights = weights.clamp_min(min_weight)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        log_weights = torch.log(weights.to(torch.float32))
        x_sq = x_sq_all
        use_triton_estep = (
            gmm_use_triton_estep_enabled
            and _HAS_TRITON_ASSIGN
            and x.is_cuda
        )
        use_triton_streaming_update = (
            gmm_use_triton_streaming_update_enabled
        )
        use_triton_labels = (
            gmm_use_triton_labels_enabled
            and _HAS_TRITON_ASSIGN
            and x.is_cuda
        )
        means_sq = (
            (means.to(torch.float32) ** 2).sum(dim=-1)
            if (
                use_triton_estep
                or use_triton_streaming_update
                or spherical_fused_config is not None
                or approx_triton_config is not None
            )
            else None
        )

        nk = nk_buffer.zero_()
        sum_x = sum_x_buffer.zero_()
        sum_x_sq = sum_x_sq_buffer.zero_()
        total_log_likelihood = torch.zeros((), device=x.device, dtype=torch.float32)

        for n_start in range(0, n, effective_chunk_size_N):
            n_end = min(n_start + effective_chunk_size_N, n)
            x_chunk = x[:, n_start:n_end, :]
            x_sq_chunk = x_sq[:, n_start:n_end]
            if effective_approx_top_k is not None:
                if approx_triton_config is not None:
                    try:
                        _, _, _, ll_tile = triton_approx_topk_update_spherical(
                            x_chunk,
                            means,
                            variances.to(torch.float32),
                            weights.to(torch.float32),
                            top_k=effective_approx_top_k,
                            x_sq=x_sq_chunk,
                            means_sq=means_sq,
                            log_weights=log_weights,
                            nk=nk,
                            sum_x=sum_x,
                            sum_x_sq=sum_x_sq,
                            partial_log_likelihood=approx_partial_ll_buffer,
                            **approx_triton_config,
                        )
                        total_log_likelihood = total_log_likelihood + ll_tile
                        triton_approx_update_used = True
                        continue
                    except Exception:
                        triton_approx_update_failed = True
                        approx_triton_config = None
                topk_logits, topk_indices = _topk_logits_for_chunk(
                    x_chunk,
                    means,
                    variances,
                    covariance_type="spherical",
                    chunk_size_K=chunk_size_K,
                    top_k=effective_approx_top_k,
                    terms=(x_sq_chunk, log_weights),
                )
                total_log_likelihood = _accumulate_topk_stats(
                    x_chunk,
                    topk_logits,
                    topk_indices,
                    covariance_type="spherical",
                    nk=nk,
                    sum_x=sum_x,
                    sum_x_sq=sum_x_sq,
                    x_sq=x_sq_chunk,
                    total_log_likelihood=total_log_likelihood,
                )
                continue
            if spherical_fused_config is not None:
                try:
                    means_sq_for_update = means_sq
                    if means_sq_for_update is None:
                        means_sq_for_update = (means.to(torch.float32) ** 2).sum(dim=-1)
                    nk_tile, sum_x_tile, sum_x_sq_tile, ll_tile = (
                        triton_fused_single_tile_update_spherical(
                            x_chunk,
                            means,
                            variances.to(torch.float32),
                            weights.to(torch.float32),
                            x_sq=x_sq_chunk,
                            means_sq=means_sq_for_update,
                            log_weights=log_weights,
                            partial_nk=None if blocked_partial_buffers is None else blocked_partial_buffers[0],
                            partial_sum_x=None if blocked_partial_buffers is None else blocked_partial_buffers[1],
                            partial_sum_x_sq=None if blocked_partial_buffers is None else blocked_partial_buffers[2],
                            partial_log_likelihood=None if blocked_partial_buffers is None else blocked_partial_buffers[3],
                            **spherical_fused_config,
                        )
                    )
                    total_log_likelihood = total_log_likelihood + ll_tile
                    nk += nk_tile
                    sum_x += sum_x_tile
                    sum_x_sq += sum_x_sq_tile
                    triton_fused_update_used = True
                    continue
                except Exception:
                    triton_streaming_update_failed = True
                    use_triton_streaming_update = False
                    spherical_fused_config = None

            log_norm_already_summed = False
            if use_triton_estep:
                try:
                    log_norm_out = (
                        log_norm_buffer[:, : n_end - n_start]
                        if log_norm_buffer is not None
                        else None
                    )
                    log_norm = spherical_logsumexp_triton(
                        x_chunk,
                        means,
                        variances.to(torch.float32),
                        weights.to(torch.float32),
                        x_sq=x_sq_chunk,
                        out=log_norm_out,
                        out_sum=total_log_likelihood,
                        means_sq=means_sq,
                        log_weights=log_weights,
                    )
                    log_norm_already_summed = True
                except Exception:
                    triton_estep_failed = True
                    use_triton_estep = False
                    log_norm = _stream_log_normalizer(
                        x_chunk,
                        x_sq_chunk,
                        means,
                        variances,
                        weights,
                        chunk_size_K=chunk_size_K,
                    )
            else:
                log_norm = _stream_log_normalizer(
                    x_chunk,
                    x_sq_chunk,
                    means,
                    variances,
                    weights,
                    chunk_size_K=chunk_size_K,
                )
            if not log_norm_already_summed:
                total_log_likelihood = total_log_likelihood + log_norm.sum()

            if use_triton_streaming_update:
                try:
                    means_sq_for_update = means_sq
                    if means_sq_for_update is None:
                        means_sq_for_update = (means.to(torch.float32) ** 2).sum(dim=-1)
                    block_n, block_d, block_k = _triton_blocked_update_config(d, n_components)
                    (
                        nk_tile,
                        sum_x_tile,
                        sum_x_sq_tile,
                    ) = triton_blocked_update_spherical(
                        x_chunk,
                        means,
                        variances.to(torch.float32),
                        weights.to(torch.float32),
                        log_norm,
                        x_sq=x_sq_chunk,
                        means_sq=means_sq_for_update,
                        log_weights=log_weights,
                        partial_nk=None if blocked_partial_buffers is None else blocked_partial_buffers[0],
                        partial_sum_x=None if blocked_partial_buffers is None else blocked_partial_buffers[1],
                        partial_sum_x_sq=None if blocked_partial_buffers is None else blocked_partial_buffers[2],
                        BLOCK_N=block_n,
                        BLOCK_D=block_d,
                        BLOCK_K=block_k,
                    )
                    nk += nk_tile
                    sum_x += sum_x_tile
                    sum_x_sq += sum_x_sq_tile
                    continue
                except Exception:
                    triton_streaming_update_failed = True
                    use_triton_streaming_update = False

            for k_start in range(0, n_components, chunk_size_K):
                k_end = min(k_start + chunk_size_K, n_components)
                if use_triton_estep:
                    try:
                        resp = spherical_resp_triton(
                            x_chunk,
                            means[:, k_start:k_end, :],
                            variances[:, k_start:k_end].to(torch.float32),
                            weights[:, k_start:k_end].to(torch.float32),
                            log_norm,
                            x_sq=x_sq_chunk,
                            means_sq=means_sq[:, k_start:k_end],
                            log_weights=log_weights[:, k_start:k_end],
                        )
                    except Exception:
                        triton_estep_failed = True
                        use_triton_estep = False
                        logits = _compute_chunk_logits(
                            x_chunk,
                            x_sq_chunk,
                            means[:, k_start:k_end, :],
                            variances[:, k_start:k_end],
                            log_weights[:, k_start:k_end],
                        )
                        resp = torch.exp(logits - log_norm.unsqueeze(-1))
                else:
                    logits = _compute_chunk_logits(
                        x_chunk,
                        x_sq_chunk,
                        means[:, k_start:k_end, :],
                        variances[:, k_start:k_end],
                        log_weights[:, k_start:k_end],
                    )
                    resp = torch.exp(logits - log_norm.unsqueeze(-1))
                nk[:, k_start:k_end] += resp.sum(dim=1)
                sum_x[:, k_start:k_end, :] += torch.bmm(resp.transpose(1, 2), x_chunk.to(torch.float32))
                sum_x_sq[:, k_start:k_end] += (resp * x_sq_chunk.unsqueeze(-1)).sum(dim=1)

        active_mask = nk > min_weight
        nk_safe = nk.clamp_min(min_weight)
        means_new = (sum_x / nk_safe.unsqueeze(-1)).to(x.dtype)
        means_new = torch.where(active_mask.unsqueeze(-1), means_new, means)

        mean_sq = means_new.square().sum(dim=-1)
        variances_new = (sum_x_sq - nk * mean_sq).clamp_min(0.0) / (nk_safe * float(d))
        variances_new = variances_new.clamp_min(reg_covar)
        variances_new = torch.where(active_mask, variances_new, variances)

        weights_new = nk / float(n)
        weights_new = weights_new.clamp_min(min_weight)
        weights_new = weights_new / weights_new.sum(dim=-1, keepdim=True)

        lower_bound_tensor = total_log_likelihood / float(bsz * n)
        lower_bound_history.append(lower_bound_tensor.detach())

        if verbose:
            lower_bound = lower_bound_tensor.item()
            shift = (means_new - means).norm(dim=-1).max().item()
            print(
                f"Iter {iteration}, lower_bound: {lower_bound:.6f}, "
                f"mean_shift: {shift:.6f}"
            )

        means = means_new
        variances = variances_new
        weights = weights_new

        should_check_convergence = iteration + 1 < max_iters
        if (
            should_check_convergence
            and prev_lower_bound_tensor is not None
            and bool(torch.abs(lower_bound_tensor - prev_lower_bound_tensor).lt(tol).item())
        ):
            break
        if should_check_convergence:
            prev_lower_bound_tensor = lower_bound_tensor.detach()

    if lower_bound_history and torch.is_tensor(lower_bound_history[0]):
        lower_bound_history = [
            float(value) for value in torch.stack(lower_bound_history).detach().cpu().tolist()
        ]

    if not compute_labels:
        labels = None
    elif use_triton_labels:
        try:
            labels = spherical_assign_triton(
                x,
                means,
                variances.to(torch.float32),
                weights.to(torch.float32),
                x_sq=x_sq_all,
                means_sq=(means.to(torch.float32) ** 2).sum(dim=-1),
                log_weights=torch.log(weights.to(torch.float32)),
                config=(
                    {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
                    if d <= 64 and n_components <= 256
                    else None
                ),
            )
            triton_labels_used = True
        except Exception:
            labels = spherical_assign_torch_native_chunked(
                x,
                means,
                variances,
                weights,
                chunk_size_N=chunk_size_N,
                chunk_size_K=chunk_size_K,
            )
    else:
        labels = spherical_assign_torch_native_chunked(
            x,
            means,
            variances,
            weights,
            chunk_size_N=chunk_size_N,
            chunk_size_K=chunk_size_K,
        )
    info: Dict[str, object] = {
        "n_iter": iteration + 1,
        "lower_bound": lower_bound_history[-1],
        "lower_bound_history": lower_bound_history,
        "init_source": init_source,
        "triton_estep_enabled": bool(
            (gmm_use_triton_estep_enabled and not triton_estep_failed)
            or (triton_fused_update_used and not triton_streaming_update_failed)
            or (triton_approx_update_used and not triton_approx_update_failed)
        ),
        "triton_fused_update_enabled": bool(triton_fused_update_used and not triton_streaming_update_failed),
        "triton_approx_topk_enabled": bool(triton_approx_update_used and not triton_approx_update_failed),
        "triton_streaming_update_enabled": bool(
            (
                (gmm_use_triton_streaming_update_enabled or triton_fused_update_used)
                and not triton_streaming_update_failed
            )
            or (triton_approx_update_used and not triton_approx_update_failed)
        ),
        "triton_labels_enabled": bool(triton_labels_used),
        "approximate_em_enabled": bool(effective_approx_top_k is not None),
        "approx_top_k": effective_approx_top_k,
    }
    return labels, means, variances, weights, info


@torch.no_grad()
def batch_gmm_Diagonal_torch_native(
    x: torch.Tensor,
    n_components: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_means: Optional[torch.Tensor] = None,
    init_variances: Optional[torch.Tensor] = None,
    init_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
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
    approx_top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    if x.ndim != 3:
        raise ValueError("x must have shape (B, N, D)")

    bsz, n, d = x.shape
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n == 0:
        raise ValueError("x must contain at least one sample")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if chunk_size_N <= 0 or chunk_size_K <= 0:
        raise ValueError("chunk_size_N and chunk_size_K must be positive")
    if min_weight <= 0.0:
        raise ValueError("min_weight must be positive")
    effective_approx_top_k = _resolve_approx_top_k(approx_top_k, n_components)

    means, variances, weights, init_source = _initialize_diag_parameters(
        x,
        n_components,
        init_means=init_means,
        init_variances=init_variances,
        init_weights=init_weights,
        init_params=init_params,
        reg_covar=reg_covar,
        kmeans_max_iters=kmeans_init_iters,
        kmeans_tol=kmeans_init_tol,
        kmeans_use_triton=kmeans_use_triton,
        verbose=verbose,
    )

    prev_lower_bound = None
    lower_bound_history = []
    diag_auto_triton = _auto_use_triton_diag(x, n_components)
    diag_use_triton = _resolve_triton_option(
        gmm_use_triton,
        diag_auto_triton,
        _triton_diag_supported(x, n_components),
        "gmm_use_triton",
    )
    if effective_approx_top_k is not None:
        diag_use_triton = False
    diag_triton_failed = False
    diag_fused_used = False
    diag_log_norm_buffer = None
    diag_partial_buffers = None
    if diag_use_triton:
        diag_block_n, _, _ = _triton_diag_update_config(d, n_components)
        max_chunk_n = min(chunk_size_N, n)
        max_n_blocks = (max_chunk_n + diag_block_n - 1) // diag_block_n
        diag_log_norm_buffer = torch.empty(
            (bsz, max_chunk_n),
            device=x.device,
            dtype=torch.float32,
        )
        diag_partial_buffers = (
            torch.empty((bsz, max_n_blocks, n_components), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks, n_components, d), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks, n_components, d), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks), device=x.device, dtype=torch.float32),
        )

    for iteration in range(max_iters):
        collect_lower_bound = bool(tol > 0.0 or verbose or iteration + 1 == max_iters)
        variances = variances.clamp_min(reg_covar)
        weights = weights.clamp_min(min_weight)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        log_weights = torch.log(weights.to(torch.float32))
        precision = variances.to(torch.float32).clamp_min(1e-30).reciprocal()
        logdet = torch.log(variances.to(torch.float32).clamp_min(1e-30)).sum(dim=-1)
        weighted_means = means.to(torch.float32) * precision
        mean_precision_mean = (means.to(torch.float32) * weighted_means).sum(dim=-1)
        nk = torch.zeros((bsz, n_components), device=x.device, dtype=torch.float32)
        sum_x = torch.zeros((bsz, n_components, d), device=x.device, dtype=torch.float32)
        sum_x_sq = torch.zeros((bsz, n_components, d), device=x.device, dtype=torch.float32)
        total_log_likelihood = torch.zeros((), device=x.device, dtype=torch.float32)

        for n_start in range(0, n, chunk_size_N):
            n_end = min(n_start + chunk_size_N, n)
            x_chunk = x[:, n_start:n_end, :]
            x_chunk_f = x_chunk.to(torch.float32)
            if effective_approx_top_k is not None:
                topk_logits, topk_indices = _topk_logits_for_chunk(
                    x_chunk,
                    means,
                    variances,
                    covariance_type="diag",
                    chunk_size_K=chunk_size_K,
                    top_k=effective_approx_top_k,
                    terms=(
                        precision,
                        logdet,
                        weighted_means,
                        mean_precision_mean,
                        log_weights,
                    ),
                )
                total_log_likelihood = _accumulate_topk_stats(
                    x_chunk,
                    topk_logits,
                    topk_indices,
                    covariance_type="diag",
                    nk=nk,
                    sum_x=sum_x,
                    sum_x_sq=sum_x_sq,
                    total_log_likelihood=total_log_likelihood,
                )
                continue
            if diag_use_triton:
                try:
                    fused_config = (
                        fused_single_tile_update_config(d, n_components, "diag")
                        if (_HAS_TRITON_FUSED_UPDATE and x.is_cuda)
                        else None
                    )
                    if fused_config is not None:
                        nk_tile, sum_x_tile, sum_x_sq_tile, ll_tile = (
                            triton_fused_single_tile_update_diag(
                                x_chunk,
                                precision,
                                weighted_means,
                                mean_precision_mean,
                                logdet,
                                log_weights,
                                partial_nk=None if diag_partial_buffers is None else diag_partial_buffers[0],
                                partial_sum_x=None if diag_partial_buffers is None else diag_partial_buffers[1],
                                partial_sum_x_sq=None if diag_partial_buffers is None else diag_partial_buffers[2],
                                partial_log_likelihood=None if diag_partial_buffers is None else diag_partial_buffers[3],
                                **fused_config,
                            )
                        )
                        if collect_lower_bound:
                            total_log_likelihood = total_log_likelihood + ll_tile
                        nk += nk_tile
                        sum_x += sum_x_tile
                        sum_x_sq += sum_x_sq_tile
                        diag_fused_used = True
                        continue

                    log_norm_out = (
                        diag_log_norm_buffer[:, : n_end - n_start]
                        if diag_log_norm_buffer is not None
                        else None
                    )
                    log_norm = diag_logsumexp_triton(
                        x_chunk,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        out=log_norm_out,
                    )
                    block_n, block_d, block_k = _triton_diag_update_config(d, n_components)
                    nk_tile, sum_x_tile, sum_x_sq_tile = triton_blocked_update_diag(
                        x_chunk,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        log_norm,
                        partial_nk=None if diag_partial_buffers is None else diag_partial_buffers[0],
                        partial_sum_x=None if diag_partial_buffers is None else diag_partial_buffers[1],
                        partial_sum_x_sq=None if diag_partial_buffers is None else diag_partial_buffers[2],
                        BLOCK_N=block_n,
                        BLOCK_D=block_d,
                        BLOCK_K=block_k,
                    )
                    if collect_lower_bound:
                        total_log_likelihood = total_log_likelihood + log_norm.sum()
                    nk += nk_tile
                    sum_x += sum_x_tile
                    sum_x_sq += sum_x_sq_tile
                    continue
                except Exception:
                    diag_triton_failed = True
                    diag_use_triton = False

            if n_components <= chunk_size_K:
                logits = _compute_diag_chunk_logits(
                    x_chunk,
                    means,
                    variances,
                    log_weights,
                    precision_chunk=precision,
                    logdet_chunk=logdet,
                    weighted_means_chunk=weighted_means,
                    mean_precision_mean_chunk=mean_precision_mean,
                )
                log_norm = torch.logsumexp(logits, dim=-1)
                if collect_lower_bound:
                    total_log_likelihood = total_log_likelihood + log_norm.sum()
                resp = logits.sub(log_norm.unsqueeze(-1)).exp_()
                nk += resp.sum(dim=1)
                sum_x += torch.bmm(resp.transpose(1, 2), x_chunk_f)
                sum_x_sq += torch.bmm(resp.transpose(1, 2), x_chunk_f.square())
                continue

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
            if collect_lower_bound:
                total_log_likelihood = total_log_likelihood + log_norm.sum()

            for k_start in range(0, n_components, chunk_size_K):
                k_end = min(k_start + chunk_size_K, n_components)
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
                resp = torch.exp(logits - log_norm.unsqueeze(-1))
                nk[:, k_start:k_end] += resp.sum(dim=1)
                sum_x[:, k_start:k_end, :] += torch.bmm(resp.transpose(1, 2), x_chunk_f)
                sum_x_sq[:, k_start:k_end, :] += torch.bmm(resp.transpose(1, 2), x_chunk_f.square())

        active_mask = nk > min_weight
        nk_safe = nk.clamp_min(min_weight)
        means_new = (sum_x / nk_safe.unsqueeze(-1)).to(x.dtype)
        means_new = torch.where(active_mask.unsqueeze(-1), means_new, means)

        second_moment = sum_x_sq / nk_safe.unsqueeze(-1)
        variances_new = (second_moment - means_new.to(torch.float32).square()).clamp_min(reg_covar)
        variances_new = torch.where(active_mask.unsqueeze(-1), variances_new.to(x.dtype), variances)

        weights_new = nk / float(n)
        weights_new = weights_new.clamp_min(min_weight)
        weights_new = weights_new / weights_new.sum(dim=-1, keepdim=True)
        weights_new = weights_new.to(x.dtype)

        lower_bound_tensor = (
            total_log_likelihood / float(bsz * n)
            if collect_lower_bound
            else torch.zeros((), device=x.device, dtype=torch.float32)
        )
        need_lower_bound_host = bool(verbose or tol > 0.0)
        lower_bound_value = (
            float(lower_bound_tensor.item())
            if need_lower_bound_host
            else lower_bound_tensor.detach()
        )
        lower_bound_history.append(lower_bound_value)

        if verbose:
            mean_shift = (means_new - means).norm(dim=-1).max().item()
            var_shift = (
                variances_new.to(torch.float32) - variances.to(torch.float32)
            ).abs().max().item()
            print(
                f"Iter {iteration}, lower_bound: {float(lower_bound_value):.6f}, "
                f"mean_shift: {mean_shift:.6f}, variance_shift: {var_shift:.6f}"
            )

        means = means_new
        variances = variances_new
        weights = weights_new

        if tol > 0.0 and prev_lower_bound is not None and abs(float(lower_bound_value) - prev_lower_bound) < tol:
            break
        if tol > 0.0:
            prev_lower_bound = float(lower_bound_value)

    if lower_bound_history and torch.is_tensor(lower_bound_history[0]):
        lower_bound_history = [
            float(value) for value in torch.stack(lower_bound_history).detach().cpu().tolist()
        ]

    labels = (
        diagonal_assign_torch_native_chunked(
            x,
            means,
            variances,
            weights,
            chunk_size_N=chunk_size_N,
            chunk_size_K=chunk_size_K,
        )
        if compute_labels
        else None
    )
    info: Dict[str, object] = {
        "n_iter": iteration + 1,
        "lower_bound": lower_bound_history[-1],
        "lower_bound_history": lower_bound_history,
        "init_source": init_source,
        "triton_estep_enabled": bool((diag_use_triton or diag_fused_used) and not diag_triton_failed),
        "triton_fused_update_enabled": bool(diag_fused_used and not diag_triton_failed),
        "triton_streaming_update_enabled": bool((diag_use_triton or diag_fused_used) and not diag_triton_failed),
        "approximate_em_enabled": bool(effective_approx_top_k is not None),
        "approx_top_k": effective_approx_top_k,
    }
    return labels, means, variances, weights, info


@torch.no_grad()
def _batch_gmm_matrix_torch_native(
    x: torch.Tensor,
    n_components: int,
    *,
    covariance_type: str,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_means: Optional[torch.Tensor] = None,
    init_covariances: Optional[torch.Tensor] = None,
    init_weights: Optional[torch.Tensor] = None,
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
    approx_top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    if x.ndim != 3:
        raise ValueError("x must have shape (B, N, D)")
    if covariance_type not in {"full", "tied"}:
        raise ValueError("covariance_type must be 'full' or 'tied'")

    bsz, n, d = x.shape
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n == 0:
        raise ValueError("x must contain at least one sample")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if chunk_size_N <= 0 or chunk_size_K <= 0:
        raise ValueError("chunk_size_N and chunk_size_K must be positive")
    if min_weight <= 0.0:
        raise ValueError("min_weight must be positive")
    effective_approx_top_k = _resolve_approx_top_k(approx_top_k, n_components)

    if covariance_type == "full":
        means, covariances, weights, init_source = _initialize_full_parameters(
            x,
            n_components,
            init_means=init_means,
            init_covariances=init_covariances,
            init_weights=init_weights,
            init_params=init_params,
            reg_covar=reg_covar,
            kmeans_max_iters=kmeans_init_iters,
            kmeans_tol=kmeans_init_tol,
            kmeans_use_triton=kmeans_use_triton,
            verbose=verbose,
        )
    else:
        means, covariances, weights, init_source = _initialize_tied_parameters(
            x,
            n_components,
            init_means=init_means,
            init_covariance=init_covariances,
            init_weights=init_weights,
            init_params=init_params,
            reg_covar=reg_covar,
            kmeans_max_iters=kmeans_init_iters,
            kmeans_tol=kmeans_init_tol,
            kmeans_use_triton=kmeans_use_triton,
            verbose=verbose,
        )

    prev_lower_bound = None
    lower_bound_history = []
    eye = _eye_like_covariance(d, x.device, torch.float32)
    effective_chunk_size_N = chunk_size_N
    if covariance_type == "tied" and x.is_cuda:
        effective_chunk_size_N = _auto_tied_chunk_size(n, n_components, chunk_size_N)
    full_auto_triton = covariance_type == "full" and _auto_use_triton_full(x, n_components)
    full_use_triton = covariance_type == "full" and _resolve_triton_option(
        gmm_use_triton,
        full_auto_triton,
        _triton_full_supported(x, n_components),
        "gmm_use_triton",
    )
    full_triton_failed = False
    full_triton_used = False
    full_log_norm_buffer = None
    full_partial_buffers = None
    if full_use_triton:
        full_block_n, full_block_d, _ = _triton_full_update_config(d, n_components)
        max_chunk_n = min(effective_chunk_size_N, n)
        max_n_blocks = (max_chunk_n + full_block_n - 1) // full_block_n
        full_log_norm_buffer = torch.empty((bsz, max_chunk_n), device=x.device, dtype=torch.float32)
        full_partial_buffers = (
            torch.empty((bsz, max_n_blocks, n_components), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks, n_components, d), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks, n_components, d, d), device=x.device, dtype=torch.float32),
        )
    tied_auto_triton = covariance_type == "tied" and _auto_use_triton_tied(x, n_components)
    tied_use_triton = covariance_type == "tied" and _resolve_triton_option(
        gmm_use_triton,
        tied_auto_triton,
        _triton_tied_supported(x, n_components),
        "gmm_use_triton",
    )
    if effective_approx_top_k is not None:
        full_use_triton = False
        tied_use_triton = False
    tied_triton_failed = False
    tied_triton_used = False
    tied_fused_used = False
    tied_log_norm_buffer = None
    tied_partial_buffers = None
    tied_unit_variances = None
    if tied_use_triton:
        tied_block_n, _, _ = _triton_tied_update_config(d, n_components)
        max_chunk_n = min(effective_chunk_size_N, n)
        max_n_blocks = (max_chunk_n + tied_block_n - 1) // tied_block_n
        tied_log_norm_buffer = torch.empty((bsz, max_chunk_n), device=x.device, dtype=torch.float32)
        tied_partial_buffers = (
            torch.empty((bsz, max_n_blocks, n_components), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks, n_components, d), device=x.device, dtype=torch.float32),
            torch.empty((bsz, max_n_blocks), device=x.device, dtype=torch.float32),
        )
        tied_unit_variances = torch.ones((bsz, n_components), device=x.device, dtype=torch.float32)
    tied_total_xx = None
    if covariance_type == "tied":
        tied_total_xx = torch.zeros((bsz, d, d), device=x.device, dtype=torch.float32)
        for n_start in range(0, n, effective_chunk_size_N):
            n_end = min(n_start + effective_chunk_size_N, n)
            x_chunk_f = x[:, n_start:n_end, :].to(torch.float32)
            tied_total_xx += torch.bmm(x_chunk_f.transpose(1, 2), x_chunk_f)

    for iteration in range(max_iters):
        collect_lower_bound = bool(tol > 0.0 or verbose or iteration + 1 == max_iters)
        weights = weights.clamp_min(min_weight)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        log_weights = torch.log(weights.to(torch.float32))
        precision, logdet = _precision_and_logdet(covariances)
        if covariance_type == "full":
            precision_means = torch.einsum("bkde,bke->bkd", precision, means.to(torch.float32))
        else:
            precision_means = torch.bmm(means.to(torch.float32), precision.transpose(1, 2))
        mean_precision_mean = (means.to(torch.float32) * precision_means).sum(dim=-1)
        tied_x_projected = None
        tied_x_projected_sq = None
        tied_means_projected = None
        tied_means_projected_sq = None
        tied_fused_config = (
            fused_single_tile_update_config(d, n_components, "tied")
            if (
                tied_use_triton
                and _HAS_TRITON_FUSED_UPDATE
                and x.is_cuda
            )
            else None
        )
        if tied_use_triton:
            try:
                chol_precision = torch.linalg.cholesky(precision)
                tied_means_projected = torch.bmm(means.to(torch.float32), chol_precision)
                tied_means_projected_sq = tied_means_projected.square().sum(dim=-1)
                if tied_fused_config is None:
                    tied_x_projected = torch.bmm(x.to(torch.float32), chol_precision)
                    tied_x_projected_sq = tied_x_projected.square().sum(dim=-1)
            except Exception:
                tied_triton_failed = True
                tied_use_triton = False
                tied_fused_config = None

        nk = torch.zeros((bsz, n_components), device=x.device, dtype=torch.float32)
        sum_x = torch.zeros((bsz, n_components, d), device=x.device, dtype=torch.float32)
        if covariance_type == "full":
            sum_xx = torch.zeros((bsz, n_components, d, d), device=x.device, dtype=torch.float32)
            total_xx = None
        else:
            sum_xx = None
            total_xx = tied_total_xx
        total_log_likelihood = torch.zeros((), device=x.device, dtype=torch.float32)

        for n_start in range(0, n, effective_chunk_size_N):
            n_end = min(n_start + effective_chunk_size_N, n)
            x_chunk = x[:, n_start:n_end, :]
            x_chunk_f = x_chunk.to(torch.float32)
            if effective_approx_top_k is not None:
                topk_logits, topk_indices = _topk_logits_for_chunk(
                    x_chunk,
                    means,
                    covariances,
                    covariance_type=covariance_type,
                    chunk_size_K=chunk_size_K,
                    top_k=effective_approx_top_k,
                    terms=(
                        precision,
                        logdet,
                        precision_means,
                        mean_precision_mean,
                        log_weights,
                    ),
                )
                total_log_likelihood = _accumulate_topk_stats(
                    x_chunk,
                    topk_logits,
                    topk_indices,
                    covariance_type=covariance_type,
                    nk=nk,
                    sum_x=sum_x,
                    sum_xx=sum_xx,
                    total_log_likelihood=total_log_likelihood,
                )
                continue
            if full_use_triton:
                try:
                    log_norm_out = (
                        full_log_norm_buffer[:, : n_end - n_start]
                        if full_log_norm_buffer is not None
                        else None
                    )
                    log_norm = full_logsumexp_triton(
                        x_chunk,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        out=log_norm_out,
                    )
                    block_n, block_d, block_k = _triton_full_update_config(d, n_components)
                    nk_tile, sum_x_tile, sum_xx_tile = triton_blocked_update_full(
                        x_chunk,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        log_norm,
                        partial_nk=None if full_partial_buffers is None else full_partial_buffers[0],
                        partial_sum_x=None if full_partial_buffers is None else full_partial_buffers[1],
                        partial_sum_xx=None if full_partial_buffers is None else full_partial_buffers[2],
                        BLOCK_N=block_n,
                        BLOCK_D=block_d,
                        BLOCK_K=block_k,
                    )
                    if collect_lower_bound:
                        total_log_likelihood = total_log_likelihood + log_norm.sum()
                    nk += nk_tile
                    sum_x += sum_x_tile
                    sum_xx += sum_xx_tile
                    full_triton_used = True
                    continue
                except Exception:
                    full_triton_failed = True
                    full_use_triton = False

            if tied_use_triton:
                try:
                    if tied_fused_config is not None:
                        nk_tile, sum_x_tile, ll_tile = triton_fused_single_tile_update_tied_native(
                            x_chunk,
                            chol_precision,
                            tied_means_projected,
                            tied_means_projected_sq,
                            logdet,
                            log_weights,
                            partial_nk=None if tied_partial_buffers is None else tied_partial_buffers[0],
                            partial_sum_x=None if tied_partial_buffers is None else tied_partial_buffers[1],
                            partial_log_likelihood=None if tied_partial_buffers is None else tied_partial_buffers[2],
                            **tied_fused_config,
                        )
                        if collect_lower_bound:
                            total_log_likelihood = total_log_likelihood + ll_tile
                        nk += nk_tile
                        sum_x += sum_x_tile
                        tied_triton_used = True
                        tied_fused_used = True
                        continue

                    log_norm_out = (
                        tied_log_norm_buffer[:, : n_end - n_start]
                        if tied_log_norm_buffer is not None
                        else None
                    )
                    log_norm = spherical_logsumexp_triton(
                        tied_x_projected[:, n_start:n_end, :],
                        tied_means_projected,
                        tied_unit_variances,
                        weights.to(torch.float32),
                        x_sq=tied_x_projected_sq[:, n_start:n_end],
                        out=log_norm_out,
                        means_sq=tied_means_projected_sq,
                        log_weights=log_weights,
                        config=_triton_tied_logsum_config(d, n_components),
                        unit_variance=True,
                    )
                    block_n, block_d, block_k = _triton_tied_update_config(d, n_components)
                    nk_tile, sum_x_tile = triton_blocked_update_tied_projected(
                        tied_x_projected[:, n_start:n_end, :],
                        x_chunk,
                        tied_means_projected,
                        log_weights,
                        log_norm,
                        x_projected_sq=tied_x_projected_sq[:, n_start:n_end],
                        means_projected_sq=tied_means_projected_sq,
                        partial_nk=None if tied_partial_buffers is None else tied_partial_buffers[0],
                        partial_sum_x=None if tied_partial_buffers is None else tied_partial_buffers[1],
                        BLOCK_N=block_n,
                        BLOCK_D=block_d,
                        BLOCK_K=block_k,
                    )
                    if collect_lower_bound:
                        total_log_likelihood = (
                            total_log_likelihood
                            + log_norm.sum()
                            - 0.5 * float(n_end - n_start) * logdet.sum()
                        )
                    nk += nk_tile
                    sum_x += sum_x_tile
                    tied_triton_used = True
                    continue
                except Exception:
                    tied_triton_failed = True
                    tied_use_triton = False

            if n_components <= chunk_size_K:
                if covariance_type == "full":
                    logits = _compute_full_chunk_logits(
                        x_chunk,
                        means,
                        covariances,
                        log_weights,
                        precision_chunk=precision,
                        logdet_chunk=logdet,
                        precision_means_chunk=precision_means,
                        mean_precision_mean_chunk=mean_precision_mean,
                    )
                else:
                    logits = _compute_tied_chunk_logits(
                        x_chunk,
                        means,
                        covariances,
                        log_weights,
                        precision=precision,
                        logdet=logdet,
                        precision_means_chunk=precision_means,
                        mean_precision_mean_chunk=mean_precision_mean,
                    )
                log_norm = torch.logsumexp(logits, dim=-1)
                if collect_lower_bound:
                    total_log_likelihood = total_log_likelihood + log_norm.sum()
                resp = logits.sub(log_norm.unsqueeze(-1)).exp_()
                nk += resp.sum(dim=1)
                sum_x += torch.bmm(resp.transpose(1, 2), x_chunk_f)
                if covariance_type == "full":
                    sum_xx += torch.einsum(
                        "bnk,bnd,bne->bkde",
                        resp,
                        x_chunk_f,
                        x_chunk_f,
                    )
                continue

            log_norm = _matrix_stream_log_normalizer(
                x_chunk,
                means,
                covariances,
                weights,
                chunk_size_K=chunk_size_K,
                covariance_type=covariance_type,
                precision=precision,
                logdet=logdet,
                log_weights=log_weights,
                precision_means=precision_means,
                mean_precision_mean=mean_precision_mean,
            )
            if collect_lower_bound:
                total_log_likelihood = total_log_likelihood + log_norm.sum()

            for k_start in range(0, n_components, chunk_size_K):
                k_end = min(k_start + chunk_size_K, n_components)
                if covariance_type == "full":
                    logits = _compute_full_chunk_logits(
                        x_chunk,
                        means[:, k_start:k_end, :],
                        covariances[:, k_start:k_end, :, :],
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
                        covariances,
                        log_weights[:, k_start:k_end],
                        precision=precision,
                        logdet=logdet,
                        precision_means_chunk=precision_means[:, k_start:k_end, :],
                        mean_precision_mean_chunk=mean_precision_mean[:, k_start:k_end],
                    )
                resp = torch.exp(logits - log_norm.unsqueeze(-1))
                nk[:, k_start:k_end] += resp.sum(dim=1)
                sum_x[:, k_start:k_end, :] += torch.bmm(resp.transpose(1, 2), x_chunk_f)
                if covariance_type == "full":
                    sum_xx[:, k_start:k_end, :, :] += torch.einsum(
                        "bnk,bnd,bne->bkde",
                        resp,
                        x_chunk_f,
                        x_chunk_f,
                    )

        active_mask = nk > min_weight
        nk_safe = nk.clamp_min(min_weight)
        means_new = (sum_x / nk_safe.unsqueeze(-1)).to(x.dtype)
        means_new = torch.where(active_mask.unsqueeze(-1), means_new, means)
        means_f = means_new.to(torch.float32)
        means_outer = means_f.unsqueeze(-1) * means_f.unsqueeze(-2)

        if covariance_type == "full":
            scatter = sum_xx - nk[..., None, None] * means_outer
            covariances_new = scatter / nk_safe[..., None, None]
            covariances_new = _symmetrize_matrix(covariances_new) + reg_covar * eye
            covariances_new = torch.where(
                active_mask[..., None, None],
                covariances_new.to(x.dtype),
                covariances,
            )
        else:
            scatter = total_xx - (nk[..., None, None] * means_outer).sum(dim=1)
            covariances_new = scatter / float(n)
            covariances_new = (_symmetrize_matrix(covariances_new) + reg_covar * eye).to(x.dtype)

        weights_new = nk / float(n)
        weights_new = weights_new.clamp_min(min_weight)
        weights_new = weights_new / weights_new.sum(dim=-1, keepdim=True)
        weights_new = weights_new.to(x.dtype)

        lower_bound_tensor = (
            total_log_likelihood / float(bsz * n)
            if collect_lower_bound
            else torch.zeros((), device=x.device, dtype=torch.float32)
        )
        need_lower_bound_host = bool(verbose or tol > 0.0)
        lower_bound_value = (
            float(lower_bound_tensor.item())
            if need_lower_bound_host
            else lower_bound_tensor.detach()
        )
        lower_bound_history.append(lower_bound_value)

        if verbose:
            mean_shift = (means_new - means).norm(dim=-1).max().item()
            covariance_shift = (
                covariances_new.to(torch.float32) - covariances.to(torch.float32)
            ).abs().max().item()
            print(
                f"Iter {iteration}, lower_bound: {float(lower_bound_value):.6f}, "
                f"mean_shift: {mean_shift:.6f}, covariance_shift: {covariance_shift:.6f}"
            )

        means = means_new
        covariances = covariances_new
        weights = weights_new

        if tol > 0.0 and prev_lower_bound is not None and abs(float(lower_bound_value) - prev_lower_bound) < tol:
            break
        if tol > 0.0:
            prev_lower_bound = float(lower_bound_value)

    if lower_bound_history and torch.is_tensor(lower_bound_history[0]):
        lower_bound_history = [
            float(value) for value in torch.stack(lower_bound_history).detach().cpu().tolist()
        ]

    if not compute_labels:
        labels = None
    elif covariance_type == "full":
        labels = full_assign_torch_native_chunked(
            x,
            means,
            covariances,
            weights,
            chunk_size_N=chunk_size_N,
            chunk_size_K=chunk_size_K,
        )
    else:
        labels = tied_assign_torch_native_chunked(
            x,
            means,
            covariances,
            weights,
            chunk_size_N=effective_chunk_size_N,
            chunk_size_K=chunk_size_K,
        )

    info: Dict[str, object] = {
        "n_iter": iteration + 1,
        "lower_bound": lower_bound_history[-1],
        "lower_bound_history": lower_bound_history,
        "init_source": init_source,
        "triton_estep_enabled": bool(
            (tied_triton_used and not tied_triton_failed)
            or (full_triton_used and not full_triton_failed)
        ),
        "triton_fused_update_enabled": bool(tied_fused_used and not tied_triton_failed),
        "triton_streaming_update_enabled": bool(
            (tied_triton_used and not tied_triton_failed)
            or (full_triton_used and not full_triton_failed)
        ),
        "approximate_em_enabled": bool(effective_approx_top_k is not None),
        "approx_top_k": effective_approx_top_k,
    }
    return labels, means, covariances, weights, info


def batch_gmm_Full_torch_native(
    x: torch.Tensor,
    n_components: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_means: Optional[torch.Tensor] = None,
    init_variances: Optional[torch.Tensor] = None,
    init_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
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
    approx_top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    return _batch_gmm_matrix_torch_native(
        x,
        n_components,
        covariance_type="full",
        max_iters=max_iters,
        tol=tol,
        init_means=init_means,
        init_covariances=init_variances,
        init_weights=init_weights,
        verbose=verbose,
        init_params=init_params,
        reg_covar=reg_covar,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
        kmeans_init_iters=kmeans_init_iters,
        kmeans_init_tol=kmeans_init_tol,
        kmeans_use_triton=kmeans_use_triton,
        gmm_use_triton=gmm_use_triton,
        min_weight=min_weight,
        compute_labels=compute_labels,
        approx_top_k=approx_top_k,
    )


def batch_gmm_Tied_torch_native(
    x: torch.Tensor,
    n_components: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_means: Optional[torch.Tensor] = None,
    init_variances: Optional[torch.Tensor] = None,
    init_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
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
    approx_top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    return _batch_gmm_matrix_torch_native(
        x,
        n_components,
        covariance_type="tied",
        max_iters=max_iters,
        tol=tol,
        init_means=init_means,
        init_covariances=init_variances,
        init_weights=init_weights,
        verbose=verbose,
        init_params=init_params,
        reg_covar=reg_covar,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
        kmeans_init_iters=kmeans_init_iters,
        kmeans_init_tol=kmeans_init_tol,
        kmeans_use_triton=kmeans_use_triton,
        gmm_use_triton=gmm_use_triton,
        min_weight=min_weight,
        compute_labels=compute_labels,
        approx_top_k=approx_top_k,
    )


def batch_gmm_Spherical(
    x: torch.Tensor,
    n_components: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_means: Optional[torch.Tensor] = None,
    init_variances: Optional[torch.Tensor] = None,
    init_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    init_params: str = "kmeans",
    reg_covar: float = 1e-6,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
    kmeans_init_iters: int = 10,
    kmeans_init_tol: float = 1e-4,
    use_triton: bool = True,
    min_weight: float = 1e-8,
    compute_labels: bool = True,
    approx_top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    return batch_gmm_Spherical_torch_native(
        x,
        n_components,
        max_iters=max_iters,
        tol=tol,
        init_means=init_means,
        init_variances=init_variances,
        init_weights=init_weights,
        verbose=verbose,
        init_params=init_params,
        reg_covar=reg_covar,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
        kmeans_init_iters=kmeans_init_iters,
        kmeans_init_tol=kmeans_init_tol,
        kmeans_use_triton=use_triton,
        gmm_use_triton_estep="auto" if use_triton else False,
        gmm_use_triton_streaming_update="auto" if use_triton else False,
        min_weight=min_weight,
        compute_labels=compute_labels,
        approx_top_k=approx_top_k,
    )


def batch_gmm_Diagonal(
    x: torch.Tensor,
    n_components: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_means: Optional[torch.Tensor] = None,
    init_variances: Optional[torch.Tensor] = None,
    init_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    init_params: str = "kmeans",
    reg_covar: float = 1e-6,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
    kmeans_init_iters: int = 10,
    kmeans_init_tol: float = 1e-4,
    use_triton: bool = True,
    min_weight: float = 1e-8,
    compute_labels: bool = True,
    approx_top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    return batch_gmm_Diagonal_torch_native(
        x,
        n_components,
        max_iters=max_iters,
        tol=tol,
        init_means=init_means,
        init_variances=init_variances,
        init_weights=init_weights,
        verbose=verbose,
        init_params=init_params,
        reg_covar=reg_covar,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
        kmeans_init_iters=kmeans_init_iters,
        kmeans_init_tol=kmeans_init_tol,
        kmeans_use_triton=use_triton,
        gmm_use_triton="auto" if use_triton else False,
        min_weight=min_weight,
        compute_labels=compute_labels,
        approx_top_k=approx_top_k,
    )


def batch_gmm_Full(
    x: torch.Tensor,
    n_components: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_means: Optional[torch.Tensor] = None,
    init_variances: Optional[torch.Tensor] = None,
    init_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    init_params: str = "kmeans",
    reg_covar: float = 1e-6,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
    kmeans_init_iters: int = 10,
    kmeans_init_tol: float = 1e-4,
    use_triton: bool = True,
    min_weight: float = 1e-8,
    compute_labels: bool = True,
    approx_top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    return batch_gmm_Full_torch_native(
        x,
        n_components,
        max_iters=max_iters,
        tol=tol,
        init_means=init_means,
        init_variances=init_variances,
        init_weights=init_weights,
        verbose=verbose,
        init_params=init_params,
        reg_covar=reg_covar,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
        kmeans_init_iters=kmeans_init_iters,
        kmeans_init_tol=kmeans_init_tol,
        kmeans_use_triton=use_triton,
        gmm_use_triton="auto" if use_triton else False,
        min_weight=min_weight,
        compute_labels=compute_labels,
        approx_top_k=approx_top_k,
    )


def batch_gmm_Tied(
    x: torch.Tensor,
    n_components: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_means: Optional[torch.Tensor] = None,
    init_variances: Optional[torch.Tensor] = None,
    init_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    init_params: str = "kmeans",
    reg_covar: float = 1e-6,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
    kmeans_init_iters: int = 10,
    kmeans_init_tol: float = 1e-4,
    use_triton: bool = True,
    min_weight: float = 1e-8,
    compute_labels: bool = True,
    approx_top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    return batch_gmm_Tied_torch_native(
        x,
        n_components,
        max_iters=max_iters,
        tol=tol,
        init_means=init_means,
        init_variances=init_variances,
        init_weights=init_weights,
        verbose=verbose,
        init_params=init_params,
        reg_covar=reg_covar,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
        kmeans_init_iters=kmeans_init_iters,
        kmeans_init_tol=kmeans_init_tol,
        kmeans_use_triton=use_triton,
        gmm_use_triton="auto" if use_triton else False,
        min_weight=min_weight,
        compute_labels=compute_labels,
        approx_top_k=approx_top_k,
    )


batch_gmm_spherical = batch_gmm_Spherical
batch_gmm_diagonal = batch_gmm_Diagonal
batch_gmm_diag = batch_gmm_Diagonal
batch_gmm_full = batch_gmm_Full
batch_gmm_tied = batch_gmm_Tied
