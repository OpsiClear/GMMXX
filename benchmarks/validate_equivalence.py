from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_gmm2 import FlashGMM
from flash_gmm2.torch_fallback import (
    _compute_chunk_logits,
    batch_gmm_Diagonal_torch_native,
    batch_gmm_Full_torch_native,
    batch_gmm_Spherical_torch_native,
    batch_gmm_Tied_torch_native,
    _compute_diag_chunk_logits,
    _compute_full_chunk_logits,
    _compute_tied_chunk_logits,
    _precision_and_logdet,
    diagonal_assign_torch_native_chunked,
    diagonal_score_samples_torch_native_chunked,
    full_assign_torch_native_chunked,
    full_score_samples_torch_native_chunked,
    spherical_assign_torch_native_chunked,
    spherical_predict_proba_torch_native_chunked,
    spherical_score_samples_torch_native_chunked,
    tied_assign_torch_native_chunked,
    tied_score_samples_torch_native_chunked,
)


@dataclass
class Check:
    name: str
    value: float | bool
    threshold: float | None
    passed: bool


def _require_sklearn():
    try:
        from sklearn.datasets import make_blobs
        from sklearn.metrics import adjusted_rand_score
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise RuntimeError(
            "Install benchmark extras first: python -m pip install -e \".[benchmark]\""
        ) from exc
    return make_blobs, adjusted_rand_score, GaussianMixture


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    denom = torch.maximum(a_f.abs(), b_f.abs()).clamp_min(1e-8)
    return float(((a_f - b_f).abs() / denom).max().item())


def _record(checks: list[Check], name: str, value: float | bool, threshold: float | None = None) -> None:
    if isinstance(value, bool):
        passed = value
    else:
        passed = threshold is not None and math.isfinite(value) and value <= threshold
    checks.append(Check(name, value, threshold, passed))


def _record_info(checks: list[Check], name: str, value: float | bool) -> None:
    checks.append(Check(name, value, None, True))


def _kernel_equivalence(args: argparse.Namespace, device: torch.device) -> list[Check]:
    checks: list[Check] = []
    if device.type != "cuda":
        _record(checks, "kernel_equivalence.skipped_no_cuda", True)
        return checks

    from flash_gmm2 import (
        fused_single_tile_update_config,
        spherical_assign_triton,
        spherical_logsumexp_triton,
        spherical_resp_triton,
        triton_fused_single_tile_update_diag,
        triton_fused_single_tile_update_spherical,
        triton_fused_single_tile_update_tied_native,
        approx_topk_update_spherical_config,
        triton_approx_topk_update_spherical,
        triton_weighted_update_spherical,
        triton_streaming_update_spherical,
    )

    if spherical_assign_triton is None or triton_weighted_update_spherical is None:
        _record(checks, "kernel_equivalence.triton_imported", False)
        return checks

    torch.manual_seed(args.seed)
    bsz, n, d, k = args.batch_size, args.n_samples, args.n_features, args.n_components
    x = torch.randn((bsz, n, d), device=device, dtype=torch.float32)
    means = torch.randn((bsz, k, d), device=device, dtype=torch.float32)
    variances = (0.25 + torch.rand((bsz, k), device=device, dtype=torch.float32) * 2.0).contiguous()
    weights = torch.softmax(torch.randn((bsz, k), device=device, dtype=torch.float32), dim=-1).contiguous()

    scores_torch = spherical_score_samples_torch_native_chunked(
        x, means, variances, weights, chunk_size_N=args.chunk_size_n, chunk_size_K=args.chunk_size_k
    )
    probs_torch = spherical_predict_proba_torch_native_chunked(
        x, means, variances, weights, chunk_size_N=args.chunk_size_n, chunk_size_K=args.chunk_size_k
    )
    labels_torch = spherical_assign_torch_native_chunked(
        x, means, variances, weights, chunk_size_N=args.chunk_size_n, chunk_size_K=args.chunk_size_k
    )

    scores_triton = spherical_logsumexp_triton(x, means, variances, weights)
    probs_triton = spherical_resp_triton(x, means, variances, weights, scores_triton)
    labels_triton = spherical_assign_triton(x, means, variances, weights)
    _sync(device)

    nk_torch = probs_torch.sum(dim=1)
    sum_x_torch = torch.bmm(probs_torch.transpose(1, 2), x)
    x_sq = x.square().sum(dim=-1)
    sum_x_sq_torch = (probs_torch * x_sq.unsqueeze(-1)).sum(dim=1)
    nk_triton, sum_x_triton, sum_x_sq_triton = triton_weighted_update_spherical(x, probs_triton)
    nk_stream, sum_x_stream, sum_x_sq_stream = triton_streaming_update_spherical(
        x,
        means,
        variances,
        weights,
        scores_triton,
        x_sq=x_sq,
        means_sq=(means.to(torch.float32) ** 2).sum(dim=-1),
        log_weights=torch.log(weights.to(torch.float32)),
        BLOCK_N=64 if d > 64 else 128,
        BLOCK_D=128 if d > 64 else 64,
        BLOCK_K=16,
    )
    _sync(device)

    _record(checks, "kernel.labels_equal", bool(torch.equal(labels_torch, labels_triton)))
    _record(checks, "kernel.score_max_abs", _max_abs(scores_torch, scores_triton), args.score_atol)
    _record(checks, "kernel.proba_max_abs", _max_abs(probs_torch, probs_triton), args.proba_atol)
    _record(checks, "kernel.proba_row_sum_max_abs", _max_abs(probs_triton.sum(dim=-1), torch.ones_like(scores_triton)), args.proba_sum_atol)
    _record(checks, "kernel.nk_max_abs", _max_abs(nk_torch, nk_triton), args.stats_atol)
    _record(checks, "kernel.sum_x_max_abs", _max_abs(sum_x_torch, sum_x_triton), args.stats_atol)
    _record(checks, "kernel.sum_x_sq_max_rel", _max_rel(sum_x_sq_torch, sum_x_sq_triton), args.stats_rtol)
    _record(checks, "kernel.stream_nk_max_abs", _max_abs(nk_torch, nk_stream), args.stats_atol)
    _record(checks, "kernel.stream_sum_x_max_abs", _max_abs(sum_x_torch, sum_x_stream), args.stats_atol)
    _record(checks, "kernel.stream_sum_x_sq_max_rel", _max_rel(sum_x_sq_torch, sum_x_sq_stream), args.stats_rtol)
    fused_config = fused_single_tile_update_config(d, k) if fused_single_tile_update_config is not None else None
    if fused_config is not None:
        nk_fused, sum_x_fused, sum_x_sq_fused, ll_fused = triton_fused_single_tile_update_spherical(
            x,
            means,
            variances,
            weights,
            x_sq=x_sq,
            means_sq=(means.to(torch.float32) ** 2).sum(dim=-1),
            log_weights=torch.log(weights.to(torch.float32)),
            **fused_config,
        )
        _sync(device)
        _record(checks, "kernel.fused_nk_max_abs", _max_abs(nk_torch, nk_fused), args.stats_atol)
        _record(checks, "kernel.fused_sum_x_max_abs", _max_abs(sum_x_torch, sum_x_fused), args.stats_atol)
        _record(checks, "kernel.fused_sum_x_sq_max_rel", _max_rel(sum_x_sq_torch, sum_x_sq_fused), args.stats_rtol)
        _record(checks, "kernel.fused_ll_max_abs", abs(float(ll_fused.item() - scores_torch.sum().item())), args.score_atol)

    approx_top_k = min(4, k - 1)
    approx_config = (
        approx_topk_update_spherical_config(d, k, approx_top_k)
        if approx_topk_update_spherical_config is not None and approx_top_k > 0
        else None
    )
    if approx_config is not None:
        logits = _compute_chunk_logits(
            x,
            x_sq,
            means,
            variances,
            torch.log(weights.to(torch.float32)),
        )
        top_vals, top_idx = logits.topk(approx_top_k, dim=-1)
        approx_log_norm = torch.logsumexp(top_vals, dim=-1)
        approx_resp = torch.exp(top_vals - approx_log_norm.unsqueeze(-1))
        nk_approx_torch = torch.zeros_like(nk_torch)
        sum_x_approx_torch = torch.zeros_like(sum_x_torch)
        sum_x_sq_approx_torch = torch.zeros_like(sum_x_sq_torch)
        for slot in range(approx_top_k):
            idx = top_idx[:, :, slot]
            resp_slot = approx_resp[:, :, slot]
            nk_approx_torch.scatter_add_(1, idx, resp_slot)
            sum_x_approx_torch.scatter_add_(
                1,
                idx.unsqueeze(-1).expand(-1, -1, d),
                resp_slot.unsqueeze(-1) * x,
            )
            sum_x_sq_approx_torch.scatter_add_(1, idx, resp_slot * x_sq)
        nk_approx, sum_x_approx, sum_x_sq_approx, ll_approx = triton_approx_topk_update_spherical(
            x,
            means,
            variances,
            weights,
            top_k=approx_top_k,
            x_sq=x_sq,
            means_sq=(means.to(torch.float32) ** 2).sum(dim=-1),
            log_weights=torch.log(weights.to(torch.float32)),
            **approx_config,
        )
        _sync(device)
        _record(checks, "kernel.approx_topk_nk_max_abs", _max_abs(nk_approx_torch, nk_approx), args.stats_atol)
        _record(checks, "kernel.approx_topk_sum_x_max_abs", _max_abs(sum_x_approx_torch, sum_x_approx), args.stats_atol)
        _record(checks, "kernel.approx_topk_sum_x_sq_max_rel", _max_rel(sum_x_sq_approx_torch, sum_x_sq_approx), args.stats_rtol)
        _record(checks, "kernel.approx_topk_ll_max_abs", abs(float(ll_approx.item() - approx_log_norm.sum().item())), args.score_atol)

    from flash_gmm2.assign_diag_triton import diag_logsumexp_triton
    from flash_gmm2.assign_full_triton import full_assign_triton, full_logsumexp_triton, full_resp_triton
    from flash_gmm2.weighted_update_triton import (
        triton_blocked_update_diag,
        triton_blocked_update_full,
        triton_blocked_update_tied_projected,
    )

    diag_variances = (0.25 + torch.rand((bsz, k, d), device=device, dtype=torch.float32) * 2.0).contiguous()
    diag_log_weights = torch.log(weights.to(torch.float32))
    diag_precision = diag_variances.clamp_min(1e-30).reciprocal()
    diag_logdet = torch.log(diag_variances.clamp_min(1e-30)).sum(dim=-1)
    diag_weighted_means = means.to(torch.float32) * diag_precision
    diag_mpm = (means.to(torch.float32) * diag_weighted_means).sum(dim=-1)
    diag_logits = _compute_diag_chunk_logits(
        x,
        means,
        diag_variances,
        diag_log_weights,
        precision_chunk=diag_precision,
        logdet_chunk=diag_logdet,
        weighted_means_chunk=diag_weighted_means,
        mean_precision_mean_chunk=diag_mpm,
    )
    diag_log_norm_torch = torch.logsumexp(diag_logits, dim=-1)
    diag_resp = torch.exp(diag_logits - diag_log_norm_torch.unsqueeze(-1))
    diag_nk_torch = diag_resp.sum(dim=1)
    diag_sum_x_torch = torch.bmm(diag_resp.transpose(1, 2), x.to(torch.float32))
    diag_sum_x_sq_torch = torch.bmm(diag_resp.transpose(1, 2), x.to(torch.float32).square())
    diag_log_norm_triton = diag_logsumexp_triton(
        x,
        diag_precision,
        diag_weighted_means,
        diag_mpm,
        diag_logdet,
        diag_log_weights,
    )
    diag_nk, diag_sum_x, diag_sum_x_sq = triton_blocked_update_diag(
        x,
        diag_precision,
        diag_weighted_means,
        diag_mpm,
        diag_logdet,
        diag_log_weights,
        diag_log_norm_triton,
        BLOCK_N=64 if d <= 32 else 32,
        BLOCK_D=32 if d <= 32 else 64,
        BLOCK_K=64,
    )
    _sync(device)
    _record(checks, "kernel_diag.score_max_abs", _max_abs(diag_log_norm_torch, diag_log_norm_triton), args.score_atol)
    _record(checks, "kernel_diag.nk_max_abs", _max_abs(diag_nk_torch, diag_nk), args.stats_atol)
    _record(checks, "kernel_diag.sum_x_max_abs", _max_abs(diag_sum_x_torch, diag_sum_x), args.stats_atol)
    _record(checks, "kernel_diag.sum_x_sq_max_abs", _max_abs(diag_sum_x_sq_torch, diag_sum_x_sq), args.stats_atol)
    if fused_config is not None:
        diag_nk_fused, diag_sum_x_fused, diag_sum_x_sq_fused, diag_ll_fused = triton_fused_single_tile_update_diag(
            x,
            diag_precision,
            diag_weighted_means,
            diag_mpm,
            diag_logdet,
            diag_log_weights,
            **fused_config,
        )
        _sync(device)
        _record(checks, "kernel_diag.fused_nk_max_abs", _max_abs(diag_nk_torch, diag_nk_fused), args.stats_atol)
        _record(checks, "kernel_diag.fused_sum_x_max_abs", _max_abs(diag_sum_x_torch, diag_sum_x_fused), args.stats_atol)
        _record(checks, "kernel_diag.fused_sum_x_sq_max_abs", _max_abs(diag_sum_x_sq_torch, diag_sum_x_sq_fused), args.stats_atol)
        _record(checks, "kernel_diag.fused_ll_max_abs", abs(float(diag_ll_fused.item() - diag_log_norm_torch.sum().item())), args.score_atol)

    tied_a = torch.randn((bsz, d, d), device=device, dtype=torch.float32)
    tied_cov = torch.bmm(tied_a, tied_a.transpose(1, 2)) + torch.eye(d, device=device, dtype=torch.float32).unsqueeze(0) * 0.5
    tied_precision, tied_logdet = _precision_and_logdet(tied_cov)
    tied_pm = torch.bmm(means.to(torch.float32), tied_precision.transpose(1, 2))
    tied_mpm = (means.to(torch.float32) * tied_pm).sum(dim=-1)
    tied_logits = _compute_tied_chunk_logits(
        x,
        means,
        tied_cov,
        diag_log_weights,
        precision=tied_precision,
        logdet=tied_logdet,
        precision_means_chunk=tied_pm,
        mean_precision_mean_chunk=tied_mpm,
    )
    tied_log_norm_torch = torch.logsumexp(tied_logits, dim=-1)
    tied_resp = torch.exp(tied_logits - tied_log_norm_torch.unsqueeze(-1))
    tied_nk_torch = tied_resp.sum(dim=1)
    tied_sum_x_torch = torch.bmm(tied_resp.transpose(1, 2), x.to(torch.float32))
    tied_chol = torch.linalg.cholesky(tied_precision)
    tied_x_projected = torch.bmm(x.to(torch.float32), tied_chol)
    tied_means_projected = torch.bmm(means.to(torch.float32), tied_chol)
    tied_x_projected_sq = tied_x_projected.square().sum(dim=-1)
    tied_means_projected_sq = tied_means_projected.square().sum(dim=-1)
    tied_log_norm_projected = spherical_logsumexp_triton(
        tied_x_projected,
        tied_means_projected,
        torch.ones((bsz, k), device=device, dtype=torch.float32),
        weights.to(torch.float32),
        x_sq=tied_x_projected_sq,
        means_sq=tied_means_projected_sq,
        log_weights=diag_log_weights,
        unit_variance=True,
    )
    tied_log_norm_triton = tied_log_norm_projected - 0.5 * tied_logdet[:, None]
    tied_nk, tied_sum_x = triton_blocked_update_tied_projected(
        tied_x_projected,
        x,
        tied_means_projected,
        diag_log_weights,
        tied_log_norm_projected,
        x_projected_sq=tied_x_projected_sq,
        means_projected_sq=tied_means_projected_sq,
        BLOCK_N=64 if d <= 32 else 32,
        BLOCK_D=32 if d <= 32 else 64,
        BLOCK_K=64,
    )
    _sync(device)
    _record(checks, "kernel_tied.score_max_abs", _max_abs(tied_log_norm_torch, tied_log_norm_triton), args.score_atol)
    _record(checks, "kernel_tied.nk_max_abs", _max_abs(tied_nk_torch, tied_nk), args.stats_atol)
    _record(checks, "kernel_tied.sum_x_max_abs", _max_abs(tied_sum_x_torch, tied_sum_x), args.stats_atol)
    if fused_config is not None:
        tied_nk_fused, tied_sum_x_fused, tied_ll_fused = triton_fused_single_tile_update_tied_native(
            x,
            tied_chol,
            tied_means_projected,
            tied_means_projected_sq,
            tied_logdet,
            diag_log_weights,
            **fused_config,
        )
        _sync(device)
        _record(checks, "kernel_tied.fused_nk_max_abs", _max_abs(tied_nk_torch, tied_nk_fused), args.stats_atol)
        _record(checks, "kernel_tied.fused_sum_x_max_abs", _max_abs(tied_sum_x_torch, tied_sum_x_fused), args.stats_atol)
        _record(checks, "kernel_tied.fused_ll_max_abs", abs(float(tied_ll_fused.item() - tied_log_norm_torch.sum().item())), args.score_atol)

    full_d = min(8, d)
    full_k = min(8, k)
    x_full = x[:, :, :full_d].contiguous()
    means_full = means[:, :full_k, :full_d].contiguous()
    weights_full = torch.softmax(torch.randn((bsz, full_k), device=device, dtype=torch.float32), dim=-1).contiguous()
    full_a = torch.randn((bsz, full_k, full_d, full_d), device=device, dtype=torch.float32)
    full_cov = torch.matmul(full_a, full_a.transpose(-1, -2)) + torch.eye(full_d, device=device, dtype=torch.float32).view(1, 1, full_d, full_d) * 0.5
    full_log_weights = torch.log(weights_full)
    full_precision, full_logdet = _precision_and_logdet(full_cov)
    full_pm = torch.einsum("bkde,bke->bkd", full_precision, means_full.to(torch.float32))
    full_mpm = (means_full.to(torch.float32) * full_pm).sum(dim=-1)
    full_logits = _compute_full_chunk_logits(
        x_full,
        means_full,
        full_cov,
        full_log_weights,
        precision_chunk=full_precision,
        logdet_chunk=full_logdet,
        precision_means_chunk=full_pm,
        mean_precision_mean_chunk=full_mpm,
    )
    full_log_norm_torch = torch.logsumexp(full_logits, dim=-1)
    full_resp = torch.exp(full_logits - full_log_norm_torch.unsqueeze(-1))
    full_nk_torch = full_resp.sum(dim=1)
    full_sum_x_torch = torch.bmm(full_resp.transpose(1, 2), x_full.to(torch.float32))
    full_sum_xx_torch = torch.einsum("bnk,bnd,bne->bkde", full_resp, x_full.to(torch.float32), x_full.to(torch.float32))
    full_log_norm_triton = full_logsumexp_triton(
        x_full,
        full_precision,
        full_pm,
        full_mpm,
        full_logdet,
        full_log_weights,
    )
    full_resp_triton_values = full_resp_triton(
        x_full,
        full_precision,
        full_pm,
        full_mpm,
        full_logdet,
        full_log_weights,
        full_log_norm_triton,
    )
    full_labels_triton = full_assign_triton(
        x_full,
        full_precision,
        full_pm,
        full_mpm,
        full_logdet,
        full_log_weights,
    )
    full_nk, full_sum_x, full_sum_xx = triton_blocked_update_full(
        x_full,
        full_precision,
        full_pm,
        full_mpm,
        full_logdet,
        full_log_weights,
        full_log_norm_triton,
        BLOCK_N=64,
        BLOCK_D=16,
        BLOCK_K=32,
    )
    _sync(device)
    _record(checks, "kernel_full.score_max_abs", _max_abs(full_log_norm_torch, full_log_norm_triton), args.score_atol)
    _record(checks, "kernel_full.labels_equal", bool(torch.equal(full_labels_triton, full_logits.argmax(dim=-1).to(torch.int32))))
    _record(checks, "kernel_full.proba_max_abs", _max_abs(full_resp, full_resp_triton_values), args.proba_atol)
    _record(checks, "kernel_full.proba_row_sum_max_abs", _max_abs(full_resp_triton_values.sum(dim=-1), torch.ones_like(full_log_norm_triton)), args.proba_sum_atol)
    _record(checks, "kernel_full.nk_max_abs", _max_abs(full_nk_torch, full_nk), args.stats_atol)
    _record(checks, "kernel_full.sum_x_max_abs", _max_abs(full_sum_x_torch, full_sum_x), args.stats_atol)
    _record(checks, "kernel_full.sum_xx_max_abs", _max_abs(full_sum_xx_torch, full_sum_xx), args.stats_atol)
    if d >= 16:
        full16_d = 16
        full16_k = min(8, k)
        x_full16 = x[:, :, :full16_d].contiguous()
        means_full16 = means[:, :full16_k, :full16_d].contiguous()
        weights_full16 = torch.softmax(
            torch.randn((bsz, full16_k), device=device, dtype=torch.float32),
            dim=-1,
        ).contiguous()
        full16_a = torch.randn((bsz, full16_k, full16_d, full16_d), device=device, dtype=torch.float32)
        full16_cov = (
            torch.matmul(full16_a, full16_a.transpose(-1, -2))
            + torch.eye(full16_d, device=device, dtype=torch.float32).view(1, 1, full16_d, full16_d) * 0.5
        )
        full16_log_weights = torch.log(weights_full16)
        full16_precision, full16_logdet = _precision_and_logdet(full16_cov)
        full16_pm = torch.einsum("bkde,bke->bkd", full16_precision, means_full16.to(torch.float32))
        full16_mpm = (means_full16.to(torch.float32) * full16_pm).sum(dim=-1)
        full16_logits = _compute_full_chunk_logits(
            x_full16,
            means_full16,
            full16_cov,
            full16_log_weights,
            precision_chunk=full16_precision,
            logdet_chunk=full16_logdet,
            precision_means_chunk=full16_pm,
            mean_precision_mean_chunk=full16_mpm,
        )
        full16_log_norm_torch = torch.logsumexp(full16_logits, dim=-1)
        full16_resp = torch.exp(full16_logits - full16_log_norm_torch.unsqueeze(-1))
        full16_log_norm_triton = full_logsumexp_triton(
            x_full16,
            full16_precision,
            full16_pm,
            full16_mpm,
            full16_logdet,
            full16_log_weights,
        )
        full16_resp_triton = full_resp_triton(
            x_full16,
            full16_precision,
            full16_pm,
            full16_mpm,
            full16_logdet,
            full16_log_weights,
            full16_log_norm_triton,
        )
        full16_labels_triton = full_assign_triton(
            x_full16,
            full16_precision,
            full16_pm,
            full16_mpm,
            full16_logdet,
            full16_log_weights,
        )
        _sync(device)
        _record(checks, "kernel_full16_infer.score_max_abs", _max_abs(full16_log_norm_torch, full16_log_norm_triton), args.score_atol)
        _record(checks, "kernel_full16_infer.labels_equal", bool(torch.equal(full16_labels_triton, full16_logits.argmax(dim=-1).to(torch.int32))))
        _record(checks, "kernel_full16_infer.proba_max_abs", _max_abs(full16_resp, full16_resp_triton), args.proba_atol)
        _record(checks, "kernel_full16_infer.proba_row_sum_max_abs", _max_abs(full16_resp_triton.sum(dim=-1), torch.ones_like(full16_log_norm_triton)), args.proba_sum_atol)
    return checks


def _fit_equivalence(args: argparse.Namespace, device: torch.device) -> list[Check]:
    checks: list[Check] = []
    make_blobs, adjusted_rand_score, _ = _require_sklearn()
    x_np, _ = make_blobs(
        n_samples=args.fit_samples,
        n_features=args.fit_features,
        centers=args.fit_components,
        cluster_std=args.cluster_std,
        random_state=args.seed,
    )
    x = torch.as_tensor(x_np.astype(np.float32), device=device)

    common = dict(
        d=args.fit_features,
        k=args.fit_components,
        niter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        chunk_size_data=args.chunk_size_n,
        chunk_size_centroids=args.chunk_size_k,
        init_params="random",
        reg_covar=args.reg_covar,
    )
    torch_model = FlashGMM(**common, use_triton=False).fit(x)
    triton_model = FlashGMM(**common, use_triton=True).fit(x)
    labels_torch = torch_model.predict(x).detach().cpu().numpy()
    labels_triton = triton_model.predict(x).detach().cpu().numpy()

    def compare(prefix: str, model: FlashGMM, labels) -> None:
        ari = float(adjusted_rand_score(labels_torch, labels))
        _record(checks, f"{prefix}.label_ari", 1.0 - ari, args.fit_ari_error)
        _record(checks, f"{prefix}.score_abs_diff", abs(float(torch_model.score(x)) - float(model.score(x))), args.fit_score_atol)
        _record(checks, f"{prefix}.means_max_abs", _max_abs(torch_model.means_b, model.means_b), args.fit_param_atol)
        _record(checks, f"{prefix}.variances_max_abs", _max_abs(torch_model.variances_b, model.variances_b), args.fit_param_atol)
        _record(checks, f"{prefix}.weights_max_abs", _max_abs(torch_model.weights_b, model.weights_b), args.fit_param_atol)

    compare("fit.flash_auto", triton_model, labels_triton)
    return checks


def _highdim_equivalence(args: argparse.Namespace, device: torch.device) -> list[Check]:
    checks: list[Check] = []
    if args.highdim_samples <= 0:
        return checks

    _, adjusted_rand_score, _ = _require_sklearn()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    centers = torch.randn(
        (args.highdim_components, args.highdim_features),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) * 5.0
    labels = torch.arange(args.highdim_samples, device=device) % args.highdim_components
    perm = torch.randperm(args.highdim_samples, device=device, generator=generator)
    labels = labels[perm]
    x = centers[labels] + args.cluster_std * torch.randn(
        (args.highdim_samples, args.highdim_features),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    common = dict(
        d=args.highdim_features,
        k=args.highdim_components,
        niter=args.highdim_max_iter,
        tol=args.tol,
        seed=args.seed,
        chunk_size_data=args.chunk_size_n,
        chunk_size_centroids=args.chunk_size_k,
        init_params="random",
        reg_covar=args.reg_covar,
    )
    torch_model = FlashGMM(**common, use_triton=False).fit(x)
    auto_model = FlashGMM(**common, use_triton=True).fit(x)

    labels_torch = torch_model.predict(x).detach().cpu().numpy()
    labels_auto = auto_model.predict(x).detach().cpu().numpy()

    def compare(prefix: str, model: FlashGMM, labels) -> None:
        ari = float(adjusted_rand_score(labels_torch, labels))
        _record_info(checks, f"{prefix}.triton_estep_enabled", bool(model.triton_estep_enabled_))
        _record_info(checks, f"{prefix}.triton_labels_enabled", bool(getattr(model, "triton_labels_enabled_", False)))
        _record(checks, f"{prefix}.label_ari", 1.0 - ari, args.highdim_ari_error)
        _record(checks, f"{prefix}.score_abs_diff", abs(float(torch_model.score(x)) - float(model.score(x))), args.fit_score_atol)
        _record(checks, f"{prefix}.means_max_abs", _max_abs(torch_model.means_b, model.means_b), args.fit_param_atol)
        _record(checks, f"{prefix}.variances_max_abs", _max_abs(torch_model.variances_b, model.variances_b), args.fit_param_atol)
        _record(checks, f"{prefix}.weights_max_abs", _max_abs(torch_model.weights_b, model.weights_b), args.fit_param_atol)

    compare("highdim.auto", auto_model, labels_auto)
    return checks


def _sklearn_equivalence(args: argparse.Namespace, device: torch.device) -> list[Check]:
    checks: list[Check] = []
    make_blobs, adjusted_rand_score, GaussianMixture = _require_sklearn()
    x_np, labels_true = make_blobs(
        n_samples=args.sklearn_samples,
        n_features=args.sklearn_features,
        centers=args.sklearn_components,
        cluster_std=args.cluster_std,
        random_state=args.seed,
    )
    x_np = x_np.astype(np.float32)
    x = torch.as_tensor(x_np, device=device).unsqueeze(0)

    rng = np.random.default_rng(args.seed)
    init_indices = rng.choice(x_np.shape[0], size=args.sklearn_components, replace=False)
    init_means_np = np.ascontiguousarray(x_np[init_indices], dtype=np.float32)
    centered = x_np - x_np.mean(axis=0, keepdims=True)
    init_var = float(np.mean(np.sum(centered * centered, axis=1)) / x_np.shape[1])
    init_variances_np = np.full((args.sklearn_components,), max(init_var, args.reg_covar), dtype=np.float32)
    init_weights_np = np.full((args.sklearn_components,), 1.0 / args.sklearn_components, dtype=np.float32)

    _, means, variances, weights, _ = batch_gmm_Spherical_torch_native(
        x,
        args.sklearn_components,
        max_iters=args.max_iter,
        tol=0.0,
        init_means=torch.as_tensor(init_means_np, device=device).unsqueeze(0),
        init_variances=torch.as_tensor(init_variances_np, device=device).unsqueeze(0),
        init_weights=torch.as_tensor(init_weights_np, device=device).unsqueeze(0),
        reg_covar=args.reg_covar,
        chunk_size_N=args.chunk_size_n,
        chunk_size_K=args.chunk_size_k,
        kmeans_use_triton=False,
        gmm_use_triton_estep=False,
    )
    flash_labels = spherical_assign_torch_native_chunked(x, means, variances, weights).squeeze(0).detach().cpu().numpy()
    flash_score = float(
        spherical_score_samples_torch_native_chunked(x, means, variances, weights).mean().item()
    )

    sklearn_model = GaussianMixture(
        n_components=args.sklearn_components,
        covariance_type="spherical",
        tol=0.0,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        n_init=1,
        means_init=init_means_np,
        weights_init=init_weights_np,
        precisions_init=1.0 / init_variances_np,
        random_state=args.seed,
    ).fit(x_np)
    sklearn_labels = sklearn_model.predict(x_np)
    sklearn_score = float(sklearn_model.score(x_np))

    _record(checks, "sklearn.label_ari_error", 1.0 - float(adjusted_rand_score(flash_labels, sklearn_labels)), args.sklearn_ari_error)
    _record_info(checks, "sklearn.true_label_ari_flash", float(adjusted_rand_score(labels_true, flash_labels)))
    _record(checks, "sklearn.score_abs_diff", abs(flash_score - sklearn_score), args.sklearn_score_atol)

    init_variances_diag_np = np.tile(
        np.maximum(x_np.var(axis=0), args.reg_covar).astype(np.float32),
        (args.sklearn_components, 1),
    )
    _, means_diag, variances_diag, weights_diag, _ = batch_gmm_Diagonal_torch_native(
        x,
        args.sklearn_components,
        max_iters=args.max_iter,
        tol=0.0,
        init_means=torch.as_tensor(init_means_np, device=device).unsqueeze(0),
        init_variances=torch.as_tensor(init_variances_diag_np, device=device).unsqueeze(0),
        init_weights=torch.as_tensor(init_weights_np, device=device).unsqueeze(0),
        reg_covar=args.reg_covar,
        chunk_size_N=args.chunk_size_n,
        chunk_size_K=args.chunk_size_k,
        kmeans_use_triton=False,
    )
    flash_diag_labels = diagonal_assign_torch_native_chunked(
        x,
        means_diag,
        variances_diag,
        weights_diag,
        chunk_size_N=args.chunk_size_n,
        chunk_size_K=args.chunk_size_k,
    ).squeeze(0).detach().cpu().numpy()
    flash_diag_score = float(
        diagonal_score_samples_torch_native_chunked(
            x,
            means_diag,
            variances_diag,
            weights_diag,
            chunk_size_N=args.chunk_size_n,
            chunk_size_K=args.chunk_size_k,
        ).mean().item()
    )

    sklearn_diag_model = GaussianMixture(
        n_components=args.sklearn_components,
        covariance_type="diag",
        tol=0.0,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        n_init=1,
        means_init=init_means_np,
        weights_init=init_weights_np,
        precisions_init=1.0 / init_variances_diag_np,
        random_state=args.seed,
    ).fit(x_np)
    sklearn_diag_labels = sklearn_diag_model.predict(x_np)
    sklearn_diag_score = float(sklearn_diag_model.score(x_np))

    _record(checks, "sklearn_diag.label_ari_error", 1.0 - float(adjusted_rand_score(flash_diag_labels, sklearn_diag_labels)), args.sklearn_ari_error)
    _record_info(checks, "sklearn_diag.true_label_ari_flash", float(adjusted_rand_score(labels_true, flash_diag_labels)))
    _record(checks, "sklearn_diag.score_abs_diff", abs(flash_diag_score - sklearn_diag_score), args.sklearn_score_atol)

    global_cov_np = np.matmul(centered.T, centered) / float(x_np.shape[0])
    global_cov_np = global_cov_np + args.reg_covar * np.eye(x_np.shape[1], dtype=np.float32)
    init_full_cov_np = np.broadcast_to(
        global_cov_np.astype(np.float32),
        (args.sklearn_components, x_np.shape[1], x_np.shape[1]),
    ).copy()
    init_tied_cov_np = np.ascontiguousarray(global_cov_np.astype(np.float32))

    _, means_full, covariances_full, weights_full, _ = batch_gmm_Full_torch_native(
        x,
        args.sklearn_components,
        max_iters=args.max_iter,
        tol=0.0,
        init_means=torch.as_tensor(init_means_np, device=device).unsqueeze(0),
        init_variances=torch.as_tensor(init_full_cov_np, device=device).unsqueeze(0),
        init_weights=torch.as_tensor(init_weights_np, device=device).unsqueeze(0),
        reg_covar=args.reg_covar,
        chunk_size_N=args.chunk_size_n,
        chunk_size_K=args.chunk_size_k,
        kmeans_use_triton=False,
    )
    flash_full_labels = full_assign_torch_native_chunked(
        x,
        means_full,
        covariances_full,
        weights_full,
        chunk_size_N=args.chunk_size_n,
        chunk_size_K=args.chunk_size_k,
    ).squeeze(0).detach().cpu().numpy()
    flash_full_score = float(
        full_score_samples_torch_native_chunked(
            x,
            means_full,
            covariances_full,
            weights_full,
            chunk_size_N=args.chunk_size_n,
            chunk_size_K=args.chunk_size_k,
        ).mean().item()
    )
    sklearn_full_model = GaussianMixture(
        n_components=args.sklearn_components,
        covariance_type="full",
        tol=0.0,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        n_init=1,
        means_init=init_means_np,
        weights_init=init_weights_np,
        precisions_init=np.linalg.inv(init_full_cov_np),
        random_state=args.seed,
    ).fit(x_np)
    sklearn_full_labels = sklearn_full_model.predict(x_np)
    sklearn_full_score = float(sklearn_full_model.score(x_np))

    _record(checks, "sklearn_full.label_ari_error", 1.0 - float(adjusted_rand_score(flash_full_labels, sklearn_full_labels)), args.sklearn_ari_error)
    _record_info(checks, "sklearn_full.true_label_ari_flash", float(adjusted_rand_score(labels_true, flash_full_labels)))
    _record(checks, "sklearn_full.score_abs_diff", abs(flash_full_score - sklearn_full_score), args.sklearn_score_atol)

    _, means_tied, covariance_tied, weights_tied, _ = batch_gmm_Tied_torch_native(
        x,
        args.sklearn_components,
        max_iters=args.max_iter,
        tol=0.0,
        init_means=torch.as_tensor(init_means_np, device=device).unsqueeze(0),
        init_variances=torch.as_tensor(init_tied_cov_np, device=device).unsqueeze(0),
        init_weights=torch.as_tensor(init_weights_np, device=device).unsqueeze(0),
        reg_covar=args.reg_covar,
        chunk_size_N=args.chunk_size_n,
        chunk_size_K=args.chunk_size_k,
        kmeans_use_triton=False,
    )
    flash_tied_labels = tied_assign_torch_native_chunked(
        x,
        means_tied,
        covariance_tied,
        weights_tied,
        chunk_size_N=args.chunk_size_n,
        chunk_size_K=args.chunk_size_k,
    ).squeeze(0).detach().cpu().numpy()
    flash_tied_score = float(
        tied_score_samples_torch_native_chunked(
            x,
            means_tied,
            covariance_tied,
            weights_tied,
            chunk_size_N=args.chunk_size_n,
            chunk_size_K=args.chunk_size_k,
        ).mean().item()
    )
    sklearn_tied_model = GaussianMixture(
        n_components=args.sklearn_components,
        covariance_type="tied",
        tol=0.0,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        n_init=1,
        means_init=init_means_np,
        weights_init=init_weights_np,
        precisions_init=np.linalg.inv(init_tied_cov_np),
        random_state=args.seed,
    ).fit(x_np)
    sklearn_tied_labels = sklearn_tied_model.predict(x_np)
    sklearn_tied_score = float(sklearn_tied_model.score(x_np))

    _record(checks, "sklearn_tied.label_ari_error", 1.0 - float(adjusted_rand_score(flash_tied_labels, sklearn_tied_labels)), args.sklearn_ari_error)
    _record_info(checks, "sklearn_tied.true_label_ari_flash", float(adjusted_rand_score(labels_true, flash_tied_labels)))
    _record(checks, "sklearn_tied.score_abs_diff", abs(flash_tied_score - sklearn_tied_score), args.sklearn_score_atol)
    return checks


def _print_checks(checks: list[Check]) -> bool:
    print(f"{'check':<34} {'value':>14} {'threshold':>14}  status")
    all_passed = True
    for check in checks:
        all_passed = all_passed and check.passed
        value = str(check.value) if isinstance(check.value, bool) else f"{check.value:.8g}"
        threshold = "n/a" if check.threshold is None else f"{check.threshold:.8g}"
        status = "PASS" if check.passed else "FAIL"
        print(f"{check.name:<34} {value:>14} {threshold:>14}  {status}")
    return all_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GMMXX numerical equivalence.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--fit-samples", type=int, default=2048)
    parser.add_argument("--fit-features", type=int, default=16)
    parser.add_argument("--fit-components", type=int, default=6)
    parser.add_argument("--sklearn-samples", type=int, default=2048)
    parser.add_argument("--sklearn-features", type=int, default=8)
    parser.add_argument("--sklearn-components", type=int, default=4)
    parser.add_argument("--highdim-samples", type=int, default=0)
    parser.add_argument("--highdim-features", type=int, default=256)
    parser.add_argument("--highdim-components", type=int, default=8)
    parser.add_argument("--highdim-max-iter", type=int, default=3)
    parser.add_argument("--highdim-ari-error", type=float, default=1e-2)
    parser.add_argument("--cluster-std", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--chunk-size-n", type=int, default=32768)
    parser.add_argument("--chunk-size-k", type=int, default=1024)
    parser.add_argument("--score-atol", type=float, default=5e-2)
    parser.add_argument("--proba-atol", type=float, default=5e-3)
    parser.add_argument("--proba-sum-atol", type=float, default=1e-4)
    parser.add_argument("--stats-atol", type=float, default=5e-2)
    parser.add_argument("--stats-rtol", type=float, default=5e-4)
    parser.add_argument("--fit-score-atol", type=float, default=5e-2)
    parser.add_argument("--fit-param-atol", type=float, default=2e-1)
    parser.add_argument("--fit-ari-error", type=float, default=1e-3)
    parser.add_argument("--sklearn-score-atol", type=float, default=1e-2)
    parser.add_argument("--sklearn-ari-error", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")

    checks: list[Check] = []
    checks.extend(_kernel_equivalence(args, device))
    checks.extend(_fit_equivalence(args, device))
    checks.extend(_highdim_equivalence(args, device))
    checks.extend(_sklearn_equivalence(args, device))
    passed = _print_checks(checks)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
