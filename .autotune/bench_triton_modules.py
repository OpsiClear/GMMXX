from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flash_gmm2.assign_diag_triton import diag_logsumexp_triton
from flash_gmm2.assign_full_triton import full_assign_triton, full_logsumexp_triton, full_resp_triton
from flash_gmm2.assign_spherical_triton import spherical_assign_triton, spherical_logsumexp_triton, spherical_resp_triton
from flash_gmm2.torch_fallback import (
    _precision_and_logdet,
    _triton_blocked_update_config,
    _triton_diag_update_config,
    _triton_full_update_config,
    _triton_tied_logsum_config,
    _triton_tied_update_config,
)
from flash_gmm2.weighted_update_triton import (
    triton_blocked_update_diag,
    triton_blocked_update_full,
    triton_blocked_update_spherical,
    triton_blocked_update_tied_projected,
)


def _sync() -> None:
    torch.cuda.synchronize()


def _time_ms(fn, repeats: int) -> float:
    fn()
    _sync()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / float(repeats)


def _weights(batch: int, k: int, device: torch.device) -> torch.Tensor:
    raw = torch.randn((batch, k), device=device, dtype=torch.float32)
    return torch.softmax(raw, dim=-1).contiguous()


def bench_spherical(device: torch.device, repeats: int, n: int, d: int, k: int) -> dict[str, float]:
    b = 1
    torch.manual_seed(101 + n + d + k)
    x = torch.randn((b, n, d), device=device, dtype=torch.float32)
    means = torch.randn((b, k, d), device=device, dtype=torch.float32)
    variances = (0.25 + 2.0 * torch.rand((b, k), device=device, dtype=torch.float32)).contiguous()
    weights = _weights(b, k, device)
    x_sq = x.square().sum(dim=-1)
    means_sq = means.square().sum(dim=-1)
    log_weights = torch.log(weights)
    log_norm = spherical_logsumexp_triton(
        x, means, variances, weights, x_sq=x_sq, means_sq=means_sq, log_weights=log_weights
    )
    _sync()
    block_n, block_d, block_k = _triton_blocked_update_config(d, k)
    n_blocks = math.ceil(n / block_n)
    partial_nk = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
    partial_sum_x_sq = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
    return {
        f"spherical_logsum_N{n}_D{d}_K{k}": _time_ms(
            lambda: spherical_logsumexp_triton(
                x, means, variances, weights, x_sq=x_sq, means_sq=means_sq, log_weights=log_weights
            ),
            repeats,
        ),
        f"spherical_assign_N{n}_D{d}_K{k}": _time_ms(
            lambda: spherical_assign_triton(
                x, means, variances, weights, x_sq=x_sq, means_sq=means_sq, log_weights=log_weights
            ),
            repeats,
        ),
        f"spherical_resp_N{n}_D{d}_K{k}": _time_ms(
            lambda: spherical_resp_triton(
                x, means, variances, weights, log_norm, x_sq=x_sq, means_sq=means_sq, log_weights=log_weights
            ),
            repeats,
        ),
        f"spherical_update_N{n}_D{d}_K{k}": _time_ms(
            lambda: triton_blocked_update_spherical(
                x,
                means,
                variances,
                weights,
                log_norm,
                x_sq=x_sq,
                means_sq=means_sq,
                log_weights=log_weights,
                partial_nk=partial_nk,
                partial_sum_x=partial_sum_x,
                partial_sum_x_sq=partial_sum_x_sq,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                BLOCK_K=block_k,
            ),
            repeats,
        ),
    }


def bench_diag(device: torch.device, repeats: int, n: int, d: int, k: int) -> dict[str, float]:
    b = 1
    torch.manual_seed(201 + n + d + k)
    x = torch.randn((b, n, d), device=device, dtype=torch.float32)
    means = torch.randn((b, k, d), device=device, dtype=torch.float32)
    variances = (0.25 + 2.0 * torch.rand((b, k, d), device=device, dtype=torch.float32)).contiguous()
    weights = _weights(b, k, device)
    log_weights = torch.log(weights)
    precision = variances.reciprocal()
    logdet = torch.log(variances).sum(dim=-1)
    weighted_means = means * precision
    mean_precision_mean = (means * weighted_means).sum(dim=-1)
    log_norm = diag_logsumexp_triton(x, precision, weighted_means, mean_precision_mean, logdet, log_weights)
    _sync()
    block_n, block_d, block_k = _triton_diag_update_config(d, k)
    n_blocks = math.ceil(n / block_n)
    partial_nk = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
    partial_sum_x_sq = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
    return {
        f"diag_logsum_N{n}_D{d}_K{k}": _time_ms(
            lambda: diag_logsumexp_triton(x, precision, weighted_means, mean_precision_mean, logdet, log_weights),
            repeats,
        ),
        f"diag_update_N{n}_D{d}_K{k}": _time_ms(
            lambda: triton_blocked_update_diag(
                x,
                precision,
                weighted_means,
                mean_precision_mean,
                logdet,
                log_weights,
                log_norm,
                partial_nk=partial_nk,
                partial_sum_x=partial_sum_x,
                partial_sum_x_sq=partial_sum_x_sq,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                BLOCK_K=block_k,
            ),
            repeats,
        ),
    }


def bench_tied(device: torch.device, repeats: int, n: int, d: int, k: int) -> dict[str, float]:
    b = 1
    torch.manual_seed(301 + n + d + k)
    x = torch.randn((b, n, d), device=device, dtype=torch.float32)
    means = torch.randn((b, k, d), device=device, dtype=torch.float32)
    weights = _weights(b, k, device)
    log_weights = torch.log(weights)
    a = torch.randn((b, d, d), device=device, dtype=torch.float32)
    covariance = torch.bmm(a, a.transpose(1, 2)) + torch.eye(d, device=device).unsqueeze(0) * 0.5
    precision, _ = _precision_and_logdet(covariance)
    chol = torch.linalg.cholesky(precision)
    x_projected = torch.bmm(x, chol)
    means_projected = torch.bmm(means, chol)
    x_projected_sq = x_projected.square().sum(dim=-1)
    means_projected_sq = means_projected.square().sum(dim=-1)
    unit_variances = torch.ones((b, k), device=device, dtype=torch.float32)
    tied_logsum_config = _triton_tied_logsum_config(d, k)
    log_norm = spherical_logsumexp_triton(
        x_projected,
        means_projected,
        unit_variances,
        weights,
        x_sq=x_projected_sq,
        means_sq=means_projected_sq,
        log_weights=log_weights,
        config=tied_logsum_config,
        unit_variance=True,
    )
    _sync()
    block_n, block_d, block_k = _triton_tied_update_config(d, k)
    n_blocks = math.ceil(n / block_n)
    partial_nk = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
    return {
        f"tied_projected_logsum_N{n}_D{d}_K{k}": _time_ms(
            lambda: spherical_logsumexp_triton(
                x_projected,
                means_projected,
                unit_variances,
                weights,
                x_sq=x_projected_sq,
                means_sq=means_projected_sq,
                log_weights=log_weights,
                config=tied_logsum_config,
                unit_variance=True,
            ),
            repeats,
        ),
        f"tied_update_N{n}_D{d}_K{k}": _time_ms(
            lambda: triton_blocked_update_tied_projected(
                x_projected,
                x,
                means_projected,
                log_weights,
                log_norm,
                x_projected_sq=x_projected_sq,
                means_projected_sq=means_projected_sq,
                partial_nk=partial_nk,
                partial_sum_x=partial_sum_x,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                BLOCK_K=block_k,
            ),
            repeats,
        ),
    }


def bench_full(device: torch.device, repeats: int, n: int, d: int, k: int) -> dict[str, float]:
    b = 1
    torch.manual_seed(401 + n + d + k)
    x = torch.randn((b, n, d), device=device, dtype=torch.float32)
    means = torch.randn((b, k, d), device=device, dtype=torch.float32)
    weights = _weights(b, k, device)
    log_weights = torch.log(weights)
    a = torch.randn((b, k, d, d), device=device, dtype=torch.float32)
    covariance = torch.matmul(a, a.transpose(-1, -2)) + torch.eye(d, device=device).view(1, 1, d, d) * 0.5
    precision, logdet = _precision_and_logdet(covariance)
    precision_means = torch.einsum("bkde,bke->bkd", precision, means)
    mean_precision_mean = (means * precision_means).sum(dim=-1)
    log_norm = full_logsumexp_triton(x, precision, precision_means, mean_precision_mean, logdet, log_weights)
    _sync()
    block_n, block_d, block_k = _triton_full_update_config(d, k)
    n_blocks = math.ceil(n / block_n)
    partial_nk = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
    partial_sum_xx = torch.empty((b, n_blocks, k, d, d), device=device, dtype=torch.float32)
    return {
        f"full_logsum_N{n}_D{d}_K{k}": _time_ms(
            lambda: full_logsumexp_triton(x, precision, precision_means, mean_precision_mean, logdet, log_weights),
            repeats,
        ),
        f"full_assign_N{n}_D{d}_K{k}": _time_ms(
            lambda: full_assign_triton(x, precision, precision_means, mean_precision_mean, logdet, log_weights),
            repeats,
        ),
        f"full_resp_N{n}_D{d}_K{k}": _time_ms(
            lambda: full_resp_triton(x, precision, precision_means, mean_precision_mean, logdet, log_weights, log_norm),
            repeats,
        ),
        f"full_update_N{n}_D{d}_K{k}": _time_ms(
            lambda: triton_blocked_update_full(
                x,
                precision,
                precision_means,
                mean_precision_mean,
                logdet,
                log_weights,
                log_norm,
                partial_nk=partial_nk,
                partial_sum_x=partial_sum_x,
                partial_sum_xx=partial_sum_xx,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                BLOCK_K=block_k,
            ),
            repeats,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", choices=["quick", "standard"], default="quick")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()

    results: dict[str, float] = {}
    results.update(bench_spherical(device, args.repeats, 131072, 32, 128))
    results.update(bench_spherical(device, args.repeats, 65536, 128, 64))
    results.update(bench_diag(device, args.repeats, 131072, 32, 64))
    results.update(bench_diag(device, args.repeats, 65536, 64, 128))
    results.update(bench_tied(device, args.repeats, 131072, 32, 64))
    results.update(bench_full(device, args.repeats, 65536, 8, 32))
    if args.profile == "standard":
        results.update(bench_tied(device, args.repeats, 65536, 64, 128))
        results.update(bench_full(device, args.repeats, 131072, 8, 32))

    total_ms = sum(results.values())
    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    for name, value in sorted(results.items()):
        print(f"CASE {name} {value:.6f} ms")
    print(f"METRIC total_ms={total_ms:.6f} peak_mb={peak_mb:.1f} repeats={args.repeats} profile={args.profile}")


if __name__ == "__main__":
    main()
