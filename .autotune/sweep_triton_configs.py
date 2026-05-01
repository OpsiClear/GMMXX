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
from flash_gmm2.torch_fallback import _precision_and_logdet
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


def _print(name: str, ms: float, cfg: dict[str, int]) -> None:
    parts = " ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
    print(f"RESULT {name} {ms:.6f} {parts}")


def sweep_spherical_infer(device: torch.device, repeats: int, n: int, d: int, k: int) -> None:
    b = 1
    torch.manual_seed(1001 + n + d + k)
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
    configs = [
        {"BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw, "num_stages": ns}
        for bn in (32, 64, 128)
        for bk in (32, 64, 128)
        for nw in (4, 8)
        for ns in (1, 2)
    ]
    for cfg in configs:
        try:
            _print(
                f"spherical_logsum_N{n}_D{d}_K{k}",
                _time_ms(
                    lambda cfg=cfg: spherical_logsumexp_triton(
                        x,
                        means,
                        variances,
                        weights,
                        x_sq=x_sq,
                        means_sq=means_sq,
                        log_weights=log_weights,
                        config=cfg,
                    ),
                    repeats,
                ),
                cfg,
            )
            _print(
                f"spherical_assign_N{n}_D{d}_K{k}",
                _time_ms(
                    lambda cfg=cfg: spherical_assign_triton(
                        x,
                        means,
                        variances,
                        weights,
                        x_sq=x_sq,
                        means_sq=means_sq,
                        log_weights=log_weights,
                        config=cfg,
                    ),
                    repeats,
                ),
                cfg,
            )
            if cfg["BLOCK_K"] >= k:
                _print(
                    f"spherical_resp_N{n}_D{d}_K{k}",
                    _time_ms(
                        lambda cfg=cfg: spherical_resp_triton(
                            x,
                            means,
                            variances,
                            weights,
                            log_norm,
                            x_sq=x_sq,
                            means_sq=means_sq,
                            log_weights=log_weights,
                            config=cfg,
                        ),
                        repeats,
                    ),
                    cfg,
                )
        except Exception as exc:
            print(f"ERROR spherical_infer {cfg} {type(exc).__name__}: {exc}")


def sweep_spherical_update(device: torch.device, repeats: int, n: int, d: int, k: int) -> None:
    b = 1
    torch.manual_seed(1101 + n + d + k)
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
    for block_n in (16, 32, 64, 128):
        for block_d in (d, 64, 128, 256):
            if block_d < d:
                continue
            for block_k in (16, 32, 64, 128):
                cfg = {"BLOCK_N": block_n, "BLOCK_D": block_d, "BLOCK_K": block_k}
                n_blocks = math.ceil(n / block_n)
                partial_nk = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
                partial_sum_x = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
                partial_sum_x_sq = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
                try:
                    ms = _time_ms(
                        lambda cfg=cfg: triton_blocked_update_spherical(
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
                            **cfg,
                        ),
                        repeats,
                    )
                    _print(f"spherical_update_N{n}_D{d}_K{k}", ms, cfg)
                except Exception as exc:
                    print(f"ERROR spherical_update {cfg} {type(exc).__name__}: {exc}")


def sweep_diag(device: torch.device, repeats: int, n: int, d: int, k: int) -> None:
    b = 1
    torch.manual_seed(1201 + n + d + k)
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
    for cfg in (
        {"BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw, "num_stages": 1}
        for bn in (16, 32, 64, 128)
        for bk in (32, 64, 128)
        for nw in (4, 8)
    ):
        try:
            ms = _time_ms(
                lambda cfg=cfg: diag_logsumexp_triton(
                    x, precision, weighted_means, mean_precision_mean, logdet, log_weights, config=cfg
                ),
                repeats,
            )
            _print(f"diag_logsum_N{n}_D{d}_K{k}", ms, cfg)
        except Exception as exc:
            print(f"ERROR diag_logsum {cfg} {type(exc).__name__}: {exc}")
    for block_n in (16, 32, 64, 128):
        for block_d in (d, 64, 128):
            if block_d < d:
                continue
            for block_k in (16, 32, 64, 128):
                cfg = {"BLOCK_N": block_n, "BLOCK_D": block_d, "BLOCK_K": block_k}
                n_blocks = math.ceil(n / block_n)
                partial_nk = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
                partial_sum_x = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
                partial_sum_x_sq = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
                try:
                    ms = _time_ms(
                        lambda cfg=cfg: triton_blocked_update_diag(
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
                            **cfg,
                        ),
                        repeats,
                    )
                    _print(f"diag_update_N{n}_D{d}_K{k}", ms, cfg)
                except Exception as exc:
                    print(f"ERROR diag_update {cfg} {type(exc).__name__}: {exc}")


def sweep_tied(device: torch.device, repeats: int, n: int, d: int, k: int) -> None:
    b = 1
    torch.manual_seed(1301 + n + d + k)
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
    log_norm = spherical_logsumexp_triton(
        x_projected,
        means_projected,
        unit_variances,
        weights,
        x_sq=x_projected_sq,
        means_sq=means_projected_sq,
        log_weights=log_weights,
        config={"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1},
        unit_variance=True,
    )
    for cfg in (
        {"BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw, "num_stages": 1}
        for bn in (32, 64, 128)
        for bk in (32, 64, 128)
        for nw in (4, 8)
    ):
        try:
            ms = _time_ms(
                lambda cfg=cfg: spherical_logsumexp_triton(
                    x_projected,
                    means_projected,
                    unit_variances,
                    weights,
                    x_sq=x_projected_sq,
                    means_sq=means_projected_sq,
                    log_weights=log_weights,
                    config=cfg,
                    unit_variance=True,
                ),
                repeats,
            )
            _print(f"tied_projected_logsum_N{n}_D{d}_K{k}", ms, cfg)
        except Exception as exc:
            print(f"ERROR tied_projected_logsum {cfg} {type(exc).__name__}: {exc}")
    for block_n in (16, 32, 64, 128):
        for block_d in (d, 64, 128):
            if block_d < d:
                continue
            for block_k in (16, 32, 64, 128):
                cfg = {"BLOCK_N": block_n, "BLOCK_D": block_d, "BLOCK_K": block_k}
                n_blocks = math.ceil(n / block_n)
                partial_nk = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
                partial_sum_x = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
                try:
                    ms = _time_ms(
                        lambda cfg=cfg: triton_blocked_update_tied_projected(
                            x_projected,
                            x,
                            means_projected,
                            log_weights,
                            log_norm,
                            x_projected_sq=x_projected_sq,
                            means_projected_sq=means_projected_sq,
                            partial_nk=partial_nk,
                            partial_sum_x=partial_sum_x,
                            **cfg,
                        ),
                        repeats,
                    )
                    _print(f"tied_update_N{n}_D{d}_K{k}", ms, cfg)
                except Exception as exc:
                    print(f"ERROR tied_update {cfg} {type(exc).__name__}: {exc}")


def sweep_full(device: torch.device, repeats: int, n: int, d: int, k: int) -> None:
    b = 1
    torch.manual_seed(1401 + n + d + k)
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
    for cfg in (
        {"BLOCK_N": bn, "BLOCK_K": bk, "BLOCK_D": bd, "num_warps": nw, "num_stages": 1}
        for bn in (32, 64, 128)
        for bk in (16, 32, 64)
        for bd in (8, 16)
        for nw in (4, 8)
    ):
        try:
            _print(
                f"full_logsum_N{n}_D{d}_K{k}",
                _time_ms(
                    lambda cfg=cfg: full_logsumexp_triton(
                        x, precision, precision_means, mean_precision_mean, logdet, log_weights, config=cfg
                    ),
                    repeats,
                ),
                cfg,
            )
            _print(
                f"full_assign_N{n}_D{d}_K{k}",
                _time_ms(
                    lambda cfg=cfg: full_assign_triton(
                        x, precision, precision_means, mean_precision_mean, logdet, log_weights, config=cfg
                    ),
                    repeats,
                ),
                cfg,
            )
            _print(
                f"full_resp_N{n}_D{d}_K{k}",
                _time_ms(
                    lambda cfg=cfg: full_resp_triton(
                        x, precision, precision_means, mean_precision_mean, logdet, log_weights, log_norm, config=cfg
                    ),
                    repeats,
                ),
                cfg,
            )
        except Exception as exc:
            print(f"ERROR full_infer {cfg} {type(exc).__name__}: {exc}")
    for block_n in (16, 32, 64, 128):
        for block_d in (8, 16):
            for block_k in (8, 16, 32, 64):
                cfg = {"BLOCK_N": block_n, "BLOCK_D": block_d, "BLOCK_K": block_k}
                n_blocks = math.ceil(n / block_n)
                partial_nk = torch.empty((b, n_blocks, k), device=device, dtype=torch.float32)
                partial_sum_x = torch.empty((b, n_blocks, k, d), device=device, dtype=torch.float32)
                partial_sum_xx = torch.empty((b, n_blocks, k, d, d), device=device, dtype=torch.float32)
                try:
                    ms = _time_ms(
                        lambda cfg=cfg: triton_blocked_update_full(
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
                            **cfg,
                        ),
                        repeats,
                    )
                    _print(f"full_update_N{n}_D{d}_K{k}", ms, cfg)
                except Exception as exc:
                    print(f"ERROR full_update {cfg} {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        choices=[
            "spherical-infer-d32",
            "spherical-infer-d128",
            "spherical-update-d32",
            "spherical-update-d128",
            "diag-d32",
            "diag-d64",
            "tied-d32",
            "tied-d64",
            "full-n65",
            "full-n131",
        ],
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()

    if args.target == "spherical-infer-d32":
        sweep_spherical_infer(device, args.repeats, 131072, 32, 128)
    elif args.target == "spherical-infer-d128":
        sweep_spherical_infer(device, args.repeats, 65536, 128, 64)
    elif args.target == "spherical-update-d32":
        sweep_spherical_update(device, args.repeats, 131072, 32, 128)
    elif args.target == "spherical-update-d128":
        sweep_spherical_update(device, args.repeats, 65536, 128, 64)
    elif args.target == "diag-d32":
        sweep_diag(device, args.repeats, 131072, 32, 64)
    elif args.target == "diag-d64":
        sweep_diag(device, args.repeats, 65536, 64, 128)
    elif args.target == "tied-d32":
        sweep_tied(device, args.repeats, 131072, 32, 64)
    elif args.target == "tied-d64":
        sweep_tied(device, args.repeats, 65536, 64, 128)
    elif args.target == "full-n65":
        sweep_full(device, args.repeats, 65536, 8, 32)
    elif args.target == "full-n131":
        sweep_full(device, args.repeats, 131072, 8, 32)


if __name__ == "__main__":
    main()
