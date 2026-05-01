from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gmmxx import GMMXX
from gmmxx._runtime import triton_spherical_supported


@dataclass(frozen=True)
class SizeCase:
    n: int
    d: int
    k: int
    b: int = 1


@dataclass
class CaseResult:
    case: SizeCase
    torch_seconds: float
    auto_seconds: float
    score_diff: float
    ari_error: float | None
    means_max_abs: float
    variances_max_abs: float
    weights_max_abs: float
    path: str
    passed: bool
    note: str = ""


SMOKE_CASES = [
    SizeCase(256, 1, 1),
    SizeCase(1024, 2, 2),
    SizeCase(1024, 16, 8, b=2),
    SizeCase(4096, 32, 16),
    SizeCase(8192, 129, 64),
    SizeCase(8192, 128, 2049),
]

STANDARD_CASES = [
    SizeCase(256, 1, 1),
    SizeCase(1024, 2, 2),
    SizeCase(2048, 16, 8, b=2),
    SizeCase(4096, 32, 16),
    SizeCase(16384, 32, 64),
    SizeCase(65536, 32, 64),
    SizeCase(262144, 32, 64),
    SizeCase(65536, 128, 64),
    SizeCase(262144, 128, 256),
    SizeCase(262144, 128, 1024),
    SizeCase(262144, 128, 2048),
    SizeCase(65536, 129, 64),
    SizeCase(65536, 256, 64),
    SizeCase(65536, 128, 2049),
    SizeCase(65536, 128, 4096),
]

LARGE_EXTRA_CASES = [
    SizeCase(524288, 128, 1024),
    SizeCase(1048576, 128, 128),
    SizeCase(2097152, 32, 64),
]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("integer lists must contain at least one value")
    if any(item <= 0 for item in parsed):
        raise ValueError("integer lists must contain positive values")
    return parsed


def _profile_cases(profile: str, device: torch.device) -> list[SizeCase]:
    if profile == "auto":
        profile = "standard" if device.type == "cuda" else "smoke"
    if profile == "smoke":
        return list(SMOKE_CASES)
    if profile == "standard":
        return list(STANDARD_CASES)
    if profile == "large":
        return list(STANDARD_CASES) + list(LARGE_EXTRA_CASES)
    raise ValueError(f"Unknown profile: {profile}")


def _cartesian_cases(args: argparse.Namespace) -> list[SizeCase]:
    n_values = _parse_int_list(args.n_values)
    d_values = _parse_int_list(args.d_values)
    k_values = _parse_int_list(args.k_values)
    return [SizeCase(n, d, k, b=args.batch_size) for n in n_values for d in d_values for k in k_values]


def _make_data(case: SizeCase, device: torch.device, seed: int, cluster_std: float) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + case.n * 13 + case.d * 17 + case.k * 19 + case.b * 23)
    centers = torch.randn((case.b, case.k, case.d), device=device, dtype=torch.float32, generator=generator) * 5.0
    labels = torch.randint(0, case.k, (case.b, case.n), device=device, generator=generator)
    gather_idx = labels.unsqueeze(-1).expand(case.b, case.n, case.d)
    x = torch.gather(centers, 1, gather_idx)
    noise = torch.randn((case.b, case.n, case.d), device=device, dtype=torch.float32, generator=generator)
    return x + noise * cluster_std


def _fit_timed(model: GMMXX, x: torch.Tensor, device: torch.device) -> tuple[float, GMMXX]:
    _sync(device)
    start = time.perf_counter()
    model.fit(x)
    _sync(device)
    return time.perf_counter() - start, model


def _new_model(case: SizeCase, args: argparse.Namespace, *, use_triton: bool) -> GMMXX:
    return GMMXX(
        d=case.d,
        k=case.k,
        niter=args.max_iter,
        tol=0.0,
        seed=args.seed,
        init_params=args.init_params,
        chunk_size_data=args.chunk_size_n,
        chunk_size_centroids=args.chunk_size_k,
        reg_covar=args.reg_covar,
        use_triton=use_triton,
    )


def _ari_error(labels_a: torch.Tensor, labels_b: torch.Tensor) -> float | None:
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return None
    ari = adjusted_rand_score(labels_a.detach().cpu().reshape(-1), labels_b.detach().cpu().reshape(-1))
    return 1.0 - float(ari)


def _path(model: GMMXX) -> str:
    estep = "triton" if model.triton_estep_enabled_ else "torch"
    mstep = "triton" if model.triton_streaming_update_enabled_ else "torch"
    labels = "triton" if model.triton_labels_enabled_ else "torch"
    return f"e={estep},m={mstep},l={labels}"


def _check_case(case: SizeCase, args: argparse.Namespace, device: torch.device) -> CaseResult:
    x_b = _make_data(case, device, args.seed, args.cluster_std)
    x = x_b.squeeze(0) if case.b == 1 and not args.keep_batch_dim else x_b

    for _ in range(args.warmup_runs):
        _new_model(case, args, use_triton=False).fit(x)
        _sync(device)
        _new_model(case, args, use_triton=True).fit(x)
        _sync(device)

    torch_seconds, torch_model = _fit_timed(_new_model(case, args, use_triton=False), x, device)
    auto_seconds, auto_model = _fit_timed(_new_model(case, args, use_triton=True), x, device)

    labels_torch = torch_model.predict(x)
    labels_auto = auto_model.predict(x)
    score_diff = abs(float(torch_model.score(x)) - float(auto_model.score(x)))
    ari_error = _ari_error(labels_torch, labels_auto)
    means_max_abs = float((torch_model.means_b - auto_model.means_b).abs().max().item())
    variances_max_abs = float((torch_model.variances_b - auto_model.variances_b).abs().max().item())
    weights_max_abs = float((torch_model.weights_b - auto_model.weights_b).abs().max().item())

    finite = all(
        torch.isfinite(tensor).all().item()
        for tensor in (auto_model.means_b, auto_model.variances_b, auto_model.weights_b)
    )
    fallback_ok = True
    if device.type == "cuda" and not triton_spherical_supported(case.d, case.k):
        fallback_ok = not (
            bool(auto_model.triton_estep_enabled_)
            or bool(auto_model.triton_streaming_update_enabled_)
            or bool(auto_model.triton_labels_enabled_)
        )

    ari_ok = True if ari_error is None else ari_error <= args.ari_error
    passed = bool(
        finite
        and fallback_ok
        and math.isfinite(score_diff)
        and score_diff <= args.score_atol
        and ari_ok
        and means_max_abs <= args.param_atol
        and variances_max_abs <= args.var_atol
        and weights_max_abs <= args.weight_atol
    )
    note = ""
    if not fallback_ok:
        note = "unsupported shape used Triton"
    elif not finite:
        note = "non-finite parameter"
    return CaseResult(
        case=case,
        torch_seconds=torch_seconds,
        auto_seconds=auto_seconds,
        score_diff=score_diff,
        ari_error=ari_error,
        means_max_abs=means_max_abs,
        variances_max_abs=variances_max_abs,
        weights_max_abs=weights_max_abs,
        path=_path(auto_model),
        passed=passed,
        note=note,
    )


def _print_results(results: list[CaseResult]) -> None:
    print(
        f"{'B':>3} {'N':>9} {'D':>5} {'K':>6} "
        f"{'auto_s':>9} {'torch_s':>9} {'speedup':>8} "
        f"{'score_d':>10} {'ari_err':>10} {'mean_d':>10} {'var_d':>10} {'w_d':>10} "
        f"{'path':>28} status"
    )
    for result in results:
        speedup = result.torch_seconds / result.auto_seconds if result.auto_seconds > 0 else float("inf")
        ari = "n/a" if result.ari_error is None else f"{result.ari_error:.3g}"
        status = "PASS" if result.passed else "FAIL"
        if result.note:
            status = f"{status} {result.note}"
        print(
            f"{result.case.b:>3d} {result.case.n:>9d} {result.case.d:>5d} {result.case.k:>6d} "
            f"{result.auto_seconds:>9.4f} {result.torch_seconds:>9.4f} {speedup:>8.3f} "
            f"{result.score_diff:>10.3g} {ari:>10} "
            f"{result.means_max_abs:>10.3g} {result.variances_max_abs:>10.3g} {result.weights_max_abs:>10.3g} "
            f"{result.path:>28} {status}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GMMXX size coverage against the torch path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--profile", choices=["auto", "smoke", "standard", "large"], default="auto")
    parser.add_argument("--cartesian", action="store_true", help="Use the Cartesian product of --n-values/--d-values/--k-values.")
    parser.add_argument("--n-values", default="256,4096,65536")
    parser.add_argument("--d-values", default="1,32,64,128,129,256")
    parser.add_argument("--k-values", default="1,16,64,256,2048,2049")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for --cartesian cases.")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap after case generation. 0 means no cap.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cluster-std", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--init-params", choices=["random", "kmeans"], default="random")
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--chunk-size-n", type=int, default=32768)
    parser.add_argument("--chunk-size-k", type=int, default=1024)
    parser.add_argument("--score-atol", type=float, default=1e-1)
    parser.add_argument("--ari-error", type=float, default=1e-2)
    parser.add_argument("--param-atol", type=float, default=1e-1)
    parser.add_argument("--var-atol", type=float, default=5e-2)
    parser.add_argument("--weight-atol", type=float, default=1e-4)
    parser.add_argument("--keep-batch-dim", action="store_true", help="Keep B=1 inputs as (1, N, D) instead of (N, D).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_cases < 0:
        raise ValueError("--max-cases must be non-negative")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")

    device = _resolve_device(args.device)
    cases = _cartesian_cases(args) if args.cartesian else _profile_cases(args.profile, device)
    if args.max_cases:
        cases = cases[: args.max_cases]

    print(
        f"device={device} torch={torch.__version__} cuda={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()} cases={len(cases)}"
    )
    results = [_check_case(case, args, device) for case in cases]
    _print_results(results)
    failed = [result for result in results if not result.passed]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
