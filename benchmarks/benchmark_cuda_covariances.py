"""CUDA vs Triton vs torch benchmark for non-spherical GMM covariances.

This harness is intentionally informational. It prints fit/predict timings and
backend flags for diag, tied, and full covariance so native-kernel work can be
driven by measured bottlenecks instead of guesses.

Usage:
  python benchmarks/benchmark_cuda_covariances.py
  python benchmarks/benchmark_cuda_covariances.py --shapes "diag:8192,16,32,fp32"
  python benchmarks/benchmark_cuda_covariances.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }[name]


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }.get(dtype, str(dtype).split(".")[-1])


def _has_cuda_backend() -> bool:
    try:
        from gmmxx._cuda import has_cuda

        return has_cuda()
    except Exception:
        return False


def _has_triton_backend() -> bool:
    try:
        import triton  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_shapes(value: str) -> list[tuple[str, int, int, int, torch.dtype]]:
    shapes: list[tuple[str, int, int, int, torch.dtype]] = []
    for raw in value.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        cov, rest = raw.split(":", 1)
        n_s, d_s, k_s, dtype_s = rest.split(",")
        cov = cov.strip()
        if cov not in {"diag", "tied", "full"}:
            raise ValueError("shape covariance must be diag, tied, or full")
        shapes.append((cov, int(n_s), int(d_s), int(k_s), _dtype(dtype_s.strip())))
    return shapes


def _make_input(n: int, d: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    half = n // 2
    x0 = torch.randn((half, d), generator=gen, device=device, dtype=dtype) * 0.35
    x1 = torch.randn((n - half, d), generator=gen, device=device, dtype=dtype) * 0.45 + 2.5
    return torch.cat((x0, x1), dim=0).contiguous()


def _time_call(device: torch.device, fn):
    _sync(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    return time.perf_counter() - start, value


def _run_one(
    *,
    covariance: str,
    backend: str,
    n: int,
    d: int,
    k: int,
    dtype: torch.dtype,
    max_iter: int,
    seed: int,
    warmup: int,
) -> dict[str, Any]:
    from gmmxx import GMMXX

    if backend in {"cuda", "triton"} and not torch.cuda.is_available():
        raise RuntimeError(f"{backend} backend requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = _make_input(n, d, dtype, device, seed)

    def fit_once() -> GMMXX:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = GMMXX(
            d=d,
            k=k,
            niter=max_iter,
            tol=0.0,
            seed=seed,
            init_params="random",
            covariance_type=covariance,
            backend=backend,
            device=device,
            compute_labels_on_fit=False,
            dtype=dtype,
        )
        return model.fit(x)

    for _ in range(warmup):
        fit_once()
        _sync(device)

    fit_seconds, model = _time_call(device, fit_once)
    predict_seconds, labels = _time_call(device, lambda: model.predict(x))
    score_seconds, score = _time_call(device, lambda: model.score(x))
    del labels

    return {
        "covariance": covariance,
        "backend": backend,
        "n": n,
        "d": d,
        "k": k,
        "dtype": _dtype_name(dtype),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "score_seconds": score_seconds,
        "score": float(score),
        "lower_bound": float(model.lower_bound_),
        "n_iter": int(model.n_iter_),
        "last_backend_used": model.last_backend_used_,
        "cuda_estep_enabled": bool(getattr(model, "cuda_estep_enabled_", False)),
        "cuda_fused_update_enabled": bool(getattr(model, "cuda_fused_update_enabled_", False)),
        "cuda_approx_topk_enabled": bool(getattr(model, "cuda_approx_topk_enabled_", False)),
        "triton_estep_enabled": bool(getattr(model, "triton_estep_enabled_", False)),
        "triton_fused_update_enabled": bool(getattr(model, "triton_fused_update_enabled_", False)),
        "triton_streaming_update_enabled": bool(getattr(model, "triton_streaming_update_enabled_", False)),
        "fallback_reason": getattr(model, "last_fallback_reason_", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--shapes",
        default="diag:8192,16,32,fp32;tied:4096,16,16,fp32;full:2048,8,8,fp32",
        help="Semicolon-separated cov:N,D,K,dtype entries.",
    )
    parser.add_argument(
        "--backends",
        default="cuda,triton,torch",
        help="Comma-separated backend list. Unavailable cuda/triton entries are skipped.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Print cuda/triton ratios. Informational only in Plan 12.",
    )
    args = parser.parse_args()

    requested_backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    backends = []
    for backend in requested_backends:
        if backend == "cuda" and not _has_cuda_backend():
            print("skip cuda: gmmxx._C unavailable or torch.cuda unavailable", file=sys.stderr)
            continue
        if backend == "triton" and not _has_triton_backend():
            print("skip triton: Triton or CUDA unavailable", file=sys.stderr)
            continue
        if backend not in {"cuda", "triton", "torch"}:
            raise ValueError("backends must be cuda, triton, or torch")
        backends.append(backend)
    if not backends:
        raise RuntimeError("no runnable backends")

    results: list[dict[str, Any]] = []
    shapes = _parse_shapes(args.shapes)
    for covariance, n, d, k, dtype in shapes:
        print(f"\n{covariance} N={n} D={d} K={k} dtype={_dtype_name(dtype)}")
        for backend in backends:
            try:
                row = _run_one(
                    covariance=covariance,
                    backend=backend,
                    n=n,
                    d=d,
                    k=k,
                    dtype=dtype,
                    max_iter=args.n_iter,
                    seed=args.seed,
                    warmup=args.warmup,
                )
            except Exception as exc:
                row = {
                    "covariance": covariance,
                    "backend": backend,
                    "n": n,
                    "d": d,
                    "k": k,
                    "dtype": _dtype_name(dtype),
                    "error": f"{type(exc).__name__}: {exc}",
                    "fit_seconds": math.inf,
                }
            results.append(row)
            if "error" in row:
                print(f"  {backend:7s} ERROR {row['error']}")
            else:
                print(
                    f"  {backend:7s} fit={row['fit_seconds']:.4f}s "
                    f"predict={row['predict_seconds']:.4f}s "
                    f"score={row['score_seconds']:.4f}s "
                    f"last={row['last_backend_used']} "
                    f"lb={row['lower_bound']:.6f}"
                )

        if args.gate:
            by_backend = {
                row["backend"]: row
                for row in results
                if row["covariance"] == covariance
                and row["n"] == n
                and row["d"] == d
                and row["k"] == k
                and row["dtype"] == _dtype_name(dtype)
                and "error" not in row
            }
            if "cuda" in by_backend and "triton" in by_backend:
                ratio = by_backend["cuda"]["fit_seconds"] / by_backend["triton"]["fit_seconds"]
                print(f"  informational cuda/triton fit ratio: {ratio:.3f}")

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
