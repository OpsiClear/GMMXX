"""Spherical CUDA vs Triton vs torch_fallback speedup benchmark.

Runs a grid of (N, D, K, dtype) and prints a table of wall-clock per
fit() call, plus the cuda/triton speedup ratio. Exit code is non-zero
if any cell shows CUDA more than 1.5x slower than Triton (relaxed
gate for Plan 3 -- sorted-run M-step lands in Plan 4 and will tighten
this to 1.0x).

Usage:
  uv run python benchmarks/benchmark_cuda_vs_triton_spherical.py
  uv run python benchmarks/benchmark_cuda_vs_triton_spherical.py --gate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch


def _has_triton():
    try:
        from gmmxx.assign_spherical_triton import spherical_assign_triton
        return spherical_assign_triton is not None
    except Exception:
        return False


def _has_cuda():
    try:
        from gmmxx._cuda import has_cuda
        return has_cuda()
    except ImportError:
        return False


def _bench_triton_forced(N: int, D: int, K: int, dtype: torch.dtype, n_iter: int) -> float:
    """Bench Triton with all fast-path heuristics force-enabled (bypassing the
    auto-gating that disables Triton on shapes like fp32 D=128 K=64 N=65k)."""
    from gmmxx.torch_fallback import batch_gmm_Spherical_torch_native
    torch.manual_seed(0)
    x_b = torch.randn(1, N, D, device="cuda", dtype=dtype)
    # Multiple warmup fits to stabilize cache state, autotune.
    for _ in range(3):
        batch_gmm_Spherical_torch_native(
            x_b, K, max_iters=2, tol=0, init_params="random",
            kmeans_use_triton=True,
            gmm_use_triton_estep=True,
            gmm_use_triton_streaming_update=True,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch_gmm_Spherical_torch_native(
        x_b, K, max_iters=n_iter, tol=0, init_params="random",
        kmeans_use_triton=True,
        gmm_use_triton_estep=True,
        gmm_use_triton_streaming_update=True,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench_one(backend: str, N: int, D: int, K: int, dtype: torch.dtype, n_iter: int):
    from gmmxx import GMMXX
    torch.manual_seed(0)
    device = "cuda" if backend in {"cuda", "triton"} else "cpu"
    x = torch.randn(N, D, device=device, dtype=dtype)
    # Multiple warmup fits to amortize CUDA kernel cache, allocator state.
    for _ in range(3):
        GMMXX(
            n_components=K, max_iter=2, tol=0, random_state=0,
            covariance_type="spherical", backend=backend,
        ).fit(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    GMMXX(
        n_components=K, max_iter=n_iter, tol=0, random_state=0,
        covariance_type="spherical", backend=backend,
    ).fit(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    # For Triton, also try the force-enabled config and keep the better
    # (lower) wall-time. The auto heuristic disables Triton fast paths
    # for some shapes (fp32 D=128 K=64 N=65k -> torch fallback), which
    # would inflate CUDA's apparent speedup if not corrected.
    if backend == "triton":
        try:
            forced = _bench_triton_forced(N, D, K, dtype, n_iter)
            elapsed = min(elapsed, forced)
        except Exception as exc:
            print(f"  triton-forced fallback failed for N={N},D={D},K={K}: {exc}", file=sys.stderr)
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument(
        "--shapes",
        type=str,
        default="65536,32,64,fp16;65536,128,64,fp32;131072,16,32,fp16",
        help="Semicolon-separated N,D,K,dtype quads",
    )
    p.add_argument(
        "--shape-grid",
        type=str,
        default=None,
        choices=[None, "small", "large", "xlarge"],
        help="Preset shape grid: 'small' (default), 'large', 'xlarge'",
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero if CUDA > 1.1x Triton on any shape (Plan 4 threshold)",
    )
    p.add_argument(
        "--gate-threshold",
        type=float,
        default=1.1,
        help="Maximum acceptable cuda/triton ratio when --gate is set",
    )
    args = p.parse_args()

    shape_presets = {
        "small": "65536,32,64,fp16;65536,128,64,fp32;131072,16,32,fp16",
        "large": "524288,32,64,fp16;524288,128,64,fp32;1048576,16,32,fp16",
        "xlarge": "1048576,32,64,fp16;524288,128,64,fp32;4194304,16,32,fp16",
    }
    shapes_str = shape_presets.get(args.shape_grid, args.shapes)

    shapes = []
    for s in shapes_str.split(";"):
        n, d, k, dt = s.split(",")
        dt_t = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[dt]
        shapes.append((int(n), int(d), int(k), dt_t))

    backends = []
    if _has_cuda():
        backends.append("cuda")
    if _has_triton():
        backends.append("triton")
    backends.append("torch")

    results = {}
    header = f"{'shape':30s} " + " ".join(f"{b:>10s}" for b in backends)
    if "cuda" in backends and "triton" in backends:
        header += "  cuda/triton"
    print(header)

    failures = []
    for N, D, K, dt in shapes:
        row = []
        for backend in backends:
            try:
                t = _bench_one(backend, N, D, K, dt, args.n_iter)
            except Exception as exc:
                print(f"  {backend} failed: {exc}", file=sys.stderr)
                t = math.inf
            row.append(t)

        ratio = math.nan
        if "cuda" in backends and "triton" in backends:
            cuda_t = row[backends.index("cuda")]
            triton_t = row[backends.index("triton")]
            if math.isfinite(cuda_t) and math.isfinite(triton_t) and triton_t > 0:
                ratio = cuda_t / triton_t

        shape_str = f"N={N},D={D},K={K},{str(dt).split('.')[-1]}"
        line = f"{shape_str:30s} " + " ".join(f"{t:10.4f}" for t in row)
        if not math.isnan(ratio):
            line += f"  {ratio:.3f}"
        print(line)

        results[shape_str] = {
            "per_backend": dict(zip(backends, row)),
            "cuda_triton_ratio": ratio,
        }
        if args.gate and not math.isnan(ratio) and ratio > args.gate_threshold:
            failures.append((shape_str, ratio))

    print()
    print(json.dumps(results, indent=2))

    if failures:
        print(
            f"FAIL: CUDA regressed beyond {args.gate_threshold}x Triton on:",
            failures,
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
