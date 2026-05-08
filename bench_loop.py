"""Auto-tune harness: run benchmark N times, take median of max(cuda/triton ratio).

Reduces benchmark noise that can make small improvements look like regressions
or vice versa. Uses higher --n-iter for more stable per-run measurements.

Usage:
  uv run python bench_loop.py --runs 5 --n-iter 30
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys


def _extract_max_ratio(text: str) -> float:
    m = re.search(r'\{\s*\n\s*"N=', text)
    if m is None:
        return float("nan")
    js = text[m.start():]
    depth = 0
    end = -1
    for i, c in enumerate(js):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return float("nan")
    try:
        data = json.loads(js[:end])
    except json.JSONDecodeError:
        return float("nan")
    ratios = []
    for v in data.values():
        r = v.get("cuda_triton_ratio")
        if r is None or r != r:
            continue
        ratios.append(r)
    return max(ratios) if ratios else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=30)
    p.add_argument("--shape-grid", type=str, default=None,
                   choices=[None, "small", "large", "xlarge"])
    p.add_argument("--no-uv", action="store_true",
                   help="invoke benchmark with sys.executable instead of `uv run python`")
    args = p.parse_args()

    if args.no_uv:
        runner = [sys.executable]
    else:
        runner = ["uv", "run", "python"]
    cmd = runner + [
        "benchmarks/benchmark_cuda_vs_triton_spherical.py",
        "--n-iter", str(args.n_iter),
    ]
    if args.shape_grid:
        cmd += ["--shape-grid", args.shape_grid]

    metrics = []
    for i in range(args.runs):
        r = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            print(f"run {i}: bench exit={r.returncode}", file=sys.stderr)
            print(r.stderr[-500:], file=sys.stderr)
            return 1
        metric = _extract_max_ratio(r.stdout)
        metrics.append(metric)
        print(f"run {i}: max_ratio={metric:.4f}", file=sys.stderr)

    metrics_sorted = sorted(metrics)
    median = statistics.median(metrics)
    print(f"runs: {metrics}", file=sys.stderr)
    print(f"min={metrics_sorted[0]:.4f} median={median:.4f} max={metrics_sorted[-1]:.4f}", file=sys.stderr)
    print(f"{median:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
