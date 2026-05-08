"""Compare GMMXX (CUDA backend) against scikit-learn's GaussianMixture
across a representative shape grid. Emits a markdown table suitable for
pasting into README.md.

Usage:
    python benchmarks/bench_vs_sklearn.py
    python benchmarks/bench_vs_sklearn.py --csv results.csv
    python benchmarks/bench_vs_sklearn.py --grid quick     # smaller grid
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gmmxx import GMMXX


@dataclass
class Cell:
    cov: str           # spherical/diag/tied/full
    n: int
    d: int
    k: int
    dtype: str         # fp32/fp16/bf16
    sklearn_s: float   # sklearn fit time
    gmmxx_s: float     # gmmxx fit time
    speedup: float     # sklearn_s / gmmxx_s
    backend_used: str

    def row(self) -> str:
        return (
            f"| {self.cov} | N={self.n:>7,} | D={self.d:>3} | K={self.k:>4} "
            f"| {self.dtype} "
            f"| {self.sklearn_s*1000:>7.0f} ms "
            f"| {self.gmmxx_s*1000:>7.0f} ms "
            f"| **{self.speedup:>5.1f}x** "
            f"| {self.backend_used} |"
        )


def _make_blobs(n: int, d: int, k: int, seed: int = 0):
    """Reproducible synthetic data with k well-separated clusters."""
    from sklearn.datasets import make_blobs
    x, _ = make_blobs(
        n_samples=n, n_features=d, centers=k,
        cluster_std=1.0, random_state=seed, return_centers=False,
    )
    return x.astype(np.float32)


def _sklearn_fit(x_np, k: int, cov: str, max_iter: int = 20, seed: int = 0):
    from sklearn.mixture import GaussianMixture
    # sklearn full-cov + init_params='random' frequently hits singular
    # covariances on synthetic blobs and raises. Use kmeans init for full
    # to give it a fair shot; this also matches sklearn's documented
    # default for production use.
    init = "kmeans" if cov == "full" else "random"
    sk = GaussianMixture(
        n_components=k, covariance_type=cov,
        max_iter=max_iter, tol=0.0, init_params=init,
        random_state=seed, reg_covar=1e-4 if cov == "full" else 1e-6,
        n_init=1,
    )
    t0 = time.perf_counter()
    sk.fit(x_np)
    return time.perf_counter() - t0


def _gmmxx_fit(x_np, k: int, cov: str, dtype_str: str,
                max_iter: int = 20, seed: int = 0):
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
    x = torch.from_numpy(x_np).to(device="cuda", dtype=dtype).contiguous()
    # Warmup: 3 small fits to amortize CUDA caches and triton autotune.
    for _ in range(3):
        GMMXX(
            n_components=k, covariance_type=cov, backend="cuda",
            max_iter=2, tol=0.0, init_params="random",
            random_state=seed, dtype=dtype,
        ).fit(x)
    torch.cuda.synchronize()
    # Timed run.
    t0 = time.perf_counter()
    m = GMMXX(
        n_components=k, covariance_type=cov, backend="cuda",
        max_iter=max_iter, tol=0.0, init_params="random",
        random_state=seed, dtype=dtype,
    ).fit(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, m.last_backend_used_


def _run_cell(cov: str, n: int, d: int, k: int, dtype_str: str,
              max_iter: int = 20, seed: int = 0, n_repeat: int = 3) -> Cell:
    x_np = _make_blobs(n, d, k, seed)
    sk_times = []
    for _ in range(min(2, n_repeat)):
        sk_times.append(_sklearn_fit(x_np, k, cov, max_iter, seed))
    sk_med = statistics.median(sk_times)

    gx_times = []
    backend = "cuda"
    for _ in range(n_repeat):
        t, backend = _gmmxx_fit(x_np, k, cov, dtype_str, max_iter, seed)
        gx_times.append(t)
    gx_med = statistics.median(gx_times)

    return Cell(
        cov=cov, n=n, d=d, k=k, dtype=dtype_str,
        sklearn_s=sk_med, gmmxx_s=gx_med,
        speedup=sk_med / gx_med, backend_used=backend,
    )


# sklearn's GaussianMixture scales as O(N*K*D + K*D^3 per iter); shapes
# above ~131k samples × K=128 take many minutes per fit, which is too slow
# to use for repeated benchmarking. The default grid keeps sklearn fits
# under ~60s each so the full sweep finishes in a few minutes.
GRIDS = {
    "quick": [
        ("spherical",  65_536,  32,  64, "fp16"),
        ("spherical", 131_072, 128,  64, "fp32"),
        ("diag",       65_536,  32,  64, "fp32"),
    ],
    "default": [
        # Spherical — the path with the biggest CUDA gains.
        ("spherical",  16_384,  32,  64, "fp16"),
        ("spherical",  65_536,  32,  64, "fp16"),
        ("spherical", 131_072, 128,  64, "fp32"),
        ("spherical", 131_072,  16,  32, "fp16"),
        # Diag — chunked CUDA-tensor path
        ("diag",       16_384,  32,  64, "fp32"),
        ("diag",       65_536,  32,  64, "fp32"),
        # Tied — projected coords
        ("tied",       16_384,  32,  64, "fp32"),
        ("tied",       65_536,  32,  64, "fp32"),
        # Full — small D bound
        ("full",       16_384,  16,  32, "fp32"),
        ("full",       65_536,  16,  32, "fp32"),
    ],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", choices=list(GRIDS), default="default")
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--n-repeat", type=int, default=3)
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required for this benchmark")

    print(
        f"# device: {torch.cuda.get_device_name(0)}\n"
        f"# torch: {torch.__version__}\n"
        f"# max_iter: {args.max_iter}, n_repeat: {args.n_repeat}\n"
    )
    print("| cov | N | D | K | dtype | sklearn | gmmxx (cuda) | speedup | backend |")
    print("|---|---|---|---|---|---|---|---|---|")

    cells = []
    for spec in GRIDS[args.grid]:
        try:
            cell = _run_cell(*spec, max_iter=args.max_iter, n_repeat=args.n_repeat)
            print(cell.row())
            cells.append(cell)
        except Exception as exc:
            print(f"| {spec[0]} | N={spec[1]} | D={spec[2]} | K={spec[3]} "
                  f"| {spec[4]} | (failed: {type(exc).__name__}) | | | |")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("cov,n,d,k,dtype,sklearn_s,gmmxx_s,speedup,backend\n")
            for c in cells:
                f.write(
                    f"{c.cov},{c.n},{c.d},{c.k},{c.dtype},"
                    f"{c.sklearn_s},{c.gmmxx_s},{c.speedup},{c.backend_used}\n"
                )
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
