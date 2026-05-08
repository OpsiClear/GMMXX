"""Wall-clock fits at large shapes that scikit-learn cannot reasonably
handle. Reports gmmxx-only timings to show the practical operating
window the CUDA backend reaches.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gmmxx import GMMXX


SHAPES = [
    # cov,        N,           D,    K,   dtype
    ("spherical",   524_288,   128,   64, "fp32"),
    ("spherical", 1_048_576,    32,   64, "fp16"),
    ("spherical", 4_194_304,    16,   32, "fp16"),
    ("diag",        524_288,   128,  128, "fp16"),
    ("tied",        524_288,    64,  128, "fp16"),
]


def bench(cov, n, d, k, dtype_str, max_iter=20, n_repeat=3, seed=0):
    from sklearn.datasets import make_blobs
    x_np, _ = make_blobs(n_samples=n, n_features=d, centers=k,
                          cluster_std=1.0, random_state=seed)
    x_np = x_np.astype(np.float32)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
    x = torch.from_numpy(x_np).to("cuda", dtype=dtype).contiguous()

    # Warmup.
    for _ in range(3):
        GMMXX(n_components=k, covariance_type=cov, backend="cuda",
              max_iter=2, tol=0.0, init_params="random",
              random_state=seed, dtype=dtype).fit(x)
    torch.cuda.synchronize()

    times = []
    backend = None
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = GMMXX(n_components=k, covariance_type=cov, backend="cuda",
                  max_iter=max_iter, tol=0.0, init_params="random",
                  random_state=seed, dtype=dtype).fit(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        backend = m.last_backend_used_

    med = statistics.median(times)
    return med, backend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--n-repeat", type=int, default=3)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# torch: {torch.__version__}")
    print(f"# max_iter: {args.max_iter}, n_repeat: {args.n_repeat}\n")
    print("| cov | N | D | K | dtype | gmmxx (cuda) | backend |")
    print("|---|---|---|---|---|---|---|")
    for cov, n, d, k, dt in SHAPES:
        try:
            t, backend = bench(cov, n, d, k, dt,
                               max_iter=args.max_iter, n_repeat=args.n_repeat)
            print(f"| {cov} | N={n:>9,} | D={d:>3} | K={k:>4} | {dt} "
                  f"| {t*1000:>7.0f} ms | {backend} |")
        except Exception as exc:
            print(f"| {cov} | N={n} | D={d} | K={k} | {dt} "
                  f"| (failed: {type(exc).__name__}) | |")


if __name__ == "__main__":
    main()
