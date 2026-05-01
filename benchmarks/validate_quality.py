from __future__ import annotations

import argparse
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
class QualityResult:
    dataset: str
    covariance: str
    seconds: float
    score: float
    ari: float
    n_iter: int
    passed: bool
    note: str = ""


def _require_sklearn():
    try:
        from sklearn.datasets import make_blobs
        from sklearn.metrics import adjusted_rand_score
    except ImportError as exc:
        raise RuntimeError("Install benchmark extras first: python -m pip install -e \".[benchmark]\"") from exc
    return make_blobs, adjusted_rand_score


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_dataset(
    name: str,
    *,
    n_samples: int,
    n_features: int,
    n_components: int,
    cluster_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    make_blobs, _ = _require_sklearn()
    x_np, labels_np = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_components,
        cluster_std=cluster_std,
        random_state=seed,
    )
    if name == "anisotropic-blobs":
        rng = np.random.default_rng(seed)
        transform = rng.normal(size=(n_features, n_features)).astype(np.float32)
        q, _ = np.linalg.qr(transform)
        scales = np.linspace(0.25, 3.0, n_features, dtype=np.float32)
        x_np = x_np @ (q * scales)
    return x_np.astype(np.float32), labels_np


def _fit_quality(
    x_np: np.ndarray,
    labels_np: np.ndarray,
    *,
    dataset: str,
    covariance_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> QualityResult:
    _, adjusted_rand_score = _require_sklearn()
    x = torch.as_tensor(x_np, device=device)
    model = GMMXX(
        d=x_np.shape[1],
        k=args.n_components,
        niter=args.max_iter,
        tol=args.tol,
        use_triton=True,
        seed=args.seed,
        init_params="kmeans",
        init_kmeans_iters=args.init_kmeans_iters,
        covariance_type=covariance_type,
        chunk_size_data=args.chunk_size_n,
        chunk_size_centroids=args.chunk_size_k,
        reg_covar=args.reg_covar,
    )
    _sync(device)
    start = time.perf_counter()
    model.fit(x)
    _sync(device)
    seconds = time.perf_counter() - start
    pred = model.predict(x).detach().cpu().numpy()
    score = float(model.score(x))
    ari = float(adjusted_rand_score(labels_np, pred))
    passed = ari >= args.min_ari
    return QualityResult(dataset, covariance_type, seconds, score, ari, int(model.n_iter_), passed)


def _print_results(results: list[QualityResult]) -> None:
    print(f"{'dataset':<20} {'cov':<10} {'fit_s':>9} {'score':>14} {'ari':>10} {'iters':>7} status")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        if result.note:
            status = f"{status} {result.note}"
        print(
            f"{result.dataset:<20} {result.covariance:<10} {result.seconds:>9.4f} "
            f"{result.score:>14.6f} {result.ari:>10.4f} {result.n_iter:>7d} {status}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate clustering quality on labeled synthetic datasets.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-samples", type=int, default=32768)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--cluster-std", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-kmeans-iters", type=int, default=10)
    parser.add_argument("--chunk-size-n", type=int, default=32768)
    parser.add_argument("--chunk-size-k", type=int, default=1024)
    parser.add_argument("--min-ari", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")

    results: list[QualityResult] = []
    x_blobs, y_blobs = _make_dataset(
        "blobs",
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_components=args.n_components,
        cluster_std=args.cluster_std,
        seed=args.seed,
    )
    results.append(_fit_quality(x_blobs, y_blobs, dataset="blobs", covariance_type="spherical", args=args, device=device))

    x_aniso, y_aniso = _make_dataset(
        "anisotropic-blobs",
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_components=args.n_components,
        cluster_std=args.cluster_std,
        seed=args.seed,
    )
    for covariance_type in ("spherical", "diag", "tied", "full"):
        results.append(
            _fit_quality(
                x_aniso,
                y_aniso,
                dataset="anisotropic-blobs",
                covariance_type=covariance_type,
                args=args,
                device=device,
            )
        )

    spherical_aniso = next(result for result in results if result.dataset == "anisotropic-blobs" and result.covariance == "spherical")
    for result in results:
        if result.dataset == "anisotropic-blobs" and result.covariance in {"diag", "tied", "full"}:
            if result.score <= spherical_aniso.score:
                result.passed = False
                result.note = "score did not beat spherical"

    _print_results(results)
    raise SystemExit(0 if all(result.passed for result in results) else 1)


if __name__ == "__main__":
    main()
