from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gmmxx import GMMXX


@dataclass
class AccuracyResult:
    baseline: str
    covariance: str
    fit_seconds: float
    score: float
    ari: float
    n_iter: int | None
    status: str = "PASS"


def _require_sklearn():
    try:
        from sklearn import datasets
        from sklearn.datasets import make_blobs
        from sklearn.metrics import adjusted_rand_score
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Install benchmark extras first: python -m pip install -e \".[benchmark]\"") from exc
    return datasets, make_blobs, adjusted_rand_score, GaussianMixture, StandardScaler


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_dataset(args: argparse.Namespace) -> tuple[str, np.ndarray, np.ndarray, int]:
    datasets, make_blobs, _, _, StandardScaler = _require_sklearn()
    rng = np.random.default_rng(args.seed)

    if args.dataset in {"blobs", "anisotropic-blobs"}:
        x_np, labels_np = make_blobs(
            n_samples=args.n_samples,
            n_features=args.n_features,
            centers=args.n_components,
            cluster_std=args.cluster_std,
            random_state=args.seed,
        )
        if args.dataset == "anisotropic-blobs":
            transform = rng.normal(size=(args.n_features, args.n_features)).astype(np.float32)
            q, _ = np.linalg.qr(transform)
            scales = np.linspace(0.25, 3.0, args.n_features, dtype=np.float32)
            x_np = x_np @ (q * scales)
        return args.dataset, x_np.astype(np.float32), labels_np, args.n_components

    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
    }
    bunch = loaders[args.dataset]()
    x_np = StandardScaler().fit_transform(bunch.data).astype(np.float32)
    labels_np = np.asarray(bunch.target)
    return args.dataset, x_np, labels_np, int(np.unique(labels_np).size)


def _time_fit(device: torch.device, fit_fn):
    _sync(device)
    start = time.perf_counter()
    fitted = fit_fn()
    _sync(device)
    return time.perf_counter() - start, fitted


def _run_flash(
    x_np: np.ndarray,
    labels_np: np.ndarray,
    n_components: int,
    covariance: str,
    args: argparse.Namespace,
) -> AccuracyResult:
    _, _, adjusted_rand_score, _, _ = _require_sklearn()
    device = _resolve_device(args.device)
    x = torch.as_tensor(x_np, device=device, dtype=torch.float32)

    def fit_model() -> GMMXX:
        model = GMMXX(
            d=x.shape[1],
            k=n_components,
            niter=args.max_iter,
            tol=args.tol,
            use_triton=True,
            seed=args.seed,
            chunk_size_data=args.chunk_size_n,
            chunk_size_centroids=args.chunk_size_k,
            init_params=args.init_params,
            reg_covar=args.reg_covar,
            covariance_type=covariance,
        )
        return model.fit(x)

    fit_seconds, model = _time_fit(device, fit_model)
    pred = model.predict(x).detach().cpu().numpy()
    score = float(model.score(x))
    ari = float(adjusted_rand_score(labels_np, pred))
    status = "PASS" if ari >= args.min_ari else "LOW_ARI"
    return AccuracyResult("flash", covariance, fit_seconds, score, ari, int(model.n_iter_), status)


def _run_sklearn(
    x_np: np.ndarray,
    labels_np: np.ndarray,
    n_components: int,
    covariance: str,
    args: argparse.Namespace,
) -> AccuracyResult:
    _, _, adjusted_rand_score, GaussianMixture, _ = _require_sklearn()
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance,
        tol=args.tol,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        n_init=1,
        init_params=args.sklearn_init_params,
        random_state=args.seed,
    )
    fit_seconds, fitted = _time_fit(torch.device("cpu"), lambda: model.fit(x_np))
    pred = fitted.predict(x_np)
    score = float(fitted.score(x_np))
    ari = float(adjusted_rand_score(labels_np, pred))
    status = "PASS" if ari >= args.min_ari else "LOW_ARI"
    return AccuracyResult("sklearn", covariance, fit_seconds, score, ari, int(fitted.n_iter_), status)


def _print_results(dataset: str, x_np: np.ndarray, n_components: int, results: Iterable[AccuracyResult]) -> None:
    print(f"dataset={dataset} n={x_np.shape[0]} d={x_np.shape[1]} k={n_components}")
    print(f"{'baseline':<10} {'cov':<10} {'fit_s':>10} {'score':>14} {'ari':>10} {'iters':>7} status")
    for result in results:
        n_iter = "n/a" if result.n_iter is None else str(result.n_iter)
        score = "nan" if not math.isfinite(result.score) else f"{result.score:.6f}"
        print(
            f"{result.baseline:<10} {result.covariance:<10} {result.fit_seconds:>10.4f} "
            f"{score:>14} {result.ari:>10.4f} {n_iter:>7} {result.status}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark clustering accuracy and likelihood quality.")
    parser.add_argument("--dataset", choices=["blobs", "anisotropic-blobs", "iris", "wine", "digits"], default="anisotropic-blobs")
    parser.add_argument("--n-samples", type=int, default=32768)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--cluster-std", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size-n", type=int, default=32768)
    parser.add_argument("--chunk-size-k", type=int, default=1024)
    parser.add_argument("--init-params", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument(
        "--sklearn-init-params",
        choices=["kmeans", "k-means++", "random", "random_from_data"],
        default="kmeans",
    )
    parser.add_argument("--min-ari", type=float, default=0.95)
    parser.add_argument("--fail-on-low-ari", action="store_true")
    parser.add_argument("--include-sklearn", action="store_true")
    parser.add_argument(
        "--covariances",
        nargs="+",
        choices=["spherical", "diag", "tied", "full"],
        default=["spherical", "diag", "tied", "full"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset, x_np, labels_np, n_components = _make_dataset(args)
    print(
        f"device={_resolve_device(args.device)} torch={torch.__version__} "
        f"cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}"
    )

    results: list[AccuracyResult] = []
    for covariance in args.covariances:
        results.append(_run_flash(x_np, labels_np, n_components, covariance, args))
        if args.include_sklearn:
            sklearn_covariance = "diag" if covariance == "diag" else covariance
            results.append(_run_sklearn(x_np, labels_np, n_components, sklearn_covariance, args))

    _print_results(dataset, x_np, n_components, results)
    if args.fail_on_low_ari:
        raise SystemExit(0 if all(result.status == "PASS" for result in results) else 1)


if __name__ == "__main__":
    main()
