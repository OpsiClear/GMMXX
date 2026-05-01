from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_gmm2 import FlashGMM


@dataclass
class Dataset:
    name: str
    x_np: np.ndarray
    labels_np: np.ndarray | None
    n_components: int


@dataclass
class Result:
    baseline: str
    fit_seconds: float
    score: float
    ari: float | None
    n_iter: int | None
    note: str = ""


def _require_sklearn():
    try:
        from sklearn import datasets
        from sklearn.datasets import make_blobs
        from sklearn.metrics import adjusted_rand_score
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Install benchmark extras first: python -m pip install -e \".[benchmark]\""
        ) from exc
    return datasets, make_blobs, adjusted_rand_score, GaussianMixture, StandardScaler


def _load_dataset(args: argparse.Namespace) -> Dataset:
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
        return Dataset(args.dataset, x_np.astype(np.float32), labels_np, args.n_components)

    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
    }
    bunch = loaders[args.dataset]()
    x_np = StandardScaler().fit_transform(bunch.data).astype(np.float32)
    labels_np = np.asarray(bunch.target)
    return Dataset(args.dataset, x_np, labels_np, int(np.unique(labels_np).size))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _resolve_dtype(value: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtypes[value]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_fit(device: torch.device, fit_fn: Callable[[], object], *, warmup_runs: int = 0) -> tuple[float, object]:
    for _ in range(warmup_runs):
        fit_fn()
        _sync(device)
    _sync(device)
    start = time.perf_counter()
    fitted = fit_fn()
    _sync(device)
    return time.perf_counter() - start, fitted


def _ari(labels_true: np.ndarray | None, labels_pred: np.ndarray, adjusted_rand_score) -> float | None:
    if labels_true is None:
        return None
    return float(adjusted_rand_score(labels_true, labels_pred))


def _run_flash(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    use_triton: bool,
    covariance_type: str = "spherical",
    baseline: str | None = None,
) -> Result:
    _, _, adjusted_rand_score, _, _ = _require_sklearn()
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    x = torch.as_tensor(dataset.x_np, device=device, dtype=dtype)

    def fit_model() -> FlashGMM:
        model = FlashGMM(
            d=x.shape[1],
            k=dataset.n_components,
            niter=args.max_iter,
            tol=args.tol,
            use_triton=use_triton,
            seed=args.seed,
            chunk_size_data=args.chunk_size_n,
            chunk_size_centroids=args.chunk_size_k,
            init_params=args.init_params,
            reg_covar=args.reg_covar,
            covariance_type=covariance_type,
            matmul_precision=args.matmul_precision,
            compute_labels_on_fit=not args.skip_fit_labels,
            approx_top_k=args.approx_top_k,
        )
        return model.fit(x)

    fit_seconds, model = _time_fit(device, fit_model, warmup_runs=args.warmup_runs)
    labels = model.predict(x).detach().cpu().numpy()
    score = float(model.score(x))
    return Result(
        baseline=baseline or ("flash-auto" if use_triton else "flash-torch"),
        fit_seconds=fit_seconds,
        score=score,
        ari=_ari(dataset.labels_np, labels, adjusted_rand_score),
        n_iter=model.n_iter_,
        note=(
            f"cov={model.covariance_type} init={model.init_source_} "
            f"estep={'triton' if model.triton_estep_enabled_ else 'torch'} "
            f"mstep={'approx-triton' if getattr(model, 'triton_approx_topk_enabled_', False) else ('fused' if getattr(model, 'triton_fused_update_enabled_', False) else ('streaming' if model.triton_streaming_update_enabled_ else 'torch'))}"
            f" labels={'triton' if getattr(model, 'triton_labels_enabled_', False) else 'torch'}"
            f"{' approx_top_k=' + str(model.approx_top_k_) if getattr(model, 'approximate_em_enabled_', False) else ''}"
            f"{' matmul=' + args.matmul_precision if args.matmul_precision else ''}"
            f"{' no_fit_labels' if args.skip_fit_labels else ''}"
        ),
    )


def _run_sklearn(dataset: Dataset, args: argparse.Namespace, covariance_type: str) -> Result:
    _, _, adjusted_rand_score, GaussianMixture, _ = _require_sklearn()
    model = GaussianMixture(
        n_components=dataset.n_components,
        covariance_type=covariance_type,
        tol=args.tol,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        n_init=1,
        init_params=args.sklearn_init_params,
        random_state=args.seed,
    )
    fit_seconds, fitted = _time_fit(torch.device("cpu"), lambda: model.fit(dataset.x_np), warmup_runs=args.warmup_runs)
    labels = fitted.predict(dataset.x_np)
    score = float(fitted.score(dataset.x_np))
    return Result(
        baseline=f"sklearn-{covariance_type}",
        fit_seconds=fit_seconds,
        score=score,
        ari=_ari(dataset.labels_np, labels, adjusted_rand_score),
        n_iter=int(fitted.n_iter_),
    )


def _run_torchgmm(dataset: Dataset, args: argparse.Namespace, covariance_type: str) -> Result:
    try:
        from torchgmm.bayes import GaussianMixture as TorchGMM
    except ImportError as exc:
        raise RuntimeError("Install optional baseline first: python -m pip install torchgmm") from exc

    _, _, adjusted_rand_score, _, _ = _require_sklearn()
    device = _resolve_device(args.device)
    x = torch.as_tensor(dataset.x_np, device=device, dtype=torch.float32)
    trainer_params = {
        "max_epochs": args.max_iter,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
    }
    if device.type == "cuda":
        trainer_params.update({"accelerator": "gpu", "devices": 1})
    else:
        trainer_params.update({"accelerator": "cpu"})

    def fit_model() -> TorchGMM:
        model = TorchGMM(
            num_components=dataset.n_components,
            covariance_type=covariance_type,
            init_strategy=args.torchgmm_init_strategy,
            convergence_tolerance=args.tol,
            covariance_regularization=args.reg_covar,
            batch_size=args.batch_size,
            trainer_params=trainer_params,
        )
        return model.fit(x)

    fit_seconds, model = _time_fit(device, fit_model, warmup_runs=args.warmup_runs)
    labels = model.predict(x).detach().cpu().numpy()
    score = -float(model.score(x))
    return Result(
        baseline=f"torchgmm-{covariance_type}",
        fit_seconds=fit_seconds,
        score=score,
        ari=_ari(dataset.labels_np, labels, adjusted_rand_score),
        n_iter=int(getattr(model, "num_iter_", -1)),
    )


def _format_float(value: float | None, width: int, precision: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{precision}f}"


def _print_results(dataset: Dataset, results: Iterable[Result]) -> None:
    print(f"dataset={dataset.name} n={dataset.x_np.shape[0]} d={dataset.x_np.shape[1]} k={dataset.n_components}")
    print(f"{'baseline':<20} {'fit_s':>10} {'score':>14} {'ari':>10} {'iters':>8}  note")
    for result in results:
        ari = _format_float(result.ari, 10)
        n_iter = "n/a" if result.n_iter is None or result.n_iter < 0 else str(result.n_iter)
        print(
            f"{result.baseline:<20} "
            f"{result.fit_seconds:>10.4f} "
            f"{result.score:>14.6f} "
            f"{ari} "
            f"{n_iter:>8}  {result.note}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GMMXX against GMM baselines.")
    parser.add_argument("--dataset", choices=["blobs", "anisotropic-blobs", "iris", "wine", "digits"], default="blobs")
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--cluster-std", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size-n", type=int, default=32768)
    parser.add_argument("--chunk-size-k", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--init-params", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument(
        "--sklearn-init-params",
        choices=["kmeans", "k-means++", "random", "random_from_data"],
        default="kmeans",
    )
    parser.add_argument("--torchgmm-init-strategy", default="kmeans")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--skip-fit-labels", action="store_true")
    parser.add_argument(
        "--approx-top-k",
        type=int,
        default=None,
        help="Approximate EM by normalizing responsibilities over each sample's top-k components.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["flash-auto", "flash-torch", "sklearn-spherical"],
        choices=[
            "flash-auto",
            "flash-torch",
            "flash-diag",
            "flash-diag-torch",
            "flash-tied",
            "flash-tied-torch",
            "flash-full",
            "flash-full-torch",
            "sklearn-spherical",
            "sklearn-diag",
            "sklearn-tied",
            "sklearn-full",
            "torchgmm-spherical",
            "torchgmm-diag",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = _load_dataset(args)
    results: list[Result] = []
    for baseline in args.baselines:
        try:
            if baseline == "flash-auto":
                results.append(_run_flash(
                    dataset,
                    args,
                    use_triton=True,
                    baseline=baseline,
                ))
            elif baseline == "flash-torch":
                results.append(_run_flash(
                    dataset,
                    args,
                    use_triton=False,
                ))
            elif baseline == "flash-diag":
                results.append(_run_flash(
                    dataset,
                    args,
                    use_triton=True,
                    covariance_type="diag",
                    baseline=baseline,
                ))
            elif baseline == "flash-diag-torch":
                results.append(_run_flash(
                    dataset,
                    args,
                    use_triton=False,
                    covariance_type="diag",
                    baseline=baseline,
                ))
            elif baseline == "flash-tied":
                results.append(_run_flash(
                    dataset,
                    args,
                    use_triton=True,
                    covariance_type="tied",
                    baseline=baseline,
                ))
            elif baseline == "flash-tied-torch":
                results.append(_run_flash(
                    dataset,
                    args,
                    use_triton=False,
                    covariance_type="tied",
                    baseline=baseline,
                ))
            elif baseline == "flash-full":
                results.append(_run_flash(
                    dataset,
                    args,
                    use_triton=True,
                    covariance_type="full",
                    baseline=baseline,
                ))
            elif baseline == "flash-full-torch":
                results.append(_run_flash(
                    dataset,
                    args,
                    use_triton=False,
                    covariance_type="full",
                    baseline=baseline,
                ))
            elif baseline.startswith("sklearn-"):
                results.append(_run_sklearn(dataset, args, baseline.removeprefix("sklearn-")))
            elif baseline.startswith("torchgmm-"):
                results.append(_run_torchgmm(dataset, args, baseline.removeprefix("torchgmm-")))
        except Exception as exc:
            results.append(Result(baseline=baseline, fit_seconds=float("nan"), score=float("nan"), ari=None, n_iter=None, note=str(exc)))

    _print_results(dataset, results)


if __name__ == "__main__":
    main()
