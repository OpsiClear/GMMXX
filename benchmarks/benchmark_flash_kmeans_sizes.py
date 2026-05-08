from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_LOCAL_TMP = REPO_ROOT / ".tmp" / "triton-tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(_LOCAL_TMP))
os.environ.setdefault("TEMP", str(_LOCAL_TMP))
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
tempfile.tempdir = str(_LOCAL_TMP)
_LOCAL_TRITON_CACHE = REPO_ROOT / ".triton-cache"
_LOCAL_TRITON_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", str(_LOCAL_TRITON_CACHE))

import torch

from gmmxx import GMMXX
from gmmxx._runtime import CUDA_FULL_STREAMED_MAX_COV_ELEMENTS


SHAPES = {
    # Names and dimensions mirror ../flash-kmeans-cuda benchmark presets.
    # Values are (N, D, K). B is fixed at 1 because GMMXX accepts unbatched
    # (N, D) tensors for these in-memory CUDA benchmarks.
    "tiny": (1024, 64, 32),
    "small": (8192, 64, 128),
    "med": (32768, 128, 256),
    "big": (131072, 128, 2048),
    "huge": (262144, 128, 4096),
    "mega": (524288, 128, 8192),
}
COVARIANCES = ("spherical", "diag", "tied", "full")
DEFAULT_MAX_FULL_COV_ELEMENTS = CUDA_FULL_STREAMED_MAX_COV_ELEMENTS


def _parse_csv(value: str, valid: set[str], name: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError(f"--{name} must contain at least one value")
    unknown = [item for item in items if item not in valid]
    if unknown:
        raise ValueError(f"unknown --{name} value(s): {unknown}; valid={sorted(valid)}")
    return items


def _parse_covariances(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(COVARIANCES)
    return _parse_csv(value, set(COVARIANCES), "covariances")


def _dtype(value: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[value]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fit_once(
    x: torch.Tensor,
    *,
    k: int,
    covariance_type: str,
    backend: str,
    max_iter: int,
    seed: int,
    approx_top_k: int | None,
    compute_labels: bool,
) -> GMMXX:
    model = GMMXX(
        n_components=k,
        max_iter=max_iter,
        tol=0.0,
        random_state=seed,
        init_params="random",
        covariance_type=covariance_type,
        backend=backend,
        compute_labels_on_fit=compute_labels,
        approx_top_k=approx_top_k,
        matmul_precision="high",
        dtype=x.dtype,
    )
    return model.fit(x)


def _case_mode(requested: str, k: int, approx_top_k: int) -> tuple[str, int | None]:
    if requested == "exact":
        return "exact", None
    if requested == "approx":
        return f"approx_topk={approx_top_k}", approx_top_k
    return "exact", None


def _full_skip_reason(
    covariance_type: str,
    d: int,
    k: int,
    *,
    allow_large_full: bool,
    max_full_cov_elements: int,
) -> str | None:
    if covariance_type != "full" or allow_large_full:
        return None
    cov_elements = k * d * d
    if cov_elements <= max_full_cov_elements:
        return None
    return (
        f"skipped_full_state_KD2={cov_elements} "
        f"> max_full_cov_elements={max_full_cov_elements}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GMMXX EM on flash-kmeans-cuda-sized shapes."
    )
    parser.add_argument("--shapes", default="med,big", help=f"Comma list from {','.join(SHAPES)}.")
    parser.add_argument(
        "--covariances",
        default="spherical",
        help="Comma list from spherical,diag,tied,full, or 'all'.",
    )
    parser.add_argument("--backends", default="cuda,triton,torch", help="Comma list from cuda,triton,torch,auto.")
    parser.add_argument("--mode", choices=["auto", "exact", "approx"], default="auto")
    parser.add_argument("--approx-top-k", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute-labels", action="store_true")
    parser.add_argument(
        "--allow-large-full",
        action="store_true",
        help="Run full covariance even when K*D*D exceeds the safety guard.",
    )
    parser.add_argument(
        "--max-full-cov-elements",
        type=int,
        default=DEFAULT_MAX_FULL_COV_ELEMENTS,
        help="Skip full covariance cases whose K*D*D state exceeds this value.",
    )
    args = parser.parse_args()

    if args.approx_top_k <= 0:
        raise ValueError("--approx-top-k must be positive")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.max_full_cov_elements <= 0:
        raise ValueError("--max-full-cov-elements must be positive")

    shapes = _parse_csv(args.shapes, set(SHAPES), "shapes")
    covariances = _parse_covariances(args.covariances)
    backends = _parse_csv(args.backends, {"auto", "cuda", "triton", "torch"}, "backends")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"torch={torch.__version__} dtype={args.dtype} max_iter={args.max_iter}")
    print(
        "mode\tcovariance\tcase\tshape\tbackend\tfit_s\tms_per_iter\tspeedup_vs_torch\t"
        "last_backend\ttriton_approx\tnote"
    )

    for shape_name in shapes:
        n, d, k = SHAPES[shape_name]
        torch.manual_seed(args.seed + n * 13 + d * 17 + k * 19)
        x = torch.randn(n, d, device=device, dtype=dtype)

        for covariance_type in covariances:
            mode_label, approx_top_k = _case_mode(args.mode, k, args.approx_top_k)
            skip_reason = _full_skip_reason(
                covariance_type,
                d,
                k,
                allow_large_full=args.allow_large_full,
                max_full_cov_elements=args.max_full_cov_elements,
            )
            if skip_reason is not None:
                for backend in backends:
                    print(
                        f"{mode_label}\t{covariance_type}\t{shape_name}\tN={n},D={d},K={k}\t"
                        f"{backend}\tnan\tnan\tn/a\tskipped\tFalse\t{skip_reason}"
                    )
                continue

            rows: list[tuple[str, float, object, bool, str]] = []
            times: dict[str, float] = {}
            for backend in backends:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                try:
                    for _ in range(args.warmup_runs):
                        _fit_once(
                            x,
                            k=k,
                            covariance_type=covariance_type,
                            backend=backend,
                            max_iter=args.max_iter,
                            seed=args.seed,
                            approx_top_k=approx_top_k,
                            compute_labels=args.compute_labels,
                        )
                        _sync(device)

                    _sync(device)
                    start = time.perf_counter()
                    model = _fit_once(
                        x,
                        k=k,
                        covariance_type=covariance_type,
                        backend=backend,
                        max_iter=args.max_iter,
                        seed=args.seed,
                        approx_top_k=approx_top_k,
                        compute_labels=args.compute_labels,
                    )
                    _sync(device)
                    elapsed = time.perf_counter() - start
                    info = getattr(model, "fit_info_", {}) or {}
                    actual = info.get("backend_breakdown", getattr(model, "last_backend_used_", backend))
                    note = str(getattr(model, "last_fallback_reason_", "") or "")
                    if info.get("cuda_tensor_streamed_enabled"):
                        chunk_note = (
                            f"cuda_chunks="
                            f"{info.get('cuda_tensor_streamed_chunk_size_N')}/"
                            f"{info.get('cuda_tensor_streamed_chunk_size_K')}"
                        )
                        note = chunk_note if not note else f"{note}; {chunk_note}"
                    rows.append(
                        (
                            backend,
                            elapsed,
                            actual,
                            bool(getattr(model, "triton_approx_topk_enabled_", False)),
                            note,
                        )
                    )
                    times[backend] = elapsed
                except Exception as exc:
                    rows.append((backend, math.nan, "failed", False, f"{type(exc).__name__}: {exc}"))
                    times[backend] = math.nan

            torch_seconds = times.get("torch", math.nan)
            for backend, elapsed, actual, triton_approx, note in rows:
                speedup = "n/a"
                if math.isfinite(torch_seconds) and math.isfinite(elapsed) and elapsed > 0:
                    speedup = f"{torch_seconds / elapsed:.2f}x"
                fit_s = f"{elapsed:.4f}" if math.isfinite(elapsed) else "nan"
                ms_per_iter = f"{elapsed * 1000.0 / args.max_iter:.3f}" if math.isfinite(elapsed) else "nan"
                print(
                    f"{mode_label}\t{covariance_type}\t{shape_name}\tN={n},D={d},K={k}\t"
                    f"{backend}\t{fit_s}\t{ms_per_iter}\t{speedup}\t"
                    f"{actual}\t{triton_approx}\t{note}"
                )

        del x
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
