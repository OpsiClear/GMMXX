from __future__ import annotations

import math
from typing import Any, Optional

import torch

from ._runtime import triton_spherical_supported

try:
    from .assign_spherical_triton import (
        spherical_assign_triton,
        spherical_logsumexp_triton,
        spherical_resp_triton,
    )
    _HAS_TRITON_ASSIGN = True
except Exception:
    _HAS_TRITON_ASSIGN = False

try:
    from .assign_full_triton import (
        full_assign_triton,
        full_logsumexp_triton,
        full_resp_triton,
    )
    _HAS_TRITON_FULL_INFERENCE = True
except Exception:
    _HAS_TRITON_FULL_INFERENCE = False

try:
    from .assign_diag_triton import (
        diag_assign_triton,
        diag_logsumexp_triton,
        diag_resp_triton,
    )
    _HAS_TRITON_DIAG_INFERENCE = True
except Exception:
    _HAS_TRITON_DIAG_INFERENCE = False

from .torch_fallback import (
    _precision_and_logdet,
    batch_gmm_Diagonal_torch_native,
    batch_gmm_Full_torch_native,
    batch_gmm_Spherical_torch_native,
    batch_gmm_Tied_torch_native,
    diagonal_assign_torch_native_chunked,
    diagonal_predict_proba_torch_native_chunked,
    diagonal_score_samples_torch_native_chunked,
    full_assign_torch_native_chunked,
    full_predict_proba_torch_native_chunked,
    full_score_samples_torch_native_chunked,
    _resolve_approx_top_k,
    spherical_assign_torch_native_chunked,
    spherical_predict_proba_torch_native_chunked,
    spherical_score_samples_torch_native_chunked,
    tied_assign_torch_native_chunked,
    tied_predict_proba_torch_native_chunked,
    tied_score_samples_torch_native_chunked,
)
from .large_n import (
    batch_gmm_largeN_cpu,
    large_n_predict_cpu,
    large_n_predict_proba_cpu,
    large_n_score_samples_cpu,
)


_VALID_COVARIANCE_TYPES = {"spherical", "diag", "tied", "full"}
_VALID_BACKENDS = {"auto", "cuda", "triton", "torch"}
_COVARIANCE_ALIASES = {"diagonal": "diag"}
_SKLEARN_PARAM_ALIASES = {
    "n_components": "k",
    "max_iter": "niter",
    "random_state": "seed",
    "n_features": "d",
}


def _normalize_covariance_type(covariance_type: str) -> str:
    covariance_type = _COVARIANCE_ALIASES.get(covariance_type, covariance_type)
    if covariance_type not in _VALID_COVARIANCE_TYPES:
        raise ValueError("covariance_type must be 'spherical', 'diag', 'diagonal', 'tied', or 'full'")
    return covariance_type


def _optional_positive_int(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _resolve_alias(
    primary: Optional[int],
    alias: Optional[int],
    *,
    primary_name: str,
    alias_name: str,
) -> Optional[int]:
    if primary is not None and alias is not None and int(primary) != int(alias):
        raise ValueError(f"Received conflicting {primary_name}={primary} and {alias_name}={alias}")
    return primary if primary is not None else alias


class GMMXX:
    """
    Flash-kmeans-style interface for Gaussian Mixture Models.

    Notes
    -----
    - `covariance_type` supports `spherical`, `diag`, `tied`, and `full`.
      Spherical, diagonal, tied, and small-D full covariance use flash-kmeans-style
      Triton EM kernels where the validated shape policy supports them.
    - `use_triton=True` enables Triton paths when CUDA and shape policy support them;
      unsupported shapes use PyTorch automatically.
    - Runtime Triton failures fall back to the chunked PyTorch implementation.
    - `approx_top_k` is an opt-in training approximation; prediction remains exact
      for the fitted parameters.
    """

    def __init__(
        self,
        d: Optional[int] = None,
        k: Optional[int] = None,
        niter: Optional[int] = None,
        tol: float = 1e-4,
        use_triton: Optional[bool] = None,  # deprecated; see below
        seed: Optional[int] = None,
        chunk_size_data: int = 32768,
        chunk_size_centroids: int = 1024,
        chunk_size_data_cpu: int = 1048576,
        verbose: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        init_params: str = "kmeans",
        reg_covar: float = 1e-6,
        init_kmeans_iters: int = 10,
        init_kmeans_tol: float = 1e-4,
        covariance_type: str = "spherical",
        matmul_precision: Optional[str] = None,
        compute_labels_on_fit: bool = True,
        approx_top_k: Optional[int] = None,
        n_components: Optional[int] = None,
        max_iter: Optional[int] = None,
        random_state: Optional[int] = None,
        n_features: Optional[int] = None,
        backend: str = "auto",
    ):
        d = _resolve_alias(d, n_features, primary_name="d", alias_name="n_features")
        k = _resolve_alias(k, n_components, primary_name="k", alias_name="n_components")
        niter = _resolve_alias(niter, max_iter, primary_name="niter", alias_name="max_iter")
        seed = _resolve_alias(seed, random_state, primary_name="seed", alias_name="random_state")

        # Backend selection + use_triton deprecation shim.
        # use_triton=True  -> backend="auto",  _legacy_no_triton=False
        # use_triton=False -> backend="auto",  _legacy_no_triton=True
        # If the user passes both backend= and use_triton=, raise.
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
        if use_triton is not None:
            import warnings as _warnings
            if backend != "auto":
                # User passed both. Conflict.
                raise ValueError(
                    "Cannot pass both backend= and use_triton= to GMMXX. "
                    "Drop use_triton; backend= is the canonical kwarg."
                )
            _warnings.warn(
                "use_triton is deprecated; use backend='auto'|'cuda'|'triton'|'torch' instead. "
                "use_triton=True maps to backend='auto'; use_triton=False maps to backend='auto' "
                "with Triton filtered from the dispatch chain.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._legacy_no_triton = (use_triton is False)
        else:
            self._legacy_no_triton = False
        self.backend = backend

        if k is None:
            raise ValueError("k or n_components is required")

        self.d = _optional_positive_int(d, "d")
        self.k = _positive_int(k, "k")
        self.niter = _positive_int(100 if niter is None else niter, "niter")
        self.tol = float(tol)
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative")
        # use_triton is no longer stored as an instance attr; the deprecation
        # shim above translated it into self.backend + self._legacy_no_triton.
        self.seed = int(0 if seed is None else seed)
        self.chunk_size_data = _positive_int(chunk_size_data, "chunk_size_data")
        self.chunk_size_centroids = _positive_int(chunk_size_centroids, "chunk_size_centroids")
        self.chunk_size_data_cpu = _positive_int(chunk_size_data_cpu, "chunk_size_data_cpu")
        self.verbose = bool(verbose)
        self.dtype = dtype
        self.init_params = init_params
        self.reg_covar = float(reg_covar)
        if self.reg_covar < 0.0:
            raise ValueError("reg_covar must be non-negative")
        self.init_kmeans_iters = _positive_int(init_kmeans_iters, "init_kmeans_iters")
        self.init_kmeans_tol = float(init_kmeans_tol)
        if self.init_kmeans_tol < 0.0:
            raise ValueError("init_kmeans_tol must be non-negative")
        self.covariance_type = _normalize_covariance_type(covariance_type)
        if matmul_precision is not None and matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("matmul_precision must be None, 'highest', 'high', or 'medium'")
        self.matmul_precision = matmul_precision
        self.compute_labels_on_fit = bool(compute_labels_on_fit)
        self.approx_top_k = _optional_positive_int(approx_top_k, "approx_top_k")

        self._raw_device = device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.means_b: Optional[torch.Tensor] = None
        self.variances_b: Optional[torch.Tensor] = None
        self.covariances_b: Optional[torch.Tensor] = None
        self.weights_b: Optional[torch.Tensor] = None
        self.cluster_ids_b: Optional[torch.Tensor] = None
        self.means_: Optional[torch.Tensor] = None
        self.weights_: Optional[torch.Tensor] = None
        self.covariances_: Optional[torch.Tensor] = None
        self.labels_: Optional[torch.Tensor] = None
        self.lower_bound_: Optional[float] = None
        self.lower_bound_history_: Optional[list] = None
        self.n_iter_: Optional[int] = None
        self.init_source_: Optional[str] = None
        self.fit_info_: Optional[dict[str, object]] = None
        self.triton_estep_enabled_: Optional[bool] = None
        self.triton_fused_update_enabled_: Optional[bool] = None
        self.triton_approx_topk_enabled_: Optional[bool] = None
        self.triton_streaming_update_enabled_: Optional[bool] = None
        self.triton_labels_enabled_: Optional[bool] = None
        self.approximate_em_enabled_: Optional[bool] = None
        self.approx_top_k_: Optional[int] = None
        self.large_n_streaming_enabled_: Optional[bool] = None
        self.copy_stream_prefetch_enabled_: Optional[bool] = None
        self.last_fallback_reason_: Optional[str] = None
        self.last_backend_used_: Optional[str] = None
        self.cuda_estep_enabled_: Optional[bool] = None
        self.cuda_fused_update_enabled_: Optional[bool] = None
        self.cuda_approx_topk_enabled_: Optional[bool] = None
        self._batch_size: Optional[int] = None
        self._diag_inference_cache: Optional[tuple] = None
        self._full_inference_cache: Optional[tuple] = None
        self._tied_inference_cache: Optional[tuple] = None

    @property
    def use_triton(self) -> bool:
        """Derived: True iff the Triton path may be used at all from this estimator.

        Returns True only when the user requested 'auto' or 'triton' (and didn't
        use the legacy use_triton=False shim). Explicit backend='cuda' returns
        False so internal Triton-gated inference paths fall to torch (not triton)
        when the user explicitly picked CUDA. Plan 3 rewires those inference paths
        to consult the dispatcher directly.
        """
        if self._legacy_no_triton:
            return False
        return self.backend in {"auto", "triton"}

    def _apply_matmul_precision(self) -> None:
        if self.matmul_precision is not None:
            torch.set_float32_matmul_precision(self.matmul_precision)

    def _invalidate_inference_caches(self) -> None:
        self._diag_inference_cache = None
        self._full_inference_cache = None
        self._tied_inference_cache = None

    def _squeeze_if_unbatched(self, value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.squeeze(0) if self._batch_size is None else value

    def _reset_fit_state(self) -> None:
        self.means_b = None
        self.variances_b = None
        self.covariances_b = None
        self.weights_b = None
        self.cluster_ids_b = None
        self.means_ = None
        self.weights_ = None
        self.covariances_ = None
        self.labels_ = None
        self.lower_bound_ = None
        self.lower_bound_history_ = None
        self.n_iter_ = None
        self.init_source_ = None
        self.fit_info_ = None
        self.triton_estep_enabled_ = None
        self.triton_fused_update_enabled_ = None
        self.triton_approx_topk_enabled_ = None
        self.triton_streaming_update_enabled_ = None
        self.triton_labels_enabled_ = None
        self.approximate_em_enabled_ = None
        self.approx_top_k_ = None
        self.large_n_streaming_enabled_ = None
        self.copy_stream_prefetch_enabled_ = None
        self.last_fallback_reason_ = None
        self.last_backend_used_ = None
        self.cuda_estep_enabled_ = None
        self.cuda_fused_update_enabled_ = None
        self.cuda_approx_topk_enabled_ = None
        self._batch_size = None
        self._invalidate_inference_caches()

    def _set_fit_result(
        self,
        *,
        labels_b: Optional[torch.Tensor],
        means_b: torch.Tensor,
        variances_b: torch.Tensor,
        weights_b: torch.Tensor,
        info: dict[str, object],
        batch_size: Optional[int],
    ) -> None:
        self.cluster_ids_b = labels_b
        self.means_b = means_b
        self.variances_b = variances_b
        self.covariances_b = variances_b
        self.weights_b = weights_b
        self._batch_size = batch_size

        self.means_ = self._squeeze_if_unbatched(self.means_b)
        self.weights_ = self._squeeze_if_unbatched(self.weights_b)
        self.covariances_ = self._squeeze_if_unbatched(self.covariances_b)
        self.labels_ = self._squeeze_if_unbatched(self.cluster_ids_b)

        self.lower_bound_ = float(info["lower_bound"])
        self.lower_bound_history_ = list(info["lower_bound_history"])
        self.n_iter_ = int(info["n_iter"])
        self.init_source_ = str(info["init_source"])
        self.fit_info_ = dict(info)
        self.triton_estep_enabled_ = bool(info["triton_estep_enabled"])
        self.triton_fused_update_enabled_ = bool(info.get("triton_fused_update_enabled", False))
        self.triton_approx_topk_enabled_ = bool(info.get("triton_approx_topk_enabled", False))
        self.triton_streaming_update_enabled_ = bool(info["triton_streaming_update_enabled"])
        self.triton_labels_enabled_ = bool(info.get("triton_labels_enabled", False))
        self.approximate_em_enabled_ = bool(info.get("approximate_em_enabled", False))
        self.approx_top_k_ = info.get("approx_top_k")
        self.large_n_streaming_enabled_ = bool(info.get("large_n_streaming_enabled", False))
        self.copy_stream_prefetch_enabled_ = bool(info.get("copy_stream_prefetch_enabled", False))
        self.last_fallback_reason_ = info.get("fallback_reason")
        self._invalidate_inference_caches()

    def _record_fallback(self, context: str, exc: Exception) -> None:
        self.last_fallback_reason_ = f"{context}: {type(exc).__name__}: {exc}"

    def _normalize_input(self, data: torch.Tensor) -> tuple[torch.Tensor, Optional[int]]:
        if not torch.is_tensor(data):
            raise TypeError("data must be a torch.Tensor")
        if not data.dtype.is_floating_point:
            raise TypeError("data must be a floating point tensor")
        if data.ndim == 2:
            n, d = data.shape
            if n == 0:
                raise ValueError("data must contain at least one sample")
            if self.d is None:
                self.d = int(d)
            elif d != self.d:
                raise ValueError(f"Expected d={self.d}, got d={d}")
            return data.unsqueeze(0), None
        if data.ndim == 3:
            bsz, _, d = data.shape
            if bsz == 0 or data.shape[1] == 0:
                raise ValueError("data must contain at least one sample")
            if self.d is None:
                self.d = int(d)
            elif d != self.d:
                raise ValueError(f"Expected d={self.d}, got d={d}")
            return data, bsz
        raise ValueError("data must have shape (N, D) or (B, N, D)")

    def _compute_dtype_for_input(self, x_b: torch.Tensor) -> torch.dtype:
        compute_dtype = self.dtype
        if compute_dtype is None:
            if x_b.dtype in (torch.float16, torch.bfloat16):
                compute_dtype = torch.float32
            else:
                compute_dtype = x_b.dtype
        return compute_dtype

    def _is_large_cpu_stream_input(self, x_b: torch.Tensor) -> bool:
        return (
            x_b.device.type == "cpu"
            and self.device.type == "cuda"
            and x_b.shape[1] > self.chunk_size_data_cpu
        )

    def _prepare_compute_tensor(self, x_b: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._compute_dtype_for_input(x_b)

        if self._is_large_cpu_stream_input(x_b):
            raise NotImplementedError(
                "Large-N CPU-to-GPU streaming is only available through train/predict/score APIs."
            )

        return x_b.to(device=self.device, dtype=compute_dtype, copy=False)

    def _train_large_cpu_streaming(self, x_b: torch.Tensor, batch_size: Optional[int]) -> None:
        compute_dtype = self._compute_dtype_for_input(x_b)
        labels_b, means_b, variances_b, weights_b, info = batch_gmm_largeN_cpu(
            x_b,
            self.k,
            covariance_type=self.covariance_type,
            device=self.device,
            dtype=compute_dtype,
            max_iters=self.niter,
            tol=self.tol,
            verbose=self.verbose,
            init_params=self.init_params,
            reg_covar=self.reg_covar,
            chunk_size_N=self.chunk_size_data,
            chunk_size_K=self.chunk_size_centroids,
            kmeans_init_iters=self.init_kmeans_iters,
            kmeans_init_tol=self.init_kmeans_tol,
            kmeans_use_triton=self.use_triton,
            gmm_use_triton="auto" if self.use_triton else False,
            compute_labels=self.compute_labels_on_fit,
            approx_top_k=self.approx_top_k,
            backend=self.backend,
            legacy_no_triton=self._legacy_no_triton,
        )
        self._set_fit_result(
            labels_b=labels_b,
            means_b=means_b,
            variances_b=variances_b,
            weights_b=weights_b,
            info=info,
            batch_size=batch_size,
        )
        # Set last_backend_used_ from backend_breakdown if available.
        bd = info.get("backend_breakdown", {})
        if bd.get("cuda", 0) > 0:
            self.last_backend_used_ = "cuda"
        elif bd.get("triton", 0) > 0:
            self.last_backend_used_ = "triton"
        else:
            self.last_backend_used_ = "torch"

    def _train_spherical_cuda(self, x_b: torch.Tensor, batch_size: Optional[int]) -> None:
        """Spherical EM loop on the CUDA backend (Plan 2 safe path).

        Mirrors batch_gmm_Spherical_torch_native's structure:
          1. Initialize means via random sampling.
          2. Initialize variances and weights uniformly.
          3. EM loop: assign -> blocked_update -> finalize -> check ELBO.
          4. Call _set_fit_result with the final tensors.

        When approx_top_k is set to a value in [1, K-1], each EM iteration
        uses the CUDA approximate top-k soft-stat update. Otherwise, when
        cuda_spherical_fused_supported(D, K, dtype) is True (D <= 64, K <=
        128), each EM iteration uses the fused single-tile exact kernel.
        """
        from . import _cuda as _cuda_mod
        from . import _runtime as _gm_runtime

        B, N, D = x_b.shape
        K = self.k
        device = x_b.device

        effective_approx_top_k = _resolve_approx_top_k(self.approx_top_k, K)
        use_approx = effective_approx_top_k is not None
        # Decide once per fit() whether the fused path is available for this
        # (D, K, dtype). Approx-topK is a separate soft-EM approximation and
        # intentionally disables the exact fused path.
        # Exp3: fused kernel is SIMT-only (no sm80 mma); for fp16/bf16 the
        # separate sm80-mma E-step + soft M-step path beats fused-SIMT by
        # leveraging tensor cores via cuBLAS / mma.sync.
        # Exp13: for fp32 with D >= 16 the cuBLAS GEMM fastpath inside
        # spherical_logsumexp/_resp uses TF32 tensor cores and beats the
        # SIMT fused kernel. Disable fused there so we route through
        # use_soft_update -> cuBLAS path.
        from . import _cuda as _cm_local
        cublas_fastpath = (
            x_b.dtype == torch.float32
            and _cm_local._use_torch_fastpath_spherical(x_b)
        )
        use_fused = (
            not use_approx
            and x_b.dtype == torch.float32
            and not cublas_fastpath
            and _gm_runtime.cuda_spherical_fused_supported(D, K, x_b.dtype)
        )
        use_soft_update = not use_approx and not use_fused and D <= 128 and K <= 256
        self.cuda_fused_update_enabled_ = bool(use_fused)
        self.cuda_approx_topk_enabled_ = bool(use_approx)

        # Initialize means by sampling from the data.
        rng = torch.Generator(device=device).manual_seed(self.seed)
        init_idx = torch.randint(0, N, (B, K), generator=rng, device=device)
        means = torch.gather(
            x_b, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)
        ).contiguous()

        # Initialize variances to per-feature data variance averaged over D, divided by K.
        var = (
            x_b.float().var(dim=1).mean(dim=-1, keepdim=True)
            .expand(B, K).contiguous() / K
        ).clamp_min(self.reg_covar)
        log_w = torch.full((B, K), -math.log(K), dtype=torch.float32, device=device)
        weights = torch.full((B, K), 1.0 / K, dtype=torch.float32, device=device)

        lower_bound_history: list[float] = []
        n_iter = 0
        prev_lb = -math.inf
        lb = float("-inf")
        ids: Optional[torch.Tensor] = None

        # Hoist x.float() and |x|^2 out of the EM loop — both depend only on
        # x and otherwise get recomputed every iteration inside soft_update.
        x_f_cached: Optional[torch.Tensor] = None
        x_sq_cached: Optional[torch.Tensor] = None
        if use_soft_update:
            x_f_cached = x_b.float() if x_b.dtype != torch.float32 else x_b
            x_sq_cached = x_f_cached.square().sum(dim=-1)

        # Hoist hot-path attribute and module-attribute lookups out of
        # the Python loop. The interpreter's name resolution per iter on
        # small shapes (where each iter is < 1 ms) is non-trivial.
        _niter = self.niter
        _reg_covar = self.reg_covar
        _tol = self.tol
        _last_iter = _niter - 1
        _soft_update = _cuda_mod.soft_update_spherical
        _fused = _cuda_mod.fused_spherical
        _approx = _cuda_mod.approx_topk_update_spherical
        for it in range(_niter):
            n_iter += 1
            is_last = (it == _last_iter)
            if use_fused:
                means, var, weights, lse, ids = _fused(
                    x_b, means, var, log_w, _reg_covar
                )
                log_w = torch.log(weights.clamp_min(1e-30))
                lb = float(lse.mean().item())
            elif use_soft_update:
                means, var, weights, lse, ids = _soft_update(
                    x_b, means, var, log_w, _reg_covar,
                    x_f_cached=x_f_cached, x_sq_cached=x_sq_cached,
                    compute_ids=is_last,
                )
                log_w = torch.log(weights.clamp_min(1e-30))
                lb = float(lse.mean().item())
            elif use_approx:
                nk, sum_x, sum_x_sq, ll_sum = _cuda_mod.approx_topk_update_spherical(
                    x_b,
                    means,
                    var,
                    log_w,
                    top_k=int(effective_approx_top_k),
                    chunk_size_K=self.chunk_size_centroids,
                )
                active_mask = nk > 1e-8
                nk_safe = nk.clamp_min(1e-8)
                means_new = (sum_x / nk_safe.unsqueeze(-1)).to(x_b.dtype)
                means_new = torch.where(active_mask.unsqueeze(-1), means_new, means)

                mean_sq = means_new.float().square().sum(dim=-1)
                var_new = (sum_x_sq - nk * mean_sq).clamp_min(0.0) / (
                    nk_safe * float(D)
                )
                var_new = var_new.clamp_min(self.reg_covar)
                var_new = torch.where(active_mask, var_new, var)

                weights = (nk / float(N)).clamp_min(1e-8)
                weights = weights / weights.sum(dim=-1, keepdim=True)
                means, var = means_new, var_new
                log_w = torch.log(weights.clamp_min(1e-30))
                lb = float((ll_sum / float(B * N)).item())
                ids = None
            else:
                ids = _cuda_mod.spherical_assign(x_b, means, var, log_w)
                lse = _cuda_mod.spherical_logsumexp(x_b, means, var, log_w)
                lb = float(lse.mean().item())

                sums, sumsq, counts = _cuda_mod.blocked_update_spherical(
                    x_b, ids, K,
                    force_sort=getattr(self, "_force_sort", None),
                )
                means, var, weights = _cuda_mod.finalize_spherical(
                    sums, sumsq, counts, means, var, N, self.reg_covar
                )
                log_w = torch.log(weights.clamp_min(1e-30))

            lower_bound_history.append(lb)
            if abs(lb - prev_lb) < _tol:
                break
            prev_lb = lb

        if self.compute_labels_on_fit and ids is None:
            ids = _cuda_mod.spherical_assign(x_b, means, var, log_w)
        labels_b = ids if self.compute_labels_on_fit else None
        info = {
            "lower_bound": lb,
            "lower_bound_history": lower_bound_history,
            "n_iter": n_iter,
            "init_source": "cuda_random",
            "triton_estep_enabled": False,
            "triton_streaming_update_enabled": False,
            "triton_fused_update_enabled": False,
            "triton_approx_topk_enabled": False,
            "triton_labels_enabled": False,
            "approximate_em_enabled": bool(use_approx),
            "approx_top_k": effective_approx_top_k,
            "cuda_approx_topk_enabled": bool(use_approx),
            "large_n_streaming_enabled": False,
            "copy_stream_prefetch_enabled": False,
            "backend_breakdown": {"cuda": n_iter},
        }
        self._set_fit_result(
            labels_b=labels_b,
            means_b=means,
            variances_b=var,
            weights_b=weights,
            info=info,
            batch_size=batch_size,
        )

    def _train_diag_cuda(self, x_b: torch.Tensor, batch_size: Optional[int]) -> None:
        """Diagonal-covariance EM loop on the CUDA backend.

        Mirrors _train_spherical_cuda but with per-feature variance (B, K, D).
        """
        from . import _cuda as _cuda_mod

        B, N, D = x_b.shape
        K = self.k
        device = x_b.device

        # Initialize means by sampling from the data.
        rng = torch.Generator(device=device).manual_seed(self.seed)
        init_idx = torch.randint(0, N, (B, K), generator=rng, device=device)
        means = torch.gather(
            x_b, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)
        ).contiguous()

        # Initialize variances per-feature: data variance per feature,
        # broadcast to K. x_b.float().var(dim=1) is (B, D); expand to (B, K, D).
        feat_var = x_b.float().var(dim=1).unsqueeze(1).expand(B, K, D).contiguous() / K
        var = feat_var.clamp_min(self.reg_covar)
        log_w = torch.full((B, K), -math.log(K), dtype=torch.float32, device=device)
        weights = torch.full((B, K), 1.0 / K, dtype=torch.float32, device=device)
        x_b_f = x_b.float()

        lower_bound_history: list[float] = []
        n_iter = 0
        prev_lb = -math.inf
        lb = float("-inf")
        ids: Optional[torch.Tensor] = None

        for _ in range(self.niter):
            n_iter += 1
            lse = _cuda_mod.diag_logsumexp(x_b, means, var, log_w)
            lb = float(lse.mean().item())
            lower_bound_history.append(lb)

            resp = _cuda_mod.diag_resp(x_b, means, var, log_w, lse)
            nk = resp.sum(dim=1)
            resp_t = resp.transpose(1, 2)
            sums = torch.bmm(resp_t, x_b_f)
            sumsq = torch.bmm(resp_t, x_b_f.square())

            active_mask = nk > 1e-8
            nk_safe = nk.clamp_min(1e-8)
            means_new = (sums / nk_safe.unsqueeze(-1)).to(x_b.dtype)
            means_new = torch.where(active_mask.unsqueeze(-1), means_new, means)
            var_new = (sumsq / nk_safe.unsqueeze(-1) - means_new.float().square()).clamp_min(
                self.reg_covar
            )
            var_new = torch.where(active_mask.unsqueeze(-1), var_new, var)

            weights = (nk / float(N)).clamp_min(1e-8)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            means, var = means_new, var_new
            log_w = torch.log(weights.clamp_min(1e-30))

            if abs(lb - prev_lb) < self.tol:
                break
            prev_lb = lb

        if self.compute_labels_on_fit:
            ids = _cuda_mod.diag_assign(x_b, means, var, log_w)
        labels_b = ids if self.compute_labels_on_fit else None
        info = {
            "lower_bound": lb,
            "lower_bound_history": lower_bound_history,
            "n_iter": n_iter,
            "init_source": "cuda_random",
            "triton_estep_enabled": False,
            "triton_streaming_update_enabled": False,
            "triton_fused_update_enabled": False,
            "triton_approx_topk_enabled": False,
            "triton_labels_enabled": False,
            "approximate_em_enabled": False,
            "approx_top_k": None,
            "large_n_streaming_enabled": False,
            "copy_stream_prefetch_enabled": False,
            "backend_breakdown": {"cuda": n_iter},
        }
        self._set_fit_result(
            labels_b=labels_b,
            means_b=means,
            variances_b=var,
            weights_b=weights,
            info=info,
            batch_size=batch_size,
        )

    def _train_tied_cuda(self, x_b: torch.Tensor, batch_size: Optional[int]) -> None:
        """Tied-covariance EM loop on the CUDA backend.

        Reuses the spherical CUDA kernels via projected coordinates. Only
        the projection step + tied finalize live on the host.
        """
        import math
        from . import _cuda as _cuda_mod

        B, N, D = x_b.shape
        K = self.k
        device = x_b.device

        # Init: random means; L = sqrt(data_var) * I (isotropic).
        rng = torch.Generator(device=device).manual_seed(self.seed)
        init_idx = torch.randint(0, N, (B, K), generator=rng, device=device)
        means = torch.gather(
            x_b, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)
        ).contiguous()
        data_var_scalar = x_b.float().var(dim=1).mean(-1, keepdim=True)  # (B, 1)
        sqrt_var = data_var_scalar.clamp_min(self.reg_covar).sqrt().unsqueeze(-1)  # (B, 1, 1)
        eye = torch.eye(D, device=device, dtype=x_b.dtype).unsqueeze(0)  # (1, D, D)
        L = (sqrt_var.to(x_b.dtype) * eye).contiguous()  # (B, D, D)
        log_w = torch.full((B, K), -math.log(K), dtype=torch.float32, device=device)
        weights = torch.full((B, K), 1.0 / K, dtype=torch.float32, device=device)

        # Precompute X^T X for the M-step (constant across iterations).
        x_b_f = x_b.float()
        xx_total = x_b_f.transpose(-1, -2) @ x_b_f  # (B, D, D)

        lower_bound_history: list[float] = []
        n_iter = 0
        prev_lb = -math.inf
        lb = float("-inf")
        ids: Optional[torch.Tensor] = None

        for _ in range(self.niter):
            n_iter += 1
            lse = _cuda_mod.tied_logsumexp(x_b, means, L, log_w)
            lb = float(lse.mean().item())
            lower_bound_history.append(lb)

            resp = _cuda_mod.tied_resp(x_b, means, L, log_w, lse)
            counts = resp.sum(dim=1)
            sums = torch.bmm(resp.transpose(1, 2), x_b_f)
            means, L, weights = _cuda_mod.tied_finalize(
                sums, xx_total, counts, N, self.reg_covar
            )
            log_w = torch.log(weights.clamp_min(1e-30))

            if abs(lb - prev_lb) < self.tol:
                break
            prev_lb = lb

        # GMMXX exposes covariances_ as the full (D, D) covariance matrix.
        cov_b = L @ L.transpose(-1, -2)  # (B, D, D)

        if self.compute_labels_on_fit:
            ids = _cuda_mod.tied_assign(x_b, means, L, log_w)
        labels_b = ids if self.compute_labels_on_fit else None
        info = {
            "lower_bound": lb,
            "lower_bound_history": lower_bound_history,
            "n_iter": n_iter,
            "init_source": "cuda_random",
            "triton_estep_enabled": False,
            "triton_streaming_update_enabled": False,
            "triton_fused_update_enabled": False,
            "triton_approx_topk_enabled": False,
            "triton_labels_enabled": False,
            "approximate_em_enabled": False,
            "approx_top_k": None,
            "large_n_streaming_enabled": False,
            "copy_stream_prefetch_enabled": False,
            "backend_breakdown": {"cuda": n_iter},
        }
        self._set_fit_result(
            labels_b=labels_b,
            means_b=means,
            variances_b=cov_b,  # (B, D, D)
            weights_b=weights,
            info=info,
            batch_size=batch_size,
        )

    def _train_full_cuda(self, x_b: torch.Tensor, batch_size: Optional[int]) -> None:
        """Full-covariance EM loop on the CUDA backend (D <= 16, K <= 32)."""
        import math
        from . import _cuda as _cuda_mod

        B, N, D = x_b.shape
        K = self.k
        device = x_b.device

        # Init: random means; per-cluster L_k = sqrt(data_var) * I (isotropic).
        rng = torch.Generator(device=device).manual_seed(self.seed)
        init_idx = torch.randint(0, N, (B, K), generator=rng, device=device)
        means = torch.gather(
            x_b, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)
        ).contiguous()
        data_var_scalar = x_b.float().var(dim=1).mean(-1, keepdim=True)  # (B, 1)
        sqrt_var = data_var_scalar.clamp_min(self.reg_covar).sqrt().view(B, 1, 1, 1)
        eye = torch.eye(D, device=device, dtype=x_b.dtype).view(1, 1, D, D)
        L = (sqrt_var.to(x_b.dtype) * eye.expand(B, K, D, D)).contiguous()  # (B, K, D, D)
        log_w = torch.full((B, K), -math.log(K), dtype=torch.float32, device=device)
        weights = torch.full((B, K), 1.0 / K, dtype=torch.float32, device=device)
        x_b_f = x_b.float()

        lower_bound_history: list[float] = []
        n_iter = 0
        prev_lb = -math.inf
        lb = float("-inf")
        ids: Optional[torch.Tensor] = None

        for _ in range(self.niter):
            n_iter += 1
            lse = _cuda_mod.full_logsumexp(x_b, means, L, log_w)
            lb = float(lse.mean().item())
            lower_bound_history.append(lb)

            resp = _cuda_mod.full_resp(x_b, means, L, log_w, lse)
            counts = resp.sum(dim=1)
            sums = torch.bmm(resp.transpose(1, 2), x_b_f)
            outer_sums = torch.einsum("bnk,bnd,bne->bkde", resp, x_b_f, x_b_f)
            means, L, weights = _cuda_mod.full_finalize(
                sums, outer_sums, counts, means, L, N, self.reg_covar
            )
            log_w = torch.log(weights.clamp_min(1e-30))

            if abs(lb - prev_lb) < self.tol:
                break
            prev_lb = lb

        # GMMXX exposes covariances_ as the full (B, K, D, D) Σ_k.
        cov_b = L @ L.transpose(-1, -2)  # (B, K, D, D)

        if self.compute_labels_on_fit:
            ids = _cuda_mod.full_assign(x_b, means, L, log_w)
        labels_b = ids if self.compute_labels_on_fit else None
        info = {
            "lower_bound": lb,
            "lower_bound_history": lower_bound_history,
            "n_iter": n_iter,
            "init_source": "cuda_random",
            "triton_estep_enabled": False,
            "triton_streaming_update_enabled": False,
            "triton_fused_update_enabled": False,
            "triton_approx_topk_enabled": False,
            "triton_labels_enabled": False,
            "approximate_em_enabled": False,
            "approx_top_k": None,
            "large_n_streaming_enabled": False,
            "copy_stream_prefetch_enabled": False,
            "backend_breakdown": {"cuda": n_iter},
        }
        self._set_fit_result(
            labels_b=labels_b,
            means_b=means,
            variances_b=cov_b,  # (B, K, D, D)
            weights_b=weights,
            info=info,
            batch_size=batch_size,
        )

    def train(self, data: torch.Tensor):
        self._apply_matmul_precision()
        self._reset_fit_state()
        x_b, batch_size = self._normalize_input(data)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if self._is_large_cpu_stream_input(x_b):
            self._train_large_cpu_streaming(x_b, batch_size)
            return

        x_b = self._prepare_compute_tensor(x_b)

        if self.covariance_type == "spherical":
            # CUDA backend dispatch — when resolver picks "cuda", run the
            # standalone EM loop in _train_spherical_cuda and return early.
            # This path handles both exact EM and approximate top-k EM.
            from . import _dispatch
            shape_for_dispatch = (
                x_b.shape[0],
                x_b.shape[1],
                x_b.shape[2],
                self.k,
            )
            resolved = _dispatch.resolve_backend_with_env(
                requested=self.backend,
                covariance="spherical",
                shape=shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if resolved == "cuda":
                self._train_spherical_cuda(x_b, batch_size)
                self.last_backend_used_ = "cuda"
                self.cuda_estep_enabled_ = True
                return
            self.last_backend_used_ = resolved

            labels_b, means_b, variances_b, weights_b, info = batch_gmm_Spherical_torch_native(
                x_b,
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                verbose=self.verbose,
                init_params=self.init_params,
                reg_covar=self.reg_covar,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
                kmeans_init_iters=self.init_kmeans_iters,
                kmeans_init_tol=self.init_kmeans_tol,
                kmeans_use_triton=self.use_triton,
                gmm_use_triton_estep="auto" if self.use_triton else False,
                gmm_use_triton_streaming_update="auto" if self.use_triton else False,
                compute_labels=self.compute_labels_on_fit,
                approx_top_k=self.approx_top_k,
            )
        elif self.covariance_type == "diag":
            # CUDA backend dispatch — when resolver picks "cuda", run the
            # standalone EM loop in _train_diag_cuda and return early.
            # The CUDA path does not implement approximate top-k EM, so
            # when approx_top_k is requested we skip the CUDA dispatch and let
            # the existing Triton/torch path handle it.
            if self.approx_top_k is None:
                from . import _dispatch
                shape_for_dispatch = (
                    x_b.shape[0],
                    x_b.shape[1],
                    x_b.shape[2],
                    self.k,
                )
                resolved = _dispatch.resolve_backend_with_env(
                    requested=self.backend,
                    covariance="diag",
                    shape=shape_for_dispatch,
                    dtype=x_b.dtype,
                    legacy_no_triton=self._legacy_no_triton,
                )
                if resolved == "cuda":
                    self._train_diag_cuda(x_b, batch_size)
                    self.last_backend_used_ = "cuda"
                    self.cuda_estep_enabled_ = True
                    return
                self.last_backend_used_ = resolved

            labels_b, means_b, variances_b, weights_b, info = batch_gmm_Diagonal_torch_native(
                x_b,
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                verbose=self.verbose,
                init_params=self.init_params,
                reg_covar=self.reg_covar,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
                kmeans_init_iters=self.init_kmeans_iters,
                kmeans_init_tol=self.init_kmeans_tol,
                kmeans_use_triton=self.use_triton,
                gmm_use_triton="auto" if self.use_triton else False,
                compute_labels=self.compute_labels_on_fit,
                approx_top_k=self.approx_top_k,
            )
        elif self.covariance_type == "tied":
            # CUDA backend dispatch — when resolver picks "cuda", run the
            # standalone EM loop in _train_tied_cuda and return early.
            # The CUDA path does not implement approximate top-k EM, so
            # when approx_top_k is requested we skip the CUDA dispatch and let
            # the existing Triton/torch path handle it.
            if self.approx_top_k is None:
                from . import _dispatch
                shape_for_dispatch = (
                    x_b.shape[0],
                    x_b.shape[1],
                    x_b.shape[2],
                    self.k,
                )
                resolved = _dispatch.resolve_backend_with_env(
                    requested=self.backend,
                    covariance="tied",
                    shape=shape_for_dispatch,
                    dtype=x_b.dtype,
                    legacy_no_triton=self._legacy_no_triton,
                )
                if resolved == "cuda":
                    self._train_tied_cuda(x_b, batch_size)
                    self.last_backend_used_ = "cuda"
                    self.cuda_estep_enabled_ = True
                    return
                self.last_backend_used_ = resolved
            labels_b, means_b, variances_b, weights_b, info = batch_gmm_Tied_torch_native(
                x_b,
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                verbose=self.verbose,
                init_params=self.init_params,
                reg_covar=self.reg_covar,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
                kmeans_init_iters=self.init_kmeans_iters,
                kmeans_init_tol=self.init_kmeans_tol,
                kmeans_use_triton=self.use_triton,
                gmm_use_triton="auto" if self.use_triton else False,
                compute_labels=self.compute_labels_on_fit,
                approx_top_k=self.approx_top_k,
            )
        else:
            # covariance_type == "full"
            # CUDA backend dispatch — when resolver picks "cuda", run the
            # standalone EM loop in _train_full_cuda and return early.
            # The CUDA path does not implement approximate top-k EM, so
            # when approx_top_k is requested we skip the CUDA dispatch and let
            # the existing Triton/torch path handle it.
            if self.approx_top_k is None:
                from . import _dispatch
                shape_for_dispatch = (
                    x_b.shape[0],
                    x_b.shape[1],
                    x_b.shape[2],
                    self.k,
                )
                resolved = _dispatch.resolve_backend_with_env(
                    requested=self.backend,
                    covariance="full",
                    shape=shape_for_dispatch,
                    dtype=x_b.dtype,
                    legacy_no_triton=self._legacy_no_triton,
                )
                if resolved == "cuda":
                    self._train_full_cuda(x_b, batch_size)
                    self.last_backend_used_ = "cuda"
                    self.cuda_estep_enabled_ = True
                    return
                self.last_backend_used_ = resolved
            labels_b, means_b, variances_b, weights_b, info = batch_gmm_Full_torch_native(
                x_b,
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                verbose=self.verbose,
                init_params=self.init_params,
                reg_covar=self.reg_covar,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
                kmeans_init_iters=self.init_kmeans_iters,
                kmeans_init_tol=self.init_kmeans_tol,
                kmeans_use_triton=self.use_triton,
                gmm_use_triton="auto" if self.use_triton else False,
                compute_labels=self.compute_labels_on_fit,
                approx_top_k=self.approx_top_k,
            )

        self._set_fit_result(
            labels_b=labels_b,
            means_b=means_b,
            variances_b=variances_b,
            weights_b=weights_b,
            info=info,
            batch_size=batch_size,
        )

    def fit(self, data: torch.Tensor, y: Any = None):
        del y
        self.train(data)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "d": self.d,
            "n_components": self.k,
            "max_iter": self.niter,
            "tol": self.tol,
            "random_state": self.seed,
            "chunk_size_data": self.chunk_size_data,
            "chunk_size_centroids": self.chunk_size_centroids,
            "chunk_size_data_cpu": self.chunk_size_data_cpu,
            "verbose": self.verbose,
            "dtype": self.dtype,
            "device": self.device,
            "init_params": self.init_params,
            "reg_covar": self.reg_covar,
            "init_kmeans_iters": self.init_kmeans_iters,
            "init_kmeans_tol": self.init_kmeans_tol,
            "covariance_type": self.covariance_type,
            "matmul_precision": self.matmul_precision,
            "compute_labels_on_fit": self.compute_labels_on_fit,
            "approx_top_k": self.approx_top_k,
            "backend": self.backend,
            # Note: use_triton is intentionally NOT returned. It's the deprecated
            # alias; clone()-style round-trip uses 'backend' as the canonical key.
            # _legacy_no_triton is also intentionally not exposed — it's an
            # implementation detail of the deprecation shim.
        }

    def set_params(self, **params: Any):
        if not params:
            return self
        valid = set(self.get_params().keys()) | {
            "k",
            "niter",
            "seed",
            "n_features",
            "n_components",
            "max_iter",
            "random_state",
            "use_triton",   # deprecated alias
            "backend",
        }
        for raw_name, value in params.items():
            if raw_name not in valid:
                raise ValueError(f"Invalid parameter {raw_name!r} for GMMXX")
            name = _SKLEARN_PARAM_ALIASES.get(raw_name, raw_name)
            if name == "d":
                value = _optional_positive_int(value, "d")
            elif name in {"k", "niter", "chunk_size_data", "chunk_size_centroids", "chunk_size_data_cpu", "init_kmeans_iters"}:
                value = _positive_int(value, name)
            elif name in {"tol", "reg_covar", "init_kmeans_tol"}:
                value = float(value)
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative")
            elif name == "seed":
                value = int(value)
            elif name == "use_triton":
                # Deprecated; route through the same shim as __init__.
                if value is not None:
                    if self.backend != "auto":
                        raise ValueError(
                            "Cannot set use_triton when backend is already explicit "
                            f"(self.backend={self.backend!r}). Drop use_triton; "
                            "backend= is the canonical kwarg."
                        )
                    import warnings as _warnings
                    _warnings.warn(
                        "use_triton is deprecated; use backend='auto'|'cuda'|'triton'|'torch' instead. "
                        "use_triton=True maps to backend='auto'; use_triton=False maps to backend='auto' "
                        "with Triton filtered from the dispatch chain.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    self.backend = "auto"
                    self._legacy_no_triton = not bool(value)
                continue  # don't fall through to setattr
            elif name == "backend":
                if value not in _VALID_BACKENDS:
                    raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {value!r}")
                self.backend = value
                continue  # don't fall through to setattr
            elif name in {"verbose", "compute_labels_on_fit"}:
                value = bool(value)
            elif name == "covariance_type":
                value = _normalize_covariance_type(value)
            elif name == "matmul_precision" and value is not None and value not in {"highest", "high", "medium"}:
                raise ValueError("matmul_precision must be None, 'highest', 'high', or 'medium'")
            elif name == "approx_top_k":
                value = _optional_positive_int(value, "approx_top_k")
            elif name == "device":
                value = torch.device(value)
                self._raw_device = value
            setattr(self, name, value)
        self._reset_fit_state()
        return self

    def _require_trained(self) -> None:
        if self.means_b is None or self.variances_b is None or self.weights_b is None:
            raise RuntimeError("Model not trained. Call train() or fit() first.")

    def _prepare_predict_input(self, data: torch.Tensor) -> tuple[torch.Tensor, Optional[int]]:
        self._apply_matmul_precision()
        self._require_trained()
        x_b, batch_size = self._normalize_input(data)
        if batch_size != self._batch_size:
            raise ValueError(
                f"Model was trained with batch size B={self._batch_size}, "
                f"but received B={batch_size}."
            )
        if self._is_large_cpu_stream_input(x_b):
            return x_b, batch_size
        return self._prepare_compute_tensor(x_b), batch_size

    def _large_cpu_predict_kwargs(self, x_b: torch.Tensor) -> dict:
        return {
            "covariance_type": self.covariance_type,
            "device": self.device,
            "dtype": self._compute_dtype_for_input(x_b),
            "chunk_size_N": self.chunk_size_data,
            "chunk_size_K": self.chunk_size_centroids,
            "use_triton": self.use_triton,
            "backend": self.backend,
            "legacy_no_triton": self._legacy_no_triton,
        }

    def _use_triton_spherical_inference(self, x_b: torch.Tensor) -> bool:
        if not (self.use_triton and _HAS_TRITON_ASSIGN and x_b.is_cuda):
            return False
        return triton_spherical_supported(x_b.shape[-1], self.means_b.shape[1])

    def _use_triton_spherical_labels(self, x_b: torch.Tensor) -> bool:
        if not (self.use_triton and _HAS_TRITON_ASSIGN and x_b.is_cuda):
            return False
        return triton_spherical_supported(x_b.shape[-1], self.means_b.shape[1])

    def _use_triton_full_inference(self, x_b: torch.Tensor) -> bool:
        if not (self.use_triton and _HAS_TRITON_FULL_INFERENCE and x_b.is_cuda):
            return False
        return x_b.shape[-1] <= 16 and self.means_b.shape[1] <= 128

    def _use_triton_diag_inference(self, x_b: torch.Tensor) -> bool:
        if not (self.use_triton and _HAS_TRITON_DIAG_INFERENCE and x_b.is_cuda):
            return False
        d = x_b.shape[-1]
        return 16 <= d <= 64 and self.means_b.shape[1] <= 512

    def _use_triton_tied_inference(self, x_b: torch.Tensor) -> bool:
        if not (self.use_triton and _HAS_TRITON_ASSIGN and x_b.is_cuda):
            return False
        d = x_b.shape[-1]
        return d >= 16 and triton_spherical_supported(d, self.means_b.shape[1])

    def _diag_inference_terms(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._diag_inference_cache is not None:
            return self._diag_inference_cache
        variances = self.variances_b.to(torch.float32).clamp_min(1e-30)
        precision = variances.reciprocal()
        logdet = torch.log(variances).sum(dim=-1)
        means_f = self.means_b.to(torch.float32)
        weighted_means = means_f * precision
        mean_precision_mean = (means_f * weighted_means).sum(dim=-1)
        log_weights = torch.log(self.weights_b.to(torch.float32))
        self._diag_inference_cache = (
            precision,
            weighted_means,
            mean_precision_mean,
            logdet,
            log_weights,
        )
        return self._diag_inference_cache

    def _tied_static_inference_terms(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._tied_inference_cache is not None:
            return self._tied_inference_cache
        precision, logdet = _precision_and_logdet(self.variances_b)
        chol_precision = torch.linalg.cholesky(precision)
        means_projected = torch.bmm(self.means_b.to(torch.float32), chol_precision)
        means_projected_sq = means_projected.square().sum(dim=-1)
        unit_variances = torch.ones(
            self.weights_b.shape,
            device=self.weights_b.device,
            dtype=torch.float32,
        )
        log_weights = torch.log(self.weights_b.to(torch.float32))
        self._tied_inference_cache = (
            chol_precision,
            means_projected,
            means_projected_sq,
            unit_variances,
            logdet,
            log_weights,
        )
        return self._tied_inference_cache

    def _tied_projected_inference_terms(
        self,
        x_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            chol_precision,
            means_projected,
            means_projected_sq,
            unit_variances,
            logdet,
            log_weights,
        ) = self._tied_static_inference_terms()
        x_projected = torch.bmm(x_b.to(torch.float32), chol_precision)
        x_projected_sq = x_projected.square().sum(dim=-1)
        return (
            x_projected,
            means_projected,
            x_projected_sq,
            means_projected_sq,
            unit_variances,
            logdet,
            log_weights,
        )

    def _full_inference_terms(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._full_inference_cache is not None:
            return self._full_inference_cache
        precision, logdet = _precision_and_logdet(self.variances_b)
        means_f = self.means_b.to(torch.float32)
        precision_means = torch.einsum("bkde,bke->bkd", precision, means_f)
        mean_precision_mean = (means_f * precision_means).sum(dim=-1)
        log_weights = torch.log(self.weights_b.to(torch.float32))
        self._full_inference_cache = (
            precision,
            precision_means,
            mean_precision_mean,
            logdet,
            log_weights,
        )
        return self._full_inference_cache

    def predict(self, data: torch.Tensor) -> torch.LongTensor:
        x_b, batch_size = self._prepare_predict_input(data)
        if self._is_large_cpu_stream_input(x_b):
            labels_b, backend_used = large_n_predict_cpu(
                x_b,
                self.means_b,
                self.variances_b,
                self.weights_b,
                return_backend_used=True,
                **self._large_cpu_predict_kwargs(x_b),
            )
            self.last_backend_used_ = backend_used
            return labels_b.squeeze(0) if batch_size is None else labels_b
        if self.covariance_type == "diag":
            # CUDA branch (Plan 6 Task 8)
            from . import _dispatch as _dispatch_mod
            _shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
            _resolved = _dispatch_mod.resolve_backend_with_env(
                requested=self.backend,
                covariance="diag",
                shape=_shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if _resolved == "cuda":
                log_w = torch.log(self.weights_b.clamp_min(1e-30))
                if self.dtype is not None and x_b.dtype != self.dtype:
                    x_b_compute = x_b.to(self.dtype)
                    means_b = self.means_b.to(self.dtype)
                else:
                    x_b_compute = x_b
                    means_b = self.means_b
                ids_b = _dispatch_mod.dispatch_kernel(
                    "diag_assign", "cuda",
                    x_b_compute, means_b, self.variances_b, log_w,
                )
                self.last_backend_used_ = "cuda"
                return self._squeeze_if_unbatched(ids_b.to(torch.long))
            self.last_backend_used_ = _resolved

            if self._use_triton_diag_inference(x_b):
                try:
                    precision, weighted_means, mean_precision_mean, logdet, log_weights = self._diag_inference_terms()
                    labels_b = diag_assign_triton(
                        x_b,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                    ).to(torch.long)
                except Exception as exc:
                    self._record_fallback("diag Triton predict failed", exc)
                    labels_b = diagonal_assign_torch_native_chunked(
                        x_b,
                        self.means_b,
                        self.variances_b,
                        self.weights_b,
                        chunk_size_N=self.chunk_size_data,
                        chunk_size_K=self.chunk_size_centroids,
                    )
            else:
                labels_b = diagonal_assign_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
            return labels_b.squeeze(0) if batch_size is None else labels_b
        if self.covariance_type == "tied":
            # CUDA branch (Plan 7 Task 5).
            from . import _dispatch
            shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
            resolved = _dispatch.resolve_backend_with_env(
                requested=self.backend,
                covariance="tied",
                shape=shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if resolved == "cuda":
                from . import _cuda as _cuda_mod
                if self.dtype is not None and x_b.dtype != self.dtype:
                    x_b_compute = x_b.to(self.dtype)
                    means_b = self.means_b.to(self.dtype)
                else:
                    x_b_compute = x_b
                    means_b = self.means_b
                log_w = torch.log(self.weights_b.clamp_min(1e-30))
                L = torch.linalg.cholesky(self.covariances_b)
                ids_b = _cuda_mod.tied_assign(x_b_compute, means_b, L, log_w)
                self.last_backend_used_ = "cuda"
                return self._squeeze_if_unbatched(ids_b).long()
            self.last_backend_used_ = resolved
            labels_b = tied_assign_torch_native_chunked(
                x_b,
                self.means_b,
                self.variances_b,
                self.weights_b,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
            )
            return labels_b.squeeze(0) if batch_size is None else labels_b
        if self.covariance_type == "full":
            # CUDA branch (Plan 8 Task 5).
            from . import _dispatch
            shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
            resolved = _dispatch.resolve_backend_with_env(
                requested=self.backend,
                covariance="full",
                shape=shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if resolved == "cuda":
                from . import _cuda as _cuda_mod
                if self.dtype is not None and x_b.dtype != self.dtype:
                    x_b_compute = x_b.to(self.dtype)
                    means_b = self.means_b.to(self.dtype)
                else:
                    x_b_compute = x_b
                    means_b = self.means_b
                log_w = torch.log(self.weights_b.clamp_min(1e-30))
                L, _ = torch.linalg.cholesky_ex(self.covariances_b)
                ids_b = _cuda_mod.full_assign(x_b_compute, means_b, L, log_w)
                self.last_backend_used_ = "cuda"
                return self._squeeze_if_unbatched(ids_b).long()
            self.last_backend_used_ = resolved
            if self._use_triton_full_inference(x_b):
                try:
                    precision, precision_means, mean_precision_mean, logdet, log_weights = self._full_inference_terms()
                    labels_b = full_assign_triton(
                        x_b,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                    ).to(torch.long)
                except Exception as exc:
                    self._record_fallback("full Triton predict failed", exc)
                    labels_b = full_assign_torch_native_chunked(
                        x_b,
                        self.means_b,
                        self.variances_b,
                        self.weights_b,
                        chunk_size_N=self.chunk_size_data,
                        chunk_size_K=self.chunk_size_centroids,
                    )
            else:
                labels_b = full_assign_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
            return labels_b.squeeze(0) if batch_size is None else labels_b

        # CUDA inference branch for spherical covariance.
        from . import _dispatch as _dispatch_mod
        _shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
        _resolved = _dispatch_mod.resolve_backend_with_env(
            requested=self.backend,
            covariance="spherical",
            shape=_shape_for_dispatch,
            dtype=x_b.dtype,
            legacy_no_triton=self._legacy_no_triton,
        )
        if _resolved == "cuda":
            log_w = torch.log(self.weights_b.clamp_min(1e-30))
            if self.dtype is not None and x_b.dtype != self.dtype:
                x_b_compute = x_b.to(self.dtype)
                means_b_compute = self.means_b.to(self.dtype)
            else:
                x_b_compute = x_b
                means_b_compute = self.means_b
            ids_b = _dispatch_mod.dispatch_kernel(
                "spherical_assign", "cuda",
                x_b_compute, means_b_compute, self.variances_b, log_w,
            )
            self.last_backend_used_ = "cuda"
            return self._squeeze_if_unbatched(ids_b.to(torch.long))

        if self._use_triton_spherical_labels(x_b):
            try:
                labels_b = spherical_assign_triton(
                    x_b,
                    self.means_b,
                    self.variances_b.to(torch.float32),
                    self.weights_b.to(torch.float32),
                )
            except Exception as exc:
                self._record_fallback("spherical Triton predict failed", exc)
                labels_b = spherical_assign_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
        else:
            labels_b = spherical_assign_torch_native_chunked(
                x_b,
                self.means_b,
                self.variances_b,
                self.weights_b,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
            )
        return labels_b.squeeze(0) if batch_size is None else labels_b

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        x_b, batch_size = self._prepare_predict_input(data)
        if self._is_large_cpu_stream_input(x_b):
            probs_b, backend_used = large_n_predict_proba_cpu(
                x_b,
                self.means_b,
                self.variances_b,
                self.weights_b,
                return_backend_used=True,
                **self._large_cpu_predict_kwargs(x_b),
            )
            self.last_backend_used_ = backend_used
            return probs_b.squeeze(0) if batch_size is None else probs_b
        if self.covariance_type == "diag":
            # CUDA branch (Plan 6 Task 8)
            from . import _dispatch as _dispatch_mod
            _shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
            _resolved = _dispatch_mod.resolve_backend_with_env(
                requested=self.backend,
                covariance="diag",
                shape=_shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if _resolved == "cuda":
                log_w = torch.log(self.weights_b.clamp_min(1e-30))
                if self.dtype is not None and x_b.dtype != self.dtype:
                    x_b_compute = x_b.to(self.dtype)
                    means_b = self.means_b.to(self.dtype)
                else:
                    x_b_compute = x_b
                    means_b = self.means_b
                log_norm_b = _dispatch_mod.dispatch_kernel(
                    "diag_logsumexp", "cuda",
                    x_b_compute, means_b, self.variances_b, log_w,
                )
                probs_b = _dispatch_mod.dispatch_kernel(
                    "diag_resp", "cuda",
                    x_b_compute, means_b, self.variances_b, log_w, log_norm_b,
                )
                self.last_backend_used_ = "cuda"
                return self._squeeze_if_unbatched(probs_b)
            self.last_backend_used_ = _resolved

            if self._use_triton_diag_inference(x_b):
                try:
                    precision, weighted_means, mean_precision_mean, logdet, log_weights = self._diag_inference_terms()
                    log_norm_b = diag_logsumexp_triton(
                        x_b,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                    )
                    probs_b = diag_resp_triton(
                        x_b,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        log_norm_b,
                    )
                except Exception as exc:
                    self._record_fallback("diag Triton predict_proba failed", exc)
                    probs_b = diagonal_predict_proba_torch_native_chunked(
                        x_b,
                        self.means_b,
                        self.variances_b,
                        self.weights_b,
                        chunk_size_N=self.chunk_size_data,
                        chunk_size_K=self.chunk_size_centroids,
                    )
            else:
                probs_b = diagonal_predict_proba_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
            return probs_b.squeeze(0) if batch_size is None else probs_b
        if self.covariance_type == "tied":
            # CUDA branch (Plan 7 Task 5).
            from . import _dispatch
            shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
            resolved = _dispatch.resolve_backend_with_env(
                requested=self.backend,
                covariance="tied",
                shape=shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if resolved == "cuda":
                from . import _cuda as _cuda_mod
                if self.dtype is not None and x_b.dtype != self.dtype:
                    x_b_compute = x_b.to(self.dtype)
                    means_b = self.means_b.to(self.dtype)
                else:
                    x_b_compute = x_b
                    means_b = self.means_b
                log_w = torch.log(self.weights_b.clamp_min(1e-30))
                L = torch.linalg.cholesky(self.covariances_b)
                log_norm = _cuda_mod.tied_logsumexp(x_b_compute, means_b, L, log_w)
                proba_b = _cuda_mod.tied_resp(x_b_compute, means_b, L, log_w, log_norm)
                self.last_backend_used_ = "cuda"
                return self._squeeze_if_unbatched(proba_b)
            self.last_backend_used_ = resolved
            if self._use_triton_tied_inference(x_b):
                try:
                    (
                        x_projected,
                        means_projected,
                        x_projected_sq,
                        means_projected_sq,
                        unit_variances,
                        _,
                        log_weights,
                    ) = self._tied_projected_inference_terms(x_b)
                    log_norm_b = spherical_logsumexp_triton(
                        x_projected,
                        means_projected,
                        unit_variances,
                        self.weights_b.to(torch.float32),
                        x_sq=x_projected_sq,
                        means_sq=means_projected_sq,
                        log_weights=log_weights,
                        unit_variance=True,
                    )
                    probs_b = spherical_resp_triton(
                        x_projected,
                        means_projected,
                        unit_variances,
                        self.weights_b.to(torch.float32),
                        log_norm_b,
                        x_sq=x_projected_sq,
                        means_sq=means_projected_sq,
                        log_weights=log_weights,
                    )
                except Exception as exc:
                    self._record_fallback("tied projected Triton predict_proba failed", exc)
                    probs_b = tied_predict_proba_torch_native_chunked(
                        x_b,
                        self.means_b,
                        self.variances_b,
                        self.weights_b,
                        chunk_size_N=self.chunk_size_data,
                        chunk_size_K=self.chunk_size_centroids,
                    )
            else:
                probs_b = tied_predict_proba_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
            return probs_b.squeeze(0) if batch_size is None else probs_b
        if self.covariance_type == "full":
            # CUDA branch (Plan 8 Task 5).
            from . import _dispatch
            shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
            resolved = _dispatch.resolve_backend_with_env(
                requested=self.backend,
                covariance="full",
                shape=shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if resolved == "cuda":
                from . import _cuda as _cuda_mod
                if self.dtype is not None and x_b.dtype != self.dtype:
                    x_b_compute = x_b.to(self.dtype)
                    means_b = self.means_b.to(self.dtype)
                else:
                    x_b_compute = x_b
                    means_b = self.means_b
                log_w = torch.log(self.weights_b.clamp_min(1e-30))
                L, _ = torch.linalg.cholesky_ex(self.covariances_b)
                log_norm = _cuda_mod.full_logsumexp(x_b_compute, means_b, L, log_w)
                proba_b = _cuda_mod.full_resp(x_b_compute, means_b, L, log_w, log_norm)
                self.last_backend_used_ = "cuda"
                return self._squeeze_if_unbatched(proba_b)
            self.last_backend_used_ = resolved
            if self._use_triton_full_inference(x_b):
                try:
                    precision, precision_means, mean_precision_mean, logdet, log_weights = self._full_inference_terms()
                    log_norm_b = full_logsumexp_triton(
                        x_b,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                    )
                    probs_b = full_resp_triton(
                        x_b,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                        log_norm_b,
                    )
                except Exception as exc:
                    self._record_fallback("full Triton predict_proba failed", exc)
                    probs_b = full_predict_proba_torch_native_chunked(
                        x_b,
                        self.means_b,
                        self.variances_b,
                        self.weights_b,
                        chunk_size_N=self.chunk_size_data,
                        chunk_size_K=self.chunk_size_centroids,
                    )
            else:
                probs_b = full_predict_proba_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
            return probs_b.squeeze(0) if batch_size is None else probs_b

        # CUDA inference branch for spherical covariance.
        from . import _dispatch as _dispatch_mod
        _shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
        _resolved = _dispatch_mod.resolve_backend_with_env(
            requested=self.backend,
            covariance="spherical",
            shape=_shape_for_dispatch,
            dtype=x_b.dtype,
            legacy_no_triton=self._legacy_no_triton,
        )
        if _resolved == "cuda":
            log_w = torch.log(self.weights_b.clamp_min(1e-30))
            if self.dtype is not None and x_b.dtype != self.dtype:
                x_b_compute = x_b.to(self.dtype)
                means_b_compute = self.means_b.to(self.dtype)
            else:
                x_b_compute = x_b
                means_b_compute = self.means_b
            log_norm_b = _dispatch_mod.dispatch_kernel(
                "spherical_logsumexp", "cuda",
                x_b_compute, means_b_compute, self.variances_b, log_w,
            )
            probs_b = _dispatch_mod.dispatch_kernel(
                "spherical_resp", "cuda",
                x_b_compute, means_b_compute, self.variances_b, log_w, log_norm_b,
            )
            self.last_backend_used_ = "cuda"
            return self._squeeze_if_unbatched(probs_b)

        if self._use_triton_spherical_inference(x_b):
            try:
                log_norm_b = spherical_logsumexp_triton(
                    x_b,
                    self.means_b,
                    self.variances_b.to(torch.float32),
                    self.weights_b.to(torch.float32),
                )
                probs_b = spherical_resp_triton(
                    x_b,
                    self.means_b,
                    self.variances_b.to(torch.float32),
                    self.weights_b.to(torch.float32),
                    log_norm_b,
                )
            except Exception as exc:
                self._record_fallback("spherical Triton predict_proba failed", exc)
                probs_b = spherical_predict_proba_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
        else:
            probs_b = spherical_predict_proba_torch_native_chunked(
                x_b,
                self.means_b,
                self.variances_b,
                self.weights_b,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
            )
        return probs_b.squeeze(0) if batch_size is None else probs_b

    def score_samples(self, data: torch.Tensor) -> torch.Tensor:
        x_b, batch_size = self._prepare_predict_input(data)
        if self._is_large_cpu_stream_input(x_b):
            scores_b, backend_used = large_n_score_samples_cpu(
                x_b,
                self.means_b,
                self.variances_b,
                self.weights_b,
                return_backend_used=True,
                **self._large_cpu_predict_kwargs(x_b),
            )
            self.last_backend_used_ = backend_used
            return scores_b.squeeze(0) if batch_size is None else scores_b
        if self.covariance_type == "diag":
            # CUDA branch (Plan 6 Task 8)
            from . import _dispatch as _dispatch_mod
            _shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
            _resolved = _dispatch_mod.resolve_backend_with_env(
                requested=self.backend,
                covariance="diag",
                shape=_shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if _resolved == "cuda":
                log_w = torch.log(self.weights_b.clamp_min(1e-30))
                if self.dtype is not None and x_b.dtype != self.dtype:
                    x_b_compute = x_b.to(self.dtype)
                    means_b = self.means_b.to(self.dtype)
                else:
                    x_b_compute = x_b
                    means_b = self.means_b
                ll_b = _dispatch_mod.dispatch_kernel(
                    "diag_logsumexp", "cuda",
                    x_b_compute, means_b, self.variances_b, log_w,
                )
                self.last_backend_used_ = "cuda"
                return self._squeeze_if_unbatched(ll_b)
            self.last_backend_used_ = _resolved

            if self._use_triton_diag_inference(x_b):
                try:
                    precision, weighted_means, mean_precision_mean, logdet, log_weights = self._diag_inference_terms()
                    scores_b = diag_logsumexp_triton(
                        x_b,
                        precision,
                        weighted_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                    )
                except Exception as exc:
                    self._record_fallback("diag Triton score_samples failed", exc)
                    scores_b = diagonal_score_samples_torch_native_chunked(
                        x_b,
                        self.means_b,
                        self.variances_b,
                        self.weights_b,
                        chunk_size_N=self.chunk_size_data,
                        chunk_size_K=self.chunk_size_centroids,
                    )
            else:
                scores_b = diagonal_score_samples_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
            return scores_b.squeeze(0) if batch_size is None else scores_b
        if self.covariance_type == "tied":
            # CUDA branch (Plan 7 Task 5).
            from . import _dispatch
            shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
            resolved = _dispatch.resolve_backend_with_env(
                requested=self.backend,
                covariance="tied",
                shape=shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if resolved == "cuda":
                from . import _cuda as _cuda_mod
                if self.dtype is not None and x_b.dtype != self.dtype:
                    x_b_compute = x_b.to(self.dtype)
                    means_b = self.means_b.to(self.dtype)
                else:
                    x_b_compute = x_b
                    means_b = self.means_b
                log_w = torch.log(self.weights_b.clamp_min(1e-30))
                L = torch.linalg.cholesky(self.covariances_b)
                ll_b = _cuda_mod.tied_logsumexp(x_b_compute, means_b, L, log_w)
                self.last_backend_used_ = "cuda"
                return self._squeeze_if_unbatched(ll_b)
            self.last_backend_used_ = resolved
            if self._use_triton_tied_inference(x_b):
                try:
                    (
                        x_projected,
                        means_projected,
                        x_projected_sq,
                        means_projected_sq,
                        unit_variances,
                        logdet,
                        log_weights,
                    ) = self._tied_projected_inference_terms(x_b)
                    scores_b = spherical_logsumexp_triton(
                        x_projected,
                        means_projected,
                        unit_variances,
                        self.weights_b.to(torch.float32),
                        x_sq=x_projected_sq,
                        means_sq=means_projected_sq,
                        log_weights=log_weights,
                        unit_variance=True,
                    )
                    scores_b = scores_b - 0.5 * logdet.unsqueeze(-1)
                except Exception as exc:
                    self._record_fallback("tied projected Triton score_samples failed", exc)
                    scores_b = tied_score_samples_torch_native_chunked(
                        x_b,
                        self.means_b,
                        self.variances_b,
                        self.weights_b,
                        chunk_size_N=self.chunk_size_data,
                        chunk_size_K=self.chunk_size_centroids,
                    )
            else:
                scores_b = tied_score_samples_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
            return scores_b.squeeze(0) if batch_size is None else scores_b
        if self.covariance_type == "full":
            # CUDA branch (Plan 8 Task 5).
            from . import _dispatch
            shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
            resolved = _dispatch.resolve_backend_with_env(
                requested=self.backend,
                covariance="full",
                shape=shape_for_dispatch,
                dtype=x_b.dtype,
                legacy_no_triton=self._legacy_no_triton,
            )
            if resolved == "cuda":
                from . import _cuda as _cuda_mod
                if self.dtype is not None and x_b.dtype != self.dtype:
                    x_b_compute = x_b.to(self.dtype)
                    means_b = self.means_b.to(self.dtype)
                else:
                    x_b_compute = x_b
                    means_b = self.means_b
                log_w = torch.log(self.weights_b.clamp_min(1e-30))
                L, _ = torch.linalg.cholesky_ex(self.covariances_b)
                ll_b = _cuda_mod.full_logsumexp(x_b_compute, means_b, L, log_w)
                self.last_backend_used_ = "cuda"
                return self._squeeze_if_unbatched(ll_b)
            self.last_backend_used_ = resolved
            if self._use_triton_full_inference(x_b):
                try:
                    precision, precision_means, mean_precision_mean, logdet, log_weights = self._full_inference_terms()
                    scores_b = full_logsumexp_triton(
                        x_b,
                        precision,
                        precision_means,
                        mean_precision_mean,
                        logdet,
                        log_weights,
                    )
                except Exception as exc:
                    self._record_fallback("full Triton score_samples failed", exc)
                    scores_b = full_score_samples_torch_native_chunked(
                        x_b,
                        self.means_b,
                        self.variances_b,
                        self.weights_b,
                        chunk_size_N=self.chunk_size_data,
                        chunk_size_K=self.chunk_size_centroids,
                    )
            else:
                scores_b = full_score_samples_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
            return scores_b.squeeze(0) if batch_size is None else scores_b

        # CUDA inference branch for spherical covariance.
        from . import _dispatch as _dispatch_mod
        _shape_for_dispatch = (x_b.shape[0], x_b.shape[1], x_b.shape[2], self.k)
        _resolved = _dispatch_mod.resolve_backend_with_env(
            requested=self.backend,
            covariance="spherical",
            shape=_shape_for_dispatch,
            dtype=x_b.dtype,
            legacy_no_triton=self._legacy_no_triton,
        )
        if _resolved == "cuda":
            log_w = torch.log(self.weights_b.clamp_min(1e-30))
            if self.dtype is not None and x_b.dtype != self.dtype:
                x_b_compute = x_b.to(self.dtype)
                means_b_compute = self.means_b.to(self.dtype)
            else:
                x_b_compute = x_b
                means_b_compute = self.means_b
            ll_b = _dispatch_mod.dispatch_kernel(
                "spherical_logsumexp", "cuda",
                x_b_compute, means_b_compute, self.variances_b, log_w,
            )
            self.last_backend_used_ = "cuda"
            return self._squeeze_if_unbatched(ll_b)

        if self._use_triton_spherical_inference(x_b):
            try:
                scores_b = spherical_logsumexp_triton(
                    x_b,
                    self.means_b,
                    self.variances_b.to(torch.float32),
                    self.weights_b.to(torch.float32),
                )
            except Exception as exc:
                self._record_fallback("spherical Triton score_samples failed", exc)
                scores_b = spherical_score_samples_torch_native_chunked(
                    x_b,
                    self.means_b,
                    self.variances_b,
                    self.weights_b,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
        else:
            scores_b = spherical_score_samples_torch_native_chunked(
                x_b,
                self.means_b,
                self.variances_b,
                self.weights_b,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
            )
        return scores_b.squeeze(0) if batch_size is None else scores_b

    def score(self, data: torch.Tensor, y: Any = None) -> float:
        del y
        return float(self.score_samples(data).mean().item())

    def fit_predict(self, data: torch.Tensor, y: Any = None) -> torch.LongTensor:
        del y
        self.train(data)
        if self.cluster_ids_b is None:
            labels = self.predict(data)
            self.cluster_ids_b = labels.unsqueeze(0) if self._batch_size is None else labels
            self.labels_ = labels
            return labels
        return self.cluster_ids_b.squeeze(0) if self._batch_size is None else self.cluster_ids_b

    def _n_parameters(self, batch_size: int = 1) -> int:
        self._require_trained()
        d = int(self.means_b.shape[-1])
        k = int(self.means_b.shape[1])
        covariance_params = {
            "spherical": k,
            "diag": k * d,
            "tied": d * (d + 1) // 2,
            "full": k * d * (d + 1) // 2,
        }[self.covariance_type]
        return int(batch_size * (k * d + covariance_params + k - 1))

    def bic(self, data: torch.Tensor) -> float:
        scores = self.score_samples(data)
        n_samples = int(scores.numel())
        batch_size = int(scores.shape[0]) if scores.ndim == 2 else 1
        return float(-2.0 * scores.sum().item() + self._n_parameters(batch_size) * math.log(float(n_samples)))

    def aic(self, data: torch.Tensor) -> float:
        scores = self.score_samples(data)
        batch_size = int(scores.shape[0]) if scores.ndim == 2 else 1
        return float(-2.0 * scores.sum().item() + 2.0 * self._n_parameters(batch_size))
