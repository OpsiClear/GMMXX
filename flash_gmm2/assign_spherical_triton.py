import math
from typing import Optional
import torch
import triton
import triton.language as tl

# ===============================================================
# Copied and modified from flash-kmeans:
# Triton kernel for spherical-GMM component assignment.
# Inputs:
#   x            : (B, N, D)  float16 / float32
#   means        : (B, K, D)  same dtype as x
#   variances    : (B, K)     float32
#   weights      : (B, K)     float32
# Output:
#   cluster_ids  : (B, N)     int32   – max-logit component index per point
# ===============================================================


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# -----------------------------------------------------------------------------
# Auto-tuning setup – explore various tile sizes / warp counts
# -----------------------------------------------------------------------------

_TUNE_CONFIGS = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=num_stages, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [32, 64, 128]
    for wp in [4, 8]
    for num_stages in [1, 2, 4]
]


def _cfg_keep(conf):
    """Basic heuristic to prune unbalanced configs."""
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    # Avoid tiny tiles on many warps
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    return True

_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _dtype_bytes(dtype) -> int:
    """Element size in bytes for a torch / numpy-ish dtype.

    Falls back to 2 (fp16) when the dtype is unknown to keep prior behaviour
    (the heuristic was originally tuned with fp16 in mind).
    """
    if dtype is None:
        return 2
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    # Allow callers to pass a raw byte size.
    if isinstance(dtype, int):
        return dtype
    return 2


def _is_half_dtype(dtype) -> bool:
    """True for fp16/bf16 (the original tuning regime).

    For these dtypes we skip the SMEM-fitting fallback entirely so heuristic
    selection on already-validated GPUs (H100/H200/A100) is byte-for-byte
    identical to the previous behaviour.
    """
    if dtype is None:
        return True
    if isinstance(dtype, torch.dtype):
        return dtype in _HALF_DTYPES
    return False


def _smem_bytes(D: int, BN: int, BK: int, num_stages: int, dtype_bytes: int) -> int:
    """Approximate dynamic shared-memory usage of `_euclid_assign_kernel`.

    The kernel materialises:
    - one ``x_tile`` of shape (BN, D) outside the K loop, and
    - ``num_stages`` copies of ``c_tile`` of shape (D, BK) for the software
      pipelined K loop.

    Other buffers (x_sq, c_sq, masks, accumulators) are negligible compared
    to these and are ignored.
    """
    return D * dtype_bytes * (BN + num_stages * BK)


def _smem_limit(device) -> int:
    """Per-block dynamic shared-memory budget for ``device``.

    Triton uses opt-in dynamic shared memory; prefer that attribute when
    available, fall back to the static limit, and finally to a conservative
    48 KiB for very old PyTorch builds.
    """
    props = torch.cuda.get_device_properties(device)
    for attr in (
        "shared_memory_per_block_optin",
        "max_shared_memory_per_block_optin",
        "shared_memory_per_block",
        "max_shared_memory_per_block",
    ):
        v = getattr(props, attr, None)
        if v:
            return int(v)
    return 48 * 1024


def _fit_config_to_smem(
    cfg: dict,
    D: int,
    dtype_bytes: int,
    smem_limit: int,
) -> dict:
    """Return a config that fits ``smem_limit`` and is closest to ``cfg``.

    The original config is returned unchanged whenever it already fits. If
    not, we enumerate all power-of-two ``(BLOCK_N, BLOCK_K, num_stages)``
    that are no larger than the original and pick the one that maximises
    work-per-program tile (``BLOCK_N * BLOCK_K * num_stages``), breaking
    ties towards the original aspect ratio. This avoids the pitfall of a
    pure greedy halving (e.g. shrinking BLOCK_K all the way to 16 when only
    a single halving was needed).

    Raises ``RuntimeError`` if even ``(BN=16, BK=16, S=1)`` does not fit –
    this only happens for absurdly large D combined with fp32 on tiny-SMEM
    GPUs.
    """
    BN0 = int(cfg["BLOCK_N"])
    BK0 = int(cfg["BLOCK_K"])
    W0 = int(cfg["num_warps"])
    S0 = int(cfg["num_stages"])

    if _smem_bytes(D, BN0, BK0, S0, dtype_bytes) <= smem_limit:
        return {"BLOCK_N": BN0, "BLOCK_K": BK0, "num_warps": W0, "num_stages": S0}

    def _pow2_down_to_16(v):
        out = []
        x = v
        while x >= 16:
            out.append(x)
            x //= 2
        return out

    best = None
    best_key = None
    for BN in _pow2_down_to_16(BN0):
        for BK in _pow2_down_to_16(BK0):
            for S in range(S0, 0, -1):
                if _smem_bytes(D, BN, BK, S, dtype_bytes) > smem_limit:
                    continue
                # Prefer larger total tile work, then closer aspect ratio
                # to the original, then larger BLOCK_N (more parallelism
                # along N), then larger num_stages (better pipelining).
                aspect_penalty = abs(
                    (BN / max(BK, 1)) - (BN0 / max(BK0, 1))
                )
                key = (BN * BK * S, -aspect_penalty, BN, S)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (BN, BK, S)

    if best is None:
        raise RuntimeError(
            f"euclid_assign_triton: cannot fit kernel into shared memory "
            f"(D={D}, dtype_bytes={dtype_bytes}, smem_limit={smem_limit}). "
            f"Even BLOCK_N=16, BLOCK_K=16, num_stages=1 needs "
            f"{_smem_bytes(D, 16, 16, 1, dtype_bytes)} bytes."
        )

    BN, BK, S = best
    W = W0
    # Tiny tiles do not benefit from many warps and may even fail to compile
    # for some Triton versions; cap to 4.
    if BN * BK <= 32 * 32 and W > 4:
        W = 4

    return {"BLOCK_N": BN, "BLOCK_K": BK, "num_warps": W, "num_stages": S}


def _heuristic_euclid_config(
    N: int,
    K: int,
    D: int,
    *,
    device: Optional[torch.device] = None,
    dtype=None,
):
    """Architecture-aware heuristic config selection without autotune.

    Keep one unified heuristic entry and diverge inside by GPU family:
    - H200: existing hand-tuned heuristic
    - H100: heuristic derived from H100 grid tuning results
    - A100: heuristic derived from A100 grid tuning results
    - GB10: heuristic derived from GB10 grid tuning results
    - others: conservative fallback to reduce OOR risk

    For half-precision dtypes (fp16/bf16, the regime the per-arch tables
    were tuned in) the picked config is returned as-is so behaviour on
    H100/H200/A100 stays byte-for-byte identical. For wider dtypes (fp32
    and friends) we additionally pass the config through
    ``_fit_config_to_smem`` so the kernel does not OOR on small-SMEM GPUs.
    """
    if device is None:
        device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_properties(device).name.upper()

    if _is_half_dtype(dtype):
        # Original code path: trust the per-arch table without SMEM checks.
        def _finalize(cfg):
            return cfg
    else:
        dtype_bytes = _dtype_bytes(dtype)
        smem_limit = _smem_limit(device)

        def _finalize(cfg):
            return _fit_config_to_smem(cfg, D, dtype_bytes, smem_limit)

    if "H200" in gpu_name:
        # Keep the original H200 heuristic as-is.
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 1

        if D >= 512:
            block_n = 128
            block_k = 64
            num_warps = 8
            num_stages = 1
        elif D >= 256:
            block_n = 128
            block_k = 64
            num_warps = 4
            num_stages = 2
        else:
            # D <= 128
            if K >= 4096:
                block_k = 128
                if D >= 128:
                    num_warps = 8
                    num_stages = 2
                else:
                    num_warps = 4
                    num_stages = 4
            else:
                block_k = 64
                num_warps = 4
                num_stages = 1

        # D=64 with large K tends to prefer smaller BLOCK_N and deeper pipeline.
        if D <= 64 and K >= 4096:
            block_n = 64
            block_k = 128
            num_warps = 4
            num_stages = 4

        # Smaller N favors smaller BLOCK_N to reduce wasted work.
        if N < 65536:
            block_n = 64

        return _finalize({
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        })

    if "H100" in gpu_name:
        # H100 tuned heuristic (more conservative on D=64 mid-K vs H200).
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 1

        if D >= 512:
            block_n = 128
            block_k = 64
            num_warps = 8
            num_stages = 1
        elif D >= 256:
            block_n = 128
            block_k = 64
            if K <= 1024:
                num_warps = 8
                num_stages = 1
            elif K <= 16384:
                num_warps = 4
                num_stages = 1
            else:
                num_warps = 8
                num_stages = 1
        else:
            # D <= 128
            if D <= 64:
                if K <= 1024:
                    block_k = 64
                    num_warps = 4
                    num_stages = 2
                elif K <= 16384:
                    block_k = 64
                    num_warps = 4
                    num_stages = 2
                elif K <= 65536:
                    block_k = 128
                    num_warps = 4
                    num_stages = 4
                else:
                    block_k = 64
                    num_warps = 4
                    num_stages = 4
            else:
                # D == 128
                if K <= 1024:
                    block_k = 64
                    num_warps = 4
                    num_stages = 1
                elif K <= 65536:
                    block_k = 128
                    num_warps = 8
                    num_stages = 2
                else:
                    block_k = 64
                    num_warps = 4
                    num_stages = 4

        if N < 65536:
            block_n = 64

        return _finalize({
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        })

    if "A100" in gpu_name:
        # Robust default on A100 across tuned grid.
        block_n = 128
        block_k = 32
        num_warps = 4
        num_stages = 2

        if D == 128:
            # Small-N cases tend to prefer a larger K tile.
            if N <= 65536:
                block_k = 64
        elif D == 256:
            # D=256 benefits from deeper pipeline at larger K.
            if K >= 65536:
                block_k = 32
                num_stages = 4
            elif K >= 1024 and N <= 262144:
                block_k = 64
                num_stages = 4

        return _finalize({
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        })

    if "GB10" in gpu_name:
        # GB10 (Grace Blackwell, ~80 SMs, ~99 KiB SMEM/SM) tuned heuristic.
        # Derived from a grid sweep over N in {65536, 262144, 1048576},
        # K in {256..200000}, D in {64,128,256,512}, B in {1, 32}, fp16.
        # Geomean slowdown vs. per-shape optimum is ~1% across the sweep
        # (worst case ~9% on sub-millisecond ops where timing noise
        # dominates). The selected config is post-processed by
        # _fit_config_to_smem so fp32 / large D inputs are shrunk to fit
        # GB10's modest shared-memory budget.
        if D >= 512:
            # D=512 strongly prefers a small BLOCK_N with 8 warps, except for
            # very small K where 4 warps is enough to saturate.
            if K <= 256:
                return _finalize({"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1})
            return _finalize({"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 8, "num_stages": 1})

        if D >= 256:
            if K <= 256:
                return _finalize({"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1})
            # Deeper pipeline + wider K tile pays off for K>=1024 at D=256.
            return _finalize({"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2})

        if D >= 128:
            if K <= 256:
                # Small K: a more square tile wins (BN=64, BK=64).
                return _finalize({"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1})
            if K <= 1024:
                # Transition region: small N likes a square tile, large N
                # benefits from BN=128 with deeper pipeline.
                if N <= 65536:
                    return _finalize({"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1})
                return _finalize({"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2})
            if K <= 65536:
                return _finalize({"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1})
            # K > 65536 (e.g. 200k) prefers the wider K tile to amortize loads.
            return _finalize({"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1})

        # D <= 64: BN=128, BK=32, 4 warps is robust across the full grid.
        # Only the tiniest shapes (small N + small K) shift toward a square
        # BN=64/BK=64 tile; larger N keeps the wider BN=128 default.
        if K <= 256 and N <= 65536:
            return _finalize({"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1})
        return _finalize({"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1})

    # Conservative fallback for unknown architectures (prioritize avoiding OOR).
    return _finalize({
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "num_warps": 4,
        "num_stages": 1,
    })


LOG_2PI = math.log(2.0 * math.pi)


@triton.jit
def _spherical_assign_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    means_ptr,             # *f16 / *f32 [B, K, D]
    x_sq_ptr,              # *f32         [B, N]
    means_sq_ptr,          # *f32         [B, K]
    variances_ptr,         # *f32         [B, K]
    log_weights_ptr,       # *f32         [B, K]
    out_ptr,               # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_means_b: tl.constexpr,
    stride_means_k: tl.constexpr,
    stride_means_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_meanssq_b: tl.constexpr,
    stride_meanssq_k: tl.constexpr,
    stride_var_b: tl.constexpr,
    stride_var_k: tl.constexpr,
    stride_logw_b: tl.constexpr,
    stride_logw_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_offsets < N

    offs_d = tl.arange(0, D).to(tl.int64)
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_sq_tile = tl.load(
        x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    best_logit = tl.full((BLOCK_N,), -3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)
    d_const = tl.full((1,), D, tl.float32)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)
        k_mask = k_offsets < K

        means_ptrs = (
            means_ptr
            + pid_b * stride_means_b
            + k_offsets[None, :] * stride_means_k
            + offs_d[:, None] * stride_means_d
        )
        means_tile = tl.load(means_ptrs, mask=k_mask[None, :], other=0.0)
        means_sq = tl.load(
            means_sq_ptr + pid_b * stride_meanssq_b + k_offsets * stride_meanssq_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        variances = tl.load(
            variances_ptr + pid_b * stride_var_b + k_offsets * stride_var_k,
            mask=k_mask,
            other=1.0,
        ).to(tl.float32)
        log_weights = tl.load(
            log_weights_ptr + pid_b * stride_logw_b + k_offsets * stride_logw_k,
            mask=k_mask,
            other=-3.4e38,
        ).to(tl.float32)

        cross = tl.dot(x_tile, means_tile, input_precision="tf32x3").to(tl.float32)
        dist = tl.maximum(x_sq_tile[:, None] + means_sq[None, :] - 2.0 * cross, 0.0)

        log_det_term = d_const * (log_2pi + tl.log(variances))
        logits = log_weights[None, :] - 0.5 * (
            dist / variances[None, :] + log_det_term[None, :]
        )
        logits = tl.where(k_mask[None, :], logits, -3.4e38)

        curr_max = tl.max(logits, axis=1)
        curr_idx = tl.argmax(logits, axis=1)
        update = curr_max > best_logit
        best_logit = tl.where(update, curr_max, best_logit)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


_spherical_assign_kernel_autotuned = triton.autotune(
    _TUNE_CONFIGS,
    key=["N", "K"],
)(_spherical_assign_kernel)


def spherical_assign_triton(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    x_sq: torch.Tensor = None,
    out: torch.Tensor = None,
    means_sq: torch.Tensor = None,
    log_weights: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    config: Optional[dict] = None,
    use_heuristic: bool = True,
) -> torch.Tensor:
    assert x.is_cuda and means.is_cuda and variances.is_cuda and weights.is_cuda, "All tensors must be on CUDA"
    assert means.dtype == x.dtype, "means dtype mismatch"

    B, N, D = x.shape
    K = means.shape[1]
    assert means.shape == (B, K, D), "means shape mismatch"
    assert variances.shape == (B, K), "variances shape mismatch"
    assert weights.shape == (B, K), "weights shape mismatch"

    if x_sq is None:
        x_sq = (x.to(torch.float32) ** 2).sum(dim=-1)
    if means_sq is None:
        means_sq = (means.to(torch.float32) ** 2).sum(dim=-1)
    if log_weights is None:
        log_weights = torch.log(weights.to(torch.float32))
    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_means_b, stride_means_k, stride_means_d = means.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_meanssq_b, stride_meanssq_k = means_sq.stride()
    stride_var_b, stride_var_k = variances.stride()
    stride_logw_b, stride_logw_k = log_weights.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    selected_config = None
    if config is not None:
        selected_config = config
    elif num_warps is not None or num_stages is not None:
        if num_warps is None or num_stages is None:
            raise ValueError("num_warps and num_stages must be set together")
        selected_config = {
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    elif use_heuristic:
        selected_config = _heuristic_euclid_config(N, K, D, device=x.device, dtype=x.dtype)

    launcher = _spherical_assign_kernel if selected_config is not None else _spherical_assign_kernel_autotuned
    launch_kwargs = {}
    if selected_config is not None:
        launch_kwargs = {
            "BLOCK_N": selected_config["BLOCK_N"],
            "BLOCK_K": selected_config["BLOCK_K"],
            "num_warps": selected_config["num_warps"],
            "num_stages": selected_config["num_stages"],
        }

    launcher[grid](
        x,
        means,
        x_sq,
        means_sq,
        variances,
        log_weights,
        out,
        B,
        N,
        K,
        D,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_means_b,
        stride_means_k,
        stride_means_d,
        stride_xsq_b,
        stride_xsq_n,
        stride_meanssq_b,
        stride_meanssq_k,
        stride_var_b,
        stride_var_k,
        stride_logw_b,
        stride_logw_k,
        stride_out_b,
        stride_out_n,
        **launch_kwargs,
    )
    return out


@triton.jit
def _spherical_logsumexp_kernel(
    x_ptr,
    means_ptr,
    x_sq_ptr,
    means_sq_ptr,
    variances_ptr,
    log_weights_ptr,
    out_ptr,
    sum_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_means_b: tl.constexpr,
    stride_means_k: tl.constexpr,
    stride_means_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_meanssq_b: tl.constexpr,
    stride_meanssq_k: tl.constexpr,
    stride_var_b: tl.constexpr,
    stride_var_k: tl.constexpr,
    stride_logw_b: tl.constexpr,
    stride_logw_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    HAS_SUM: tl.constexpr,
    UNIT_VARIANCE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_offsets < N

    offs_d = tl.arange(0, D).to(tl.int64)
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_sq_tile = tl.load(
        x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    running_max = tl.full((BLOCK_N,), -3.4e38, tl.float32)
    exp_sums = tl.zeros((BLOCK_N,), tl.float32)
    d_const = tl.full((1,), D, tl.float32)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)
        k_mask = k_offsets < K

        means_ptrs = (
            means_ptr
            + pid_b * stride_means_b
            + k_offsets[None, :] * stride_means_k
            + offs_d[:, None] * stride_means_d
        )
        means_tile = tl.load(means_ptrs, mask=k_mask[None, :], other=0.0)
        means_sq = tl.load(
            means_sq_ptr + pid_b * stride_meanssq_b + k_offsets * stride_meanssq_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        log_weights = tl.load(
            log_weights_ptr + pid_b * stride_logw_b + k_offsets * stride_logw_k,
            mask=k_mask,
            other=-3.4e38,
        ).to(tl.float32)

        cross = tl.dot(x_tile, means_tile, input_precision="tf32x3").to(tl.float32)
        dist = tl.maximum(x_sq_tile[:, None] + means_sq[None, :] - 2.0 * cross, 0.0)
        if UNIT_VARIANCE:
            logits = log_weights[None, :] - 0.5 * (dist + d_const * log_2pi)
        else:
            variances = tl.load(
                variances_ptr + pid_b * stride_var_b + k_offsets * stride_var_k,
                mask=k_mask,
                other=1.0,
            ).to(tl.float32)
            log_det_term = d_const * (log_2pi + tl.log(variances))
            logits = log_weights[None, :] - 0.5 * (
                dist / variances[None, :] + log_det_term[None, :]
            )
        logits = tl.where(k_mask[None, :], logits, -3.4e38)

        tile_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        exp_sums = exp_sums * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(logits - new_max[:, None]),
            axis=1,
        )
        running_max = new_max

    values = running_max + tl.log(exp_sums)
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, values, mask=n_mask)
    if HAS_SUM:
        tl.atomic_add(sum_ptr, tl.sum(tl.where(n_mask, values, 0.0), axis=0))


_spherical_logsumexp_kernel_autotuned = triton.autotune(
    _TUNE_CONFIGS,
    key=["N", "K"],
)(_spherical_logsumexp_kernel)


def spherical_logsumexp_triton(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    x_sq: torch.Tensor = None,
    out: torch.Tensor = None,
    out_sum: torch.Tensor = None,
    means_sq: torch.Tensor = None,
    log_weights: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    config: Optional[dict] = None,
    use_heuristic: bool = True,
    unit_variance: bool = False,
) -> torch.Tensor:
    assert x.is_cuda and means.is_cuda and variances.is_cuda and weights.is_cuda, "All tensors must be on CUDA"
    assert means.dtype == x.dtype, "means dtype mismatch"

    B, N, D = x.shape
    K = means.shape[1]
    assert means.shape == (B, K, D), "means shape mismatch"
    assert variances.shape == (B, K), "variances shape mismatch"
    assert weights.shape == (B, K), "weights shape mismatch"

    if x_sq is None:
        x_sq = (x.to(torch.float32) ** 2).sum(dim=-1)
    if means_sq is None:
        means_sq = (means.to(torch.float32) ** 2).sum(dim=-1)
    if log_weights is None:
        log_weights = torch.log(weights.to(torch.float32))
    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.float32)
    if out_sum is not None:
        assert out_sum.is_cuda and out_sum.numel() == 1, "out_sum must be a CUDA scalar tensor"
        assert out_sum.dtype == torch.float32, "out_sum must be float32"

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_means_b, stride_means_k, stride_means_d = means.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_meanssq_b, stride_meanssq_k = means_sq.stride()
    stride_var_b, stride_var_k = variances.stride()
    stride_logw_b, stride_logw_k = log_weights.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    selected_config = None
    if config is not None:
        selected_config = config
    elif num_warps is not None or num_stages is not None:
        if num_warps is None or num_stages is None:
            raise ValueError("num_warps and num_stages must be set together")
        selected_config = {
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    elif use_heuristic:
        selected_config = _heuristic_euclid_config(N, K, D, device=x.device, dtype=x.dtype)

    launcher = _spherical_logsumexp_kernel if selected_config is not None else _spherical_logsumexp_kernel_autotuned
    launch_kwargs = {}
    if selected_config is not None:
        launch_kwargs = {
            "BLOCK_N": selected_config["BLOCK_N"],
            "BLOCK_K": selected_config["BLOCK_K"],
            "num_warps": selected_config["num_warps"],
            "num_stages": selected_config["num_stages"],
        }

    launcher[grid](
        x,
        means,
        x_sq,
        means_sq,
        variances,
        log_weights,
        out,
        out_sum if out_sum is not None else out,
        B,
        N,
        K,
        D,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_means_b,
        stride_means_k,
        stride_means_d,
        stride_xsq_b,
        stride_xsq_n,
        stride_meanssq_b,
        stride_meanssq_k,
        stride_var_b,
        stride_var_k,
        stride_logw_b,
        stride_logw_k,
        stride_out_b,
        stride_out_n,
        HAS_SUM=out_sum is not None,
        UNIT_VARIANCE=unit_variance,
        **launch_kwargs,
    )
    return out


@triton.jit
def _spherical_resp_kernel(
    x_ptr,
    means_ptr,
    x_sq_ptr,
    means_sq_ptr,
    variances_ptr,
    log_weights_ptr,
    log_norm_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_means_b: tl.constexpr,
    stride_means_k: tl.constexpr,
    stride_means_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_meanssq_b: tl.constexpr,
    stride_meanssq_k: tl.constexpr,
    stride_var_b: tl.constexpr,
    stride_var_k: tl.constexpr,
    stride_logw_b: tl.constexpr,
    stride_logw_k: tl.constexpr,
    stride_lognorm_b: tl.constexpr,
    stride_lognorm_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    stride_out_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_offsets < N
    k_offsets = tl.arange(0, BLOCK_K).to(tl.int64)
    k_mask = k_offsets < K

    offs_d = tl.arange(0, D).to(tl.int64)
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    means_ptrs = (
        means_ptr
        + pid_b * stride_means_b
        + k_offsets[None, :] * stride_means_k
        + offs_d[:, None] * stride_means_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    means_tile = tl.load(means_ptrs, mask=k_mask[None, :], other=0.0)
    x_sq_tile = tl.load(
        x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    means_sq = tl.load(
        means_sq_ptr + pid_b * stride_meanssq_b + k_offsets * stride_meanssq_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    variances = tl.load(
        variances_ptr + pid_b * stride_var_b + k_offsets * stride_var_k,
        mask=k_mask,
        other=1.0,
    ).to(tl.float32)
    log_weights = tl.load(
        log_weights_ptr + pid_b * stride_logw_b + k_offsets * stride_logw_k,
        mask=k_mask,
        other=-3.4e38,
    ).to(tl.float32)
    log_norm = tl.load(
        log_norm_ptr + pid_b * stride_lognorm_b + n_offsets * stride_lognorm_n,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    d_const = tl.full((1,), D, tl.float32)
    log_2pi = tl.full((1,), 1.8378770664093453, tl.float32)
    cross = tl.dot(x_tile, means_tile, input_precision="tf32x3").to(tl.float32)
    dist = tl.maximum(x_sq_tile[:, None] + means_sq[None, :] - 2.0 * cross, 0.0)
    log_det_term = d_const * (log_2pi + tl.log(variances))
    logits = log_weights[None, :] - 0.5 * (
        dist / variances[None, :] + log_det_term[None, :]
    )
    resp = tl.exp(logits - log_norm[:, None])
    resp = tl.where(n_mask[:, None] & k_mask[None, :], resp, 0.0)

    out_ptrs = (
        out_ptr
        + pid_b * stride_out_b
        + n_offsets[:, None] * stride_out_n
        + k_offsets[None, :] * stride_out_k
    )
    tl.store(out_ptrs, resp, mask=n_mask[:, None] & k_mask[None, :])


_spherical_resp_kernel_autotuned = triton.autotune(
    _TUNE_CONFIGS,
    key=["N", "K"],
)(_spherical_resp_kernel)


def spherical_resp_triton(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    log_norm: torch.Tensor,
    x_sq: torch.Tensor = None,
    out: torch.Tensor = None,
    means_sq: torch.Tensor = None,
    log_weights: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    config: Optional[dict] = None,
    use_heuristic: bool = True,
) -> torch.Tensor:
    assert x.is_cuda and means.is_cuda and variances.is_cuda and weights.is_cuda and log_norm.is_cuda, "All tensors must be on CUDA"
    assert means.dtype == x.dtype, "means dtype mismatch"

    B, N, D = x.shape
    K = means.shape[1]
    assert means.shape == (B, K, D), "means shape mismatch"
    assert variances.shape == (B, K), "variances shape mismatch"
    assert weights.shape == (B, K), "weights shape mismatch"
    assert log_norm.shape == (B, N), "log_norm shape mismatch"

    if x_sq is None:
        x_sq = (x.to(torch.float32) ** 2).sum(dim=-1)
    if means_sq is None:
        means_sq = (means.to(torch.float32) ** 2).sum(dim=-1)
    if log_weights is None:
        log_weights = torch.log(weights.to(torch.float32))
    if out is None:
        out = torch.empty((B, N, K), device=x.device, dtype=torch.float32)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_means_b, stride_means_k, stride_means_d = means.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_meanssq_b, stride_meanssq_k = means_sq.stride()
    stride_var_b, stride_var_k = variances.stride()
    stride_logw_b, stride_logw_k = log_weights.stride()
    stride_lognorm_b, stride_lognorm_n = log_norm.stride()
    stride_out_b, stride_out_n, stride_out_k = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    selected_config = None
    if config is not None:
        selected_config = config
    elif num_warps is not None or num_stages is not None:
        if num_warps is None or num_stages is None:
            raise ValueError("num_warps and num_stages must be set together")
        selected_config = {
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    elif use_heuristic:
        selected_config = _heuristic_euclid_config(N, K, D, device=x.device, dtype=x.dtype)
        if D >= 128 and K <= 64 and x.dtype == torch.float32:
            gpu_name = torch.cuda.get_device_properties(x.device).name.upper()
            if "GEFORCE RTX 4090" in gpu_name:
                selected_config = {
                    "BLOCK_N": 32,
                    "BLOCK_K": 64,
                    "num_warps": 8,
                    "num_stages": 1,
                }

    if selected_config is not None and selected_config["BLOCK_K"] < K:
        selected_config = dict(selected_config)
        selected_config["BLOCK_K"] = 1 << (K - 1).bit_length()

    launcher = _spherical_resp_kernel if selected_config is not None else _spherical_resp_kernel_autotuned
    launch_kwargs = {}
    if selected_config is not None:
        launch_kwargs = {
            "BLOCK_N": selected_config["BLOCK_N"],
            "BLOCK_K": selected_config["BLOCK_K"],
            "num_warps": selected_config["num_warps"],
            "num_stages": selected_config["num_stages"],
        }

    launcher[grid](
        x,
        means,
        x_sq,
        means_sq,
        variances,
        log_weights,
        log_norm,
        out,
        B,
        N,
        K,
        D,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_means_b,
        stride_means_k,
        stride_means_d,
        stride_xsq_b,
        stride_xsq_n,
        stride_meanssq_b,
        stride_meanssq_k,
        stride_var_b,
        stride_var_k,
        stride_logw_b,
        stride_logw_k,
        stride_lognorm_b,
        stride_lognorm_n,
        stride_out_b,
        stride_out_n,
        stride_out_k,
        **launch_kwargs,
    )
    return out
