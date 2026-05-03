# GMMXX CUDA Backend — Plan 4: Sorted-run M-step + Real sm80 logsumexp/resp

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Two coupled deliverables that close the remaining spherical perf gaps:

1. **Sorted-run atomic-coalesced M-step** for spherical — replaces Plan 2's per-token `atomicAdd` accumulator with a kernel that walks contiguous runs of equal `cluster_id` and emits one atomic per (run, feature) tuple. ~256× atomic-issue reduction; closes the M-step perf gap vs Triton on K-heavy workloads.
2. **Real mma implementations of `logsumexp_sm80` and `resp_sm80`** — Plan 3 Task 2 strategically stubbed these to the safe path. This plan replaces the stubs with proper mma kernels, completing the sm80 fp16/bf16 fast path for the spherical E-step.

After Plan 4, the perf gate (Plan 3 Task 8 benchmark) tightens from 1.5× to 1.0× — CUDA must not regress vs Triton on any shape inside the spherical support window.

**Architecture:** Two new C++ TUs (`csrc/mstep/blocked_spherical_sorted.cu` + extensions to `csrc/estep/spherical_sm80.cu`). The Python `_cuda.py` wrapper for `blocked_update_spherical` adds an `argsort` step and a small heuristic to choose sorted vs naive based on `N · K` (sorted-run wins above ~2²¹ N·K product). The sm80 dispatcher in `spherical_dispatch.cu` is unchanged — `logsumexp_sm80` / `resp_sm80` take the same arguments.

**Tech Stack:** CUDA 12.8+, sm_80+ for the mma path. PowerShell + uv on the host.

**Spec sections covered:** §5d sorted-run atomic coalescing, §5d real mma logsumexp/resp, §10 perf gate tightening.

**Out of scope (deferred to Plan 5+):**
- Persistent E-step kernels (one CTA per SM, work-stealing).
- Multi-stream E/M overlap with `cudaEvent_t` plumbing.
- Fused single-tile E/M (Plan 5 deliverable).
- Approx top-K (Plan 6).
- Diag / tied / full (Plans 7–9).

**Foundation assumed:** Plans 1–3 complete (`spherical-mma-plan3` tag). The Plan 3 sm80 dispatcher already calls `assign_sm80` for fp16/bf16+sm_80+ inputs; this plan adds the real implementations of the two sibling functions and a sorted-run M-step variant.

---

## Sorted-run M-step design

### Reference

Read `C:\Users\HEQ\Projects\flash-kmeans-cuda\flash_kmeans_cuda\csrc\update\update_sorted.cu` (~150 lines). Algorithm sketch:

1. Caller pre-sorts `cluster_ids` (Python side, `torch.argsort`); also gathers `x` to match the permutation OR uses an indexed variant that reads `x[sorted_idx[i]]` inline.
2. Each CTA processes `BLOCK_N` (=256) tokens of the sorted permutation.
3. Lane 0 (per warp) scans for run boundaries (consecutive equal `cluster_ids`).
4. Once a run is identified, all 128 threads in the CTA cooperate to flush one `(run, feature)` tuple per thread to global atomics. Each thread accumulates a strided slice of D in a per-thread register, then issues a single atomicAdd per feature.
5. One additional atomic per run for the count.

The win: a typical run length ~256 tokens at K=64 means ~256× fewer atomics than per-token (in atomic-issue count, not bandwidth — atomics are much more about contention than bytes).

### Adaptation for GMM spherical

FKC's update kernel only computes the mean (`Σ x` per cluster). GMM spherical needs three accumulators:
- `sums[B, K, D]` — same as FKC.
- `sumsq[B, K]` — scalar `Σ ||x_n||²` per cluster (needed for variance later).
- `counts[B, K]` — same as FKC.

Each thread computes `||x_n||²` once per run and adds to a single per-run sumsq atomic — small extra cost.

### Threshold heuristic

For small `N·K`, the argsort cost (~5ms on H100 for N=2M) dominates the atomic-coalescing win (~1ms). Below `N · K < 2²¹` the dispatcher should prefer the existing per-token kernel that skips the sort. The Python wrapper picks the path; both kernels remain available.

---

## File Structure

### Created

| Path | Responsibility |
| --- | --- |
| `gmmxx/csrc/mstep/blocked_spherical_sorted.cu` | Sorted-run M-step kernel: walks runs, emits one atomic per (run, feature). Mirrors FKC `update_sorted.cu`. |
| `tests/test_cuda_spherical_sorted.py` | Sorted-run correctness vs the existing per-token kernel. |
| `tests/test_cuda_sm80_logsumexp_resp.py` | Per-kernel correctness for the real `logsumexp_sm80` / `resp_sm80`. |

### Modified

| Path | Change |
| --- | --- |
| `gmmxx/csrc/mstep/spherical.h` | Add `blocked_update_sorted(...)` declaration alongside the existing `blocked_update(...)`. |
| `gmmxx/csrc/bindings.cpp` | Expose `blocked_update_spherical_sorted` as a new public op. |
| `gmmxx/csrc/estep/spherical_sm80.cu` | Replace the safe-path stubs for `logsumexp_sm80` / `resp_sm80` with real mma kernels. |
| `gmmxx/_cuda.py` | `blocked_update_spherical` wrapper grows a `prefer_sorted` heuristic based on `N · K`. New `blocked_update_spherical_sorted(...)` low-level wrapper. |
| `gmmxx/cuda_ops.py` | Re-export `blocked_update_spherical_sorted`. |
| `gmmxx/interface.py` | `_train_spherical_cuda` calls the wrapper, which auto-picks sorted vs naive. Optional `gmmxx_force_sort: bool` for testing. |
| `setup.py` | Add `blocked_spherical_sorted.cu` to sources. |
| `benchmarks/benchmark_cuda_vs_triton_spherical.py` | Tighten `--gate-threshold` default from 1.5 to 1.1 (10% headroom). |
| `README.md` | Update to reflect Plan 4 closure. |

---

## Conventions

- Working directory: `C:\Users\HEQ\Projects\flashGMM2`. Branch: `GMMXX-cuda` (post-`spherical-mma-plan3`).
- Dev rebuild: `$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .` after each `.cu`/`.cpp`/setup.py change.
- Test command: `uv run pytest tests/<file> -v` (per task), full suite via `uv run pytest tests/ -q`.

---

## Task 1 — Write `blocked_spherical_sorted.cu`

**Files:** Create `gmmxx/csrc/mstep/blocked_spherical_sorted.cu`; modify `gmmxx/csrc/mstep/spherical.h` (add declaration); modify `setup.py`.

### Step 1.1 — Read FKC's update_sorted.cu

```bash
cat /c/Users/HEQ/Projects/flash-kmeans-cuda/flash_kmeans_cuda/csrc/update/update_sorted.cu
```

Take note of:
- `BLOCK_N = 256`, `THREADS_PER_CTA = 128`.
- Run-boundary scan via lane 0 broadcast.
- Per-thread strided-feature accumulation.
- The atomicAdd pattern: one per `(run, feature_chunk)` tuple.

### Step 1.2 — Add declaration to `gmmxx/csrc/mstep/spherical.h`

After the existing `blocked_update` declaration:

```cpp
// Sorted-run blocked M-step accumulator.
//
// Faster than blocked_update() above for large N*K where the argsort cost
// is amortized by the atomic-issue reduction (~256x fewer atomics within
// a CTA). Caller must pre-sort cluster_ids and provide the sorted x.
//
// x_sorted: (B, N, D) — caller-permuted x, matching sorted_ids ordering.
// sorted_ids: (B, N) int32 — sorted ascending per batch.
// sums_out, sumsq_out, counts_out: as in blocked_update — caller-zeroed.
void blocked_update_sorted(const at::Tensor& x_sorted,
                           const at::Tensor& sorted_ids,
                           at::Tensor& sums_out,
                           at::Tensor& sumsq_out,
                           at::Tensor& counts_out);
```

### Step 1.3 — Write the kernel

Mirror FKC's `update_sorted_kernel` structure. Adapt for sumsq accumulation.

```cpp
#include "spherical.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace mstep { namespace spherical {

namespace {

constexpr int BLOCK_N = 256;
constexpr int THREADS_PER_CTA = 128;

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_CTA, 4)
blocked_update_sorted_kernel(
    const T* __restrict__ x_sorted,           // (B, N, D)
    const int32_t* __restrict__ sorted_ids,   // (B, N) sorted ascending per batch
    float* __restrict__ sums,                 // (B, K, D) atomicAdd target
    float* __restrict__ sumsq,                // (B, K) atomicAdd target
    int32_t* __restrict__ counts,             // (B, K) atomicAdd target
    int B, int N, int K, int D
) {
    int b = blockIdx.y;
    int n_start = blockIdx.x * BLOCK_N;
    int n_count = min(BLOCK_N, N - n_start);
    if (n_count <= 0 || b >= B) return;

    const T* x_b = x_sorted + (size_t)b * N * D;
    const int32_t* ids_b = sorted_ids + (size_t)b * N;

    __shared__ int run_start;
    __shared__ int run_len;
    __shared__ int run_cid;

    int cursor = 0;
    while (cursor < n_count) {
        if (threadIdx.x == 0) {
            // Find next run.
            int cid = ids_b[n_start + cursor];
            int len = 1;
            while (cursor + len < n_count && ids_b[n_start + cursor + len] == cid) {
                len++;
            }
            run_start = cursor;
            run_len = len;
            run_cid = cid;
        }
        __syncthreads();

        int rs = run_start;
        int rl = run_len;
        int cid = run_cid;

        if (cid >= 0 && cid < K) {
            // Per-thread strided slice of D for sums.
            for (int d_base = threadIdx.x; d_base < D; d_base += THREADS_PER_CTA) {
                float acc = 0.0f;
                for (int r = 0; r < rl; ++r) {
                    int n_idx = n_start + rs + r;
                    acc += static_cast<float>(x_b[(size_t)n_idx * D + d_base]);
                }
                atomicAdd(sums + ((size_t)b * K + cid) * D + d_base, acc);
            }

            // Sumsq and count: one thread does the per-run reduction in parallel.
            // Strided accumulation of ||x||^2 per run.
            float local_ss = 0.0f;
            for (int rd = threadIdx.x; rd < rl * D; rd += THREADS_PER_CTA) {
                int r = rd / D;
                int d = rd % D;
                int n_idx = n_start + rs + r;
                float v = static_cast<float>(x_b[(size_t)n_idx * D + d]);
                local_ss += v * v;
            }
            // Block-reduce local_ss to thread 0.
            __shared__ float ss_smem[THREADS_PER_CTA / 32];
            // Warp reduce.
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_ss += __shfl_xor_sync(0xffffffffu, local_ss, offset);
            }
            int lane = threadIdx.x & 31;
            int warp_id = threadIdx.x >> 5;
            if (lane == 0) ss_smem[warp_id] = local_ss;
            __syncthreads();
            if (warp_id == 0) {
                float v = (lane < THREADS_PER_CTA / 32) ? ss_smem[lane] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    v += __shfl_xor_sync(0xffffffffu, v, offset);
                }
                if (threadIdx.x == 0) {
                    atomicAdd(sumsq + (size_t)b * K + cid, v);
                    atomicAdd(counts + (size_t)b * K + cid, rl);
                }
            }
            __syncthreads();
        }

        cursor = rs + rl;
    }
}

}  // anonymous namespace

void blocked_update_sorted(const at::Tensor& x_sorted,
                           const at::Tensor& sorted_ids,
                           at::Tensor& sums_out,
                           at::Tensor& sumsq_out,
                           at::Tensor& counts_out) {
    TORCH_CHECK(x_sorted.is_cuda() && x_sorted.is_contiguous(), "x_sorted must be contiguous CUDA");
    TORCH_CHECK(sorted_ids.is_cuda() && sorted_ids.is_contiguous() &&
                sorted_ids.scalar_type() == at::kInt,
                "sorted_ids must be contiguous int32 CUDA");
    TORCH_CHECK(sums_out.is_cuda() && sums_out.is_contiguous() &&
                sums_out.scalar_type() == at::kFloat,
                "sums_out must be contiguous fp32 CUDA");
    TORCH_CHECK(sumsq_out.is_cuda() && sumsq_out.is_contiguous() &&
                sumsq_out.scalar_type() == at::kFloat,
                "sumsq_out must be contiguous fp32 CUDA");
    TORCH_CHECK(counts_out.is_cuda() && counts_out.is_contiguous() &&
                counts_out.scalar_type() == at::kInt,
                "counts_out must be contiguous int32 CUDA");
    TORCH_CHECK(x_sorted.dim() == 3, "x_sorted must be (B,N,D)");

    int B = (int)x_sorted.size(0);
    int N = (int)x_sorted.size(1);
    int D = (int)x_sorted.size(2);
    int K = (int)sums_out.size(1);
    if (N == 0) return;

    c10::cuda::CUDAGuard guard(x_sorted.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, B);

    switch (x_sorted.scalar_type()) {
        case at::kFloat:
            blocked_update_sorted_kernel<float><<<grid, THREADS_PER_CTA, 0, stream>>>(
                x_sorted.data_ptr<float>(), sorted_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        case at::kHalf:
            blocked_update_sorted_kernel<at::Half><<<grid, THREADS_PER_CTA, 0, stream>>>(
                x_sorted.data_ptr<at::Half>(), sorted_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        case at::kBFloat16:
            blocked_update_sorted_kernel<at::BFloat16><<<grid, THREADS_PER_CTA, 0, stream>>>(
                x_sorted.data_ptr<at::BFloat16>(), sorted_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        default:
            TORCH_CHECK(false, "blocked_update_sorted: unsupported dtype ", x_sorted.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}}}
```

### Step 1.4 — Update `setup.py`

Add to sources:

```python
str(CSRC / "mstep" / "blocked_spherical_sorted.cu"),
```

### Step 1.5 — Add binding

In `gmmxx/csrc/bindings.cpp`, after the existing `blocked_update_spherical` m.def:

```cpp
    m.def(
        "blocked_update_spherical_sorted",
        &gmmxx::mstep::spherical::blocked_update_sorted,
        nb::arg("x_sorted"),
        nb::arg("sorted_ids"),
        nb::arg("sums_out"),
        nb::arg("sumsq_out"),
        nb::arg("counts_out"),
        "Sorted-run M-step. Caller pre-sorts cluster_ids and gathers x. "
        "Faster than blocked_update_spherical for large N*K.");
```

### Step 1.6 — Build + smoke

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .
```

```bash
uv run python -c "
import torch
from gmmxx import _C
torch.manual_seed(0)
B, N, D, K = 1, 1024, 8, 16
x = torch.randn(B, N, D, device='cuda')
ids = torch.randint(0, K, (B, N), device='cuda', dtype=torch.int32)

# Naive baseline
sums_naive = torch.zeros(B, K, D, device='cuda')
sumsq_naive = torch.zeros(B, K, device='cuda')
counts_naive = torch.zeros(B, K, device='cuda', dtype=torch.int32)
_C.blocked_update_spherical(x, ids, sums_naive, sumsq_naive, counts_naive)

# Sorted-run
sorted_ids, perm = ids.sort(dim=1)
x_sorted = torch.gather(x, 1, perm.unsqueeze(-1).expand(-1, -1, D))
sums_sorted = torch.zeros(B, K, D, device='cuda')
sumsq_sorted = torch.zeros(B, K, device='cuda')
counts_sorted = torch.zeros(B, K, device='cuda', dtype=torch.int32)
_C.blocked_update_spherical_sorted(x_sorted, sorted_ids.int(), sums_sorted, sumsq_sorted, counts_sorted)

print('counts match:', torch.equal(counts_naive, counts_sorted))
print('sums max diff:', (sums_naive - sums_sorted).abs().max().item())
print('sumsq max diff:', (sumsq_naive - sumsq_sorted).abs().max().item())
"
```

Expected: counts equal; sums and sumsq agree to within fp32 atomic ULP drift (< 1e-3 absolute on these shapes).

### Step 1.7 — Commit

```bash
git add gmmxx/csrc/mstep/blocked_spherical_sorted.cu gmmxx/csrc/mstep/spherical.h gmmxx/csrc/bindings.cpp setup.py
git commit -m "$(cat <<'EOF'
Add spherical M-step sorted-run kernel for atomic coalescing

blocked_update_spherical_sorted: caller pre-sorts cluster_ids and gathers
x; the kernel walks contiguous runs of equal cluster_id within a CTA and
emits one atomicAdd per (run, feature) tuple instead of per token.
Approximately 256x atomic-issue reduction within a CTA on K-heavy
workloads.

Mirrors flash-kmeans-cuda's update_sorted.cu pattern; adds a per-run
sumsq accumulator (Σ ||x||²) needed for the GMM variance update —
not present in FKC's k-means update.

The naive per-token blocked_update_spherical from Plan 2 stays as the
default for small N*K where the argsort cost dominates the coalescing
win. Plan 4 Task 2 wires the threshold heuristic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Wrapper + heuristic in `_cuda.py`

**Files:** Modify `gmmxx/_cuda.py`; modify `gmmxx/cuda_ops.py` (re-export).

### Step 2.1 — Update `blocked_update_spherical` wrapper

The wrapper currently allocates and zero-inits sums/sumsq/counts then calls `_C.blocked_update_spherical`. Modify it to:

1. If `N · K >= 2²¹` (configurable): argsort `cluster_ids` (Python side), gather `x`, call `_C.blocked_update_spherical_sorted`.
2. Else: call `_C.blocked_update_spherical` (existing path).

```python
_SORT_THRESHOLD_NK = 2 ** 21  # ~2M; below this, sort cost dominates


def blocked_update_spherical(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    n_components: int,
    *,
    force_sort: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M-step accumulator. Picks sorted-run vs per-token by N*K heuristic.

    force_sort: True forces sorted path; False forces per-token; None auto.
    """
    require_cuda()
    x = _check_input(x, "x")
    cluster_ids = _check_input(cluster_ids, "cluster_ids", dtype=torch.int32)
    B, N, D = x.shape
    K = int(n_components)
    sums = torch.zeros((B, K, D), dtype=torch.float32, device=x.device)
    sumsq = torch.zeros((B, K), dtype=torch.float32, device=x.device)
    counts = torch.zeros((B, K), dtype=torch.int32, device=x.device)

    use_sort = force_sort if force_sort is not None else (N * K >= _SORT_THRESHOLD_NK)

    try:
        if use_sort:
            sorted_ids, perm = cluster_ids.sort(dim=1)
            x_sorted = torch.gather(x, 1, perm.unsqueeze(-1).expand(-1, -1, D))
            _C.blocked_update_spherical_sorted(
                x_sorted.contiguous(), sorted_ids.int().contiguous(),
                sums, sumsq, counts,
            )
        else:
            _C.blocked_update_spherical(x, cluster_ids, sums, sumsq, counts)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(
            f"blocked_update_spherical (use_sort={use_sort}) failed: {exc}"
        ) from exc
    return sums, sumsq, counts


def blocked_update_spherical_sorted(
    x_sorted: torch.Tensor,
    sorted_ids: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct sorted-run wrapper. Caller is responsible for sorting cluster_ids
    and gathering x to match. For most uses, prefer blocked_update_spherical
    which handles the sort + heuristic."""
    require_cuda()
    x_sorted = _check_input(x_sorted, "x_sorted")
    sorted_ids = _check_input(sorted_ids, "sorted_ids", dtype=torch.int32)
    B, N, D = x_sorted.shape
    K = int(n_components)
    sums = torch.zeros((B, K, D), dtype=torch.float32, device=x_sorted.device)
    sumsq = torch.zeros((B, K), dtype=torch.float32, device=x_sorted.device)
    counts = torch.zeros((B, K), dtype=torch.int32, device=x_sorted.device)
    try:
        _C.blocked_update_spherical_sorted(x_sorted, sorted_ids, sums, sumsq, counts)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(
            f"blocked_update_spherical_sorted failed: {exc}"
        ) from exc
    return sums, sumsq, counts
```

### Step 2.2 — Re-export through `cuda_ops.py`

Add to the spherical block:

```python
blocked_update_spherical_sorted = _cuda.blocked_update_spherical_sorted
```

And to `__all__`.

### Step 2.3 — Smoke test

```bash
uv run python -c "
import torch
from gmmxx import _cuda
torch.manual_seed(0)
B, N, D, K = 1, 4096, 32, 16
x = torch.randn(B, N, D, device='cuda')
ids = torch.randint(0, K, (B, N), device='cuda', dtype=torch.int32)

# Auto path (N*K=65536, below threshold) — should use per-token
s1, ss1, c1 = _cuda.blocked_update_spherical(x, ids, K)
# Force sorted
s2, ss2, c2 = _cuda.blocked_update_spherical(x, ids, K, force_sort=True)

print('counts match:', torch.equal(c1, c2))
print('sums max diff:', (s1 - s2).abs().max().item())
print('sumsq max diff:', (ss1 - ss2).abs().max().item())
"
```

Expected: counts equal exactly; sums/sumsq differ by atomic ULP only (< 1e-3 absolute).

### Step 2.4 — Commit

```bash
git add gmmxx/_cuda.py gmmxx/cuda_ops.py
git commit -m "$(cat <<'EOF'
Wire sorted-run M-step into blocked_update_spherical wrapper

blocked_update_spherical now picks sorted-run vs per-token via the
heuristic N*K >= 2^21 (~2M). force_sort kwarg overrides for testing.

Adds blocked_update_spherical_sorted as a direct low-level wrapper for
callers who already have sorted cluster_ids (e.g., a future fused
sort+EM iteration).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Wire into `_train_spherical_cuda`

**Files:** Modify `gmmxx/interface.py`

The existing `_train_spherical_cuda` calls `_cuda_mod.blocked_update_spherical(data_b, ids, K)` once per EM iteration. The wrapper now handles sort vs naive automatically, so no change is strictly required — but verify the heuristic is firing for typical training shapes.

### Step 3.1 — Verify

```bash
uv run python -c "
import torch
from gmmxx import GMMXX

torch.manual_seed(0)
# N*K = 8192*16 = 131072, BELOW the 2M threshold → per-token path
x = torch.randn(8192, 16, device='cuda')
gmm = GMMXX(n_components=16, max_iter=10, tol=1e-4, random_state=0,
            covariance_type='spherical', backend='cuda')
gmm.fit(x)
print('small case last_backend_used_:', gmm.last_backend_used_)
print('lower_bound_:', gmm.lower_bound_)

# N*K = 524288*64 = 33554432, ABOVE the 2M threshold → sorted path
x_big = torch.randn(524288, 64, device='cuda')
gmm2 = GMMXX(n_components=64, max_iter=5, tol=0, random_state=0,
             covariance_type='spherical', backend='cuda')
gmm2.fit(x_big)
print('big case last_backend_used_:', gmm2.last_backend_used_)
print('lower_bound_:', gmm2.lower_bound_)
"
```

Expected: both paths run cleanly; last_backend_used_ == "cuda" in both cases.

### Step 3.2 — Optional: add force_sort path for testing

In `_train_spherical_cuda`, expose `_force_sort` as a private attribute set during testing:

```python
sums, sumsq, counts = _cuda_mod.blocked_update_spherical(
    data_b, ids, K,
    force_sort=getattr(self, "_force_sort", None),
)
```

This lets tests construct `GMMXX(...)` then `gmm._force_sort = True` to exercise the sorted path on small shapes.

### Step 3.3 — Run full test suite

```bash
uv run pytest tests/ -q
```

Expected: 118+ pass, no regressions.

### Step 3.4 — Commit

```bash
git add gmmxx/interface.py
git commit -m "$(cat <<'EOF'
_train_spherical_cuda: thread force_sort kwarg through blocked_update wrapper

The wrapper auto-selects sorted vs per-token based on N*K, but tests
need to exercise both paths on small shapes. Add a private _force_sort
attribute that gets passed to blocked_update_spherical when set.

No behavior change for users; the heuristic still picks correctly on
realistic shapes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Sorted-run correctness tests

**Files:** Create `tests/test_cuda_spherical_sorted.py`

```python
"""Sorted-run M-step kernel correctness vs the per-token kernel."""

from __future__ import annotations
import pytest
import torch


def _has_cuda():
    try:
        from gmmxx._cuda import has_cuda
        return has_cuda()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")


@pytest.mark.parametrize("N,K", [(1024, 8), (4096, 16), (16384, 32), (65536, 64)])
def test_sorted_matches_naive(N, K):
    """Sorted-run output should match per-token within fp32 atomic ULP drift."""
    from gmmxx import _cuda
    torch.manual_seed(0)
    B, D = 1, 16
    x = torch.randn(B, N, D, device="cuda")
    ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)

    s_naive, ss_naive, c_naive = _cuda.blocked_update_spherical(x, ids, K, force_sort=False)
    s_sorted, ss_sorted, c_sorted = _cuda.blocked_update_spherical(x, ids, K, force_sort=True)

    assert torch.equal(c_naive, c_sorted), "counts must match exactly"
    # Atomic order varies, so sums/sumsq agree only to fp32 ULP.
    assert torch.allclose(s_naive, s_sorted, rtol=1e-4, atol=1e-3)
    assert torch.allclose(ss_naive, ss_sorted, rtol=1e-4, atol=1e-2)


def test_force_sort_zero_N():
    """Empty N must not crash either path."""
    from gmmxx import _cuda
    x = torch.empty(1, 0, 8, device="cuda")
    ids = torch.empty(1, 0, device="cuda", dtype=torch.int32)
    s, ss, c = _cuda.blocked_update_spherical(x, ids, 4, force_sort=True)
    assert s.shape == (1, 4, 8) and (s == 0).all()
    assert (c == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sorted_handles_all_dtypes(dtype):
    from gmmxx import _cuda
    torch.manual_seed(0)
    B, N, D, K = 1, 2048, 16, 8
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    s, ss, c = _cuda.blocked_update_spherical(x, ids, K, force_sort=True)
    assert c.sum().item() == N
    assert s.shape == (B, K, D)
    assert ss.shape == (B, K)


def test_heuristic_picks_per_token_below_threshold():
    """N*K=8192 << 2^21 should pick per-token. We can't observe the path
    selection directly, but the result must be correct."""
    from gmmxx import _cuda
    torch.manual_seed(0)
    x = torch.randn(1, 1024, 8, device="cuda")
    ids = torch.randint(0, 8, (1, 1024), device="cuda", dtype=torch.int32)
    s, ss, c = _cuda.blocked_update_spherical(x, ids, 8)
    assert c.sum().item() == 1024
```

Run + commit:

```bash
uv run pytest tests/test_cuda_spherical_sorted.py -v
git add tests/test_cuda_spherical_sorted.py
git commit -m "$(cat <<'EOF'
Add correctness tests for sorted-run M-step

Compares sorted-run vs per-token outputs across (N, K) shapes and dtypes.
counts must match exactly; sums/sumsq agree to fp32 ULP (atomic order
differs but the result is the same to within ULP drift).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Real `logsumexp_sm80` (replace stub)

**Files:** Modify `gmmxx/csrc/estep/spherical_sm80.cu`

The current `logsumexp_sm80` calls `logsumexp_safe(...)`. Replace with a real mma-based kernel.

### Algorithm

The sm80 assign kernel already computes `logit_k` per (m, n) pair from `cross[m,n] + x_sq[m] + c_sq[n] → dist → logit`. The logsumexp kernel runs the same outer structure (cp.async double-buffered c_tile, mma loop over D) but the per-row reduction is a running (max_logit, sumexp) pair across K chunks instead of a running max.

Stable cross-chunk update:

```
on first chunk:
    max_so_far = max(logit over k in chunk)
    sumexp = Σ exp(logit - max_so_far)

on subsequent chunks:
    chunk_max = max(logit over k in chunk)
    new_max = max(max_so_far, chunk_max)
    sumexp = sumexp * exp(max_so_far - new_max) + Σ exp(logit - new_max)
    max_so_far = new_max

at end:
    log_norm[m] = max_so_far + log(sumexp)
```

Each thread tracks its own per-row (max_so_far, sumexp) pair across the K-chunk loop.

### Implementation

The simplest approach: copy the assign kernel body, replace the per-thread `(best_logit, best_idx)` register tile with `(max_logit, sumexp)`, and replace the in-chunk argmax with the in-chunk logsumexp followed by the cross-chunk update. The assign kernel's outer structure (cp.async, mma loop, fragment register tile) is identical.

If the implementer is uncertain about the cross-chunk fragment-register state management (sumexp must persist across iterations of the K-chunk loop), an alternative is to:
1. Run the kernel once per K-chunk, accumulating max + sumexp in global memory.
2. After the loop, finalize `log_norm = max + log(sumexp)`.

That uses one extra global write per K-chunk but is simpler to reason about. Pick this if the in-register state management is too fiddly.

### Smoke test

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .
uv run python -c "
import torch, math
from gmmxx import _C
B, N, D, K = 1, 256, 32, 16
torch.manual_seed(0)
x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
means = torch.randn(B, K, D, device='cuda', dtype=torch.float16)
var = torch.ones(B, K, device='cuda')
log_w = torch.zeros(B, K, device='cuda')

cuda_lse = _C.spherical_logsumexp(x, means, var, log_w)

diff = x.float().unsqueeze(2) - means.float().unsqueeze(1)
dist = diff.pow(2).sum(-1)
logits = log_w.unsqueeze(1) - 0.5*D*math.log(2*math.pi) - 0.5*dist
ref_lse = logits.logsumexp(-1)

print('max abs diff:', (cuda_lse - ref_lse).abs().max().item())
"
```

Expected: max abs diff < 1e-2 for fp16 (per spec §6 fp16 contract).

### Commit

```bash
git add gmmxx/csrc/estep/spherical_sm80.cu
git commit -m "$(cat <<'EOF'
Replace logsumexp_sm80 stub with real mma kernel (Plan 4 Task 5)

logsumexp_sm80 was previously stubbed to logsumexp_safe (Plan 3 Task 2
scope-down). Now implements the full mma-based path: same outer
structure as assign_sm80 (cp.async double-buffer + mma loop), with
per-row (max_logit, sumexp) running pairs updated stably across
K-chunks.

Final log_norm[m] = max_so_far[m] + log(sumexp[m]).

Smoke-tested at fp16 max-abs-diff < 1e-2 vs torch reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Real `resp_sm80` (replace stub)

**Files:** Modify `gmmxx/csrc/estep/spherical_sm80.cu`

The current `resp_sm80` calls `resp_safe(...)`. Replace with a real mma-based kernel.

### Algorithm

Even simpler than logsumexp: there's no cross-chunk reduction. Per-row, per-cluster, write `exp(logit[m,n] - log_norm[m])` directly to `out[b, n_for_m, k_for_n]`. The kernel structure is the assign kernel's mma loop with a write per (m, n) instead of a min/max reduction.

Each thread owns a fragment slice of `(BLOCK_N, BLOCK_K)`; in the epilogue, for each (m, n) it owns:
- Compute logit (same formula as assign).
- Read `log_norm[b, n_base + m]` from global.
- Write `exp(logit - log_norm)` to `out[b, n_base + m, k_chunk_base + n]`.

No per-thread state across K-chunks — all per-cluster outputs are independent.

### Smoke test

```bash
uv run python -c "
import torch, math
from gmmxx import _C
B, N, D, K = 1, 128, 32, 16
torch.manual_seed(0)
x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
means = torch.randn(B, K, D, device='cuda', dtype=torch.float16)
var = torch.ones(B, K, device='cuda')
log_w = torch.zeros(B, K, device='cuda')

lse = _C.spherical_logsumexp(x, means, var, log_w)
r = _C.spherical_resp(x, means, var, log_w, lse)

print('shape:', r.shape, 'sums per row:', r.sum(-1).mean().item())

diff = x.float().unsqueeze(2) - means.float().unsqueeze(1)
dist = diff.pow(2).sum(-1)
logits = log_w.unsqueeze(1) - 0.5*D*math.log(2*math.pi) - 0.5*dist
ref_r = (logits - lse.unsqueeze(-1)).exp()
print('max abs diff:', (r - ref_r).abs().max().item())
"
```

Expected: shape (1, 128, 16); per-row sums ~1.0; max abs diff < 1e-2 for fp16.

### Commit

```bash
git add gmmxx/csrc/estep/spherical_sm80.cu
git commit -m "$(cat <<'EOF'
Replace resp_sm80 stub with real mma kernel (Plan 4 Task 6)

Same outer structure as assign_sm80 (cp.async + mma loop), but the
epilogue writes exp(logit - log_norm) per (m, n) directly to global
output instead of doing a per-row reduction. Each (m, n) is independent
so no cross-chunk state is needed.

Spherical sm80 path is now fully populated: assign / logsumexp / resp
all run on tensor cores for fp16/bf16 inputs on sm_80+.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Tests for real sm80 logsumexp / resp

**Files:** Create `tests/test_cuda_sm80_logsumexp_resp.py`

```python
"""Per-kernel correctness for the real sm80 logsumexp_sm80 and resp_sm80.

These were stubbed to safe in Plan 3 Task 2; Plan 4 Tasks 5–6 implement
them as real mma kernels. This test file verifies the mma path matches
the safe path within the spec's fp16/bf16 tolerance window.
"""

from __future__ import annotations
import math
import pytest
import torch


def _has_sm80():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


pytestmark = pytest.mark.skipif(not _has_sm80(), reason="requires CUDA + sm_80+")


def _setup(B=1, N=256, D=32, K=16, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    means = torch.randn(B, K, D, device=device, dtype=dtype)
    var = torch.rand(B, K, device=device).clamp_min(0.5).float()
    log_w = torch.log_softmax(torch.randn(B, K, device=device), dim=-1).float()
    return x, means, var, log_w


def _ref_logits(x, means, var, log_w):
    B, N, D = x.shape
    K = means.shape[1]
    diff = x.float().unsqueeze(2) - means.float().unsqueeze(1)
    dist = diff.pow(2).sum(-1)
    return (
        log_w.unsqueeze(1)
        - 0.5 * D * torch.log(2 * math.pi * var).unsqueeze(1)
        - 0.5 * dist / var.unsqueeze(1)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D,K", [(16, 16), (32, 32), (64, 64), (128, 32)])
def test_logsumexp_matches_reference(dtype, D, K):
    from gmmxx import _cuda
    x, means, var, log_w = _setup(D=D, K=K, dtype=dtype)
    cuda_lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    ref_lse = _ref_logits(x, means, var, log_w).logsumexp(-1)
    assert torch.allclose(cuda_lse, ref_lse, rtol=5e-3, atol=5e-3), (
        f"max abs diff: {(cuda_lse - ref_lse).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D,K", [(16, 16), (32, 32), (64, 64)])
def test_resp_matches_reference(dtype, D, K):
    from gmmxx import _cuda
    x, means, var, log_w = _setup(D=D, K=K, dtype=dtype)
    lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    cuda_r = _cuda.spherical_resp(x, means, var, log_w, lse)
    ref_r = (_ref_logits(x, means, var, log_w) - lse.unsqueeze(-1)).exp()
    assert torch.allclose(cuda_r, ref_r, rtol=5e-3, atol=5e-3), (
        f"max abs diff: {(cuda_r - ref_r).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_resp_sums_to_one(dtype):
    from gmmxx import _cuda
    x, means, var, log_w = _setup(N=512, D=32, K=32, dtype=dtype)
    lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    r = _cuda.spherical_resp(x, means, var, log_w, lse)
    assert torch.allclose(r.sum(-1), torch.ones_like(r.sum(-1)), atol=5e-3)
```

Run + commit:

```bash
uv run pytest tests/test_cuda_sm80_logsumexp_resp.py -v
git add tests/test_cuda_sm80_logsumexp_resp.py
git commit -m "$(cat <<'EOF'
Add per-kernel correctness tests for real sm80 logsumexp_sm80 / resp_sm80

Validates Plan 4 Tasks 5-6 implementations against torch reference at
rtol=5e-3 for fp16/bf16 across (D, K) tile shapes 16/16, 32/32, 64/64,
128/32. Also verifies resp rows sum to 1.0 within tolerance.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — Tighten the perf gate

**Files:** Modify `benchmarks/benchmark_cuda_vs_triton_spherical.py`

Change the default `--gate-threshold` from 1.5 to 1.1 (10% headroom). Run with `--gate` and verify it still passes.

```bash
uv run python benchmarks/benchmark_cuda_vs_triton_spherical.py
uv run python benchmarks/benchmark_cuda_vs_triton_spherical.py --gate
```

If the gate passes at 1.1, commit. If it fails on some shapes, investigate (does the sorted-run path help on those? does the real logsumexp_sm80 close the gap?). Don't relax the threshold without root-cause.

```bash
git add benchmarks/benchmark_cuda_vs_triton_spherical.py
git commit -m "$(cat <<'EOF'
Tighten perf gate from 1.5x to 1.1x (Plan 4 closure)

After sorted-run M-step + real sm80 logsumexp/resp, CUDA should be
within 10% of Triton on all spherical shapes inside the support window.
Plan 5 (persistent kernels + multi-stream) will tighten further.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — README + tag

```bash
# Update README to reflect Plan 4 closure. Replace the Plan 3 status line:
# "**Spherical covariance is fully on the CUDA path** for both training and
#  inference (Plan 3). The sm_80 mma.sync optimized E-step assign runs..."
# with:
# "**Spherical covariance is fully on the CUDA path** for both training and
#  inference, with sorted-run M-step atomic coalescing and real mma E-step
#  for assign/logsumexp/resp on Ampere+ (Plans 2-4). CUDA is within 10% of
#  Triton on all supported shapes."
```

```bash
git add README.md
git commit -m "README: spherical perf parity (Plan 4 complete)"
git tag -a spherical-fast-plan4 -m "Plan 4: sorted-run M-step + real sm80 logsumexp/resp"
git log --oneline spherical-mma-plan3..spherical-fast-plan4
```

---

## Self-Review Checklist

**1. Spec coverage**

| Spec section | Plan task |
| --- | --- |
| §5d sorted-run atomic coalescing | Tasks 1, 2, 3 |
| §5d mma.sync optimized logsumexp / resp | Tasks 5, 6 |
| §6 fp16/bf16 tolerance | Tasks 4, 7 |
| §10 perf benchmark gate tightening | Task 8 |

Unaddressed (deferred):
- §5d persistent kernels → Plan 5
- §5d multi-stream events → Plan 5
- §4 fused single-tile E/M → Plan 5

**2. Placeholder scan** — Tasks 5 and 6 describe the kernel pseudocode but not the full register-fragment layout. Implementer must read the existing `assign_sm80` kernel as the structural template and adapt its epilogue. This is unavoidable for mma kernels.

**3. Type consistency** — `at::Tensor` throughout. `force_sort: Optional[bool]` threaded through `_train_spherical_cuda`. `_SORT_THRESHOLD_NK = 2 ** 21` documented.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-03-gmmxx-cuda-spherical-mstep.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Tasks 1–4 can use sonnet (sorted-run kernel structure is well-defined). Tasks 5–6 (real sm80 logsumexp/resp) benefit from opus given the in-register state management complexity. Tasks 7–9 use sonnet.

**2. Inline Execution** — Same session.

After Plan 4: **Plan 5** = persistent E-step kernels + multi-stream events + fused single-tile E/M for spherical. Then **Plan 6** = approx top-K. Then **Plans 7–9** for diag, tied, full. Then **Plan 10** = `large_n.py` integration.
