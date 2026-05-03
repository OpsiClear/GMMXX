# GMMXX CUDA Backend — Plan 3: Spherical sm_80 mma + Inference Rewire

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two coupled deliverables:

1. **sm_80 mma.sync optimized E-step** for spherical fp16/bf16 inputs — replaces the safe SIMT path with `m16n8k16.f32.f16.f16.f32` tensor-core matmul + `cp.async` double-buffered SMEM. Targets ~2–4× E-step speedup vs the Plan 2 safe path on D≥32 / K≥32 shapes.
2. **Inference-path rewire** for the I-1 issue surfaced in Plan 1's final review — `predict()`, `predict_proba()`, `score_samples()`, `score()` consult `_dispatch.resolve_backend` instead of the `self.use_triton` property, and a CUDA branch wires through `gmmxx.cuda_ops.spherical_assign` / `spherical_logsumexp` / `spherical_resp` for spherical inputs in the support window.

**Architecture:** Two new CUDA TUs (`csrc/estep/spherical_sm80.cu` + `csrc/estep/spherical_dispatch.cu`); the latter is a thin host-side dispatcher that picks safe vs sm80 by dtype + arch. Existing `_C.spherical_assign` etc. become arch-routing entry points (host-side). Python `_cuda.py` wrappers don't change shape; the dispatch happens inside C++. Inference paths in `gmmxx/interface.py` are rewritten to consult `_dispatch.dispatch_kernel(op, backend, *args)` per call; the `use_triton` property is retained as a backward-compat shim but is no longer read by internal call sites.

**Tech Stack:** CUDA 12.8+, sm_80+ for the mma path (sm_70/75 stays on safe), `cp.async.cg`, fp16/bf16 mma m16n8k16. PowerShell + uv on the host.

**Spec sections covered:** §4 (mma path for spherical), §5d optimization techniques (mma.sync + cp.async), §7 dispatch, §10 testing — perf benchmark gate.

**Out of scope (deferred):**
- Sorted-run atomic coalescing M-step → Plan 4.
- Persistent E-step kernels → Plan 4.
- Fused single-tile E/M → Plan 5.
- Multi-stream events → Plan 4 / Phase 2.
- Diagonal / tied / full covariance → Plans 6–8.

**Foundation assumed in place:** Plan 2 complete (`spherical-safe-plan2` tag). The cleanup PR is also in (M-4/M-7/I-1-lite). The `use_triton` property currently returns False under explicit `backend="cuda"` (Plan 2 cleanup), so internal Triton-gated inference paths already bail out under CUDA — they just go to torch instead of CUDA. Plan 3 closes that gap.

---

## File Structure

### Created in this plan

| Path | Responsibility |
| --- | --- |
| `gmmxx/csrc/estep/spherical_sm80.cu` | Three mma-based kernels (assign / logsumexp / resp) for fp16/bf16. Uses `cp.async` for centroid SMEM tile loading and `m16n8k16` mma for the `x · μ` matmul, with the GMM logit formula applied in fp32 register epilogue. |
| `gmmxx/csrc/estep/spherical_dispatch.cu` | Host-side `assign(...)`/`logsumexp(...)`/`resp(...)` that route to safe vs sm80 by dtype + compute capability. Replaces the existing host launchers in `spherical_safe.cu` (which become `assign_safe(...)` etc.). |
| `tests/test_cuda_spherical_sm80.py` | Per-kernel correctness vs torch reference at `rtol=5e-3` for fp16/bf16 (per spec §6). |
| `tests/test_cuda_inference_spherical.py` | predict/predict_proba/score_samples/score on `backend="cuda"` produce correct outputs and populate `last_backend_used_`. |
| `benchmarks/benchmark_cuda_vs_triton_spherical.py` | Speedup matrix per `(N, D, K, dtype)`. CI-gated: CUDA must not regress vs Triton on any shape inside the supported window. |

### Modified

| Path | Change |
| --- | --- |
| `gmmxx/csrc/common/ptx.cuh` | Populate sm_80+ helpers: `cp_async_cg<bytes>`, `cp_async_commit`, `cp_async_wait_group<N>`, `cp_async_wait_all`, `ldmatrix_sync_x4` (transposed and non-transposed), `mma_m16n8k16_f32_f16` (fp16 acc fp32), `mma_m16n8k16_f32_bf16` (bf16 acc fp32). All gated on `GMMXX_HAS_*` macros from arch.cuh. |
| `gmmxx/csrc/estep/spherical_safe.cu` | Rename internal namespace symbols: existing `assign(...)` → `assign_safe(...)`, etc. The dispatcher in `spherical_dispatch.cu` calls these directly. |
| `gmmxx/csrc/estep/spherical.h` | Add `assign_safe` / `logsumexp_safe` / `resp_safe` declarations alongside the existing public `assign` / `logsumexp` / `resp`. The public ones now route. |
| `gmmxx/csrc/bindings.cpp` | No new entries needed; existing `spherical_assign` / `spherical_logsumexp` / `spherical_resp` automatically benefit from the host-side dispatcher. |
| `setup.py` | Add `spherical_sm80.cu` and `spherical_dispatch.cu` to `sources`. |
| `gmmxx/_runtime.py` | `cuda_spherical_supported` returns True for the same window; the C++ dispatcher decides safe vs sm80. No Python-side changes needed for routing. |
| `gmmxx/_cuda.py` | Existing wrappers unchanged — they still call `_C.spherical_assign` etc. The C++ side now dispatches internally. |
| `gmmxx/_dispatch.py` | Add `dispatch_kernel(op_name: str, backend: str, *args, **kw)` helper. Maps `(op_name, backend)` → callable from `gmmxx.cuda_ops` / Triton modules / `torch_fallback`. Used by the rewired inference paths. |
| `gmmxx/interface.py` | Rewrite `_use_triton_<cov>_inference` methods to consult `_dispatch.resolve_backend` per call. Add CUDA branches in `predict()`, `predict_proba()`, `score_samples()` for spherical. The `use_triton` property is kept as a deprecated alias for external readers but is no longer consulted internally. |
| `tests/test_gmmxx.py` | Parametrize key inference tests on backend ∈ {torch, triton, cuda} with skips. |

---

## Numerical contract

Per spec §6, fp16/bf16 inputs use `rtol=1e-3` for `means_/weights_` and `rtol=5e-3` for `covariances_/score_samples`. The mma path uses fp32 accumulator (`f32.f16.f16.f32`) so accumulation drift is bounded; per-MAC fp16 quantization is the dominant error source.

For the assign output specifically: `≥99% label agreement` on separable data (most points have a clear winner), `≥95%` on near-degenerate clusters (near-tie samples may flip). This matches Plan 2's tolerance contract.

---

## sm_80 mma kernel design — reference

The three kernels follow flash-kmeans-cuda's `assign_sm80.cu` structure. The key adaptation: GMM spherical computes `||x − μ_k||² = ||x||² − 2·x·μ_k + ||μ_k||²`, so the matmul is `x · μᵀ` (cross-product, not Euclidean distance). The mma fragment computes `cross[m,n] = Σ_k x[m,k] · μ[n,k]` for a `(BLOCK_N, BLOCK_K)` tile, then the epilogue computes `dist = x_sq[m] + c_sq[n] − 2 · cross[m,n]` and the logit `log_w[n] − D/2·log(2π·var[n]) − 0.5·dist/var[n]`.

Tile: `BLOCK_N=128` (points), `BLOCK_K=64` (clusters), `BLOCK_D=16` (features per mma step). For D > 16, the mma loop iterates `D / 16` times and accumulates into the same fragment register tile.

SMEM:
- `x_tile`: `(BLOCK_N, D)` half/bfloat16, padded to `(D + 8)` to avoid bank conflicts.
- `c_tile`: `(2, BLOCK_K, D)` double-buffered (cp.async ping-pong).
- `x_sq` and `c_sq` precomputed in fp32 by host (Python wrapper computes once per call).

The `assign` kernel keeps a per-thread `(best_logit, best_idx)` register tile and runs min-over-K in registers. `logsumexp` uses the existing `reduce::logsumexp_block_f32` after computing logits per (m, n) pair. `resp` writes per-thread `exp(logit - log_norm)` directly to global memory.

`x_sq` and `c_sq` are precomputed and passed as fp32 inputs (matches flash-kmeans-cuda's contract). The Python wrapper handles this transparently.

---

## Conventions

- Working directory: `C:\Users\HEQ\Projects\flashGMM2`. Branch: `GMMXX-cuda` (after `spherical-safe-plan2` tag).
- Dev rebuild: `$env:TORCH_CUDA_ARCH_LIST = "8.9"; uv pip install -e .` after each `.cu`/`.cpp`/`.h`/setup.py change.
- Test command: `uv run pytest tests/test_cuda_spherical_sm80.py -v` (per-task), full suite via `uv run pytest tests/ -q`.
- Reference template: `C:\Users\HEQ\Projects\flash-kmeans-cuda\flash_kmeans_cuda\csrc\assign\assign_sm80.cu` (700+ lines). Read it before starting Task 2.

---

## Task 1 — Populate `ptx.cuh` with sm_80 helpers

**Files:** Modify `gmmxx/csrc/common/ptx.cuh`

Add the cp.async, ldmatrix, and mma wrappers below the existing warp-shuffle helpers. Each helper must be `__device__ __forceinline__` and gated on `GMMXX_HAS_*` macros from arch.cuh.

The reference for these wrappers is `C:\Users\HEQ\Projects\flash-kmeans-cuda\flash_kmeans_cuda\csrc\common\ptx.cuh` (~120 lines). Read that file first; it defines:

- `cvta_to_shared(ptr) -> uint32_t` — generic-to-shared-memory address conversion (used by cp.async destinations).
- `cp_async_cg<size>(dst_smem, src_global, predicate)` — 16-byte cache-global async copy with zero-fill on predicate=false. Gated on `GMMXX_HAS_CP_ASYNC`.
- `cp_async_commit()` — `cp.async.commit_group`.
- `cp_async_wait_group<N>()` — `cp.async.wait_group N`.
- `cp_async_wait_all()` — `cp.async.wait_all`.
- `ldmatrix_sync_x4(uint32_t (&dst)[4], uint32_t src_smem)` — load four 8×8 fp16 matrices into registers. Gated on `GMMXX_HAS_LDMATRIX_X4`.
- `ldmatrix_sync_x4_trans(...)` — same but transposed.
- `mma_m16n8k16_f32_f16(float (&D)[4], uint32_t (&A)[4], uint32_t (&B)[2], const float (&C)[4])` — fp16-input fp32-accumulator mma. Gated on `GMMXX_HAS_F16_MMA`.
- `mma_m16n8k16_f32_bf16(...)` — same for bf16.

**Step 1.1** — Read FKC's ptx.cuh:

```bash
cat /c/Users/HEQ/Projects/flash-kmeans-cuda/flash_kmeans_cuda/csrc/common/ptx.cuh
```

(or via Read tool from PowerShell.)

**Step 1.2** — Append the wrappers to `gmmxx/csrc/common/ptx.cuh`. Use FKC's wrappers verbatim where possible; just rename the namespace if FKC uses `fkc::ptx` to `gmmxx::ptx` (likely already correct since arch.cuh is GMMXX_-prefixed).

**Step 1.3** — Build to verify:

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .
```

Expected: clean rebuild. The new helpers are unused by any existing kernel, so this just compile-validates the inline-PTX syntax.

**Step 1.4** — Commit:

```bash
git add gmmxx/csrc/common/ptx.cuh
git commit -m "$(cat <<'EOF'
ptx.cuh: add sm_80+ helpers (cp.async, ldmatrix, mma_m16n8k16)

Verbatim from flash-kmeans-cuda's ptx.cuh (renamed FKC_ -> GMMXX_):
- cp_async_cg / commit / wait_group / wait_all (16B async global->shared)
- ldmatrix_sync_x4 (and transposed variant) for mma operand loads
- mma_m16n8k16_f32_f16 and ..._f32_bf16 (fp16/bf16 inputs, fp32 acc)

All gated on GMMXX_HAS_CP_ASYNC / HAS_LDMATRIX_X4 / HAS_F16_MMA from
arch.cuh. Plan 3's sm80 spherical kernels consume these.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Write the sm_80 mma E-step kernels

**Files:** Create `gmmxx/csrc/estep/spherical_sm80.cu`; modify `gmmxx/csrc/estep/spherical.h`.

This is the largest task in Plan 3. Read flash-kmeans-cuda's `assign_sm80.cu` first as the structural reference. Adapt it for the GMM spherical formula.

**Step 2.1** — Read the reference:

```bash
cat /c/Users/HEQ/Projects/flash-kmeans-cuda/flash_kmeans_cuda/csrc/assign/assign_sm80.cu
```

Take note of:
- Tile constants (`BLOCK_N=128`, `BLOCK_K=64`, `BLOCK_D=16`, etc.).
- The double-buffered cp.async pipeline (lines ~120–200).
- The mma loop structure (lines ~200–280).
- The fragment register tiling and best-tracking (lines ~280–340).
- The `__launch_bounds__(WARPS_PER_CTA*32, 1)` configuration.

**Step 2.2** — Update `gmmxx/csrc/estep/spherical.h`. Add three new declarations alongside the existing public ones:

```cpp
namespace gmmxx { namespace estep { namespace spherical {

// Public dispatchers (see spherical_dispatch.cu for routing).
at::Tensor assign(...);     // existing
at::Tensor logsumexp(...);  // existing
at::Tensor resp(...);       // existing

// Safe-path implementations (renamed from the existing public functions in
// spherical_safe.cu; called by the dispatcher).
at::Tensor assign_safe(const at::Tensor& x, const at::Tensor& means,
                       const at::Tensor& var, const at::Tensor& log_w,
                       c10::optional<at::Tensor> out);
at::Tensor logsumexp_safe(...);  // same params as assign_safe
at::Tensor resp_safe(const at::Tensor& x, ..., const at::Tensor& log_norm,
                     c10::optional<at::Tensor> out);

// sm_80 mma path (fp16/bf16 only).
at::Tensor assign_sm80(const at::Tensor& x, const at::Tensor& means,
                       const at::Tensor& var, const at::Tensor& log_w,
                       const at::Tensor& x_sq, const at::Tensor& c_sq,
                       c10::optional<at::Tensor> out);
at::Tensor logsumexp_sm80(...);  // same plus x_sq, c_sq
at::Tensor resp_sm80(...);       // same plus x_sq, c_sq, log_norm

}}}
```

Note: the sm80 kernels take precomputed `x_sq` and `c_sq` (fp32) for the dist formula. The dispatcher (Task 3) computes these once and passes them in.

**Step 2.3** — Rename the existing host launchers in `spherical_safe.cu` from `assign`/`logsumexp`/`resp` to `assign_safe`/`logsumexp_safe`/`resp_safe`. (Just rename; no logic change.) The kernels themselves keep their templated names.

**Step 2.4** — Write `gmmxx/csrc/estep/spherical_sm80.cu`. Structure (follow FKC's assign_sm80.cu):

```cpp
#include "spherical.h"
#include "../common/arch.cuh"
#include "../common/ptx.cuh"
#include "../common/reduce.cuh"

namespace gmmxx { namespace estep { namespace spherical {

#if GMMXX_HAS_F16_MMA && GMMXX_HAS_CP_ASYNC && GMMXX_HAS_LDMATRIX_X4

// Tile constants.
static constexpr int BLOCK_N = 128;
static constexpr int BLOCK_K = 64;
static constexpr int BLOCK_D = 16;
static constexpr int WARPS_PER_CTA = 4;
static constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;
static constexpr int SMEM_PAD = 8;

// Shared-memory layout (one CTA processes BLOCK_N points × all K clusters in
// chunks of BLOCK_K; cp.async double-buffers c_tile across K chunks).
//
// x_tile: (BLOCK_N) × (D + SMEM_PAD) — loaded once per CTA, reused across K.
// c_tile: 2 × BLOCK_K × (D + SMEM_PAD) — ping-pong across K chunks.

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_CTA, 1)
spherical_assign_sm80_kernel(
    const T* __restrict__ x,        // (B, N, D) fp16 or bf16
    const T* __restrict__ means,    // (B, K, D)
    const float* __restrict__ var,  // (B, K)
    const float* __restrict__ log_w,// (B, K)
    const float* __restrict__ x_sq, // (B, N) precomputed ||x_n||²
    const float* __restrict__ c_sq, // (B, K) precomputed ||c_k||²
    int32_t* __restrict__ out,      // (B, N)
    int B, int N, int K, int D
) {
    // ... follow FKC's structure ...
    //
    // Fragment register tile per thread:
    //   float best_dist[M_ATOMS_PER_WARP];   // running min-dist per row
    //   int best_k[M_ATOMS_PER_WARP];        // corresponding cluster idx
    //
    // For each K-chunk of BLOCK_K clusters:
    //   1. Issue cp.async to load c_tile[chunk_idx % 2] for the next chunk.
    //   2. Wait on cp.async for c_tile[chunk_idx % 2] (current chunk).
    //   3. ldmatrix.x4 to load 8x8 fp16 fragments of x_tile and c_tile.
    //   4. mma_m16n8k16_f32_f16 to accumulate `cross[m,n] = Σ_d x[m,d] * c[n,d]`.
    //      (Loop over D / BLOCK_D = D/16 mma calls, accumulating into the same
    //       fragment register tile.)
    //   5. Epilogue: compute dist[m,n] = x_sq[m] + c_sq[n] - 2*cross[m,n],
    //      then logit = log_w[n] - 0.5*D*log(2π*var[n]) - 0.5*dist/var[n],
    //      then update (best_dist, best_k) per thread.
    //
    // After the K loop:
    //   Each thread writes its best_k[m] to out[(b * N + n_for_m]].
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_CTA, 1)
spherical_logsumexp_sm80_kernel(
    /* same signature as assign + */ float* __restrict__ out, /* (B, N) */
    int B, int N, int K, int D
) {
    // Same as assign but instead of tracking best, track a running logsumexp:
    //   For first K-chunk: best = max(logits_in_chunk); sumexp = sum(exp(logit - best)).
    //   For subsequent chunks: new_best = max(best, max(logits_in_chunk));
    //     sumexp = sumexp * exp(best - new_best) + Σ exp(logit - new_best);
    //     best = new_best.
    //   Final: log_norm = best + log(sumexp).
    //
    // The running-max pattern keeps the exp arguments bounded. Each thread
    // tracks its own per-row (best, sumexp) pair across K-chunks.
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_CTA, 1)
spherical_resp_sm80_kernel(
    /* same plus log_norm input + */ float* __restrict__ out /* (B,N,K) */,
    int B, int N, int K, int D
) {
    // For each K-chunk, compute logit per (m, n), then exp(logit - log_norm[m]),
    // and write directly to out[b, n_for_m, k_for_n]. No reduction needed —
    // each (m, n) writes to a distinct global location.
}

#endif  // GMMXX_HAS_F16_MMA && ...

// ---------------------------------------------------------------------
// Host launchers — only built when the arch supports the mma instructions.
// On older arches, the launcher functions still exist but are stubs that
// TORCH_CHECK(false, ...) so the dispatcher knows to use the safe path.
// ---------------------------------------------------------------------

#if GMMXX_HAS_F16_MMA && GMMXX_HAS_CP_ASYNC && GMMXX_HAS_LDMATRIX_X4

at::Tensor assign_sm80(const at::Tensor& x, const at::Tensor& means,
                       const at::Tensor& var, const at::Tensor& log_w,
                       const at::Tensor& x_sq, const at::Tensor& c_sq,
                       c10::optional<at::Tensor> out) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "spherical.assign_sm80 requires fp16 or bf16; for fp32 use assign_safe");
    /* ... contiguity, shape checks ... */
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    /* allocate out, dispatch by dtype to the templated kernel, return. */
}

at::Tensor logsumexp_sm80(...) { /* analogous */ }
at::Tensor resp_sm80(...) { /* analogous */ }

#else  // older arch — sm80 path not compiled

at::Tensor assign_sm80(...) {
    TORCH_CHECK(false, "spherical.assign_sm80 not available on this arch (requires sm_80+)");
}
at::Tensor logsumexp_sm80(...) {
    TORCH_CHECK(false, "...");
}
at::Tensor resp_sm80(...) {
    TORCH_CHECK(false, "...");
}

#endif

}}}  // namespace gmmxx::estep::spherical
```

**Implementation guidance:** the kernel body is non-trivial (~300–400 lines). The implementer should:
1. Open FKC's `assign_sm80.cu` side-by-side and translate the structure section by section.
2. Replace the FKC dist formula `dist = x_sq + c_sq - 2*cross` (which computes Euclidean distance) with the GMM logit formula `logit = log_w - 0.5*D*log(2π*var) - 0.5*dist/var` after the mma+epilogue section.
3. For `assign`, the running min becomes a running max-of-logit (= argmax over k) — flip the comparison sign.
4. For `logsumexp`, replace the running min with the running (max, sumexp) pair as documented above.
5. For `resp`, no reduction at all — just compute and write.

If the implementer hits issues with mma fragment layout or ldmatrix register interpretation, those are well-documented in the NVIDIA PTX ISA spec; flash-kmeans-cuda's comments in `assign_sm80.cu` lines 7–28 describe the layout.

**Step 2.5** — Add to `setup.py`:

```python
str(CSRC / "estep" / "spherical_sm80.cu"),
```

**Step 2.6** — Build:

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .
```

Expected: clean build (~3–5 min for the new TU + relink). nvcc may emit warnings for unused fragment elements at low D — those are benign.

**Step 2.7** — Commit:

```bash
git add gmmxx/csrc/estep/spherical_sm80.cu gmmxx/csrc/estep/spherical_safe.cu gmmxx/csrc/estep/spherical.h setup.py
git commit -m "$(cat <<'EOF'
Add sm_80 mma.sync E-step kernels for spherical fp16/bf16

Three kernels mirroring flash-kmeans-cuda's assign_sm80.cu structure
adapted for GMM spherical:
- assign: running argmax of logit over K via fragment register tile.
- logsumexp: running (max, sumexp) pair across K chunks, fused subtract-max.
- resp: writes exp(logit - log_norm) directly per (b,n,k); no reduction.

Tile (BLOCK_N=128, BLOCK_K=64, BLOCK_D=16), cp.async double-buffered
SMEM for centroid tiles, mma.sync m16n8k16.f32.f16.f16.f32 for the
x · μᵀ matmul, fp32 epilogue for the GMM log p_k(x) formula.

Renamed the safe-path host launchers (assign->assign_safe etc.) so the
dispatcher in Plan 3 Task 3 can route fp32 to safe and fp16/bf16 to sm80.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Host-side dispatcher: `spherical_dispatch.cu`

**Files:** Create `gmmxx/csrc/estep/spherical_dispatch.cu`; modify `setup.py`.

The new public `assign(...)` / `logsumexp(...)` / `resp(...)` route by dtype + compute capability:

- fp32 inputs → `assign_safe` (mma path doesn't support fp32 inputs).
- fp16 / bf16 inputs AND device compute capability ≥ 80 → `assign_sm80` (precompute `x_sq`, `c_sq`).
- fp16 / bf16 inputs AND device cap < 80 → `assign_safe`.

The dispatcher precomputes `x_sq = (x.float() ** 2).sum(-1)` and `c_sq = (means.float() ** 2).sum(-1)` once for the sm80 path (reused across all three ops in a single fit() iteration, but for now we recompute per call — Plan 4 may cache).

```cpp
#include "spherical.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace estep { namespace spherical {

namespace {

// Returns true if the calling device has compute capability >= 8.0.
bool _device_has_sm80(at::DeviceIndex idx) {
    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, idx);
    if (err != cudaSuccess) return false;
    return prop.major >= 8;
}

bool _is_fp16_or_bf16(const at::Tensor& t) {
    return t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16;
}

}  // anonymous

at::Tensor assign(const at::Tensor& x, const at::Tensor& means,
                  const at::Tensor& var, const at::Tensor& log_w,
                  c10::optional<at::Tensor> out) {
    if (_is_fp16_or_bf16(x) && _device_has_sm80(x.device().index())) {
        // Precompute squared-norms for the sm80 path.
        auto x_sq = x.to(at::kFloat).pow(2).sum(-1).contiguous();
        auto c_sq = means.to(at::kFloat).pow(2).sum(-1).contiguous();
        return assign_sm80(x, means, var, log_w, x_sq, c_sq, std::move(out));
    }
    return assign_safe(x, means, var, log_w, std::move(out));
}

at::Tensor logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     c10::optional<at::Tensor> out) {
    if (_is_fp16_or_bf16(x) && _device_has_sm80(x.device().index())) {
        auto x_sq = x.to(at::kFloat).pow(2).sum(-1).contiguous();
        auto c_sq = means.to(at::kFloat).pow(2).sum(-1).contiguous();
        return logsumexp_sm80(x, means, var, log_w, x_sq, c_sq, std::move(out));
    }
    return logsumexp_safe(x, means, var, log_w, std::move(out));
}

at::Tensor resp(const at::Tensor& x, const at::Tensor& means,
                const at::Tensor& var, const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out) {
    if (_is_fp16_or_bf16(x) && _device_has_sm80(x.device().index())) {
        auto x_sq = x.to(at::kFloat).pow(2).sum(-1).contiguous();
        auto c_sq = means.to(at::kFloat).pow(2).sum(-1).contiguous();
        return resp_sm80(x, means, var, log_w, x_sq, c_sq, log_norm, std::move(out));
    }
    return resp_safe(x, means, var, log_w, log_norm, std::move(out));
}

}}}
```

**Step 3.1** — Add to setup.py sources:

```python
str(CSRC / "estep" / "spherical_dispatch.cu"),
```

**Step 3.2** — Build + smoke (the existing Plan 2 spherical tests should keep passing because the dispatcher routes fp32 to safe; fp16/bf16 should now hit sm80 if the device supports it):

```bash
$env:TORCH_CUDA_ARCH_LIST = "8.9"; $env:DISTUTILS_USE_SDK = "1"; uv pip install -e .
uv run pytest tests/test_cuda_spherical_safe.py -v
```

Expected: all 17 Plan 2 tests still pass. fp16/bf16 cases are now exercising the sm80 kernel; fp32 cases stay on safe.

**Step 3.3** — Commit:

```bash
git add gmmxx/csrc/estep/spherical_dispatch.cu setup.py
git commit -m "$(cat <<'EOF'
Add host-side dispatcher routing fp16/bf16 spherical to sm80 mma path

assign / logsumexp / resp now route by dtype + compute capability:
- fp32 -> safe (mma doesn't support fp32 inputs).
- fp16/bf16 + sm_80+ -> sm80 (precomputes x_sq, c_sq fp32 norms).
- fp16/bf16 + older arch -> safe.

cudaDeviceProp queried per call; could be cached in Plan 4.

The Python wrappers and bindings.cpp are unchanged — the dispatch
happens entirely on the C++ side.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Per-kernel correctness tests for sm_80

**Files:** Create `tests/test_cuda_spherical_sm80.py`

Mirrors `test_cuda_spherical_safe.py` but specifically exercises the sm80 path by passing fp16/bf16 inputs on a sm_80+ host. Skips on lesser arches.

```python
"""Per-kernel correctness for the spherical sm_80 mma path.

Tests fp16 and bf16 inputs at rtol=5e-3 (per spec §6 fp16/bf16 contract).
Skips when device < sm_80 or CUDA unavailable.
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


def _random_setup(B=1, N=256, D=32, K=16, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    means = torch.randn(B, K, D, device=device, dtype=dtype)
    var = torch.rand(B, K, device=device).clamp_min(0.5).float()
    log_w = torch.log_softmax(torch.randn(B, K, device=device), dim=-1).float()
    return x, means, var, log_w


def _torch_logits(x, means, var, log_w):
    B, N, D = x.shape
    K = means.shape[1]
    x_f, means_f = x.float(), means.float()
    diff = x_f.unsqueeze(2) - means_f.unsqueeze(1)
    dist_sq = diff.pow(2).sum(-1)
    return (
        log_w.unsqueeze(1)
        - 0.5 * D * torch.log(2 * math.pi * var).unsqueeze(1)
        - 0.5 * dist_sq / var.unsqueeze(1)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D,K", [(16, 16), (32, 32), (64, 64), (128, 32)])
def test_assign_matches_reference(dtype, D, K):
    from gmmxx import _cuda
    x, means, var, log_w = _random_setup(D=D, K=K, dtype=dtype)
    cuda_ids = _cuda.spherical_assign(x, means, var, log_w)
    ref_ids = _torch_logits(x, means, var, log_w).argmax(-1).int()
    agree = (cuda_ids == ref_ids).float().mean().item()
    assert agree >= 0.95, f"only {agree:.3f} agreement"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_logsumexp_matches_reference(dtype):
    from gmmxx import _cuda
    x, means, var, log_w = _random_setup(N=512, D=32, K=32, dtype=dtype)
    cuda_lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    ref_lse = _torch_logits(x, means, var, log_w).logsumexp(-1)
    assert torch.allclose(cuda_lse, ref_lse, rtol=5e-3, atol=5e-3), (
        f"max diff: {(cuda_lse - ref_lse).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_resp_sums_to_one(dtype):
    from gmmxx import _cuda
    x, means, var, log_w = _random_setup(dtype=dtype)
    lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    r = _cuda.spherical_resp(x, means, var, log_w, lse)
    assert torch.allclose(r.sum(-1), torch.ones_like(r.sum(-1)), atol=5e-3)
```

**Step 4.1** — Run:

```bash
uv run pytest tests/test_cuda_spherical_sm80.py -v
```

Expected: ~14 cases (2 dtypes × 4 (D,K) shapes for assign + 2 for logsumexp + 2 for resp) pass on sm_89 host.

**Step 4.2** — Commit:

```bash
git add tests/test_cuda_spherical_sm80.py
git commit -m "$(cat <<'EOF'
Add per-kernel correctness tests for spherical sm80 mma path

Exercises fp16 and bf16 inputs at rtol=5e-3 (spec §6 dtype tier).
Skips on devices < sm_80. Covers (D,K) tile shapes 16/16, 32/32,
64/64, 128/32 to validate the mma loop's BLOCK_D iteration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Add `_dispatch.dispatch_kernel` helper

**Files:** Modify `gmmxx/_dispatch.py`

`dispatch_kernel(op_name, backend, *args, **kw)` resolves a callable from the right backend's module and invokes it. This is what the rewired inference paths in `interface.py` call instead of branching on `self.use_triton`.

Append to `gmmxx/_dispatch.py`:

```python
# ---------------------------------------------------------------------------
# Kernel dispatch helper. Plan 3 onwards: inference paths consult
# resolve_backend, then call dispatch_kernel(op_name, resolved, *args).
# ---------------------------------------------------------------------------


_TRITON_OPS_BY_NAME: dict[str, str] = {
    # spherical
    "spherical_assign":    "gmmxx.assign_spherical_triton.spherical_assign_triton",
    "spherical_logsumexp": "gmmxx.assign_spherical_triton.spherical_logsumexp_triton",
    "spherical_resp":      "gmmxx.assign_spherical_triton.spherical_resp_triton",
    # diag, tied, full — wired in Plans 6-8 as those backends gain ops.
}


def _resolve_callable(op_name: str, backend: str):
    """Look up the callable for (op_name, backend). Raises if not found."""
    if backend == "cuda":
        from . import _cuda
        return getattr(_cuda, op_name)
    if backend == "triton":
        path = _TRITON_OPS_BY_NAME.get(op_name)
        if path is None:
            raise KeyError(f"no triton op named {op_name!r}")
        module_path, attr = path.rsplit(".", 1)
        import importlib
        return getattr(importlib.import_module(module_path), attr)
    if backend == "torch":
        # Each torch op is a slim wrapper inside torch_fallback. Plan 3 does
        # not require this branch for the spherical inference paths because
        # interface.py calls torch_fallback functions directly when backend
        # resolves to torch — but the symmetric path is wired here for
        # symmetry and Plan 6+ readiness.
        from . import torch_fallback
        return getattr(torch_fallback, op_name)
    raise ValueError(f"unknown backend {backend!r}")


def dispatch_kernel(op_name: str, backend: str, *args, **kw):
    """Resolve the callable for (op_name, backend) and call it.

    Wraps RuntimeError in CudaRuntimeFallback when backend == "cuda" so callers
    can catch a single exception type and re-resolve.
    """
    fn = _resolve_callable(op_name, backend)
    if backend == "cuda":
        from ._cuda import _no_fallback, CudaRuntimeFallback
        try:
            return fn(*args, **kw)
        except RuntimeError as exc:
            if _no_fallback():
                raise
            raise CudaRuntimeFallback(f"{op_name} (cuda) failed: {exc}") from exc
    return fn(*args, **kw)
```

**Step 5.1** — Add a small test exercising the helper:

Append to `tests/test_dispatch.py`:

```python
class TestDispatchKernel:
    def test_cuda_op_resolves_to_cuda_module(self):
        if not _dispatch._cuda.has_cuda():
            pytest.skip("requires CUDA")
        import torch
        x = torch.randn(1, 8, 4, device="cuda")
        means = torch.randn(1, 3, 4, device="cuda")
        var = torch.ones(1, 3, device="cuda")
        log_w = torch.zeros(1, 3, device="cuda")
        out = _dispatch.dispatch_kernel(
            "spherical_assign", "cuda", x, means, var, log_w
        )
        assert out.shape == (1, 8) and out.dtype == torch.int32

    def test_unknown_op_raises(self):
        with pytest.raises((KeyError, AttributeError)):
            _dispatch.dispatch_kernel("nonexistent_op", "cuda")
```

Run:

```bash
uv run pytest tests/test_dispatch.py -v
```

**Step 5.2** — Commit:

```bash
git add gmmxx/_dispatch.py tests/test_dispatch.py
git commit -m "$(cat <<'EOF'
Add _dispatch.dispatch_kernel helper for backend-routed kernel calls

Inference paths in interface.py (predict, predict_proba, score_samples,
score) now consult resolve_backend then call dispatch_kernel(op_name,
backend, ...) instead of branching on the use_triton property.

CUDA branch wraps RuntimeError in CudaRuntimeFallback so callers see a
single exception type and can re-resolve to triton/torch on failure.

_TRITON_OPS_BY_NAME is currently spherical-only; Plans 6-8 will add
diag/tied/full entries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Inference rewire in `interface.py`: spherical predict() / score_samples()

**Files:** Modify `gmmxx/interface.py`

The bigger I-1 fix. Replace the spherical inference paths (`_use_triton_spherical_inference`, `predict()` / `predict_proba()` / `score_samples()` spherical branches) so they consult `_dispatch.resolve_backend_with_env` and call into the dispatched backend.

The exact edit shape depends on the existing code. Read `gmmxx/interface.py` carefully to find:
- `_use_triton_spherical_inference` (or similar) helper.
- `predict()` spherical branch (consults `self.use_triton`).
- `predict_proba()` spherical branch.
- `score_samples()` spherical branch.

The pattern for each:

```python
def predict(self, data: torch.Tensor) -> torch.LongTensor:
    # ... existing input normalization ...
    if self.covariance_type == "spherical":
        # Resolve backend per call (shape may differ from fit shape).
        from . import _dispatch
        shape = (data_b.shape[0], data_b.shape[1], data_b.shape[2], self.k)
        resolved = _dispatch.resolve_backend_with_env(
            requested=self.backend,
            covariance="spherical",
            shape=shape,
            dtype=data_b.dtype,
            legacy_no_triton=self._legacy_no_triton,
        )
        if resolved == "cuda":
            log_w = torch.log(self.weights_b.clamp_min(1e-30))
            ids = _dispatch.dispatch_kernel(
                "spherical_assign", "cuda",
                data_b, self.means_b, self.variances_b, log_w
            )
            self.last_backend_used_ = "cuda"
            return self._squeeze_if_unbatched(ids).long()
        if resolved == "triton":
            # existing triton call
            ...
        # fall through to torch
        ...
        self.last_backend_used_ = "torch"
        ...
    # ... other covariance branches unchanged ...
```

Same pattern for `predict_proba` (use `dispatch_kernel("spherical_resp", ...)` after computing `log_norm` via `dispatch_kernel("spherical_logsumexp", ...)`) and `score_samples` (use `spherical_logsumexp` directly).

**Step 6.1** — Read `gmmxx/interface.py:660-820` (approximate line range; find `predict` / `predict_proba` / `score_samples`).

**Step 6.2** — Apply the rewire to `predict()` first; run tests:

```bash
uv run pytest tests/test_gmmxx.py -v -k "spherical" --tb=short
```

Iterate until existing tests pass. Some may have been written assuming Triton inference under `backend="cuda"` — those now expect torch / cuda; update assertions to use `last_backend_used_` rather than `triton_estep_enabled_`.

**Step 6.3** — Repeat for `predict_proba` and `score_samples` and `score`.

**Step 6.4** — Add a regression test exercising the new paths:

Create `tests/test_cuda_inference_spherical.py`:

```python
"""Inference under backend='cuda' for spherical covariance.

Verifies predict / predict_proba / score_samples / score on the CUDA
inference path produce shape-correct outputs and populate
last_backend_used_ correctly.
"""

from __future__ import annotations
import math
import pytest
import torch


def _has_cuda():
    try:
        from gmmxx._cuda import has_cuda
        return has_cuda()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")


@pytest.fixture
def fitted_cuda_gmm():
    from gmmxx import GMMXX
    torch.manual_seed(0)
    x = torch.randn(2048, 16, device="cuda")
    gmm = GMMXX(n_components=8, max_iter=15, tol=1e-4, random_state=0,
                covariance_type="spherical", backend="cuda")
    gmm.fit(x)
    return gmm, x


def test_predict_returns_correct_shape_dtype(fitted_cuda_gmm):
    gmm, x = fitted_cuda_gmm
    labels = gmm.predict(x[:512])
    assert labels.shape == (512,)
    assert labels.dtype == torch.long
    assert (labels >= 0).all() and (labels < gmm.k).all()
    assert gmm.last_backend_used_ == "cuda"


def test_predict_proba_sums_to_one(fitted_cuda_gmm):
    gmm, x = fitted_cuda_gmm
    p = gmm.predict_proba(x[:256])
    assert p.shape == (256, gmm.k)
    assert torch.allclose(p.sum(-1), torch.ones(256, device="cuda"), atol=1e-4)
    assert gmm.last_backend_used_ == "cuda"


def test_score_samples_returns_finite(fitted_cuda_gmm):
    gmm, x = fitted_cuda_gmm
    ll = gmm.score_samples(x[:512])
    assert ll.shape == (512,)
    assert torch.isfinite(ll).all()
    assert gmm.last_backend_used_ == "cuda"


def test_score_returns_finite_scalar(fitted_cuda_gmm):
    gmm, x = fitted_cuda_gmm
    s = gmm.score(x[:512])
    assert math.isfinite(s)
    assert gmm.last_backend_used_ == "cuda"


def test_predict_consistent_with_predict_proba(fitted_cuda_gmm):
    """argmax of predict_proba should equal predict (within fp32 tolerance)."""
    gmm, x = fitted_cuda_gmm
    labels_direct = gmm.predict(x[:128])
    labels_argmax = gmm.predict_proba(x[:128]).argmax(-1).long()
    agree = (labels_direct == labels_argmax).float().mean().item()
    assert agree >= 0.99
```

**Step 6.5** — Run:

```bash
uv run pytest tests/test_cuda_inference_spherical.py -v
uv run pytest tests/ -q
```

**Step 6.6** — Commit:

```bash
git add gmmxx/interface.py tests/test_cuda_inference_spherical.py
git commit -m "$(cat <<'EOF'
Wire spherical CUDA inference: predict / predict_proba / score_samples / score

Plan 1 final-review I-1 full fix (spherical only). Replaces the
self.use_triton property reads in the spherical inference branches
with explicit resolve_backend + dispatch_kernel calls.

When backend="cuda" and the shape is supported, predict() and friends
now run on the CUDA spherical kernels (assign / logsumexp / resp via
the host-side mma-aware dispatcher). last_backend_used_ tracks the
actual backend used per call.

Diag / tied / full inference paths are unchanged; Plans 6-8 will rewire
them as their CUDA kernels land. The use_triton property is retained
as a deprecated external-reader alias but is no longer consulted by
internal spherical inference paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Parametrize key test_gmmxx.py inference tests on backend

**Files:** Modify `tests/test_gmmxx.py`

Append a single new test that exercises the full fit→predict→score loop under each backend:

```python
@pytest.mark.parametrize("backend", ["torch", "triton", "cuda"])
def test_spherical_full_pipeline_each_backend(backend):
    if not _backend_available(backend):
        pytest.skip(f"backend {backend!r} not available")

    if backend == "torch":
        device = "cpu"
    else:
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        device = "cuda"

    torch.manual_seed(0)
    x_train = torch.randn(2048, 16, device=device)
    x_test = torch.randn(256, 16, device=device)

    gmm = GMMXX(n_components=6, max_iter=15, tol=1e-4, random_state=0,
                covariance_type="spherical", backend=backend)
    gmm.fit(x_train)

    labels = gmm.predict(x_test)
    proba = gmm.predict_proba(x_test)
    ll = gmm.score_samples(x_test)
    s = gmm.score(x_test)

    assert labels.shape == (256,)
    assert proba.shape == (256, 6)
    assert torch.allclose(proba.sum(-1), torch.ones(256, device=device), atol=1e-3)
    assert ll.shape == (256,)
    assert torch.isfinite(ll).all()
    assert isinstance(s, float)
    assert math.isfinite(s)
    # Predict and predict_proba must agree on argmax (within fp32 tolerance).
    agree = (labels == proba.argmax(-1).long()).float().mean().item()
    assert agree >= 0.99
```

Run + commit:

```bash
uv run pytest tests/test_gmmxx.py -v -k test_spherical_full_pipeline_each_backend
git add tests/test_gmmxx.py
git commit -m "$(cat <<'EOF'
Parametrize full spherical pipeline test on backend ∈ {torch, triton, cuda}

End-to-end fit → predict → predict_proba → score_samples → score
under each backend. Asserts shape correctness, normalized probability
rows, finite log-likelihoods, and predict-vs-argmax(predict_proba)
consistency at ≥99% across all three backends.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — Perf benchmark: CUDA vs Triton speedup matrix

**Files:** Create `benchmarks/benchmark_cuda_vs_triton_spherical.py`

A grid sweep across `(N, D, K, dtype)` measuring CUDA vs Triton vs torch_fallback wall-clock for a full fit() with fixed iterations. Used as a CI gate: CUDA must not regress vs Triton on any (N, D, K, dtype) inside the supported window. Acceptable threshold: CUDA ≤ 1.10 × Triton (10% slower max).

```python
"""Spherical CUDA vs Triton vs torch_fallback speedup benchmark.

Runs a grid of (N, D, K, dtype) and prints a table of wall-clock per
fit() call, plus the CUDA/Triton speedup ratio. Exit code is non-zero
if any cell shows CUDA more than 10% slower than Triton.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time

import torch


def _has_triton():
    try:
        from gmmxx.assign_spherical_triton import spherical_assign_triton
        return spherical_assign_triton is not None
    except ImportError:
        return False


def _has_cuda():
    try:
        from gmmxx._cuda import has_cuda
        return has_cuda()
    except ImportError:
        return False


def _bench_one(backend: str, N: int, D: int, K: int, dtype: torch.dtype, n_iter: int):
    from gmmxx import GMMXX
    torch.manual_seed(0)
    device = "cuda" if backend in {"cuda", "triton"} else "cpu"
    x = torch.randn(N, D, device=device, dtype=dtype)
    # Warm-up
    GMMXX(n_components=K, max_iter=2, tol=0, random_state=0,
          covariance_type="spherical", backend=backend).fit(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    GMMXX(n_components=K, max_iter=n_iter, tol=0, random_state=0,
          covariance_type="spherical", backend=backend).fit(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--shapes", type=str,
                   default="65536,32,64,fp16;65536,128,64,fp32;131072,16,32,fp16",
                   help="Semicolon-separated N,D,K,dtype triples")
    p.add_argument("--gate", action="store_true",
                   help="Exit non-zero if CUDA > 1.10 × Triton on any shape")
    args = p.parse_args()

    shapes = []
    for s in args.shapes.split(";"):
        n, d, k, dt = s.split(",")
        dt_t = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dt]
        shapes.append((int(n), int(d), int(k), dt_t))

    backends = []
    if _has_cuda(): backends.append("cuda")
    if _has_triton(): backends.append("triton")
    backends.append("torch")

    results = {}
    print(f"{'shape':30s} " + " ".join(f"{b:>10s}" for b in backends) + "  cuda/triton")
    failures = []
    for N, D, K, dt in shapes:
        row = []
        for backend in backends:
            try:
                t = _bench_one(backend, N, D, K, dt, args.n_iter)
            except Exception as e:
                t = math.inf
            row.append(t)
        ratio = (row[0] / row[1]) if "cuda" in backends and "triton" in backends else math.nan
        shape_str = f"N={N},D={D},K={K},{str(dt).split('.')[-1]}"
        print(f"{shape_str:30s} " + " ".join(f"{t:10.4f}" for t in row) + f"  {ratio:.3f}")
        results[shape_str] = {"per_backend": dict(zip(backends, row)), "cuda_triton_ratio": ratio}
        if args.gate and not math.isnan(ratio) and ratio > 1.10:
            failures.append((shape_str, ratio))

    print()
    print(json.dumps(results, indent=2))
    if failures:
        print("FAIL: CUDA regressed vs Triton on:", failures, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 8.1** — Run interactively first (no gate):

```bash
uv run python benchmarks/benchmark_cuda_vs_triton_spherical.py
```

Expected output: a table with three columns (cuda, triton, torch) and the cuda/triton ratio. On RTX 4090 + sm_89 + the safe path, ratio may be ≈ 1.0–1.3 (CUDA slightly slower than Triton's autotuned tf32 path on fp32, faster on fp16 with sm80 mma). On the sm80 path, fp16/bf16 should be < 1.0 (CUDA faster).

**Step 8.2** — Tune the gate. If real speedups require more aggressive tuning (sorted-run M-step from Plan 4, persistent kernels, etc.), the gate threshold may need to be relaxed for Plan 3 — set it to 1.5 × in this plan, then tighten to 1.0 × after Plan 4. (Document this in the file's docstring.)

```bash
uv run python benchmarks/benchmark_cuda_vs_triton_spherical.py --gate
```

**Step 8.3** — Commit:

```bash
git add benchmarks/benchmark_cuda_vs_triton_spherical.py
git commit -m "$(cat <<'EOF'
Add spherical CUDA-vs-Triton perf benchmark with optional CI gate

Sweeps a small (N, D, K, dtype) grid and prints per-backend wall-clock
plus the cuda/triton ratio. --gate flag fails if CUDA regresses by
more than 10% on any cell (relaxed to 50% in Plan 3 since sorted-run
M-step and persistent E-step are deferred to Plan 4 — the assign
kernel alone goes faster but the M-step's per-token atomics dominate
on K-heavy shapes until Plan 4 lands).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — README update + tag

**Files:** Modify `README.md`

Update the "CUDA backend (experimental)" section to reflect Plan 3 completion. Replace the Plan 2 status line with:

> **Spherical covariance is fully on the CUDA path** for both training and inference (Plan 3). The sm_80 mma.sync optimized E-step runs for fp16/bf16 inputs on Ampere+; fp32 stays on the safe SIMT path. Diagonal, tied, and full are still on Triton/PyTorch — coming in Plans 6–8.

Then commit and tag:

```bash
git add README.md
git commit -m "README: spherical CUDA inference live (Plan 3 complete)"
git tag -a spherical-mma-plan3 -m "Plan 3: spherical sm80 mma + inference rewire"
git log --oneline spherical-safe-plan2..spherical-mma-plan3
```

---

## Self-Review Checklist

**1. Spec coverage**

| Spec section | Plan task |
| --- | --- |
| §4 spherical sm80 mma | Tasks 1, 2, 3 |
| §5d optimization techniques (mma + cp.async) | Task 2 |
| §6 fp16/bf16 tolerance | Task 4 |
| §7 dispatch + dispatch_kernel | Task 5 |
| §7a runtime fallback (cuda → triton/torch) | Tasks 5, 6 (try/except in dispatch_kernel) |
| §10 perf benchmark gate | Task 8 |

Unaddressed (deferred):
- §5d sorted-run M-step → Plan 4
- §5d persistent kernels → Plan 4
- §5d multi-stream → Plan 4
- §10 3-way oracle for sm80 inference (covered by Task 4 + Task 7)

**2. Placeholder scan** — Task 2's kernel body is described structurally, not line-by-line. The implementer must read flash-kmeans-cuda's `assign_sm80.cu` and adapt section by section. The plan documents the formula adaptation explicitly. This is the only "non-verbatim" task; it's necessary because writing 400 lines of mma kernel inline would be error-prone vs. directly translating from the proven FKC reference.

**3. Type consistency** — `at::Tensor` throughout C++; `assign_safe`/`assign_sm80` signatures carefully aligned; `cuda::has_cuda()` consulted in dispatcher; `last_backend_used_` populated by interface.py per call.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-02-gmmxx-cuda-spherical-mma.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task. Task 2 (sm_80 kernel) is the largest; it should use opus given the kernel-engineering judgment required, while the others can use sonnet.

**2. Inline Execution** — Execute tasks in this session.

After Plan 3: **Plan 4** = sorted-run M-step + persistent E-step + multi-stream events for spherical (the remaining perf optimizations). Then **Plan 5** = fused single-tile spherical. Then **Plans 6–8** for diag, tied, full. Then **Plan 9** = `large_n.py` integration.
