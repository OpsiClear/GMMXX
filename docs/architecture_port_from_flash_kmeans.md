# Architecture port from flash-kmeans-cuda

flash-kmeans-cuda's spherical-assign dispatcher ships ~70 kernel variants
behind a small policy table and an in-memory autotune cache, and lets users
override the chosen variant via `FKC_*` env vars. GMMXX's spherical sm80
dispatcher currently hardcodes a 2- (now 3-) variant ladder. This doc tracks
what has been ported and what is the highest-leverage next step.

## Patterns surveyed

flash-kmeans-cuda files inspected (under
`flash_kmeans_cuda/csrc/assign/`):

- `assign_variants.h` — `Variant` struct: `name`, `try_fp16`, `try_bf16`,
  `smem`. `make_variant<BN,BK,WARPS,STAGES,NT,DFix,Raw,KMin>` produces a
  `constexpr Variant` for one kernel shape. ~70 variants enumerated this
  way; binary cost = template instantiations only.
- `assign_policy.h` / `assign_policy.cu` — `(dtype_idx, d_idx, k_bucket) ->
  PolicyRow` (8-slot ordered candidate list). `d_idx` enumerates D = 1..16,
  64, 96, 128, 192, 224, 256, 320, 384, OTHER. `k_bucket` ∈ {tiny<128,
  small<512, med<2048, large<8192, mega}.
- `assign_autotune.{h,cu}` — `AutotuneCache` keyed by
  `(dtype_idx, d_idx, k_bucket)`. First call to each cell probes the top-3
  SMEM-feasible candidates with `cudaEvent` timing (3 iters, min), then
  reorders the row by measured time. Hot path is `std::atomic<bool>`
  lock-free. Probe is mutexed.
- `assign_autotune.cu :: time_one_launch` — falls through if a candidate
  returns false (e.g. SMEM exceeded). Autotune skips those.
- `EnvKnobs` (FKC_NTILES, FKC_WIDE3, FKC_W4, FKC_NARROW, FKC_DSLAB,
  FKC_AUTOTUNE) — every override path is one switch on a bool. Force-mode
  builds a custom `VariantView` and bypasses the policy.

## Ported (Exp59)

`gmmxx/csrc/estep/spherical_sm80.cu` now instantiates a third tile shape
`(BLOCK_N=128, BLOCK_K=32, WARPS=4)` for assign / logsumexp / resp, and the
dispatcher prefers it when `K <= 32`. The original ladder was
`(128, 64, 4) -> (64, 32, 4)`; the new one is
`(128, 32, 4) [K<=32 only] -> (128, 64, 4) -> (128, 32, 4) -> (64, 32, 4)`.

Why this specific shape: at K=32 the (128, 64, 4) tile half-fills `c_smem`,
which costs occupancy. Halving BK to 32 keeps the wide BLOCK_N=128 grid
(half the CTAs vs (64, 32, 4)) while restoring SMEM headroom.

Bench impact (xl_grid 30iter): D=16 K=32 fp16 cuda/triton ratio
~0.82 -> ~0.78. The xl_grid max-ratio metric is dominated by D=32 K=64 noise
and so does not move.

## Not yet ported (high leverage, listed by ROI)

1. **Variant + PolicyRow data structures**. The current `if (bn == ... &&
   bk == ...)` ladder repeats six times across dispatchers. A
   `make_spherical_variant<BN,BK,W>(...)` and a flat
   `(dtype, D_idx, K_bucket) -> array<const Variant*, MAX_CAND>` table would
   collapse those branches and let new variants land in one line.
2. **AutotuneCache**. Drop-in copy of
   flash-kmeans-cuda/`assign_autotune.cu` keyed by `(dtype, D, K_bucket)`
   for spherical. Removes the need to hand-code per-shape dispatch (e.g.
   the K<=32 cutoff in Exp59 would be discovered automatically). Gated
   behind `GMMXX_AUTOTUNE=1` so CI runs deterministic.
3. **Persistent kernel (NT)**. flash-kmeans-cuda's `NT` template parameter
   makes one CTA process NT BLOCK_N tiles, halving (or quartering) the CTA
   grid. At our bench bottleneck (N=4M, BLOCK_N=128) the grid is 32K CTAs;
   NT=4 cuts launch + tail overhead and is consistently the autotune
   winner in flash-kmeans-cuda for the analogous shapes. Requires a
   non-trivial kernel-body refactor (the per-CTA prologue currently
   assumes a single tile).
4. **D-fixed / padded-D variants**. flash-kmeans-cuda has separate kernels
   for D=16, 32, ..., 384 with `Raw=true` (no D_SMEM padding) and
   `D_FIXED` template arg. For our `D=32` and `D=128` benches, D-fixed
   could shave a few percent by avoiding the runtime D-loop.
5. **EnvKnob overrides**. `GMMXX_SPHERICAL_FORCE_TILE=128x32x4` etc. — useful
   for debugging perf regressions and writing per-shape tests.

## Where the spherical-bench wall lives

After Exp55, Exp56, Exp59 the cuda/triton ratio at xl_grid sits at ~0.88
median. Profile data (`30 iter, max_iter=30`, N=4M D=16 K=32 fp16):

- Per-iter wall clock ~6.5 ms (212 ms total)
- Raw `_cuda.soft_update_spherical` loop without GMMXX wrapper: 2.8 ms/iter
- Wrapper overhead: ~3.7 ms/iter
- Within `soft_update_spherical`: spherical_logsumexp 0.59 ms, spherical_resp
  1.40 ms (both with cached x_sq/c_sq; without cache they are 5.0 / 4.5 ms)
- bmm M-step ~0.5 ms; finalize_spherical_soft ~0.05 ms

The wrapper overhead dominates the kernel time on the bottleneck shape.
The next big spherical-bench win is almost certainly a C++-side EM-loop
driver (one binding does N iters end-to-end, eliminating per-iter Python
dispatch), not another tile variant.

## Followup turn (Exp62 onwards)

That diagnosis turned out to be wrong. The "wrapper overhead" was CUDA
sync time attributed to `lse.mean().item()` in cProfile, not actual
Python dispatch — Python and GPU work overlap on async kernels, so
total wall time = max(Python time, GPU time), and GPU dominates.

### Bench harness fix (commit `1e85334`)

The xl_grid bench had labels like `4194304,16,32,fp16` but was
measuring the auto-promoted fp32 cuBLAS path, because
`GMMXX._compute_dtype_for_input` upcasts fp16/bf16 to fp32 unless the
caller passes `dtype=` explicitly. Pass `dtype=dtype` in the bench's
GMMXX construction so the cuda path also runs at the input dtype.
Metric drops 0.88 → 0.46 with no kernel changes.

### Spherical sm80 path (Exp62)

`spherical_logsumexp_resp_sm80` — single-pass mma kernel that runs the
GEMM once, online-reduces to LSE in registers, and writes
`resp = exp(logit - lse)` from the cached register tile. Replaces the
two-kernel `logsumexp + resp` pipeline. End-to-end fp16 fit at
N=4M D=16 K=32 went 196 ms -> 72 ms (2.7x). The xl_grid metric does
move once the bench harness is fixed.

Bonus: the existing `spherical_logsumexp_sm80` has a pre-existing
data-dependent numerical bug (LSE off by up to 8 at certain
seed-shape combinations). The fused kernel matches manual fp32
exactly. Original kept as the K > 64 fallback.

### C++ EM-loop driver (Exp63 — kept, not wired)

Built `gmmxx::em::spherical::soft_chunked` that runs the full n_iter
loop in C++. Same wall-clock as the Python loop (43 ms median) on the
bench bottleneck. Python and GPU work overlap; eliminating Python
dispatch doesn't reduce total time when GPU is the dominant cost.
Kept as available infrastructure for future cudaGraph capture.

### cuBLAS chunked path (Exp64–66)

After the bench fix, the xl_grid bottleneck moved to
`D=128 K=64 N=524k fp32`, which goes through the cuBLAS chunked path.

- **Exp64** — replace `mm(bf16) + cast + add(alpha)` with `addmm(bf16)`
  whose epilogue fuses the bias-add into the cuBLAS GEMM. 0.255 ms ->
  0.080 ms per chunk (3x). Metric 0.46 -> 0.42.
- **Exp65** — lower the chunked-path target from 32 MB to 16 MB.
  Sweep showed 16 MB best at 32.5 ms vs 34.8 ms at 32 MB and 40.3 ms
  at 64 MB. Per-chunk launch overhead dropped enough after Exp64 that
  smaller chunks win on L2 residency.
- **Exp66** — keep chunked path active when `compute_lse` or
  `compute_ids` is True (last iter of every fit). Pre-allocated
  `(B,N)` lse/ids buffers, fill per chunk. Avoids the full-N
  non-chunked fallback that was running once per fit. Metric
  0.42 -> 0.39.

Total this turn: bench harness fix + Exp62-66 took the xl_grid metric
from the misleading 0.88 to a real-fp16 0.39 (~2.3x compounded across
the cuda spherical paths once the bench measures what it claims).
