# GMMXX CUDA Backend — Plan 12: Non-Spherical Performance Pass

> **For agentic workers:** this is a performance plan, not a covariance
> enablement plan. Keep correctness tests green and require benchmark evidence
> before adding broad native kernels.

**Goal:** Improve CUDA performance for `diag` and `tied` covariance after Plan
9 restored soft-EM semantics. Avoid speculative full-covariance native kernels
until benchmark data justifies them.

---

## Scope

### Deliverable A — Benchmark Harness

Add `benchmarks/benchmark_cuda_covariances.py`:

- Runs `backend="cuda"`, `backend="triton"`, and `backend="torch"`.
- Covers `diag`, `tied`, and `full`.
- Reports fit time, prediction time, lower bound, `last_backend_used_`,
  `cuda_*_enabled_` flags, and fallback reason.
- Supports `--json` and `--gate`, but Plan 12 starts with informational gates
  only. Hard pass/fail thresholds come after baseline data is collected.

### Deliverable B — Diag sorted-run hard-stat kernel support

This is only useful for code paths that still need hard labels
(`blocked_update_diag` low-level API and possible future hard/fused variants).
It should not replace Plan 9 soft EM in `GMMXX.fit()`.

- Add `blocked_update_diag_sorted` by adapting spherical sorted-run.
- Thread `force_sort` through `_cuda.blocked_update_diag`.
- Add direct tests comparing sorted vs per-token counts/sums/sumsq.

### Deliverable C — Diag sm80 E-step design spike

Do not implement the full sm80 diag kernel until a small prototype validates
the formulation. The diag logit can be expressed as:

```text
logit = log_w - 0.5 * (
    x^2 @ precision.T - 2 * x @ (mean * precision).T
    + mean_precision_mean
    + D * log(2π)
    + logdet
)
```

Plan 12 should add a design note plus a microbenchmark proving whether
precomputing `precision`, `weighted_means`, `mean_precision_mean`, and
`logdet` is enough to beat the current safe kernel before committing to PTX.

### Deliverable D — Tied fused projected update design spike

Tied currently projects `x` and `means` repeatedly, then calls spherical E-step
helpers. A fused tied update should:

- compute projected logits in the `L⁻¹` space;
- accumulate original-space `nk` and `sum_x` using soft responsibilities;
- leave covariance finalization in `_cuda.tied_finalize`.

Plan 12 should benchmark current tied CUDA and produce the kernel interface,
shape gates, and correctness oracle before implementation.

---

## Non-goals

- Native full-covariance kernels.
- Reverting Plan 9 soft EM.
- Replacing Python/torch soft-stat accumulation with native kernels before
  benchmark evidence exists.
- Hard performance gates in CI.

---

## Tasks

1. Add `benchmarks/benchmark_cuda_covariances.py`.
2. Add direct `blocked_update_diag_sorted` CUDA binding and wrapper.
3. Add `tests/test_cuda_diag_sorted.py`.
4. Add `docs/superpowers/specs/2026-05-04-diag-sm80-design-note.md`.
5. Add `docs/superpowers/specs/2026-05-04-tied-fused-design-note.md`.
6. Run focused tests, full suite, and at least one informational benchmark row.
7. Commit and tag `nonspherical-perf-plan12`.

---

## Risk Notes

- Sorted-run can lose to sort/gather overhead below large `N*K`; keep the same
  heuristic style as spherical and include a forced test knob.
- Diag sm80 has fp16/bf16 numerical risk around near-tied logits. Safe kernels
  remain the fallback.
- Tied fused update can easily mix projected-space logits with original-space
  stats incorrectly; correctness tests must compare `nk`, `sum_x`, and log
  likelihood against a soft-EM torch reference.
