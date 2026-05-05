# GMMXX CUDA Backend — Plan 11: Spherical approx_top_k CUDA

> **For agentic workers:** use subagent-driven development or execute this
> plan task-by-task. Keep the branch green after every task.

**Goal:** `GMMXX(covariance_type="spherical", backend="cuda", approx_top_k=k)`
trains on the CUDA backend instead of falling through to Triton/PyTorch. This
plan closes the user-visible backend gap for approximate EM while preserving
the public API and the existing approximation contract: approximate top-k is a
training-time M-step approximation; prediction and scoring remain exact.

**Scope decision:** This first CUDA implementation is torch-on-CUDA
orchestration, not a new nanobind C++ kernel. It mirrors the existing
`torch_fallback._topk_logits_for_chunk` + `_accumulate_topk_stats` math, but all
work stays on CUDA tensors. A raw `csrc/approx/approx_topk_spherical.cu` kernel
is a later performance pass if profiling shows this path matters.

---

## Contract

Inputs:

- `x`: `(B, N, D)` CUDA floating tensor.
- `means`: `(B, K, D)` CUDA floating tensor.
- `var`: `(B, K)` CUDA fp32 spherical variances.
- `log_w`: `(B, K)` CUDA fp32 log weights.
- `top_k`: integer in `[1, K - 1]`.

Outputs:

- `nk`: `(B, K)` fp32 soft counts.
- `sum_x`: `(B, K, D)` fp32 sufficient statistics.
- `sum_x_sq`: `(B, K)` fp32 weighted squared norms.
- `log_likelihood_sum`: scalar fp32, sum over `(B, N)` of approximate log norm.

Numerics:

- Logits use the existing spherical formula:
  `log_w_k - 0.5 * (||x - mu_k||^2 / var_k + D * (log(2π) + log(var_k)))`.
- Only the row-wise top-k logits participate in the responsibility softmax.
- Accumulators are fp32 regardless of input dtype.
- `top_k >= K` resolves to exact EM through the existing `_resolve_approx_top_k`
  behavior and should not mark `approximate_em_enabled_`.

---

## Tasks

### 1. Add `_cuda.approx_topk_update_spherical`

- Add a pure torch CUDA helper in `gmmxx/_cuda.py`.
- Validate CUDA tensors and dtype/shape consistency.
- Stream over K with `chunk_size_K` to avoid materializing `(B, N, K)` for large
  K.
- Maintain `(best_logits, best_indices)` via `torch.topk` over
  `cat(previous_best, current_tile_logits)`.
- Accumulate `nk`, `sum_x`, and `sum_x_sq` with `scatter_add_` over the top-k
  slots.
- Return the four tensors listed in the contract.
- Wrap runtime errors in `CudaRuntimeFallback` unless
  `GMMXX_CUDA_NO_FALLBACK=1`.

### 2. Export through `gmmxx.cuda_ops`

- Re-export `approx_topk_update_spherical`.
- Add it to `__all__`.
- Keep docstring wording experimental.

### 3. Wire spherical training

- Remove the `self.approx_top_k is None` guard from the spherical CUDA dispatch
  branch in `GMMXX.train`.
- In `_train_spherical_cuda`, resolve `effective_approx_top_k`:
  - `None` when user did not request it or `approx_top_k >= K`.
  - Otherwise the requested positive integer.
- Disable fused E/M when `effective_approx_top_k is not None`.
- In each EM iteration, call `_cuda.approx_topk_update_spherical`, then finalize
  the soft sufficient statistics in Python:
  - inactive clusters preserve prior mean/variance;
  - variance is `(sum_x_sq - nk * ||mean||²) / (nk * D)`;
  - weights are normalized soft counts.
- If `compute_labels_on_fit` is true, compute final labels exactly via
  `_cuda.spherical_assign`.
- Set:
  - `cuda_approx_topk_enabled_ = True`;
  - `fit_info_["cuda_approx_topk_enabled"] = True`;
  - `approximate_em_enabled_ = True`;
  - `approx_top_k_ = top_k`.

### 4. Tests

Add `tests/test_cuda_approx_topk_spherical.py`:

- Direct helper test vs a brute torch reference for fp32/fp16/bf16 small shapes.
- Chunk-size invariance (`chunk_size_K=3` vs `chunk_size_K=K`).
- Invalid `top_k` is rejected.
- `GMMXX(... backend="cuda", approx_top_k=...)` uses CUDA and marks
  `cuda_approx_topk_enabled_`.
- `approx_top_k >= K` routes to exact CUDA and leaves
  `approximate_em_enabled_ == False`.

### 5. Docs and milestone

- README: mention spherical `approx_top_k` now stays on CUDA.
- Run focused tests and full suite.
- Commit and tag `approx-topk-plan11`.

---

## Deferred

- Raw nanobind/CUDA `csrc/approx/approx_topk_spherical.cu` kernel.
- large-N spherical approximate top-k CUDA streaming. The existing large-N
  exact CUDA route remains unchanged; large-N approx continues to use the
  existing Triton/torch path.
