# GMMXX CUDA Backend — Plan 9: Soft-EM Correction for Non-Spherical CUDA

> **For agentic workers:** execute task-by-task and keep the test suite green.

**Goal:** Align in-memory CUDA training for `diag`, `tied`, and `full`
covariance with the GMM soft-EM contract before adding lower-level performance
kernels. The current CUDA loops compute a correct logsumexp, but their M-step
uses hard `assign -> blocked_update` statistics. That is k-means-like and can
drift from the torch/Triton reference. Large-N CUDA training already uses soft
responsibilities; this plan brings the in-memory loops to the same semantics.

---

## Scope

No new C++/nanobind build is required.

- `diag`: use `diag_logsumexp` + `diag_resp`, then fp32 `bmm`/broadcasted
  reductions for `nk`, `sum_x`, and per-feature `sum_x_sq`.
- `tied`: use `tied_logsumexp` + `tied_resp`, then fp32 `bmm`; reuse
  `_cuda.tied_finalize`, which already accepts fp32 soft counts.
- `full`: use `full_logsumexp` + `full_resp`, then fp32 `bmm` and
  `einsum("bnk,bnd,bne->bkde", ...)`; update `_cuda.full_finalize` so soft
  counts are not truncated to int when checking empty clusters.

Spherical already has a soft fused path inside its primary `(D <= 64, K <= 128)`
window and Plan 11 soft approx-topK. The remaining wide-shape unfused spherical
hard path is deferred to a later spherical cleanup because this plan is focused
on non-spherical covariance.

---

## Tasks

### 1. Diagonal soft M-step

- Replace `_train_diag_cuda` hard `diag_assign + blocked_update_diag` M-step
  with:
  - `lse = diag_logsumexp(...)`;
  - `resp = diag_resp(..., lse)`;
  - `nk = resp.sum(dim=1)`;
  - `sum_x = bmm(resp.T, x.float())`;
  - `sum_x_sq = (resp[..., None] * x.float().square()[:, :, None, :]).sum(dim=1)`.
- Finalize in Python with empty-cluster preservation, reg-covar clamp, and
  normalized soft weights.
- Compute final labels exactly only when `compute_labels_on_fit` is true.

### 2. Tied soft M-step

- Replace `_train_tied_cuda` hard `tied_assign + blocked_update_spherical`
  update with `tied_logsumexp + tied_resp`.
- Use soft `nk` and `sum_x` with `_cuda.tied_finalize`.
- Compute final labels exactly only when requested.

### 3. Full soft M-step

- Replace `_train_full_cuda` hard `full_assign + full_blocked_update` update
  with `full_logsumexp + full_resp`.
- Use soft `nk`, `sum_x`, and `outer_sums`.
- Update `_cuda.full_finalize` empty-cluster/non-PD handling to use fp32 counts
  directly instead of `counts.to(int32)`.
- Compute final labels exactly only when requested.

### 4. Tests

Add `tests/test_cuda_soft_em_non_spherical.py`:

- For each covariance type, compare one CUDA EM iteration against the
  corresponding torch backend with same seed/init where feasible.
- Assert `compute_labels_on_fit=False` skips final assignment.
- Assert weights are normalized and lower bounds finite.
- Add a direct `full_finalize` soft-count test proving fractional counts do not
  get treated as empty clusters.

### 5. Docs and milestone

- README: mention non-spherical CUDA training uses soft EM.
- Run focused and full suites.
- Commit and tag `soft-em-plan9`.

---

## Deferred

- Raw sorted-run/soft-stat kernels for diag/tied/full.
- Fused single-tile diag/tied/full kernels.
- sm80 mma kernels for diag/tied/full E-steps.
- Wide-shape unfused spherical soft M-step cleanup.
