#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace mstep { namespace spherical {

// Hard-assignment M-step accumulator.
//
// Caller MUST zero-initialize sums_out, sumsq_out, counts_out before calling
// (see spec §5b). Kernel only does atomicAdd; it does not zero internally.
//
// x: (B, N, D) fp32 / fp16 / bf16
// cluster_ids: (B, N) int32 — per-point hard assignment
// sums_out: (B, K, D) fp32 — caller-zeroed accumulator (Σ x_n by cluster)
// sumsq_out: (B, K) fp32 — caller-zeroed accumulator (Σ ||x_n||²)
// counts_out: (B, K) int32 — caller-zeroed accumulator (Σ 1)
void blocked_update(const at::Tensor& x,
                    const at::Tensor& cluster_ids,
                    at::Tensor& sums_out,
                    at::Tensor& sumsq_out,
                    at::Tensor& counts_out);

// Sorted-run variant: caller pre-sorts cluster_ids and gathers x in the same
// permutation. One atomicAdd per (run, feature) tuple instead of per token.
void blocked_update_sorted(const at::Tensor& x_sorted,
                           const at::Tensor& sorted_ids,
                           at::Tensor& sums_out,
                           at::Tensor& sumsq_out,
                           at::Tensor& counts_out);

// Forward-declare finalize so spherical.h is the single header for both ops;
// finalize_spherical.cu (Plan 2 Task 5) provides the implementation.
std::tuple<at::Tensor, at::Tensor, at::Tensor> finalize(
    const at::Tensor& sums,
    const at::Tensor& sumsq,
    const at::Tensor& counts,
    const at::Tensor& old_means,
    const at::Tensor& old_var,
    int64_t total_n,
    double reg_covar);

}}}  // namespace gmmxx::mstep::spherical
