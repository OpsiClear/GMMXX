#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace mstep { namespace diag {

// Hard-assignment M-step accumulator. Caller MUST zero sums_out, sumsq_out,
// counts_out before calling.
//
// x: (B, N, D) fp32 / fp16 / bf16
// cluster_ids: (B, N) int32
// sums_out: (B, K, D) fp32 — Σ x_n by cluster
// sumsq_out: (B, K, D) fp32 — Σ x_n² PER FEATURE by cluster (NOT scalar)
// counts_out: (B, K) int32
void blocked_update(const at::Tensor& x,
                    const at::Tensor& cluster_ids,
                    at::Tensor& sums_out,
                    at::Tensor& sumsq_out,
                    at::Tensor& counts_out);

// Finalize: divide sums/counts, clamp per-feature variance to reg_covar.
//
// old_means: (B, K, D) — preserved when count[k] == 0.
// old_var: (B, K, D) — per-feature, preserved when count[k] == 0.
// Returns:
//   means: (B, K, D)
//   var: (B, K, D) — per-feature variance
//   weights: (B, K)
std::tuple<at::Tensor, at::Tensor, at::Tensor> finalize(
    const at::Tensor& sums,
    const at::Tensor& sumsq,
    const at::Tensor& counts,
    const at::Tensor& old_means,
    const at::Tensor& old_var,
    int64_t total_n,
    double reg_covar);

}}}  // namespace gmmxx::mstep::diag
