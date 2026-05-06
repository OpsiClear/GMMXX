#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace em { namespace spherical {

// Exp63: C++-side EM-loop driver for the spherical soft-EM cuBLAS chunked
// fastpath. Runs n_iter iterations end-to-end with no per-iter Python
// dispatch — each iteration only crosses the FFI boundary at the bracketing
// fit() call.
//
// Inputs:
//   x_estep_aug      (B, N, D+1) fp32   = [x | x_sq]
//   x_estep_aug_bf16 (B, N, D+1) bf16   optional bf16 cache for HMMA path.
//                                         Pass empty to use TF32 path on
//                                         x_estep_aug.
//   x_aug            (B, N, D+2) fp32   = [x | x_sq | 1]; M-step bmm input.
//   means_init       (B, K, D) fp32     starting means (modified in-place).
//   var_init         (B, K) fp32        starting variances.
//   log_w_init       (B, K) fp32        starting log-weights.
//   n_iter           number of EM iterations to run.
//   reg_covar        floor for variance.
//   chunk_size       chunk granularity along N for the L2-streaming inner
//                    loop. Pass 0 to default to ~32 MB target.
//   need_final_lse   if true, also compute and return the per-sample LSE
//                    after the last iteration.
//
// Returns (means, var, weights, lower_bound, final_lse_or_empty).
std::tuple<at::Tensor, at::Tensor, at::Tensor, double, at::Tensor>
soft_chunked(
    const at::Tensor& x_estep_aug,
    const at::Tensor& x_estep_aug_bf16,
    const at::Tensor& x_aug,
    at::Tensor means,
    at::Tensor var,
    at::Tensor log_w,
    int64_t n_iter,
    double reg_covar,
    int64_t chunk_size,
    bool need_final_lse);

}}}  // namespace gmmxx::em::spherical
