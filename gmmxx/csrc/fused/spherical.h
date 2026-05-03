#pragma once

#include "../common/torch_cuda_includes.h"
#include <tuple>

namespace gmmxx { namespace fused { namespace spherical {

// Public dispatcher — routes by dtype + compute capability.
// Returns (new_means, new_var, new_weights, lse_per_sample, labels).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused(const at::Tensor& x,
      const at::Tensor& means,
      const at::Tensor& var,
      const at::Tensor& log_w,
      double reg_covar);

// Safe SIMT path (any dtype). Plan 5 Task 1 implements this.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_safe(const at::Tensor& x,
           const at::Tensor& means,
           const at::Tensor& var,
           const at::Tensor& log_w,
           double reg_covar);

// sm80+ mma path (fp16/bf16 inputs). Plan 5 Task 4 replaces this stub.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_sm80(const at::Tensor& x,
           const at::Tensor& means,
           const at::Tensor& var,
           const at::Tensor& log_w,
           double reg_covar);

}}}  // namespace gmmxx::fused::spherical
