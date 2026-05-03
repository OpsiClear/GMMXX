#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace estep { namespace diag {

// Public dispatchers.
//
// x: (B, N, D) fp32 / fp16 / bf16, contiguous, CUDA.
// means: (B, K, D) same dtype as x.
// var: (B, K, D) fp32 — per-feature variance.
// log_w: (B, K) fp32.
at::Tensor assign(const at::Tensor& x, const at::Tensor& means,
                  const at::Tensor& var, const at::Tensor& log_w,
                  c10::optional<at::Tensor> out);

at::Tensor logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     c10::optional<at::Tensor> out);

at::Tensor resp(const at::Tensor& x, const at::Tensor& means,
                const at::Tensor& var, const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out);

// Safe path implementations.
at::Tensor assign_safe(const at::Tensor& x, const at::Tensor& means,
                       const at::Tensor& var, const at::Tensor& log_w,
                       c10::optional<at::Tensor> out);
at::Tensor logsumexp_safe(const at::Tensor& x, const at::Tensor& means,
                          const at::Tensor& var, const at::Tensor& log_w,
                          c10::optional<at::Tensor> out);
at::Tensor resp_safe(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     const at::Tensor& log_norm,
                     c10::optional<at::Tensor> out);

}}}  // namespace gmmxx::estep::diag
