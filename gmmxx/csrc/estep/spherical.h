#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace estep { namespace spherical {

// Hard-assign cluster IDs (argmax of log p_k(x) over k).
// Returns int32 tensor of shape (B, N).
//
// x: (B, N, D) fp32 / fp16 / bf16. Contiguous, CUDA.
// means: (B, K, D) same dtype as x.
// var: (B, K) fp32. Per-component scalar variance.
// log_w: (B, K) fp32. Per-component log mixture weight.
at::Tensor assign(const at::Tensor& x,
                  const at::Tensor& means,
                  const at::Tensor& var,
                  const at::Tensor& log_w,
                  c10::optional<at::Tensor> out);

// Per-row stable logsumexp over K. Returns (B, N) fp32.
at::Tensor logsumexp(const at::Tensor& x,
                     const at::Tensor& means,
                     const at::Tensor& var,
                     const at::Tensor& log_w,
                     c10::optional<at::Tensor> out);

// Soft responsibilities r_{n,k} = exp(log p_k - log_norm). Returns (B, N, K) fp32.
//
// log_norm: (B, N) fp32. Caller supplies; typically obtained from logsumexp().
at::Tensor resp(const at::Tensor& x,
                const at::Tensor& means,
                const at::Tensor& var,
                const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out);

}}}  // namespace gmmxx::estep::spherical
