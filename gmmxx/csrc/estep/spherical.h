#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace estep { namespace spherical {

// Public dispatchers (Plan 3 Task 3 implements these in spherical_dispatch.cu).
//
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

at::Tensor logsumexp(const at::Tensor& x,
                     const at::Tensor& means,
                     const at::Tensor& var,
                     const at::Tensor& log_w,
                     c10::optional<at::Tensor> out,
                     c10::optional<at::Tensor> x_sq = c10::nullopt,
                     c10::optional<at::Tensor> c_sq = c10::nullopt);

at::Tensor resp(const at::Tensor& x,
                const at::Tensor& means,
                const at::Tensor& var,
                const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out,
                c10::optional<at::Tensor> x_sq = c10::nullopt,
                c10::optional<at::Tensor> c_sq = c10::nullopt);

// Safe-path implementations (renamed from Plan 2's public functions).
at::Tensor assign_safe(const at::Tensor& x,
                       const at::Tensor& means,
                       const at::Tensor& var,
                       const at::Tensor& log_w,
                       c10::optional<at::Tensor> out);

at::Tensor logsumexp_safe(const at::Tensor& x,
                          const at::Tensor& means,
                          const at::Tensor& var,
                          const at::Tensor& log_w,
                          c10::optional<at::Tensor> out);

at::Tensor resp_safe(const at::Tensor& x,
                     const at::Tensor& means,
                     const at::Tensor& var,
                     const at::Tensor& log_w,
                     const at::Tensor& log_norm,
                     c10::optional<at::Tensor> out);

// sm_80+ mma path (fp16/bf16 only). Caller precomputes x_sq/c_sq fp32 norms.
at::Tensor assign_sm80(const at::Tensor& x,
                       const at::Tensor& means,
                       const at::Tensor& var,
                       const at::Tensor& log_w,
                       const at::Tensor& x_sq,
                       const at::Tensor& c_sq,
                       c10::optional<at::Tensor> out);

at::Tensor logsumexp_sm80(const at::Tensor& x,
                          const at::Tensor& means,
                          const at::Tensor& var,
                          const at::Tensor& log_w,
                          const at::Tensor& x_sq,
                          const at::Tensor& c_sq,
                          c10::optional<at::Tensor> out);

at::Tensor resp_sm80(const at::Tensor& x,
                     const at::Tensor& means,
                     const at::Tensor& var,
                     const at::Tensor& log_w,
                     const at::Tensor& x_sq,
                     const at::Tensor& c_sq,
                     const at::Tensor& log_norm,
                     c10::optional<at::Tensor> out);

// Soft-EM E-step preparation: builds alpha (B,K) and means_aug (B,K,D+1)
// for the augmented cuBLAS GEMM that produces logits in one matmul.
// Replaces ~10 small (B,K) torch ops with a single kernel launch.
std::tuple<at::Tensor, at::Tensor> prepare_estep(
    const at::Tensor& log_w,
    const at::Tensor& means,
    const at::Tensor& var);

// Exp85: prepare_estep variant that emits bf16 outputs in the layout the
// persistent-CTA Triton kernel needs:
//   alpha_bf16: (B, K) bf16
//   means_aug_t_bf16: (B, D+1, K) bf16, contiguous — transposed M-step input.
// Eliminates the ~25 μs/iter chain of (.to(bf16), .t(), .contiguous()) that
// the Python wrapper does after the original prepare_estep. B is restricted
// to 1 in practice; the (B, ...) shape is preserved for API symmetry.
std::tuple<at::Tensor, at::Tensor> prepare_estep_bf16_t(
    const at::Tensor& log_w,
    const at::Tensor& means,
    const at::Tensor& var);

// Exp62: fused logsumexp + resp sm80 kernel.
// Returns (lse, resp). lse is empty when need_lse is false. If no tile fits
// (e.g. K > BLOCK_K of every available tile), returns two empty tensors —
// caller must fall back to the separate logsumexp+resp pipeline.
std::tuple<at::Tensor, at::Tensor> logsumexp_resp_sm80(
    const at::Tensor& x,
    const at::Tensor& means,
    const at::Tensor& var,
    const at::Tensor& log_w,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    bool need_lse);

}}}  // namespace gmmxx::estep::spherical
