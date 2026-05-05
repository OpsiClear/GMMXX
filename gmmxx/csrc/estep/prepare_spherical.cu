// Soft-EM E-step preparation kernel.
//
// Fuses ~10 small (B,K) torch ops into a single CUDA kernel:
//   alpha[b,k]    = log_w[b,k] - 0.5*D*log(2π*var[b,k]) - 0.5*|c_k|^2 / var[b,k]
//   means_aug[b,k,:D] = inv_var[b,k] * means[b,k,:]
//   means_aug[b,k,D]  = -0.5 * inv_var[b,k]
//
// One thread per (b, k). Output is consumed by the cuBLAS E-step GEMM:
//   logits[b,n,k] = alpha[b,k] + sum_d x_aug[b,n,d] * means_aug[b,k,d]
//                 = log_w[b,k] - 0.5*D*log(2π*var[b,k]) - 0.5/var[b,k] * |x_n - c_k|^2

#include "spherical.h"
#include "../common/torch_cuda_includes.h"

#include <math_constants.h>

namespace gmmxx { namespace estep { namespace spherical {

__global__ void
prepare_spherical_estep_kernel(
    const float* __restrict__ log_w,    // (B, K)
    const float* __restrict__ means,    // (B, K, D)
    const float* __restrict__ var,      // (B, K)
    float*       __restrict__ alpha,    // (B, K)
    float*       __restrict__ means_aug,// (B, K, D+1)
    int B, int K, int D
) {
    int b = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || k >= K) return;

    size_t bk = (size_t)b * K + k;
    float v = var[bk];
    float inv_var = 1.0f / v;

    // c_sq = sum_d means[b,k,d]^2; produce means_aug[:D] = inv_var * means
    float c_sq = 0.0f;
    size_t base = bk * D;
    size_t base_aug = bk * (D + 1);
    for (int d = 0; d < D; ++d) {
        float m = means[base + d];
        c_sq += m * m;
        means_aug[base_aug + d] = m * inv_var;
    }
    means_aug[base_aug + D] = -0.5f * inv_var;

    // alpha = log_w - 0.5*D*log(2π*v) - 0.5*c_sq*inv_var
    constexpr float LOG_2PI = 1.8378770664093453f;  // log(2π)
    float half_d = 0.5f * (float)D;
    alpha[bk] = log_w[bk] - half_d * (LOG_2PI + logf(v)) - 0.5f * c_sq * inv_var;
}

std::tuple<at::Tensor, at::Tensor> prepare_estep(
    const at::Tensor& log_w,
    const at::Tensor& means,
    const at::Tensor& var
) {
    TORCH_CHECK(log_w.is_cuda() && log_w.is_contiguous() && log_w.scalar_type() == at::kFloat,
                "log_w must be contiguous fp32 CUDA");
    TORCH_CHECK(means.is_cuda() && means.is_contiguous() && means.scalar_type() == at::kFloat,
                "means must be contiguous fp32 CUDA");
    TORCH_CHECK(var.is_cuda() && var.is_contiguous() && var.scalar_type() == at::kFloat,
                "var must be contiguous fp32 CUDA");
    TORCH_CHECK(means.dim() == 3 && log_w.dim() == 2 && var.dim() == 2,
                "means must be (B,K,D); log_w, var must be (B,K)");

    int B = (int)means.size(0);
    int K = (int)means.size(1);
    int D = (int)means.size(2);

    c10::cuda::CUDAGuard guard(means.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto opts = means.options();
    auto alpha = at::empty({B, K}, opts);
    auto means_aug = at::empty({B, K, D + 1}, opts);

    constexpr int kThreads = 64;
    dim3 grid((K + kThreads - 1) / kThreads, B);

    prepare_spherical_estep_kernel<<<grid, kThreads, 0, stream>>>(
        log_w.data_ptr<float>(),
        means.data_ptr<float>(),
        var.data_ptr<float>(),
        alpha.data_ptr<float>(),
        means_aug.data_ptr<float>(),
        B, K, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(alpha, means_aug);
}

}}}  // namespace gmmxx::estep::spherical
