#include "diag.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace mstep { namespace diag {

template <typename T>
__global__ void
finalize_diag_kernel(
    const float* __restrict__ sums,    // (B, K, D)
    const float* __restrict__ sumsq,   // (B, K, D) per-feature
    const int32_t* __restrict__ counts,// (B, K)
    const T* __restrict__ old_means,   // (B, K, D)
    const float* __restrict__ old_var, // (B, K, D)
    T* __restrict__ new_means,
    float* __restrict__ new_var,       // (B, K, D)
    float* __restrict__ new_weights,   // (B, K)
    int B, int K, int D, int total_n,
    float reg_covar
) {
    int b = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || k >= K) return;

    int32_t cnt = counts[(size_t)b * K + k];
    if (cnt <= 0) {
        for (int d = 0; d < D; ++d) {
            size_t idx = ((size_t)b * K + k) * D + d;
            new_means[idx] = old_means[idx];
            new_var[idx] = old_var[idx];
        }
        new_weights[(size_t)b * K + k] = 0.0f;
        return;
    }

    float n_inv = 1.0f / (float)cnt;
    for (int d = 0; d < D; ++d) {
        size_t idx = ((size_t)b * K + k) * D + d;
        float mu_d = sums[idx] * n_inv;
        new_means[idx] = static_cast<T>(mu_d);
        float var_d = sumsq[idx] * n_inv - mu_d * mu_d;
        new_var[idx] = fmaxf(var_d, reg_covar);
    }
    new_weights[(size_t)b * K + k] = (float)cnt / (float)total_n;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> finalize(
    const at::Tensor& sums,
    const at::Tensor& sumsq,
    const at::Tensor& counts,
    const at::Tensor& old_means,
    const at::Tensor& old_var,
    int64_t total_n,
    double reg_covar
) {
    TORCH_CHECK(sums.is_cuda() && sums.is_contiguous() && sums.scalar_type() == at::kFloat,
                "sums must be contiguous fp32 CUDA");
    TORCH_CHECK(sumsq.is_cuda() && sumsq.is_contiguous() && sumsq.scalar_type() == at::kFloat,
                "sumsq must be contiguous fp32 CUDA");
    TORCH_CHECK(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type() == at::kInt,
                "counts must be contiguous int32 CUDA");
    TORCH_CHECK(old_means.is_cuda() && old_means.is_contiguous(),
                "old_means must be contiguous CUDA");
    TORCH_CHECK(old_var.is_cuda() && old_var.is_contiguous() && old_var.scalar_type() == at::kFloat,
                "old_var must be contiguous fp32 CUDA");
    TORCH_CHECK(sums.dim() == 3 && sumsq.dim() == 3,
                "sums and sumsq must be (B, K, D)");
    TORCH_CHECK(sums.sizes() == sumsq.sizes(),
                "sums and sumsq must agree in shape");
    TORCH_CHECK(old_var.sizes() == sums.sizes(),
                "old_var must be (B, K, D)");
    TORCH_CHECK(total_n > 0, "total_n must be positive");

    int B = (int)sums.size(0);
    int K = (int)sums.size(1);
    int D = (int)sums.size(2);

    c10::cuda::CUDAGuard guard(sums.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto new_means = at::empty_like(old_means);
    auto new_var = at::empty({B, K, D}, sums.options());
    auto new_weights = at::empty({B, K}, sums.options());

    constexpr int kThreads = 64;
    dim3 grid((K + kThreads - 1) / kThreads, B);

    switch (old_means.scalar_type()) {
        case at::kFloat:
            finalize_diag_kernel<float><<<grid, kThreads, 0, stream>>>(
                sums.data_ptr<float>(), sumsq.data_ptr<float>(), counts.data_ptr<int32_t>(),
                old_means.data_ptr<float>(), old_var.data_ptr<float>(),
                new_means.data_ptr<float>(), new_var.data_ptr<float>(), new_weights.data_ptr<float>(),
                B, K, D, (int)total_n, (float)reg_covar);
            break;
        case at::kHalf:
            finalize_diag_kernel<at::Half><<<grid, kThreads, 0, stream>>>(
                sums.data_ptr<float>(), sumsq.data_ptr<float>(), counts.data_ptr<int32_t>(),
                old_means.data_ptr<at::Half>(), old_var.data_ptr<float>(),
                new_means.data_ptr<at::Half>(), new_var.data_ptr<float>(), new_weights.data_ptr<float>(),
                B, K, D, (int)total_n, (float)reg_covar);
            break;
        case at::kBFloat16:
            finalize_diag_kernel<at::BFloat16><<<grid, kThreads, 0, stream>>>(
                sums.data_ptr<float>(), sumsq.data_ptr<float>(), counts.data_ptr<int32_t>(),
                old_means.data_ptr<at::BFloat16>(), old_var.data_ptr<float>(),
                new_means.data_ptr<at::BFloat16>(), new_var.data_ptr<float>(), new_weights.data_ptr<float>(),
                B, K, D, (int)total_n, (float)reg_covar);
            break;
        default:
            TORCH_CHECK(false, "finalize_diag: unsupported dtype ", old_means.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(new_means, new_var, new_weights);
}

}}}  // namespace gmmxx::mstep::diag
