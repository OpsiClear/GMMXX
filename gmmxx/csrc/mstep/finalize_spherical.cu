#include "spherical.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace mstep { namespace spherical {

// Per-cluster finalize: divides accumulators, clamps variance, preserves
// previous mean/var on empty clusters.
//
// One thread per (b, k). Each thread:
//   - reads count[b,k]; if 0, copies old_means and old_var.
//   - else computes mean[b,k,d] = sums[b,k,d] / count[b,k] for each d.
//   - and var[b,k] = max((sumsq[b,k]/count - ||mean||²) / D, reg_covar)
//   - and weight[b,k] = count / total_n
template <typename T>
__global__ void
finalize_spherical_kernel(
    const float* __restrict__ sums,
    const float* __restrict__ sumsq,
    const int32_t* __restrict__ counts,
    const T* __restrict__ old_means,
    const float* __restrict__ old_var,
    T* __restrict__ new_means,
    float* __restrict__ new_var,
    float* __restrict__ new_weights,
    int B, int K, int D, int total_n,
    float reg_covar
) {
    int b = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || k >= K) return;

    int32_t cnt = counts[(size_t)b * K + k];

    if (cnt <= 0) {
        for (int d = 0; d < D; ++d) {
            new_means[((size_t)b * K + k) * D + d] = old_means[((size_t)b * K + k) * D + d];
        }
        new_var[(size_t)b * K + k] = old_var[(size_t)b * K + k];
        new_weights[(size_t)b * K + k] = 0.0f;
        return;
    }

    float n_inv = 1.0f / (float)cnt;

    float mu_sq = 0.0f;
    for (int d = 0; d < D; ++d) {
        float mu_d = sums[((size_t)b * K + k) * D + d] * n_inv;
        mu_sq += mu_d * mu_d;
        new_means[((size_t)b * K + k) * D + d] = static_cast<T>(mu_d);
    }

    float ss = sumsq[(size_t)b * K + k];
    float var_raw = (ss * n_inv - mu_sq) / (float)D;
    new_var[(size_t)b * K + k] = fmaxf(var_raw, reg_covar);

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
    TORCH_CHECK(sums.dim() == 3 && sumsq.dim() == 2,
                "sums must be (B,K,D); sumsq must be (B,K)");
    TORCH_CHECK(total_n > 0, "total_n must be positive");

    int B = (int)sums.size(0);
    int K = (int)sums.size(1);
    int D = (int)sums.size(2);

    c10::cuda::CUDAGuard guard(sums.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto new_means = at::empty_like(old_means);
    auto new_var = at::empty({B, K}, sums.options());
    auto new_weights = at::empty({B, K}, sums.options());

    constexpr int kThreads = 64;
    dim3 grid((K + kThreads - 1) / kThreads, B);

    switch (old_means.scalar_type()) {
        case at::kFloat:
            finalize_spherical_kernel<float><<<grid, kThreads, 0, stream>>>(
                sums.data_ptr<float>(), sumsq.data_ptr<float>(), counts.data_ptr<int32_t>(),
                old_means.data_ptr<float>(), old_var.data_ptr<float>(),
                new_means.data_ptr<float>(), new_var.data_ptr<float>(), new_weights.data_ptr<float>(),
                B, K, D, (int)total_n, (float)reg_covar);
            break;
        case at::kHalf:
            finalize_spherical_kernel<at::Half><<<grid, kThreads, 0, stream>>>(
                sums.data_ptr<float>(), sumsq.data_ptr<float>(), counts.data_ptr<int32_t>(),
                old_means.data_ptr<at::Half>(), old_var.data_ptr<float>(),
                new_means.data_ptr<at::Half>(), new_var.data_ptr<float>(), new_weights.data_ptr<float>(),
                B, K, D, (int)total_n, (float)reg_covar);
            break;
        case at::kBFloat16:
            finalize_spherical_kernel<at::BFloat16><<<grid, kThreads, 0, stream>>>(
                sums.data_ptr<float>(), sumsq.data_ptr<float>(), counts.data_ptr<int32_t>(),
                old_means.data_ptr<at::BFloat16>(), old_var.data_ptr<float>(),
                new_means.data_ptr<at::BFloat16>(), new_var.data_ptr<float>(), new_weights.data_ptr<float>(),
                B, K, D, (int)total_n, (float)reg_covar);
            break;
        default:
            TORCH_CHECK(false, "finalize_spherical: unsupported dtype ", old_means.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(new_means, new_var, new_weights);
}

}}}  // namespace gmmxx::mstep::spherical
