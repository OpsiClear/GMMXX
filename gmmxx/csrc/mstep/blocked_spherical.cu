#include "spherical.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace mstep { namespace spherical {

template <typename T>
__global__ void __launch_bounds__(128)
blocked_update_spherical_kernel(
    const T* __restrict__ x,
    const int32_t* __restrict__ ids,
    float* __restrict__ sums,
    float* __restrict__ sumsq,
    int32_t* __restrict__ counts,
    int B, int N, int K, int D
) {
    int b = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || n >= N) return;

    const T* x_n = x + ((size_t)b * N + n) * D;
    int k = ids[(size_t)b * N + n];
    if (k < 0 || k >= K) return;

    float xx = 0.0f;
    for (int d = 0; d < D; ++d) {
        float v = static_cast<float>(x_n[d]);
        xx += v * v;
        atomicAdd(sums + ((size_t)b * K + k) * D + d, v);
    }
    atomicAdd(sumsq + (size_t)b * K + k, xx);
    atomicAdd(counts + (size_t)b * K + k, 1);
}

void blocked_update(const at::Tensor& x,
                    const at::Tensor& cluster_ids,
                    at::Tensor& sums_out,
                    at::Tensor& sumsq_out,
                    at::Tensor& counts_out) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    TORCH_CHECK(cluster_ids.is_cuda() && cluster_ids.is_contiguous() &&
                cluster_ids.scalar_type() == at::kInt,
                "cluster_ids must be contiguous int32 CUDA");
    TORCH_CHECK(sums_out.is_cuda() && sums_out.is_contiguous() &&
                sums_out.scalar_type() == at::kFloat,
                "sums_out must be contiguous fp32 CUDA");
    TORCH_CHECK(sumsq_out.is_cuda() && sumsq_out.is_contiguous() &&
                sumsq_out.scalar_type() == at::kFloat,
                "sumsq_out must be contiguous fp32 CUDA");
    TORCH_CHECK(counts_out.is_cuda() && counts_out.is_contiguous() &&
                counts_out.scalar_type() == at::kInt,
                "counts_out must be contiguous int32 CUDA");
    TORCH_CHECK(x.dim() == 3, "x must be (B,N,D)");

    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)sums_out.size(1);
    TORCH_CHECK(sums_out.sizes() == at::IntArrayRef({B, K, D}),
                "sums_out must be (B,K,D)");
    TORCH_CHECK(sumsq_out.sizes() == at::IntArrayRef({B, K}),
                "sumsq_out must be (B,K)");
    TORCH_CHECK(counts_out.sizes() == at::IntArrayRef({B, K}),
                "counts_out must be (B,K)");
    TORCH_CHECK(cluster_ids.sizes() == at::IntArrayRef({B, N}),
                "cluster_ids must be (B,N)");

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int kThreads = 128;
    dim3 grid((N + kThreads - 1) / kThreads, B);

    switch (x.scalar_type()) {
        case at::kFloat:
            blocked_update_spherical_kernel<float><<<grid, kThreads, 0, stream>>>(
                x.data_ptr<float>(), cluster_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        case at::kHalf:
            blocked_update_spherical_kernel<at::Half><<<grid, kThreads, 0, stream>>>(
                x.data_ptr<at::Half>(), cluster_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        case at::kBFloat16:
            blocked_update_spherical_kernel<at::BFloat16><<<grid, kThreads, 0, stream>>>(
                x.data_ptr<at::BFloat16>(), cluster_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        default:
            TORCH_CHECK(false, "blocked_update_spherical: unsupported dtype ", x.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}}}  // namespace gmmxx::mstep::spherical
