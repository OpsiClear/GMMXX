#include "diag.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace mstep { namespace diag {

namespace {

constexpr int BLOCK_N = 256;
constexpr int THREADS_PER_CTA = 128;

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_CTA, 4)
blocked_update_diag_sorted_kernel(
    const T* __restrict__ x_sorted,
    const int32_t* __restrict__ sorted_ids,
    float* __restrict__ sums,
    float* __restrict__ sumsq,
    int32_t* __restrict__ counts,
    int B, int N, int K, int D
) {
    int b = blockIdx.y;
    int n_start = blockIdx.x * BLOCK_N;
    int n_count = min(BLOCK_N, N - n_start);
    if (n_count <= 0 || b >= B) return;

    const T* x_b = x_sorted + (size_t)b * N * D;
    const int32_t* ids_b = sorted_ids + (size_t)b * N;

    __shared__ int run_start;
    __shared__ int run_len;
    __shared__ int run_cid;

    int cursor = 0;
    while (cursor < n_count) {
        if (threadIdx.x == 0) {
            int cid = ids_b[n_start + cursor];
            int len = 1;
            while (cursor + len < n_count && ids_b[n_start + cursor + len] == cid) {
                ++len;
            }
            run_start = cursor;
            run_len = len;
            run_cid = cid;
        }
        __syncthreads();

        int rs = run_start;
        int rl = run_len;
        int cid = run_cid;

        if (cid >= 0 && cid < K) {
            for (int d = threadIdx.x; d < D; d += THREADS_PER_CTA) {
                float acc = 0.0f;
                float acc_sq = 0.0f;
                for (int r = 0; r < rl; ++r) {
                    int n_idx = n_start + rs + r;
                    float v = static_cast<float>(x_b[(size_t)n_idx * D + d]);
                    acc += v;
                    acc_sq += v * v;
                }
                size_t out_idx = ((size_t)b * K + cid) * D + d;
                atomicAdd(sums + out_idx, acc);
                atomicAdd(sumsq + out_idx, acc_sq);
            }
            if (threadIdx.x == 0) {
                atomicAdd(counts + (size_t)b * K + cid, rl);
            }
        }

        __syncthreads();
        cursor = rs + rl;
    }
}

}  // namespace

void blocked_update_sorted(const at::Tensor& x_sorted,
                           const at::Tensor& sorted_ids,
                           at::Tensor& sums_out,
                           at::Tensor& sumsq_out,
                           at::Tensor& counts_out) {
    TORCH_CHECK(x_sorted.is_cuda() && x_sorted.is_contiguous(), "x_sorted must be contiguous CUDA");
    TORCH_CHECK(sorted_ids.is_cuda() && sorted_ids.is_contiguous() &&
                sorted_ids.scalar_type() == at::kInt,
                "sorted_ids must be contiguous int32 CUDA");
    TORCH_CHECK(sums_out.is_cuda() && sums_out.is_contiguous() &&
                sums_out.scalar_type() == at::kFloat,
                "sums_out must be contiguous fp32 CUDA");
    TORCH_CHECK(sumsq_out.is_cuda() && sumsq_out.is_contiguous() &&
                sumsq_out.scalar_type() == at::kFloat,
                "sumsq_out must be contiguous fp32 CUDA");
    TORCH_CHECK(counts_out.is_cuda() && counts_out.is_contiguous() &&
                counts_out.scalar_type() == at::kInt,
                "counts_out must be contiguous int32 CUDA");
    TORCH_CHECK(x_sorted.dim() == 3, "x_sorted must be (B,N,D)");

    int B = (int)x_sorted.size(0);
    int N = (int)x_sorted.size(1);
    int D = (int)x_sorted.size(2);
    int K = (int)sums_out.size(1);
    TORCH_CHECK(sums_out.sizes() == at::IntArrayRef({B, K, D}),
                "sums_out must be (B,K,D)");
    TORCH_CHECK(sumsq_out.sizes() == at::IntArrayRef({B, K, D}),
                "sumsq_out must be (B,K,D)");
    TORCH_CHECK(counts_out.sizes() == at::IntArrayRef({B, K}),
                "counts_out must be (B,K)");
    TORCH_CHECK(sorted_ids.sizes() == at::IntArrayRef({B, N}),
                "sorted_ids must be (B,N)");
    if (N == 0) return;

    c10::cuda::CUDAGuard guard(x_sorted.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, B);

    switch (x_sorted.scalar_type()) {
        case at::kFloat:
            blocked_update_diag_sorted_kernel<float><<<grid, THREADS_PER_CTA, 0, stream>>>(
                x_sorted.data_ptr<float>(), sorted_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(), B, N, K, D);
            break;
        case at::kHalf:
            blocked_update_diag_sorted_kernel<at::Half><<<grid, THREADS_PER_CTA, 0, stream>>>(
                x_sorted.data_ptr<at::Half>(), sorted_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(), B, N, K, D);
            break;
        case at::kBFloat16:
            blocked_update_diag_sorted_kernel<at::BFloat16><<<grid, THREADS_PER_CTA, 0, stream>>>(
                x_sorted.data_ptr<at::BFloat16>(), sorted_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(), B, N, K, D);
            break;
        default:
            TORCH_CHECK(false, "blocked_update_diag_sorted: unsupported dtype ", x_sorted.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}}}  // namespace gmmxx::mstep::diag
