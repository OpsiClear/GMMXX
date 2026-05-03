#include "spherical.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace mstep { namespace spherical {

namespace {

constexpr int BLOCK_N = 256;
constexpr int THREADS_PER_CTA = 128;
constexpr int N_WARPS = THREADS_PER_CTA / 32;

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_CTA, 4)
blocked_update_sorted_kernel(
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
    __shared__ float ss_smem[N_WARPS];

    int cursor = 0;
    while (cursor < n_count) {
        if (threadIdx.x == 0) {
            int cid = ids_b[n_start + cursor];
            int len = 1;
            while (cursor + len < n_count && ids_b[n_start + cursor + len] == cid) {
                len++;
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
            // Per-thread strided slice of D for sums.
            for (int d_base = threadIdx.x; d_base < D; d_base += THREADS_PER_CTA) {
                float acc = 0.0f;
                for (int r = 0; r < rl; ++r) {
                    int n_idx = n_start + rs + r;
                    acc += static_cast<float>(x_b[(size_t)n_idx * D + d_base]);
                }
                atomicAdd(sums + ((size_t)b * K + cid) * D + d_base, acc);
            }

            // Sumsq: strided accumulation of ||x||^2 then block-reduce to lane 0.
            float local_ss = 0.0f;
            for (int rd = threadIdx.x; rd < rl * D; rd += THREADS_PER_CTA) {
                int r = rd / D;
                int d = rd % D;
                int n_idx = n_start + rs + r;
                float v = static_cast<float>(x_b[(size_t)n_idx * D + d]);
                local_ss += v * v;
            }
            // Warp reduce.
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_ss += __shfl_xor_sync(0xffffffffu, local_ss, offset);
            }
            int lane = threadIdx.x & 31;
            int warp_id = threadIdx.x >> 5;
            if (lane == 0) ss_smem[warp_id] = local_ss;
            __syncthreads();
            if (warp_id == 0) {
                float v = (lane < N_WARPS) ? ss_smem[lane] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    v += __shfl_xor_sync(0xffffffffu, v, offset);
                }
                if (threadIdx.x == 0) {
                    atomicAdd(sumsq + (size_t)b * K + cid, v);
                    atomicAdd(counts + (size_t)b * K + cid, rl);
                }
            }
            __syncthreads();
        }

        cursor = rs + rl;
    }
}

}  // anonymous namespace

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
    if (N == 0) return;

    c10::cuda::CUDAGuard guard(x_sorted.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, B);

    switch (x_sorted.scalar_type()) {
        case at::kFloat:
            blocked_update_sorted_kernel<float><<<grid, THREADS_PER_CTA, 0, stream>>>(
                x_sorted.data_ptr<float>(), sorted_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        case at::kHalf:
            blocked_update_sorted_kernel<at::Half><<<grid, THREADS_PER_CTA, 0, stream>>>(
                x_sorted.data_ptr<at::Half>(), sorted_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        case at::kBFloat16:
            blocked_update_sorted_kernel<at::BFloat16><<<grid, THREADS_PER_CTA, 0, stream>>>(
                x_sorted.data_ptr<at::BFloat16>(), sorted_ids.data_ptr<int32_t>(),
                sums_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
                counts_out.data_ptr<int32_t>(),
                B, N, K, D);
            break;
        default:
            TORCH_CHECK(false, "blocked_update_sorted: unsupported dtype ", x_sorted.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}}}
