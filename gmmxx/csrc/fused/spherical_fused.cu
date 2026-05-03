// Fused single-tile spherical E/M kernel (safe SIMT path).
//
// Plan 5 Task 1. One kernel handles assignment, soft accumulation, and
// partial-sum flush for shapes D <= 64, K <= 128.
//
// Kernel overview:
//   fused_safe_kernel<T>:
//     - CTA loads centroids + var + log_w into SMEM once.
//     - Each thread owns one row of x.
//     - Phase 1: stable logsumexp over K to compute log_norm.
//     - Phase 2: compute r_k, atomicAdd into per-CTA SMEM partials.
//     - After all threads done: cooperative SMEM->global atomicAdd flush.
//
//   fused_finalize_kernel<T>:
//     - One thread per (b, k).
//     - Divides partial_sums / n_k, computes var, clamps.

#include "spherical.h"
#include "../common/arch.cuh"
#include <cmath>

namespace gmmxx { namespace fused { namespace spherical {

namespace {

constexpr int FUSED_BLOCK_N = 128;
constexpr int FUSED_THREADS = 128;
constexpr int FUSED_MAX_K   = 128;
constexpr int FUSED_MAX_D   = 64;

template <typename T>
__global__ void
fused_safe_kernel(
    const T* __restrict__     x,              // (B, N, D)
    const T* __restrict__     means_in,       // (B, K, D)
    const float* __restrict__ var_in,         // (B, K)
    const float* __restrict__ log_w_in,       // (B, K)
    float* __restrict__       partial_sums,   // (B, K, D) — caller-zeroed
    float* __restrict__       partial_sumsq,  // (B, K)    — caller-zeroed
    float* __restrict__       partial_counts, // (B, K)    — caller-zeroed
    float* __restrict__       lse_per_sample, // (B, N)
    int32_t* __restrict__     labels,         // (B, N)
    int B, int N, int K, int D
) {
    int b             = blockIdx.y;
    int n_block_start = blockIdx.x * FUSED_BLOCK_N;
    int tid           = threadIdx.x;
    int my_n          = n_block_start + tid;
    if (b >= B) return;

    // ---------------------------------------------------------------
    // Shared-memory layout (all fp32):
    //   [0 .. K*D-1]               means_smem
    //   [K*D .. K*D+K-1]           var_smem
    //   [K*D+K .. K*D+2K-1]        log_w_smem
    //   [K*D+2K .. 2*K*D+2K-1]     partial_sums_smem
    //   [2*K*D+2K .. 2*K*D+3K-1]   partial_sumsq_smem
    //   [2*K*D+3K .. 2*K*D+4K-1]   partial_counts_smem
    // Total floats: 2*K*D + 4*K
    // At K=128,D=64: 16896 floats = 67584 bytes ~ 66 KiB.
    // ---------------------------------------------------------------
    extern __shared__ float smem_buf[];
    float* means_smem          = smem_buf;
    float* var_smem            = means_smem + K * D;
    float* log_w_smem          = var_smem + K;
    float* partial_sums_smem   = log_w_smem + K;
    float* partial_sumsq_smem  = partial_sums_smem + K * D;
    float* partial_counts_smem = partial_sumsq_smem + K;

    // 1. Cooperative load centroids / var / log_w into SMEM.
    int kd_total = K * D;
    for (int i = tid; i < kd_total; i += FUSED_THREADS) {
        means_smem[i] = static_cast<float>(means_in[(size_t)b * K * D + i]);
    }
    for (int i = tid; i < K; i += FUSED_THREADS) {
        var_smem[i]   = var_in  [(size_t)b * K + i];
        log_w_smem[i] = log_w_in[(size_t)b * K + i];
    }
    // Zero per-CTA partial accumulators.
    for (int i = tid; i < kd_total; i += FUSED_THREADS) partial_sums_smem[i]   = 0.0f;
    for (int i = tid; i < K;        i += FUSED_THREADS) {
        partial_sumsq_smem[i]  = 0.0f;
        partial_counts_smem[i] = 0.0f;
    }
    __syncthreads();

    if (my_n < N) {
        // ---------------------------------------------------------------
        // Load x[my_n] into per-thread registers and compute ||x||².
        // ---------------------------------------------------------------
        float x_local[FUSED_MAX_D];
        float xx_local = 0.0f;
        const T* x_row = x + ((size_t)b * N + my_n) * D;
#pragma unroll 4
        for (int d = 0; d < FUSED_MAX_D; ++d) {
            float v = (d < D) ? static_cast<float>(x_row[d]) : 0.0f;
            x_local[d] = v;
            xx_local  += v * v;
        }

        const float TWO_PI = 6.283185307179586f;

        // ---------------------------------------------------------------
        // Phase 1: stable online logsumexp over K.
        // ---------------------------------------------------------------
        float max_so_far = -INFINITY;
        float sumexp     = 0.0f;

        for (int k = 0; k < K; ++k) {
            const float* mu_k = means_smem + k * D;
            float v    = var_smem[k];
            float dist = 0.0f;
#pragma unroll 4
            for (int d = 0; d < FUSED_MAX_D; ++d) {
                if (d < D) {
                    float dx = x_local[d] - mu_k[d];
                    dist += dx * dx;
                }
            }
            float logit = log_w_smem[k]
                        - 0.5f * (float)D * logf(TWO_PI * v)
                        - 0.5f * dist / v;

            if (logit > max_so_far) {
                sumexp     = sumexp * expf(max_so_far - logit) + 1.0f;
                max_so_far = logit;
            } else {
                sumexp += expf(logit - max_so_far);
            }
        }
        float log_norm = max_so_far + logf(sumexp);
        lse_per_sample[(size_t)b * N + my_n] = log_norm;

        // ---------------------------------------------------------------
        // Phase 2: compute r_k, accumulate partials, track argmax.
        // ---------------------------------------------------------------
        float best_logit = -INFINITY;
        int   best_k     = 0;

        for (int k = 0; k < K; ++k) {
            const float* mu_k = means_smem + k * D;
            float v    = var_smem[k];
            float dist = 0.0f;
#pragma unroll 4
            for (int d = 0; d < FUSED_MAX_D; ++d) {
                if (d < D) {
                    float dx = x_local[d] - mu_k[d];
                    dist += dx * dx;
                }
            }
            float logit = log_w_smem[k]
                        - 0.5f * (float)D * logf(TWO_PI * v)
                        - 0.5f * dist / v;

            if (logit > best_logit) {
                best_logit = logit;
                best_k     = k;
            }

            float r = expf(logit - log_norm);
            atomicAdd(partial_counts_smem + k,       r);
            atomicAdd(partial_sumsq_smem  + k,       r * xx_local);
            float* sum_k = partial_sums_smem + k * D;
#pragma unroll 4
            for (int d = 0; d < FUSED_MAX_D; ++d) {
                if (d < D) {
                    atomicAdd(sum_k + d, r * x_local[d]);
                }
            }
        }
        labels[(size_t)b * N + my_n] = best_k;
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // Cooperative SMEM -> global flush via atomicAdd.
    // ---------------------------------------------------------------
    for (int i = tid; i < kd_total; i += FUSED_THREADS) {
        int k = i / D;
        int d = i % D;
        atomicAdd(partial_sums + ((size_t)b * K + k) * D + d, partial_sums_smem[i]);
    }
    for (int i = tid; i < K; i += FUSED_THREADS) {
        atomicAdd(partial_sumsq  + (size_t)b * K + i, partial_sumsq_smem[i]);
        atomicAdd(partial_counts + (size_t)b * K + i, partial_counts_smem[i]);
    }
}

template <typename T>
__global__ void
fused_finalize_kernel(
    const float* __restrict__ partial_sums,   // (B, K, D)
    const float* __restrict__ partial_sumsq,  // (B, K)
    const float* __restrict__ partial_counts, // (B, K) soft counts
    const T*     __restrict__ old_means,      // (B, K, D)
    const float* __restrict__ old_var,        // (B, K)
    T*           __restrict__ new_means,      // (B, K, D)
    float*       __restrict__ new_var,        // (B, K)
    float*       __restrict__ new_weights,    // (B, K)
    int B, int K, int D, int N,
    float reg_covar
) {
    int b = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || k >= K) return;

    float n_k = partial_counts[(size_t)b * K + k];

    if (n_k <= 0.0f) {
        // Empty cluster: preserve old mean/var, set weight 0.
        for (int d = 0; d < D; ++d) {
            new_means[((size_t)b * K + k) * D + d] =
                old_means[((size_t)b * K + k) * D + d];
        }
        new_var    [(size_t)b * K + k] = old_var[(size_t)b * K + k];
        new_weights[(size_t)b * K + k] = 0.0f;
        return;
    }

    float n_inv  = 1.0f / n_k;
    float mu_sq  = 0.0f;
    for (int d = 0; d < D; ++d) {
        float mu_d = partial_sums[((size_t)b * K + k) * D + d] * n_inv;
        mu_sq += mu_d * mu_d;
        new_means[((size_t)b * K + k) * D + d] = static_cast<T>(mu_d);
    }

    float ss      = partial_sumsq[(size_t)b * K + k];
    float var_raw = (ss * n_inv - mu_sq) / (float)D;
    new_var    [(size_t)b * K + k] = fmaxf(var_raw, reg_covar);
    new_weights[(size_t)b * K + k] = n_k / (float)N;
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// Public: fused_safe
// -----------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_safe(const at::Tensor& x,
           const at::Tensor& means,
           const at::Tensor& var,
           const at::Tensor& log_w,
           double reg_covar)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(),
                "fused_safe: x must be contiguous CUDA");
    TORCH_CHECK(means.is_cuda() && means.is_contiguous() &&
                means.scalar_type() == x.scalar_type(),
                "fused_safe: means must be contiguous CUDA, same dtype as x");
    TORCH_CHECK(var.is_cuda() && var.is_contiguous() &&
                var.scalar_type() == at::kFloat,
                "fused_safe: var must be contiguous fp32 CUDA");
    TORCH_CHECK(log_w.is_cuda() && log_w.is_contiguous() &&
                log_w.scalar_type() == at::kFloat,
                "fused_safe: log_w must be contiguous fp32 CUDA");
    TORCH_CHECK(x.dim() == 3, "fused_safe: x must be (B, N, D)");

    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);
    TORCH_CHECK(D <= FUSED_MAX_D,
                "fused_safe requires D <= 64; got ", D);
    TORCH_CHECK(K <= FUSED_MAX_K,
                "fused_safe requires K <= 128; got ", K);
    TORCH_CHECK(D > 0 && K > 0, "fused_safe: D and K must be positive");

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto new_means      = at::empty_like(means);
    auto new_var        = at::empty({B, K}, var.options());
    auto new_weights    = at::empty({B, K}, var.options());
    auto lse_per_sample = at::empty({B, N}, var.options());
    auto labels         = at::empty({B, N}, x.options().dtype(at::kInt));

    if (N == 0) {
        new_means.copy_(means);
        new_var.copy_(var);
        new_weights.zero_();
        lse_per_sample.zero_();
        return std::make_tuple(new_means, new_var, new_weights,
                               lse_per_sample, labels);
    }

    auto partial_sums   = at::zeros({B, K, D}, var.options());
    auto partial_sumsq  = at::zeros({B, K},    var.options());
    auto partial_counts = at::zeros({B, K},    var.options());

    int    n_blocks    = (N + FUSED_BLOCK_N - 1) / FUSED_BLOCK_N;
    dim3   grid(n_blocks, B);
    size_t smem_bytes  = (size_t)(2 * K * D + 4 * K) * sizeof(float);

    auto launch = [&](auto type_tag) {
        using T = typename decltype(type_tag)::type;

        // Request enlarged dynamic SMEM (sm_80+ supports up to ~100 KiB).
        cudaFuncSetAttribute(
            (const void*)fused_safe_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes
        );

        fused_safe_kernel<T><<<grid, FUSED_THREADS, smem_bytes, stream>>>(
            x.data_ptr<T>(),
            means.data_ptr<T>(),
            var.data_ptr<float>(),
            log_w.data_ptr<float>(),
            partial_sums.data_ptr<float>(),
            partial_sumsq.data_ptr<float>(),
            partial_counts.data_ptr<float>(),
            lse_per_sample.data_ptr<float>(),
            labels.data_ptr<int32_t>(),
            B, N, K, D
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Finalize: divide by soft counts, clamp variance.
        constexpr int FIN_THREADS = 64;
        dim3 fin_grid((K + FIN_THREADS - 1) / FIN_THREADS, B);
        fused_finalize_kernel<T><<<fin_grid, FIN_THREADS, 0, stream>>>(
            partial_sums.data_ptr<float>(),
            partial_sumsq.data_ptr<float>(),
            partial_counts.data_ptr<float>(),
            means.data_ptr<T>(),
            var.data_ptr<float>(),
            new_means.data_ptr<T>(),
            new_var.data_ptr<float>(),
            new_weights.data_ptr<float>(),
            B, K, D, N, (float)reg_covar
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    struct FloatTag    { using type = float;        };
    struct HalfTag     { using type = at::Half;     };
    struct BFloat16Tag { using type = at::BFloat16; };

    switch (x.scalar_type()) {
        case at::kFloat:    launch(FloatTag{});    break;
        case at::kHalf:     launch(HalfTag{});     break;
        case at::kBFloat16: launch(BFloat16Tag{}); break;
        default: TORCH_CHECK(false,
                     "fused_safe: unsupported dtype ", x.scalar_type());
    }

    return std::make_tuple(new_means, new_var, new_weights,
                           lse_per_sample, labels);
}

// -----------------------------------------------------------------------
// Stub: fused_sm80 — Plan 5 Task 4 replaces this with the mma variant.
// -----------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_sm80(const at::Tensor& x,
           const at::Tensor& means,
           const at::Tensor& var,
           const at::Tensor& log_w,
           double reg_covar)
{
    return fused_safe(x, means, var, log_w, reg_covar);
}

// -----------------------------------------------------------------------
// Public dispatcher: fused — routes to fused_safe for now.
// Plan 5 Task 5 will add the sm80 gate.
// -----------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused(const at::Tensor& x,
      const at::Tensor& means,
      const at::Tensor& var,
      const at::Tensor& log_w,
      double reg_covar)
{
    return fused_safe(x, means, var, log_w, reg_covar);
}

}}}  // namespace gmmxx::fused::spherical
