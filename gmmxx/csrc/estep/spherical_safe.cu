#include "spherical.h"
#include "../common/arch.cuh"
#include "../common/reduce.cuh"
#include <cmath>
#include <algorithm>

namespace gmmxx { namespace estep { namespace spherical {

// One thread per (b, n) point. Each thread loops over K clusters,
// computes log p_k(x), and tracks (best_logit, best_idx) for argmax.
//
// log p_k(x) = log_w_k - D/2 * log(2π σ_k²) - 0.5/σ_k² * ||x − μ_k||²
template <typename T>
__global__ void __launch_bounds__(128)
spherical_assign_safe_kernel(
    const T* __restrict__ x,
    const T* __restrict__ means,
    const float* __restrict__ var,
    const float* __restrict__ log_w,
    int32_t* __restrict__ out,
    int B, int N, int K, int D
) {
    int b = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || b >= B) return;

    const T* x_b = x + (size_t)b * N * D;
    const T* means_b = means + (size_t)b * K * D;
    const float* var_b = var + (size_t)b * K;
    const float* log_w_b = log_w + (size_t)b * K;

    const T* x_n = x_b + (size_t)n * D;

    float best = -INFINITY;
    int best_k = 0;
    const float TWO_PI = 6.283185307179586f;

    for (int k = 0; k < K; ++k) {
        const T* mu_k = means_b + (size_t)k * D;
        float v = var_b[k];
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float dx = static_cast<float>(x_n[d]) - static_cast<float>(mu_k[d]);
            dist += dx * dx;
        }
        float logit = log_w_b[k]
                    - 0.5f * (float)D * logf(TWO_PI * v)
                    - 0.5f * dist / v;
        if (logit > best) {
            best = logit;
            best_k = k;
        }
    }
    out[(size_t)b * N + n] = best_k;
}

// One CTA per (b, n). blockDim.x = K (or padded to nearest multiple of 32 >= K).
template <typename T>
__global__ void
spherical_logsumexp_safe_kernel(
    const T* __restrict__ x,
    const T* __restrict__ means,
    const float* __restrict__ var,
    const float* __restrict__ log_w,
    float* __restrict__ out,
    int B, int N, int K, int D
) {
    extern __shared__ float smem[];

    int b = blockIdx.y;
    int n = blockIdx.x;
    int k = threadIdx.x;
    if (b >= B || n >= N) return;

    const T* x_b = x + (size_t)b * N * D;
    const T* means_b = means + (size_t)b * K * D;
    const float* var_b = var + (size_t)b * K;
    const float* log_w_b = log_w + (size_t)b * K;
    const T* x_n = x_b + (size_t)n * D;

    float logit = -INFINITY;
    if (k < K) {
        const T* mu_k = means_b + (size_t)k * D;
        float v = var_b[k];
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float dx = static_cast<float>(x_n[d]) - static_cast<float>(mu_k[d]);
            dist += dx * dx;
        }
        const float TWO_PI = 6.283185307179586f;
        logit = log_w_b[k] - 0.5f * (float)D * logf(TWO_PI * v) - 0.5f * dist / v;
    }

    float lse = ::gmmxx::reduce::logsumexp_block_f32(logit, smem);
    if (k == 0) {
        out[(size_t)b * N + n] = lse;
    }
}

// One thread per (b, n, k).
template <typename T>
__global__ void __launch_bounds__(128)
spherical_resp_safe_kernel(
    const T* __restrict__ x,
    const T* __restrict__ means,
    const float* __restrict__ var,
    const float* __restrict__ log_w,
    const float* __restrict__ log_norm,
    float* __restrict__ out,
    int B, int N, int K, int D
) {
    int b = blockIdx.y;
    int n = blockIdx.x;
    int k = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= B || n >= N || k >= K) return;

    const T* x_b = x + (size_t)b * N * D;
    const T* means_b = means + (size_t)b * K * D;
    const float* var_b = var + (size_t)b * K;
    const float* log_w_b = log_w + (size_t)b * K;
    const T* x_n = x_b + (size_t)n * D;
    const T* mu_k = means_b + (size_t)k * D;

    float v = var_b[k];
    float dist = 0.0f;
    for (int d = 0; d < D; ++d) {
        float dx = static_cast<float>(x_n[d]) - static_cast<float>(mu_k[d]);
        dist += dx * dx;
    }
    const float TWO_PI = 6.283185307179586f;
    float logit = log_w_b[k] - 0.5f * (float)D * logf(TWO_PI * v) - 0.5f * dist / v;
    float lz = log_norm[(size_t)b * N + n];
    out[((size_t)b * N + n) * K + k] = expf(logit - lz);
}

// ---------------------------------------------------------------------
// Host launchers — dispatch by dtype.
// ---------------------------------------------------------------------

namespace {

void _check_inputs(const at::Tensor& x, const at::Tensor& means,
                   const at::Tensor& var, const at::Tensor& log_w) {
    TORCH_CHECK(x.is_cuda(), "x must be on a CUDA device");
    TORCH_CHECK(x.is_contiguous() && means.is_contiguous() &&
                var.is_contiguous() && log_w.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(means.scalar_type() == x.scalar_type(),
                "means must match x dtype");
    TORCH_CHECK(var.scalar_type() == at::kFloat &&
                log_w.scalar_type() == at::kFloat,
                "var and log_w must be float32");
    TORCH_CHECK(x.dim() == 3 && means.dim() == 3,
                "x must be (B,N,D); means must be (B,K,D)");
    TORCH_CHECK(var.dim() == 2 && log_w.dim() == 2,
                "var must be (B,K); log_w must be (B,K)");
    TORCH_CHECK(x.size(0) == means.size(0),
                "x and means must agree on batch dim");
    TORCH_CHECK(x.size(2) == means.size(2),
                "x and means must agree on D");
    TORCH_CHECK(var.size(0) == x.size(0) && var.size(1) == means.size(1),
                "var must be (B,K)");
    TORCH_CHECK(log_w.sizes() == var.sizes(),
                "log_w must match var shape");
}

template <typename T>
void launch_assign(const at::Tensor& x, const at::Tensor& means,
                   const at::Tensor& var, const at::Tensor& log_w,
                   at::Tensor& out, cudaStream_t stream) {
    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);
    if (N == 0) return;  // empty input — output already has correct shape (B,0)
    constexpr int kThreads = 128;
    dim3 grid((N + kThreads - 1) / kThreads, B);
    spherical_assign_safe_kernel<T><<<grid, kThreads, 0, stream>>>(
        x.data_ptr<T>(), means.data_ptr<T>(),
        var.data_ptr<float>(), log_w.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        B, N, K, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     at::Tensor& out, cudaStream_t stream) {
    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);
    if (N == 0) return;  // empty input
    int threads = ((K + ::gmmxx::kWarp - 1) / ::gmmxx::kWarp) * ::gmmxx::kWarp;
    threads = std::min(threads, 1024);
    int n_warps = (threads + ::gmmxx::kWarp - 1) / ::gmmxx::kWarp;
    size_t smem_bytes = n_warps * sizeof(float);
    dim3 grid(N, B);
    spherical_logsumexp_safe_kernel<T><<<grid, threads, smem_bytes, stream>>>(
        x.data_ptr<T>(), means.data_ptr<T>(),
        var.data_ptr<float>(), log_w.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, K, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_resp(const at::Tensor& x, const at::Tensor& means,
                 const at::Tensor& var, const at::Tensor& log_w,
                 const at::Tensor& log_norm, at::Tensor& out, cudaStream_t stream) {
    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);
    if (N == 0) return;  // empty input
    constexpr int kThreads = 64;
    dim3 grid(N, B, (K + kThreads - 1) / kThreads);
    spherical_resp_safe_kernel<T><<<grid, kThreads, 0, stream>>>(
        x.data_ptr<T>(), means.data_ptr<T>(),
        var.data_ptr<float>(), log_w.data_ptr<float>(),
        log_norm.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, K, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // anonymous namespace

at::Tensor assign(const at::Tensor& x, const at::Tensor& means,
                  const at::Tensor& var, const at::Tensor& log_w,
                  c10::optional<at::Tensor> out) {
    _check_inputs(x, means, var, log_w);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto B = x.size(0);
    auto N = x.size(1);
    at::Tensor result;
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == at::kInt &&
                    out->sizes() == at::IntArrayRef({B, N}),
                    "out must be int32 (B,N)");
        result = *out;
    } else {
        result = at::empty({B, N}, x.options().dtype(at::kInt));
    }

    switch (x.scalar_type()) {
        case at::kFloat:    launch_assign<float>(x, means, var, log_w, result, stream); break;
        case at::kHalf:     launch_assign<at::Half>(x, means, var, log_w, result, stream); break;
        case at::kBFloat16: launch_assign<at::BFloat16>(x, means, var, log_w, result, stream); break;
        default: TORCH_CHECK(false, "spherical.assign: unsupported dtype ", x.scalar_type());
    }
    return result;
}

at::Tensor logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     c10::optional<at::Tensor> out) {
    _check_inputs(x, means, var, log_w);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto B = x.size(0);
    auto N = x.size(1);
    at::Tensor result;
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == at::kFloat &&
                    out->sizes() == at::IntArrayRef({B, N}),
                    "out must be float32 (B,N)");
        result = *out;
    } else {
        result = at::empty({B, N}, x.options().dtype(at::kFloat));
    }

    switch (x.scalar_type()) {
        case at::kFloat:    launch_logsumexp<float>(x, means, var, log_w, result, stream); break;
        case at::kHalf:     launch_logsumexp<at::Half>(x, means, var, log_w, result, stream); break;
        case at::kBFloat16: launch_logsumexp<at::BFloat16>(x, means, var, log_w, result, stream); break;
        default: TORCH_CHECK(false, "spherical.logsumexp: unsupported dtype ", x.scalar_type());
    }
    return result;
}

at::Tensor resp(const at::Tensor& x, const at::Tensor& means,
                const at::Tensor& var, const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out) {
    _check_inputs(x, means, var, log_w);
    TORCH_CHECK(log_norm.is_cuda() && log_norm.is_contiguous() &&
                log_norm.scalar_type() == at::kFloat &&
                log_norm.sizes() == at::IntArrayRef({x.size(0), x.size(1)}),
                "log_norm must be contiguous fp32 (B,N)");
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto B = x.size(0);
    auto N = x.size(1);
    auto K = means.size(1);
    at::Tensor result;
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == at::kFloat &&
                    out->sizes() == at::IntArrayRef({B, N, K}),
                    "out must be float32 (B,N,K)");
        result = *out;
    } else {
        result = at::empty({B, N, K}, x.options().dtype(at::kFloat));
    }

    switch (x.scalar_type()) {
        case at::kFloat:    launch_resp<float>(x, means, var, log_w, log_norm, result, stream); break;
        case at::kHalf:     launch_resp<at::Half>(x, means, var, log_w, log_norm, result, stream); break;
        case at::kBFloat16: launch_resp<at::BFloat16>(x, means, var, log_w, log_norm, result, stream); break;
        default: TORCH_CHECK(false, "spherical.resp: unsupported dtype ", x.scalar_type());
    }
    return result;
}

}}}  // namespace gmmxx::estep::spherical
