// Exp63: C++-side EM-loop driver for the spherical soft-EM cuBLAS chunked
// fastpath. See spherical_em.h for the public contract.
//
// Mirrors the Python loop in gmmxx/_cuda.py soft_update_spherical's chunked
// branch but stays in C++ across all iterations. The cuBLAS / softmax / bmm
// kernels themselves are unchanged — we just remove the per-iter Python
// dispatch + ~14 small (B,K) torch ops in the M-step finalize chain.

#include "spherical_em.h"
#include "../estep/spherical.h"
#include "../mstep/spherical.h"

#include <ATen/ops/addmm.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/log.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/zeros.h>

namespace gmmxx { namespace em { namespace spherical {

namespace {

inline int64_t default_chunk_size(int64_t K) {
    // Target ~32 MB per (chunk, K) fp32 logits tile so it fits in Ada's
    // 72 MB L2 across the softmax -> bmm hop (matches the Python heuristic).
    constexpr int64_t kTargetBytes = 32LL * 1024LL * 1024LL;
    int64_t cs = kTargetBytes / (K * 4LL);
    if (cs < 65536) cs = 65536;
    return cs;
}

}  // anonymous namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, double, at::Tensor>
soft_chunked(
    const at::Tensor& x_estep_aug,
    const at::Tensor& x_estep_aug_bf16,
    const at::Tensor& x_aug,
    at::Tensor means,
    at::Tensor var,
    at::Tensor log_w,
    int64_t n_iter,
    double reg_covar,
    int64_t chunk_size_arg,
    bool need_final_lse) {

    TORCH_CHECK(x_estep_aug.is_cuda() && x_estep_aug.is_contiguous() &&
                x_estep_aug.scalar_type() == at::kFloat,
                "x_estep_aug must be contiguous fp32 CUDA");
    TORCH_CHECK(x_aug.is_cuda() && x_aug.is_contiguous() &&
                x_aug.scalar_type() == at::kFloat,
                "x_aug must be contiguous fp32 CUDA");
    TORCH_CHECK(means.is_cuda() && means.is_contiguous() &&
                means.scalar_type() == at::kFloat,
                "means must be contiguous fp32 CUDA");
    TORCH_CHECK(var.is_cuda() && var.is_contiguous() &&
                var.scalar_type() == at::kFloat,
                "var must be contiguous fp32 CUDA");
    TORCH_CHECK(log_w.is_cuda() && log_w.is_contiguous() &&
                log_w.scalar_type() == at::kFloat,
                "log_w must be contiguous fp32 CUDA");
    TORCH_CHECK(x_estep_aug.dim() == 3 && x_aug.dim() == 3,
                "x_estep_aug and x_aug must be (B, N, *)");

    const int64_t B = x_estep_aug.size(0);
    const int64_t N = x_estep_aug.size(1);
    const int64_t Dp1 = x_estep_aug.size(2);
    const int64_t Dp2 = x_aug.size(2);
    TORCH_CHECK(B == 1, "Exp63 driver currently only supports B=1");
    TORCH_CHECK(Dp2 == Dp1 + 1, "x_aug width must be x_estep_aug width + 1");
    const int64_t D = Dp1 - 1;
    const int64_t K = means.size(1);

    const bool use_bf16_cache = x_estep_aug_bf16.defined()
                                && x_estep_aug_bf16.numel() > 0;
    if (use_bf16_cache) {
        TORCH_CHECK(x_estep_aug_bf16.is_cuda() && x_estep_aug_bf16.is_contiguous() &&
                    x_estep_aug_bf16.scalar_type() == at::kBFloat16,
                    "x_estep_aug_bf16 must be contiguous bf16 CUDA");
        TORCH_CHECK(x_estep_aug_bf16.size(0) == B &&
                    x_estep_aug_bf16.size(1) == N &&
                    x_estep_aug_bf16.size(2) == Dp1,
                    "x_estep_aug_bf16 shape must match x_estep_aug");
    }

    const int64_t chunk_size = chunk_size_arg > 0
                                   ? chunk_size_arg
                                   : default_chunk_size(K);

    auto means_options = means.options();
    at::Tensor weights;
    at::Tensor lse_final;
    double lower_bound = 0.0;

    // Single-batch slices for cuBLAS at::mm / at::addmm calls.
    const at::Tensor x_estep_aug_2d = x_estep_aug.select(0, 0);          // (N, D+1) fp32
    const at::Tensor x_aug_3d       = x_aug;                              // (1, N, D+2)
    at::Tensor x_estep_aug_bf16_2d;
    if (use_bf16_cache) {
        x_estep_aug_bf16_2d = x_estep_aug_bf16.select(0, 0);              // (N, D+1) bf16
    }

    // Pre-allocated accumulator reused across iters.
    auto sum_aug_acc = at::zeros({B, K, Dp2}, means_options);

    for (int64_t it = 0; it < n_iter; ++it) {
        // (1) E-step prep: alpha (B,K), means_aug (B,K,D+1).
        auto [alpha, means_aug] =
            gmmxx::estep::spherical::prepare_estep(log_w, means, var);
        const at::Tensor alpha_2d     = alpha.select(0, 0);               // (K)
        const at::Tensor means_aug_2d = means_aug.select(0, 0);           // (K, D+1)
        at::Tensor means_aug_bf16_2d;
        at::Tensor means_aug_bf16_t;
        if (use_bf16_cache) {
            means_aug_bf16_2d = means_aug_2d.to(at::kBFloat16);
            means_aug_bf16_t = means_aug_bf16_2d.transpose(0, 1);
        }
        // Python doesn't .contiguous() the transpose; cuBLAS handles strided.
        const at::Tensor means_aug_2d_t = means_aug_2d.transpose(0, 1);   // (D+1, K) view

        // (2) Chunked logits + softmax + partial bmm into sum_aug_acc.
        sum_aug_acc.zero_();
        const bool need_lse_this_iter = (need_final_lse && it == n_iter - 1);
        at::Tensor lse_acc;
        if (need_lse_this_iter) {
            lse_acc = at::empty({B, N}, means_options);
        }

        for (int64_t n_start = 0; n_start < N; n_start += chunk_size) {
            const int64_t n_end = std::min(n_start + chunk_size, N);
            const int64_t cn    = n_end - n_start;

            at::Tensor logits_chunk_2d;  // (cn, K) fp32
            if (use_bf16_cache) {
                auto x_chunk_bf16 = x_estep_aug_bf16_2d.narrow(0, n_start, cn);
                auto cross = at::mm(x_chunk_bf16, means_aug_bf16_t).to(at::kFloat);
                logits_chunk_2d = cross.add_(alpha_2d);  // broadcast (K,) over rows
            } else {
                auto x_chunk_f = x_estep_aug_2d.narrow(0, n_start, cn);
                logits_chunk_2d = at::addmm(alpha_2d, x_chunk_f, means_aug_2d_t);
            }
            auto logits_chunk = logits_chunk_2d.unsqueeze(0);  // (1, cn, K)

            if (need_lse_this_iter) {
                // logsumexp(logits_chunk_2d, dim=1) reduces the K dimension
                // (since shape is (cn, K) → reducing dim=1 = K-axis → (cn,)).
                auto lse_chunk = at::logsumexp(logits_chunk_2d, /*dim=*/1, /*keepdim=*/false);
                lse_acc.select(0, 0).narrow(0, n_start, cn).copy_(lse_chunk);
            }

            auto resp_chunk = at::softmax(logits_chunk, /*dim=*/-1);  // (1, cn, K)
            auto x_aug_chunk = x_aug_3d.narrow(1, n_start, cn);        // (1, cn, D+2)
            auto bmm_part = at::bmm(resp_chunk.transpose(1, 2), x_aug_chunk);  // (1, K, D+2)
            sum_aug_acc.add_(bmm_part);
        }

        // (3) Extract sum_x (B,K,D), sum_x_sq (B,K), nk (B,K) from sum_aug_acc.
        auto sum_x    = sum_aug_acc.narrow(2, 0,  D).contiguous();
        auto sum_x_sq = sum_aug_acc.select(2, D).contiguous();
        auto nk       = sum_aug_acc.select(2, D + 1).contiguous();

        // (4) M-step finalize. Returns means, var, weights, log_w in one launch.
        auto [means_new, var_new, weights_new, log_w_new] =
            gmmxx::mstep::spherical::finalize_soft(
                sum_x, sum_x_sq, nk, /*total_n=*/(int64_t)N, reg_covar);

        means   = means_new;
        var     = var_new;
        weights = weights_new;
        log_w   = log_w_new;

        if (need_lse_this_iter) {
            lse_final = lse_acc;
        }
    }

    if (need_final_lse && lse_final.defined()) {
        lower_bound = lse_final.mean().item<double>();
    } else {
        lse_final = at::empty({0}, means_options);
    }

    return std::make_tuple(means, var, weights, lower_bound, lse_final);
}

}}}  // namespace gmmxx::em::spherical
