#include "spherical.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace estep { namespace spherical {

namespace {

// Returns true if the calling device has compute capability >= 8.0.
bool _device_has_sm80(at::DeviceIndex idx) {
    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, idx);
    if (err != cudaSuccess) return false;
    return prop.major >= 8;
}

bool _is_fp16_or_bf16(const at::Tensor& t) {
    return t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16;
}

// assign_sm80 requires D % 16 == 0 (BLOCK_D restriction from mma tile layout).
bool _can_use_sm80(const at::Tensor& x) {
    return _is_fp16_or_bf16(x) &&
           x.size(-1) % 16 == 0 &&
           _device_has_sm80(x.device().index());
}

}  // anonymous namespace

at::Tensor assign(const at::Tensor& x, const at::Tensor& means,
                  const at::Tensor& var, const at::Tensor& log_w,
                  c10::optional<at::Tensor> out) {
    if (_can_use_sm80(x)) {
        auto x_sq = x.to(at::kFloat).pow(2).sum(-1).contiguous();
        auto c_sq = means.to(at::kFloat).pow(2).sum(-1).contiguous();
        return assign_sm80(x, means, var, log_w, x_sq, c_sq, std::move(out));
    }
    return assign_safe(x, means, var, log_w, std::move(out));
}

at::Tensor logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     c10::optional<at::Tensor> out) {
    if (_can_use_sm80(x)) {
        auto x_sq = x.to(at::kFloat).pow(2).sum(-1).contiguous();
        auto c_sq = means.to(at::kFloat).pow(2).sum(-1).contiguous();
        return logsumexp_sm80(x, means, var, log_w, x_sq, c_sq, std::move(out));
    }
    return logsumexp_safe(x, means, var, log_w, std::move(out));
}

at::Tensor resp(const at::Tensor& x, const at::Tensor& means,
                const at::Tensor& var, const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out) {
    if (_can_use_sm80(x)) {
        auto x_sq = x.to(at::kFloat).pow(2).sum(-1).contiguous();
        auto c_sq = means.to(at::kFloat).pow(2).sum(-1).contiguous();
        return resp_sm80(x, means, var, log_w, x_sq, c_sq, log_norm, std::move(out));
    }
    return resp_safe(x, means, var, log_w, log_norm, std::move(out));
}

}}}  // namespace gmmxx::estep::spherical
