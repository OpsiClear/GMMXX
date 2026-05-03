#include "diag.h"

namespace gmmxx { namespace estep { namespace diag {

// Plan 6: route everything to safe. Future task may add an sm80 mma variant.
at::Tensor assign(const at::Tensor& x, const at::Tensor& means,
                  const at::Tensor& var, const at::Tensor& log_w,
                  c10::optional<at::Tensor> out) {
    return assign_safe(x, means, var, log_w, std::move(out));
}

at::Tensor logsumexp(const at::Tensor& x, const at::Tensor& means,
                     const at::Tensor& var, const at::Tensor& log_w,
                     c10::optional<at::Tensor> out) {
    return logsumexp_safe(x, means, var, log_w, std::move(out));
}

at::Tensor resp(const at::Tensor& x, const at::Tensor& means,
                const at::Tensor& var, const at::Tensor& log_w,
                const at::Tensor& log_norm,
                c10::optional<at::Tensor> out) {
    return resp_safe(x, means, var, log_w, log_norm, std::move(out));
}

}}}
