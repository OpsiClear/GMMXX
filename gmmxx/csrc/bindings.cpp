// nanobind module entry point for gmmxx._C.
//
// Plan 1 exposes only the canary kernel (smoke test). Plan 2 onwards adds
// real E-step / M-step / fused / approx ops.

#include "nb_torch.h"
#include "canary/canary.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
    m.doc() = "gmmxx CUDA kernel bindings";

    m.def(
        "canary_add_offset",
        &gmmxx::canary::add_offset,
        nb::arg("input"),
        nb::arg("offset"),
        "Smoke-test kernel: returns input + offset element-wise (int32).");
}
