// nanobind module entry point for gmmxx._C.
//
// Plan 1 exposes only the canary kernel (smoke test). Plan 2 onwards adds
// real E-step / M-step / fused / approx ops.

#include "nb_torch.h"
#include "canary/canary.h"
#include "estep/spherical.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
    m.doc() = "gmmxx CUDA kernel bindings";

    m.def(
        "canary_add_offset",
        &gmmxx::canary::add_offset,
        nb::arg("input"),
        nb::arg("offset"),
        "Smoke-test kernel: returns input + offset element-wise (int32).");

    m.def(
        "spherical_assign",
        &gmmxx::estep::spherical::assign,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("out") = nb::none(),
        "Spherical E-step assign: argmax of log p_k(x) over k. Returns int32 (B,N).");

    m.def(
        "spherical_logsumexp",
        &gmmxx::estep::spherical::logsumexp,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("out") = nb::none(),
        "Spherical E-step stable logsumexp over k. Returns fp32 (B,N).");

    m.def(
        "spherical_resp",
        &gmmxx::estep::spherical::resp,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("log_norm"),
        nb::arg("out") = nb::none(),
        "Spherical E-step responsibilities r_{n,k}. Returns fp32 (B,N,K).");
}
