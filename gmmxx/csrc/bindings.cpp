// nanobind module entry point for gmmxx._C.
//
// Plan 1 exposes only the canary kernel (smoke test). Plan 2 onwards adds
// real E-step / M-step / fused / approx ops.

#include "nb_torch.h"
#include "canary/canary.h"
#include "estep/spherical.h"
#include "mstep/spherical.h"

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

    m.def(
        "blocked_update_spherical",
        &gmmxx::mstep::spherical::blocked_update,
        nb::arg("x"),
        nb::arg("cluster_ids"),
        nb::arg("sums_out"),
        nb::arg("sumsq_out"),
        nb::arg("counts_out"),
        "Spherical M-step accumulator (per-token atomicAdd). Caller MUST zero "
        "sums_out/sumsq_out/counts_out before calling.");

    m.def(
        "finalize_spherical",
        &gmmxx::mstep::spherical::finalize,
        nb::arg("sums"),
        nb::arg("sumsq"),
        nb::arg("counts"),
        nb::arg("old_means"),
        nb::arg("old_var"),
        nb::arg("total_n"),
        nb::arg("reg_covar"),
        "Finalize spherical M-step. Returns (means, var, weights). "
        "Empty clusters preserve previous (means, var) and get weight 0.");

    // sm_80 mma-path E-step (Plan 3 Task 2 — internal; Task 3 will route
    // through spherical_assign etc. instead of these direct entry-points).
    m.def(
        "spherical_assign_sm80",
        &gmmxx::estep::spherical::assign_sm80,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("x_sq"),
        nb::arg("c_sq"),
        nb::arg("out") = nb::none(),
        "internal — use spherical_assign instead. "
        "sm_80 mma.sync E-step assign for fp16/bf16. Returns int32 (B,N).");

    m.def(
        "spherical_logsumexp_sm80",
        &gmmxx::estep::spherical::logsumexp_sm80,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("x_sq"),
        nb::arg("c_sq"),
        nb::arg("out") = nb::none(),
        "internal — use spherical_logsumexp instead. "
        "sm_80 mma.sync E-step logsumexp (currently stubbed to safe path).");

    m.def(
        "spherical_resp_sm80",
        &gmmxx::estep::spherical::resp_sm80,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("x_sq"),
        nb::arg("c_sq"),
        nb::arg("log_norm"),
        nb::arg("out") = nb::none(),
        "internal — use spherical_resp instead. "
        "sm_80 mma.sync E-step resp (currently stubbed to safe path).");
}
