// nanobind module entry point for gmmxx._C.
//
// Plan 1 exposes only the canary kernel (smoke test). Plan 2 onwards adds
// real E-step / M-step / fused / approx ops.

#include "nb_torch.h"
#include "canary/canary.h"
#include "estep/spherical.h"
#include "estep/diag.h"
#include "mstep/spherical.h"
#include "mstep/diag.h"
#include "fused/spherical.h"
#include "em/spherical_em.h"

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
        nb::arg("x_sq") = nb::none(),
        nb::arg("c_sq") = nb::none(),
        "Spherical E-step stable logsumexp over k. Returns fp32 (B,N). "
        "Pass cached x_sq=(B,N) and/or c_sq=(B,K) fp32 to skip the "
        "per-call .to(float).pow(2).sum(-1) recompute on the sm80 path.");

    m.def(
        "spherical_resp",
        &gmmxx::estep::spherical::resp,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("log_norm"),
        nb::arg("out") = nb::none(),
        nb::arg("x_sq") = nb::none(),
        nb::arg("c_sq") = nb::none(),
        "Spherical E-step responsibilities r_{n,k}. Returns fp32 (B,N,K). "
        "Pass cached x_sq/c_sq to skip the per-call recompute on the sm80 path.");

    m.def(
        "prepare_spherical_estep",
        &gmmxx::estep::spherical::prepare_estep,
        nb::arg("log_w"),
        nb::arg("means"),
        nb::arg("var"),
        "Soft-EM E-step prep: returns (alpha (B,K), means_aug (B,K,D+1)) "
        "for the augmented cuBLAS GEMM. Single kernel; replaces ~10 small "
        "torch ops in the soft-EM E-step prep block.");

    m.def(
        "spherical_logsumexp_resp_sm80",
        &gmmxx::estep::spherical::logsumexp_resp_sm80,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("x_sq"),
        nb::arg("c_sq"),
        nb::arg("need_lse"),
        "Exp62 fused E-step: returns (lse (B,N) or empty if !need_lse, "
        "resp (B,N,K)) in one sm80 mma kernel launch — saves the "
        "redundant GEMM that the separate logsumexp+resp pipeline runs. "
        "Constraint: K must fit in one BLOCK_K of an available tile "
        "(<=64 in current build); returns two empty tensors when no tile "
        "fits, signalling the caller to fall back.");

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
        "blocked_update_spherical_sorted",
        &gmmxx::mstep::spherical::blocked_update_sorted,
        nb::arg("x_sorted"),
        nb::arg("sorted_ids"),
        nb::arg("sums_out"),
        nb::arg("sumsq_out"),
        nb::arg("counts_out"),
        "Sorted-run spherical M-step. Caller pre-sorts cluster_ids and gathers x.");

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

    m.def(
        "finalize_spherical_soft",
        &gmmxx::mstep::spherical::finalize_soft,
        nb::arg("sums"),
        nb::arg("sumsq"),
        nb::arg("nk"),
        nb::arg("total_n"),
        nb::arg("reg_covar"),
        "Soft-EM finalize. fp32 nk (fractional cluster mass). Returns "
        "(means, var, weights, log_w). Single kernel; replaces ~12 small "
        "torch ops in the spherical soft-EM finalize block.");

    m.def(
        "spherical_fused",
        &gmmxx::fused::spherical::fused,
        nb::arg("x"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("reg_covar"),
        "Fused spherical E/M single-tile kernel. Returns (means, var, weights, "
        "lse_per_sample, labels). Requires D <= 64, K <= 128.");

    m.def("blocked_update_diag", &gmmxx::mstep::diag::blocked_update,
          nb::arg("x"), nb::arg("cluster_ids"),
          nb::arg("sums_out"), nb::arg("sumsq_out"), nb::arg("counts_out"),
          "Diagonal M-step accumulator. Caller MUST zero sums_out (B,K,D), "
          "sumsq_out (B,K,D), counts_out (B,K) before calling.");

    m.def("blocked_update_diag_sorted", &gmmxx::mstep::diag::blocked_update_sorted,
          nb::arg("x_sorted"), nb::arg("sorted_ids"),
          nb::arg("sums_out"), nb::arg("sumsq_out"), nb::arg("counts_out"),
          "Sorted-run diagonal M-step. Caller pre-sorts cluster_ids and gathers x.");

    m.def("finalize_diag", &gmmxx::mstep::diag::finalize,
          nb::arg("sums"), nb::arg("sumsq"), nb::arg("counts"),
          nb::arg("old_means"), nb::arg("old_var"),
          nb::arg("total_n"), nb::arg("reg_covar"),
          "Finalize diagonal M-step. Returns (means (B,K,D), var (B,K,D), weights (B,K)).");

    m.def("diag_assign", &gmmxx::estep::diag::assign,
          nb::arg("x"), nb::arg("means"), nb::arg("var"), nb::arg("log_w"),
          nb::arg("out") = nb::none(),
          "Diagonal E-step assign. Returns int32 (B, N).");

    m.def("diag_logsumexp", &gmmxx::estep::diag::logsumexp,
          nb::arg("x"), nb::arg("means"), nb::arg("var"), nb::arg("log_w"),
          nb::arg("out") = nb::none(),
          "Diagonal E-step stable logsumexp. Returns fp32 (B, N).");

    m.def("diag_resp", &gmmxx::estep::diag::resp,
          nb::arg("x"), nb::arg("means"), nb::arg("var"), nb::arg("log_w"),
          nb::arg("log_norm"), nb::arg("out") = nb::none(),
          "Diagonal E-step responsibilities. Returns fp32 (B, N, K).");

    m.def(
        "spherical_em_soft_chunked",
        &gmmxx::em::spherical::soft_chunked,
        nb::arg("x_estep_aug"),
        nb::arg("x_estep_aug_bf16"),
        nb::arg("x_aug"),
        nb::arg("means"),
        nb::arg("var"),
        nb::arg("log_w"),
        nb::arg("n_iter"),
        nb::arg("reg_covar"),
        nb::arg("chunk_size"),
        nb::arg("need_final_lse"),
        "Exp63: C++-side EM-loop driver for spherical soft-EM cuBLAS chunked "
        "fastpath. Runs n_iter iterations end-to-end with no per-iter Python "
        "dispatch. Returns (means, var, weights, lower_bound, final_lse_or_empty).");

}
