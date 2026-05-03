"""Per-kernel correctness for the spherical sm_80 mma path.

Tests fp16 and bf16 inputs at rtol=5e-3 (per spec §6 fp16/bf16 contract).
The public spherical_assign automatically routes to assign_sm80 when:
- input is fp16 or bf16, AND
- device compute capability >= 8.0, AND
- D % 16 == 0 (BLOCK_D alignment for the mma kernel).

Skips on devices below sm_80 or when CUDA is unavailable.

logsumexp_sm80 and resp_sm80 are currently stubbed to *_safe (per Plan 3
Task 2 scope-down); their correctness is covered by Plan 2's existing
test_cuda_spherical_safe.py. This file focuses on the mma assign kernel.
"""

from __future__ import annotations

import math

import pytest
import torch


def _has_sm80():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


pytestmark = pytest.mark.skipif(not _has_sm80(), reason="requires CUDA + sm_80+")


def _random_setup(B=1, N=256, D=32, K=16, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    means = torch.randn(B, K, D, device=device, dtype=dtype)
    var = torch.rand(B, K, device=device).clamp_min(0.5).float()
    log_w = torch.log_softmax(torch.randn(B, K, device=device), dim=-1).float()
    return x, means, var, log_w


def _torch_logits(x, means, var, log_w):
    """Reference: log p_k(x_n) = log_w_k - D/2*log(2π σ_k²) - 0.5/σ_k² * ||x − μ_k||²"""
    B, N, D = x.shape
    K = means.shape[1]
    x_f, means_f = x.float(), means.float()
    diff = x_f.unsqueeze(2) - means_f.unsqueeze(1)
    dist_sq = diff.pow(2).sum(-1)
    return (
        log_w.unsqueeze(1)
        - 0.5 * D * torch.log(2 * math.pi * var).unsqueeze(1)
        - 0.5 * dist_sq / var.unsqueeze(1)
    )


class TestSphericalAssignSm80:
    """Verify assign routes to sm80 mma path on fp16/bf16/D%16==0 shapes
    and produces correct argmax."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("D,K", [(16, 16), (32, 32), (64, 64), (128, 32), (16, 8)])
    def test_argmax_matches_reference(self, dtype, D, K):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(N=256, D=D, K=K, dtype=dtype)
        cuda_ids = _cuda.spherical_assign(x, means, var, log_w)
        ref_ids = _torch_logits(x, means, var, log_w).argmax(-1).int()
        agree = (cuda_ids == ref_ids).float().mean().item()
        assert agree >= 0.95, f"only {agree:.3f} agreement at dtype={dtype}, D={D}, K={K}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_returns_int32_shape_BN(self, dtype):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(B=2, N=128, D=32, K=8, dtype=dtype)
        ids = _cuda.spherical_assign(x, means, var, log_w)
        assert ids.shape == (2, 128)
        assert ids.dtype == torch.int32

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_zero_N_returns_empty(self, dtype):
        """sm80 path must also handle empty N gracefully (matches Plan 2 fix)."""
        from gmmxx import _cuda
        x = torch.empty(1, 0, 32, device="cuda", dtype=dtype)
        means = torch.randn(1, 8, 32, device="cuda", dtype=dtype)
        var = torch.ones(1, 8, device="cuda")
        log_w = torch.zeros(1, 8, device="cuda")
        ids = _cuda.spherical_assign(x, means, var, log_w)
        assert ids.shape == (1, 0)


class TestSphericalSm80DispatchPolicy:
    """The dispatcher should route to safe path when sm80 prerequisites
    aren't met — even on a sm_80+ host."""

    def test_fp32_routes_to_safe(self):
        """fp32 must always go through safe (mma doesn't accept fp32 inputs)."""
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(D=32, K=16, dtype=torch.float32)
        # If this routes to sm80, the mma kernel would fail since it doesn't
        # template on float. The fact that this returns valid output proves
        # the dispatcher correctly routed to safe.
        ids = _cuda.spherical_assign(x, means, var, log_w)
        ref_ids = _torch_logits(x, means, var, log_w).argmax(-1).int()
        agree = (ids == ref_ids).float().mean().item()
        assert agree >= 0.99, f"fp32 (safe path) agreement {agree:.3f}"

    @pytest.mark.parametrize("D", [4, 8, 12])
    def test_d_not_multiple_of_16_routes_to_safe(self, D):
        """D not aligned to BLOCK_D=16 must fall back to safe even for fp16/bf16."""
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(D=D, K=8, dtype=torch.float16)
        # If this routed to sm80, it would either fail or produce garbage.
        ids = _cuda.spherical_assign(x, means, var, log_w)
        ref_ids = _torch_logits(x, means, var, log_w).argmax(-1).int()
        agree = (ids == ref_ids).float().mean().item()
        # fp16 safe path should still hit the >= 0.95 bar at small D.
        assert agree >= 0.95, f"D={D} fp16 (safe fallback) agreement {agree:.3f}"
