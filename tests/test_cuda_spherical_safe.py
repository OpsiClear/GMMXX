"""Per-kernel correctness tests for the spherical CUDA safe path.

Compares CUDA outputs to torch_fallback reference at fp32 rtol=1e-4 / atol=1e-4
(per spec §6 numerical contract).
"""

from __future__ import annotations

import math

import pytest
import torch


def _has_cuda() -> bool:
    try:
        from gmmxx import _cuda
        return _cuda.has_cuda()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA + gmmxx._C")


def _random_setup(B=1, N=64, D=4, K=3, dtype=torch.float32, seed=0):
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
    x_f = x.float()
    means_f = means.float()
    diff = x_f.unsqueeze(2) - means_f.unsqueeze(1)  # (B,N,K,D)
    dist_sq = diff.pow(2).sum(-1)                     # (B,N,K)
    return (
        log_w.unsqueeze(1)
        - 0.5 * D * torch.log(2 * math.pi * var).unsqueeze(1)
        - 0.5 * dist_sq / var.unsqueeze(1)
    )


class TestSphericalAssign:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_argmax_matches_torch_reference(self, dtype):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(dtype=dtype)
        cuda_ids = _cuda.spherical_assign(x, means, var, log_w)
        ref_ids = _torch_logits(x, means, var, log_w).argmax(-1).int()
        agree = (cuda_ids == ref_ids).float().mean().item()
        threshold = 0.99 if dtype == torch.float32 else 0.95
        assert agree >= threshold, f"only {agree:.3f} agreement"

    def test_returns_int32_shape_BN(self):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(B=2, N=32, D=4, K=5)
        ids = _cuda.spherical_assign(x, means, var, log_w)
        assert ids.shape == (2, 32)
        assert ids.dtype == torch.int32

    def test_zero_N_returns_empty(self):
        from gmmxx import _cuda
        x = torch.empty(1, 0, 4, device="cuda")
        means = torch.randn(1, 3, 4, device="cuda")
        var = torch.ones(1, 3, device="cuda")
        log_w = torch.zeros(1, 3, device="cuda")
        ids = _cuda.spherical_assign(x, means, var, log_w)
        assert ids.shape == (1, 0)


class TestSphericalLogsumexp:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_matches_torch_reference(self, dtype):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(dtype=dtype)
        cuda_lse = _cuda.spherical_logsumexp(x, means, var, log_w)
        ref_lse = _torch_logits(x, means, var, log_w).logsumexp(-1)
        rtol = 1e-4 if dtype == torch.float32 else 1e-2
        atol = 1e-4 if dtype == torch.float32 else 1e-2
        assert torch.allclose(cuda_lse, ref_lse, rtol=rtol, atol=atol), (
            f"max diff: {(cuda_lse - ref_lse).abs().max().item()}"
        )

    def test_returns_float32_shape_BN(self):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(B=2, N=32, K=4, D=2)
        lse = _cuda.spherical_logsumexp(x, means, var, log_w)
        assert lse.shape == (2, 32)
        assert lse.dtype == torch.float32


class TestSphericalResp:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_resp_sums_to_one_per_row(self, dtype):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(dtype=dtype)
        lse = _cuda.spherical_logsumexp(x, means, var, log_w)
        r = _cuda.spherical_resp(x, means, var, log_w, lse)
        sums = r.sum(-1)
        atol = 1e-4 if dtype == torch.float32 else 1e-2
        assert torch.allclose(sums, torch.ones_like(sums), atol=atol)

    def test_matches_torch_reference(self):
        from gmmxx import _cuda
        x, means, var, log_w = _random_setup(dtype=torch.float32)
        lse = _cuda.spherical_logsumexp(x, means, var, log_w)
        cuda_r = _cuda.spherical_resp(x, means, var, log_w, lse)
        ref_logits = _torch_logits(x, means, var, log_w)
        ref_r = (ref_logits - lse.unsqueeze(-1)).exp()
        assert torch.allclose(cuda_r, ref_r, rtol=1e-4, atol=1e-4)


class TestBlockedUpdateSpherical:
    def test_counts_sum_to_N(self):
        from gmmxx import _cuda
        torch.manual_seed(0)
        x = torch.randn(1, 256, 4, device="cuda")
        ids = torch.randint(0, 5, (1, 256), device="cuda", dtype=torch.int32)
        sums, sumsq, counts = _cuda.blocked_update_spherical(x, ids, 5)
        assert counts.sum().item() == 256

    def test_sums_match_groupby(self):
        from gmmxx import _cuda
        torch.manual_seed(0)
        B, N, D, K = 2, 128, 4, 6
        x = torch.randn(B, N, D, device="cuda")
        ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
        sums, sumsq, counts = _cuda.blocked_update_spherical(x, ids, K)

        for b in range(B):
            for k in range(K):
                mask = (ids[b] == k)
                if mask.sum() == 0:
                    assert counts[b, k].item() == 0
                    assert torch.allclose(sums[b, k], torch.zeros(D, device="cuda"))
                    continue
                ref_sum = x[b][mask].sum(0)
                ref_sumsq = x[b][mask].pow(2).sum().item()
                assert torch.allclose(sums[b, k], ref_sum, atol=1e-4)
                assert abs(sumsq[b, k].item() - ref_sumsq) < 1e-3 * max(1.0, abs(ref_sumsq))


class TestFinalizeSpherical:
    def test_basic(self):
        from gmmxx import _cuda
        sums = torch.tensor([[[10.0, 20.0], [0.0, 0.0]]], device="cuda")
        sumsq = torch.tensor([[150.0, 0.0]], device="cuda")
        counts = torch.tensor([[5, 0]], device="cuda", dtype=torch.int32)
        old_means = torch.tensor([[[0.0, 0.0], [99.0, 99.0]]], device="cuda")
        old_var = torch.tensor([[1.0, 42.0]], device="cuda")
        new_means, new_var, new_weights = _cuda.finalize_spherical(
            sums, sumsq, counts, old_means, old_var, 5, 1e-6
        )
        assert torch.allclose(new_means[0, 0], torch.tensor([2.0, 4.0], device="cuda"))
        assert abs(new_var[0, 0].item() - 5.0) < 1e-4
        assert abs(new_weights[0, 0].item() - 1.0) < 1e-4
        assert torch.allclose(new_means[0, 1], old_means[0, 1])
        assert new_var[0, 1].item() == 42.0
        assert new_weights[0, 1].item() == 0.0

    def test_reg_covar_clamps(self):
        from gmmxx import _cuda
        sums = torch.tensor([[[1.0, 1.0]]], device="cuda")
        sumsq = torch.tensor([[2.0]], device="cuda")
        counts = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        old_means = torch.zeros(1, 1, 2, device="cuda")
        old_var = torch.zeros(1, 1, device="cuda")
        _, new_var, _ = _cuda.finalize_spherical(
            sums, sumsq, counts, old_means, old_var, 1, 1e-3
        )
        assert new_var[0, 0].item() == pytest.approx(1e-3, abs=1e-9)
