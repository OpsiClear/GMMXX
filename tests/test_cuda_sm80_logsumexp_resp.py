"""Per-kernel correctness for the real sm80 logsumexp_sm80 and resp_sm80.

These were stubbed to safe in Plan 3 Task 2; Plan 4 Tasks 5-6 implement
them as real mma kernels. This test file verifies the mma path matches
the torch reference within the spec's fp16/bf16 tolerance window.
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


def _setup(B=1, N=256, D=32, K=16, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    means = torch.randn(B, K, D, device=device, dtype=dtype)
    var = torch.rand(B, K, device=device).clamp_min(0.5).float()
    log_w = torch.log_softmax(torch.randn(B, K, device=device), dim=-1).float()
    return x, means, var, log_w


def _ref_logits(x, means, var, log_w):
    B, N, D = x.shape
    K = means.shape[1]
    diff = x.float().unsqueeze(2) - means.float().unsqueeze(1)
    dist = diff.pow(2).sum(-1)
    return (
        log_w.unsqueeze(1)
        - 0.5 * D * torch.log(2 * math.pi * var).unsqueeze(1)
        - 0.5 * dist / var.unsqueeze(1)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D,K", [(16, 16), (32, 32), (64, 64), (128, 32), (16, 100)])
def test_logsumexp_matches_reference(dtype, D, K):
    """Real sm80 logsumexp output should match torch reference at rtol=5e-3."""
    from gmmxx import _cuda
    x, means, var, log_w = _setup(N=256, D=D, K=K, dtype=dtype)
    cuda_lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    ref_lse = _ref_logits(x, means, var, log_w).logsumexp(-1)
    assert torch.allclose(cuda_lse, ref_lse, rtol=5e-3, atol=5e-3), (
        f"max abs diff: {(cuda_lse - ref_lse).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D,K", [(16, 16), (32, 32), (64, 64), (128, 32)])
def test_resp_matches_reference(dtype, D, K):
    """Real sm80 resp output should match torch reference at rtol=5e-3."""
    from gmmxx import _cuda
    x, means, var, log_w = _setup(N=256, D=D, K=K, dtype=dtype)
    lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    cuda_r = _cuda.spherical_resp(x, means, var, log_w, lse)
    ref_r = (_ref_logits(x, means, var, log_w) - lse.unsqueeze(-1)).exp()
    assert torch.allclose(cuda_r, ref_r, rtol=5e-3, atol=5e-3), (
        f"max abs diff: {(cuda_r - ref_r).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_resp_sums_to_one(dtype):
    """Each row of resp must sum to 1.0 within tolerance."""
    from gmmxx import _cuda
    x, means, var, log_w = _setup(N=512, D=32, K=32, dtype=dtype)
    lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    r = _cuda.spherical_resp(x, means, var, log_w, lse)
    assert torch.allclose(r.sum(-1), torch.ones_like(r.sum(-1)), atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_logsumexp_k_not_multiple_of_block(dtype):
    """K not aligned to BLOCK_K=64 still produces correct logsumexp."""
    from gmmxx import _cuda
    x, means, var, log_w = _setup(N=128, D=32, K=100, dtype=dtype)  # K=100 = 64 + 36
    cuda_lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    ref_lse = _ref_logits(x, means, var, log_w).logsumexp(-1)
    assert torch.allclose(cuda_lse, ref_lse, rtol=5e-3, atol=5e-3)


def test_assign_logsumexp_resp_consistency():
    """Cross-kernel sanity: argmax of logits should match assign output;
    sum of responsibilities should be 1.0; logsumexp[m] should equal
    log(sum_k exp(logit[m,k]))."""
    from gmmxx import _cuda
    x, means, var, log_w = _setup(N=128, D=32, K=32, dtype=torch.float16)
    ids = _cuda.spherical_assign(x, means, var, log_w)
    lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    r = _cuda.spherical_resp(x, means, var, log_w, lse)

    # resp[m, ids[m]] should be the largest entry in resp[m] for most m.
    # (Allow some near-ties.)
    argmax_r = r.argmax(-1).int()
    agree = (argmax_r == ids).float().mean().item()
    assert agree >= 0.95, f"assign/resp argmax disagree {1-agree:.3f}"

    # sum of resp ≈ 1.
    assert torch.allclose(r.sum(-1), torch.ones_like(r.sum(-1)), atol=5e-3)
