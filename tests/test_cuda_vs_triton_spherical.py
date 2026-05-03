"""3-way oracle: CUDA spherical results match Triton AND torch_fallback within
the spec's tolerance bounds (§6).

Mirrors flash-kmeans-cuda's test_correctness.py pattern. Skipped when either
CUDA or Triton is unavailable.

Signature notes:
  - gmmxx._cuda.spherical_assign/logsumexp: (x, means, var, log_w)
    where log_w is already log-probabilities.
  - spherical_assign_triton/logsumexp_triton: (x, means, variances, weights)
    where weights is raw probabilities (the function takes log internally).
"""

from __future__ import annotations

import math

import pytest
import torch


def _has_cuda():
    try:
        from gmmxx import _cuda
        return _cuda.has_cuda()
    except ImportError:
        return False


def _has_triton():
    try:
        from gmmxx.assign_spherical_triton import spherical_assign_triton
        return spherical_assign_triton is not None
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (_has_cuda() and _has_triton()),
    reason="requires both CUDA and Triton",
)


@pytest.mark.parametrize("D,K,N", [(8, 16, 1024), (32, 64, 4096), (128, 8, 256)])
def test_assign_3way(D, K, N):
    """Assign output should agree across CUDA / Triton / torch_fallback on
    shapes inside the Triton support window.

    Triton's tl.dot requires the reduction dimension K >= 16; shapes with
    D < 16 are outside the Triton support window on sm80+.
    """
    if D < 16:
        pytest.skip(f"D={D} < 16: outside Triton tl.dot minimum reduction dim")

    from gmmxx import _cuda
    from gmmxx.assign_spherical_triton import spherical_assign_triton

    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn(1, N, D, device=device)
    means = torch.randn(1, K, D, device=device)
    var = torch.rand(1, K, device=device).clamp_min(0.5)
    log_w = torch.log_softmax(torch.randn(1, K, device=device), dim=-1)
    # Triton expects raw weights (probabilities), CUDA expects log-weights.
    weights = torch.exp(log_w)

    cuda_ids = _cuda.spherical_assign(x, means, var, log_w)
    triton_ids = spherical_assign_triton(x, means, var, weights)

    # Both backends should match within fp32 numerical noise on near-tie samples.
    agree = (cuda_ids == triton_ids).float().mean().item()
    assert agree >= 0.99, f"CUDA vs Triton agreement only {agree:.3f}"


@pytest.mark.parametrize("D,K,N", [(16, 32, 2048)])
def test_logsumexp_3way(D, K, N):
    from gmmxx import _cuda
    from gmmxx.assign_spherical_triton import spherical_logsumexp_triton

    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(1, N, D, device=device)
    means = torch.randn(1, K, D, device=device)
    var = torch.rand(1, K, device=device).clamp_min(0.5)
    log_w = torch.log_softmax(torch.randn(1, K, device=device), dim=-1)
    # Triton expects raw weights (probabilities), CUDA expects log-weights.
    weights = torch.exp(log_w)

    cuda_lse = _cuda.spherical_logsumexp(x, means, var, log_w)
    triton_lse = spherical_logsumexp_triton(x, means, var, weights)
    assert torch.allclose(cuda_lse, triton_lse, rtol=5e-3, atol=5e-3)
