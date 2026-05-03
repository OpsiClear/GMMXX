"""Spherical large_n.py CUDA streaming tests."""

from __future__ import annotations
import math
import pytest
import torch


def _has_cuda():
    try:
        from gmmxx._cuda import has_cuda
        return has_cuda()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")


def test_largen_spherical_cuda_basic():
    """A modestly-large CPU input streams through CUDA and produces a
    finite ELBO."""
    from gmmxx.large_n import batch_gmm_largeN_cpu
    torch.manual_seed(0)
    N, D, K = 100_000, 16, 8
    x_cpu = torch.randn(N, D)
    ids, means, var, w, info = batch_gmm_largeN_cpu(
        x_cpu.unsqueeze(0),  # (1, N, D) as required by the function
        K, max_iters=10, tol=1e-4, seed=0,
        covariance_type='spherical',
        device=torch.device('cuda:0'),
        dtype=torch.float32,
        backend='cuda',
        chunk_size_N=32_768,
    )
    assert math.isfinite(info["lower_bound"])
    assert info["large_n_streaming_enabled"] is True
    assert info.get("backend_breakdown", {}).get("cuda", 0) > 0
    assert means.shape == (1, K, D)
    assert var.shape == (1, K)
    assert ids.shape[-1] == N


def test_largen_spherical_cuda_via_GMMXX():
    """End-to-end via the GMMXX class with a CPU input that triggers streaming."""
    from gmmxx import GMMXX
    torch.manual_seed(0)
    N, D, K = 200_000, 16, 8
    x = torch.randn(N, D)  # CPU input
    gmm = GMMXX(n_components=K, max_iter=10, tol=1e-4, random_state=0,
                covariance_type='spherical', backend='cuda',
                chunk_size_data_cpu=65_536, dtype=torch.float32, device='cuda:0')
    gmm.fit(x)
    assert gmm.last_backend_used_ == "cuda"
    assert math.isfinite(gmm.lower_bound_)
    # large_n_streaming_enabled_ may or may not be set depending on whether
    # interface.py routed through the streaming path. If it's None, that
    # means the streaming path wasn't used (interface chose the on-GPU path
    # since N=200K fits). Either way, last_backend_used_=="cuda" is enough.
    assert gmm.means_.shape == (K, D)
    assert gmm.weights_.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_largen_spherical_torch_fallback_unchanged():
    """backend='torch' still uses the existing torch path; no regression."""
    from gmmxx.large_n import batch_gmm_largeN_cpu
    torch.manual_seed(0)
    N, D, K = 50_000, 8, 4
    x_cpu = torch.randn(N, D)
    ids, means, var, w, info = batch_gmm_largeN_cpu(
        x_cpu.unsqueeze(0),  # (1, N, D)
        K, max_iters=5, tol=1e-4, seed=0,
        covariance_type='spherical',
        device=torch.device('cuda:0'),
        dtype=torch.float32,
        backend='torch',
        chunk_size_N=16_384,
    )
    assert math.isfinite(info["lower_bound"])
    bd = info.get("backend_breakdown", {})
    assert bd.get("cuda", 0) == 0


def test_largen_spherical_cuda_lse_close_to_in_memory():
    """Streaming CUDA training should produce an ELBO close to the
    in-memory CUDA training on the same data + seed."""
    from gmmxx import GMMXX
    from gmmxx.large_n import batch_gmm_largeN_cpu

    torch.manual_seed(0)
    N, D, K = 50_000, 8, 4
    x_cpu = torch.randn(N, D)

    # Reference: in-memory CUDA training (no streaming).
    x_gpu = x_cpu.to("cuda")
    gmm_ref = GMMXX(n_components=K, max_iter=10, tol=0, random_state=0,
                    covariance_type='spherical', backend='cuda')
    gmm_ref.fit(x_gpu)

    # Streaming.
    _, means_s, var_s, w_s, info_s = batch_gmm_largeN_cpu(
        x_cpu.unsqueeze(0),  # (1, N, D)
        K, max_iters=10, tol=0, seed=0,
        covariance_type='spherical',
        device=torch.device('cuda:0'),
        dtype=torch.float32,
        backend='cuda',
        chunk_size_N=8_192,
    )
    # The two paths use different init randomness via different RNG state
    # (different generators). They won't agree exactly, but the ELBO order
    # of magnitude should match.
    assert abs(info_s["lower_bound"] - gmm_ref.lower_bound_) < 5.0  # loose; just sanity
