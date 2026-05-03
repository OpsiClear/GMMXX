"""Sorted-run M-step kernel correctness vs the per-token kernel."""

from __future__ import annotations
import pytest
import torch


def _has_cuda():
    try:
        from gmmxx._cuda import has_cuda
        return has_cuda()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")


@pytest.mark.parametrize("N,K", [(1024, 8), (4096, 16), (16384, 32), (65536, 64)])
def test_sorted_matches_naive(N, K):
    """Sorted-run output should match per-token within fp32 atomic ULP drift."""
    from gmmxx import _cuda
    torch.manual_seed(0)
    B, D = 1, 16
    x = torch.randn(B, N, D, device="cuda")
    ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)

    s_naive, ss_naive, c_naive = _cuda.blocked_update_spherical(x, ids, K, force_sort=False)
    s_sorted, ss_sorted, c_sorted = _cuda.blocked_update_spherical(x, ids, K, force_sort=True)

    assert torch.equal(c_naive, c_sorted), "counts must match exactly"
    # Atomic order varies, so sums/sumsq agree only to fp32 ULP.
    assert torch.allclose(s_naive, s_sorted, rtol=1e-4, atol=1e-3)
    # sumsq accumulates ||x||^2 with more atomic contention; allow looser tol.
    assert torch.allclose(ss_naive, ss_sorted, rtol=1e-3, atol=1e-2)


def test_force_sort_zero_N():
    """Empty N must not crash either path."""
    from gmmxx import _cuda
    x = torch.empty(1, 0, 8, device="cuda")
    ids = torch.empty(1, 0, device="cuda", dtype=torch.int32)
    s, ss, c = _cuda.blocked_update_spherical(x, ids, 4, force_sort=True)
    assert s.shape == (1, 4, 8) and (s == 0).all()
    assert (c == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sorted_handles_all_dtypes(dtype):
    from gmmxx import _cuda
    torch.manual_seed(0)
    B, N, D, K = 1, 2048, 16, 8
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    s, ss, c = _cuda.blocked_update_spherical(x, ids, K, force_sort=True)
    assert c.sum().item() == N
    assert s.shape == (B, K, D)
    assert ss.shape == (B, K)


def test_heuristic_picks_per_token_below_threshold():
    """N*K=8192 << 2^21 should pick per-token. We can't observe the path
    selection directly, but the result must be correct."""
    from gmmxx import _cuda
    torch.manual_seed(0)
    x = torch.randn(1, 1024, 8, device="cuda")
    ids = torch.randint(0, 8, (1, 1024), device="cuda", dtype=torch.int32)
    s, ss, c = _cuda.blocked_update_spherical(x, ids, 8)
    assert c.sum().item() == 1024


def test_force_sort_in_train_loop():
    """Setting _force_sort=True on a GMMXX instance routes M-step through
    the sorted kernel. Verify the EM loop still converges and produces a
    sane lower_bound."""
    from gmmxx import GMMXX
    import math
    torch.manual_seed(0)
    x = torch.randn(2048, 16, device="cuda")
    gmm = GMMXX(n_components=8, max_iter=10, tol=1e-4, random_state=0,
                covariance_type="spherical", backend="cuda")
    gmm._force_sort = True
    gmm.fit(x)
    assert math.isfinite(gmm.lower_bound_)
    assert gmm.last_backend_used_ == "cuda"
    assert gmm.means_.shape == (8, 16)
