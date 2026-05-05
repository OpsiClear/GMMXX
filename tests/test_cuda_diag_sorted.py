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


def _setup(B=2, N=513, D=17, K=9, dtype=torch.float32):
    torch.manual_seed(53)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    return x, ids, K


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_blocked_update_diag_sorted_matches_per_token(dtype):
    from gmmxx import _cuda

    x, ids, K = _setup(dtype=dtype)
    sums_ref, sumsq_ref, counts_ref = _cuda.blocked_update_diag(x, ids, K, force_sort=False)
    sums_sort, sumsq_sort, counts_sort = _cuda.blocked_update_diag(x, ids, K, force_sort=True)

    assert torch.equal(counts_sort, counts_ref)
    assert torch.allclose(sums_sort, sums_ref, atol=5e-3, rtol=5e-3)
    assert torch.allclose(sumsq_sort, sumsq_ref, atol=5e-3, rtol=5e-3)


def test_blocked_update_diag_sorted_direct_wrapper():
    from gmmxx import _cuda

    x, ids, K = _setup(B=1, N=257, D=8, K=5)
    sorted_ids, perm = ids.sort(dim=1)
    x_sorted = torch.gather(x, 1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    direct = _cuda.blocked_update_diag_sorted(
        x_sorted.contiguous(), sorted_ids.int().contiguous(), K
    )
    forced = _cuda.blocked_update_diag(x, ids, K, force_sort=True)

    for direct_t, forced_t in zip(direct, forced):
        assert torch.allclose(direct_t, forced_t, atol=1e-5, rtol=1e-5)


def test_blocked_update_diag_auto_uses_sorted_for_large_nk(monkeypatch):
    from gmmxx import _cuda

    x, ids, K = _setup(B=1, N=1024, D=4, K=4)
    calls = {"sorted": 0}
    original = _cuda._C.blocked_update_diag_sorted

    def tracking_sorted(*args, **kwargs):
        calls["sorted"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_cuda._C, "blocked_update_diag_sorted", tracking_sorted)
    old_threshold = _cuda._SORT_THRESHOLD_NK
    try:
        _cuda._SORT_THRESHOLD_NK = 1
        _cuda.blocked_update_diag(x, ids, K)
    finally:
        _cuda._SORT_THRESHOLD_NK = old_threshold

    assert calls["sorted"] >= 1
