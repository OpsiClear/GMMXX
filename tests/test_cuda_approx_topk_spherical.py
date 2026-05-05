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


def _reference_topk(x, means, var, log_w, top_k):
    B, N, D = x.shape
    K = means.shape[1]
    x_f = x.float()
    means_f = means.float()
    var_f = var.float()
    x_sq = x_f.square().sum(dim=-1)
    means_sq = means_f.square().sum(dim=-1)
    cross = torch.bmm(x_f, means_f.transpose(1, 2))
    dist = (x_sq.unsqueeze(-1) + means_sq.unsqueeze(1) - 2.0 * cross).clamp_min(0.0)
    logits = log_w.float().unsqueeze(1) - 0.5 * (
        dist / var_f.unsqueeze(1)
        + float(D) * (torch.log(torch.tensor(2.0 * torch.pi, device=x.device)) + var_f.log()).unsqueeze(1)
    )
    top_vals, top_idx = logits.topk(top_k, dim=-1)
    log_norm = top_vals.logsumexp(dim=-1)
    resp = (top_vals - log_norm.unsqueeze(-1)).exp()
    nk = torch.zeros(B, K, dtype=torch.float32, device=x.device)
    sum_x = torch.zeros(B, K, D, dtype=torch.float32, device=x.device)
    sum_x_sq = torch.zeros(B, K, dtype=torch.float32, device=x.device)
    for slot in range(top_k):
        idx = top_idx[:, :, slot]
        r = resp[:, :, slot]
        nk.scatter_add_(1, idx, r)
        sum_x.scatter_add_(
            1,
            idx.unsqueeze(-1).expand(-1, -1, D),
            r.unsqueeze(-1) * x_f,
        )
        sum_x_sq.scatter_add_(1, idx, r * x_sq)
    return nk, sum_x, sum_x_sq, log_norm.sum()


def _setup(dtype=torch.float32, B=1, N=96, D=12, K=9, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    means = torch.randn(B, K, D, device="cuda", dtype=dtype)
    var = torch.rand(B, K, device="cuda", dtype=torch.float32).clamp_min(0.5)
    log_w = torch.log_softmax(torch.randn(B, K, device="cuda"), dim=-1).float()
    return x, means, var, log_w


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_approx_topk_update_spherical_matches_reference(dtype):
    from gmmxx import _cuda

    x, means, var, log_w = _setup(dtype=dtype, B=2, N=97, D=13, K=11)
    got = _cuda.approx_topk_update_spherical(
        x, means, var, log_w, top_k=4, chunk_size_K=3
    )
    ref = _reference_topk(x, means, var, log_w, top_k=4)

    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (2e-3, 2e-3)
    for got_t, ref_t in zip(got[:3], ref[:3]):
        assert torch.allclose(got_t, ref_t, rtol=rtol, atol=atol)
    assert torch.allclose(got[3], ref[3], rtol=rtol, atol=atol)


def test_approx_topk_update_spherical_chunk_size_invariant():
    from gmmxx import _cuda

    x, means, var, log_w = _setup(dtype=torch.float32, B=1, N=65, D=7, K=10)
    small = _cuda.approx_topk_update_spherical(
        x, means, var, log_w, top_k=3, chunk_size_K=2
    )
    full = _cuda.approx_topk_update_spherical(
        x, means, var, log_w, top_k=3, chunk_size_K=10
    )
    for small_t, full_t in zip(small, full):
        assert torch.allclose(small_t, full_t, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("top_k", [0, 8])
def test_approx_topk_update_spherical_rejects_invalid_top_k(top_k):
    from gmmxx import _cuda

    x, means, var, log_w = _setup(dtype=torch.float32, B=1, N=16, D=4, K=8)
    with pytest.raises(ValueError, match="top_k"):
        _cuda.approx_topk_update_spherical(x, means, var, log_w, top_k=top_k)


def test_gmmxx_spherical_cuda_approx_topk_training_uses_cuda():
    from gmmxx import GMMXX

    torch.manual_seed(123)
    x = torch.randn(384, 10, device="cuda")
    model = GMMXX(
        d=10,
        k=12,
        niter=3,
        tol=0.0,
        seed=123,
        init_params="random",
        covariance_type="spherical",
        backend="cuda",
        approx_top_k=4,
        device=torch.device("cuda"),
        compute_labels_on_fit=True,
        chunk_size_centroids=5,
    ).fit(x)

    assert model.last_backend_used_ == "cuda"
    assert model.cuda_estep_enabled_ is True
    assert model.cuda_approx_topk_enabled_ is True
    assert model.cuda_fused_update_enabled_ is False
    assert model.approximate_em_enabled_ is True
    assert model.approx_top_k_ == 4
    assert model.triton_approx_topk_enabled_ is False
    assert model.labels_.shape == (384,)
    assert torch.isfinite(model.means_b).all()
    assert torch.isfinite(model.covariances_b).all()
    assert torch.isfinite(model.weights_b).all()
    assert torch.allclose(
        model.weights_b.sum(dim=-1),
        torch.ones(1, device="cuda", dtype=model.weights_b.dtype),
        atol=1e-5,
    )


def test_gmmxx_spherical_cuda_approx_topk_equal_k_uses_exact_cuda():
    from gmmxx import GMMXX

    x = torch.randn(192, 8, device="cuda")
    model = GMMXX(
        d=8,
        k=6,
        niter=2,
        tol=0.0,
        seed=7,
        init_params="random",
        covariance_type="spherical",
        backend="cuda",
        approx_top_k=6,
        device=torch.device("cuda"),
        compute_labels_on_fit=False,
    ).fit(x)

    assert model.last_backend_used_ == "cuda"
    assert model.cuda_approx_topk_enabled_ is False
    assert model.approximate_em_enabled_ is False
    assert model.approx_top_k_ is None


def test_large_n_spherical_cuda_does_not_ignore_approx_topk():
    from gmmxx.large_n import batch_gmm_largeN_cpu

    x = torch.randn(1, 256, 4)
    labels, means, var, weights, info = batch_gmm_largeN_cpu(
        x,
        5,
        covariance_type="spherical",
        max_iters=1,
        tol=0.0,
        dtype=torch.float32,
        device=torch.device("cuda"),
        chunk_size_N=128,
        chunk_size_K=2,
        init_params="random",
        seed=9,
        compute_labels=False,
        approx_top_k=2,
        backend="cuda",
        gmm_use_triton=False,
    )
    del labels, means, var, weights
    assert info["approximate_em_enabled"] is True
    assert info["approx_top_k"] == 2
    assert info.get("backend_breakdown", {}).get("cuda", 0) == 0
