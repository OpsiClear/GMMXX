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


def test_soft_update_spherical_matches_fused_on_small_shape():
    from gmmxx import _cuda

    torch.manual_seed(101)
    x = torch.randn(1, 512, 32, device="cuda", dtype=torch.float16)
    means = torch.randn(1, 64, 32, device="cuda", dtype=torch.float16)
    var = torch.rand(1, 64, device="cuda").clamp_min(0.5).float()
    log_w = torch.log_softmax(torch.randn(1, 64, device="cuda"), dim=-1).float()

    soft = _cuda.soft_update_spherical(x, means, var, log_w, 1e-6)
    fused = _cuda.fused_spherical(x, means, var, log_w, 1e-6)

    assert torch.allclose(soft[0].float(), fused[0].float(), atol=5e-2, rtol=5e-3)
    assert torch.allclose(soft[1], fused[1], atol=5e-3, rtol=5e-3)
    assert torch.allclose(soft[2], fused[2], atol=5e-3, rtol=5e-3)
    assert torch.allclose(soft[3], fused[3], atol=5e-3, rtol=5e-3)


def test_d64_k128_training_uses_soft_update_not_fused():
    from gmmxx import GMMXX

    x = torch.randn(1024, 64, device="cuda", dtype=torch.float16)
    model = GMMXX(
        d=64,
        k=128,
        niter=2,
        tol=0.0,
        seed=5,
        init_params="random",
        covariance_type="spherical",
        backend="cuda",
        device=torch.device("cuda"),
        compute_labels_on_fit=False,
    ).fit(x)

    assert model.last_backend_used_ == "cuda"
    assert model.cuda_estep_enabled_ is True
    assert model.cuda_fused_update_enabled_ is False
    assert model.labels_ is None
    assert math.isfinite(model.lower_bound_)
