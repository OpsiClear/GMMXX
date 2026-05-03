"""Inference under backend='cuda' for spherical covariance.

Verifies predict / predict_proba / score_samples / score on the CUDA
inference path produce shape-correct outputs and populate
last_backend_used_ correctly.
"""

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


@pytest.fixture
def fitted_cuda_gmm():
    from gmmxx import GMMXX
    torch.manual_seed(0)
    x = torch.randn(2048, 16, device="cuda")
    gmm = GMMXX(n_components=8, max_iter=15, tol=1e-4, random_state=0,
                covariance_type="spherical", backend="cuda")
    gmm.fit(x)
    return gmm, x


def test_predict_returns_correct_shape_dtype(fitted_cuda_gmm):
    gmm, x = fitted_cuda_gmm
    labels = gmm.predict(x[:512])
    assert labels.shape == (512,)
    assert labels.dtype == torch.long
    assert (labels >= 0).all() and (labels < gmm.k).all()
    assert gmm.last_backend_used_ == "cuda"


def test_predict_proba_sums_to_one(fitted_cuda_gmm):
    gmm, x = fitted_cuda_gmm
    p = gmm.predict_proba(x[:256])
    assert p.shape == (256, gmm.k)
    assert torch.allclose(p.sum(-1), torch.ones(256, device="cuda"), atol=1e-4)
    assert gmm.last_backend_used_ == "cuda"


def test_score_samples_returns_finite(fitted_cuda_gmm):
    gmm, x = fitted_cuda_gmm
    ll = gmm.score_samples(x[:512])
    assert ll.shape == (512,)
    assert torch.isfinite(ll).all()
    assert gmm.last_backend_used_ == "cuda"


def test_score_returns_finite_scalar(fitted_cuda_gmm):
    gmm, x = fitted_cuda_gmm
    s = gmm.score(x[:512])
    assert math.isfinite(s)
    assert gmm.last_backend_used_ == "cuda"


def test_predict_consistent_with_predict_proba(fitted_cuda_gmm):
    """argmax of predict_proba should equal predict (within fp32 tolerance)."""
    gmm, x = fitted_cuda_gmm
    labels_direct = gmm.predict(x[:128])
    labels_argmax = gmm.predict_proba(x[:128]).argmax(-1).long()
    agree = (labels_direct == labels_argmax).float().mean().item()
    assert agree >= 0.99
