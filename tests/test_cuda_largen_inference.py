"""large_n.py CUDA inference tests."""

from __future__ import annotations

import pytest
import torch


def _has_cuda() -> bool:
    try:
        from gmmxx._cuda import has_cuda

        return has_cuda()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")


@pytest.mark.parametrize(
    ("covariance_type", "n", "d", "k"),
    [
        ("spherical", 1024, 8, 4),
        ("diag", 1024, 8, 4),
        ("tied", 768, 6, 3),
        ("full", 512, 4, 3),
    ],
)
def test_largen_cuda_inference_matches_in_memory_cuda(covariance_type, n, d, k):
    """CPU-streamed large-N inference should use the same CUDA math as the
    in-memory path when backend='cuda' resolves inside the support window."""
    from gmmxx import GMMXX
    from gmmxx.large_n import (
        large_n_predict_cpu,
        large_n_predict_proba_cpu,
        large_n_score_samples_cpu,
    )

    torch.manual_seed(123)
    x_cpu = torch.randn(n, d, dtype=torch.float32)
    x_gpu = x_cpu.to("cuda")

    model = GMMXX(
        n_components=k,
        covariance_type=covariance_type,
        backend="cuda",
        max_iter=4,
        tol=0.0,
        random_state=123,
        dtype=torch.float32,
        device="cuda:0",
        chunk_size_data=257,
        chunk_size_data_cpu=257,
        chunk_size_centroids=2,
    ).fit(x_gpu)

    labels_ref = model.predict(x_gpu).cpu()
    probs_ref = model.predict_proba(x_gpu).cpu()
    scores_ref = model.score_samples(x_gpu).cpu()

    common = dict(
        covariance_type=covariance_type,
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        chunk_size_N=257,
        chunk_size_K=2,
        use_triton=False,
        backend="cuda",
    )
    x_stream = x_cpu.unsqueeze(0)
    labels, labels_backend = large_n_predict_cpu(
        x_stream,
        model.means_b,
        model.variances_b,
        model.weights_b,
        return_backend_used=True,
        **common,
    )
    probs, probs_backend = large_n_predict_proba_cpu(
        x_stream,
        model.means_b,
        model.variances_b,
        model.weights_b,
        return_backend_used=True,
        **common,
    )
    scores, scores_backend = large_n_score_samples_cpu(
        x_stream,
        model.means_b,
        model.variances_b,
        model.weights_b,
        return_backend_used=True,
        **common,
    )

    assert labels_backend == "cuda"
    assert probs_backend == "cuda"
    assert scores_backend == "cuda"
    assert torch.equal(labels.squeeze(0), labels_ref)
    assert torch.allclose(probs.squeeze(0), probs_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(scores.squeeze(0), scores_ref, atol=1e-4, rtol=1e-4)

    labels_via_model = model.predict(x_cpu).cpu()
    assert torch.equal(labels_via_model, labels_ref)
    assert model.last_backend_used_ == "cuda"
    probs_via_model = model.predict_proba(x_cpu).cpu()
    assert torch.allclose(probs_via_model, probs_ref, atol=1e-4, rtol=1e-4)
    assert model.last_backend_used_ == "cuda"
    scores_via_model = model.score_samples(x_cpu).cpu()
    assert torch.allclose(scores_via_model, scores_ref, atol=1e-4, rtol=1e-4)
    assert model.last_backend_used_ == "cuda"


def test_explicit_cuda_largen_inference_unsupported_shape_skips_triton():
    """Explicit backend='cuda' falls directly to torch when the CUDA shape gate
    rejects the call; it must not run Triton just because use_triton=True."""
    from gmmxx.large_n import large_n_predict_cpu

    torch.manual_seed(0)
    n, d, k = 128, 8, 64  # full CUDA supports K <= 32; full Triton would accept D=8.
    x = torch.randn(1, n, d)
    means = torch.randn(1, k, d, device="cuda")
    eye = torch.eye(d, device="cuda").view(1, 1, d, d)
    cov = eye.repeat(1, k, 1, 1)
    weights = torch.full((1, k), 1.0 / k, device="cuda")

    labels, backend_used = large_n_predict_cpu(
        x,
        means,
        cov,
        weights,
        covariance_type="full",
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        chunk_size_N=41,
        chunk_size_K=16,
        use_triton=True,
        backend="cuda",
        return_backend_used=True,
    )

    assert labels.shape == (1, n)
    assert backend_used == "torch"
