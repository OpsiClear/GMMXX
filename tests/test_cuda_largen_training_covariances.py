"""CUDA large-N training tests for non-spherical covariances."""

from __future__ import annotations

import math

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
    ("covariance_type", "n", "d", "k", "expected_cov_shape"),
    [
        ("diag", 2048, 8, 4, (1, 4, 8)),
        ("tied", 1536, 6, 3, (1, 6, 6)),
        ("full", 1024, 4, 3, (1, 3, 4, 4)),
    ],
)
def test_largen_cuda_training_direct_non_spherical(
    covariance_type, n, d, k, expected_cov_shape
):
    from gmmxx.large_n import batch_gmm_largeN_cpu

    torch.manual_seed(7)
    x_cpu = torch.randn(1, n, d, dtype=torch.float32)
    labels, means, cov, weights, info = batch_gmm_largeN_cpu(
        x_cpu,
        k,
        covariance_type=covariance_type,
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        backend="cuda",
        max_iters=4,
        tol=0.0,
        init_params="random",
        kmeans_use_triton=False,
        chunk_size_N=257,
        chunk_size_K=2,
    )

    assert labels is not None
    assert labels.shape == (1, n)
    assert means.shape == (1, k, d)
    assert cov.shape == expected_cov_shape
    assert weights.shape == (1, k)
    assert torch.allclose(weights.sum(-1).cpu(), torch.ones(1), atol=1e-4)
    assert math.isfinite(info["lower_bound"])
    assert info["large_n_streaming_enabled"] is True
    assert info["backend_breakdown"]["cuda"] == info["n_iter"]
    assert info["triton_estep_enabled"] is False


@pytest.mark.parametrize(
    ("covariance_type", "n", "d", "k"),
    [
        ("diag", 1536, 8, 4),
        ("tied", 1280, 6, 3),
        ("full", 1024, 4, 3),
    ],
)
def test_largen_cuda_training_via_gmmxx_non_spherical(covariance_type, n, d, k):
    from gmmxx import GMMXX

    torch.manual_seed(11)
    x_cpu = torch.randn(n, d, dtype=torch.float32)
    model = GMMXX(
        n_components=k,
        covariance_type=covariance_type,
        backend="cuda",
        max_iter=4,
        tol=0.0,
        random_state=11,
        init_params="random",
        dtype=torch.float32,
        device="cuda:0",
        chunk_size_data_cpu=257,
        chunk_size_data=257,
        chunk_size_centroids=2,
    ).fit(x_cpu)

    assert model.last_backend_used_ == "cuda"
    assert model.large_n_streaming_enabled_ is True
    assert math.isfinite(model.lower_bound_)
    assert model.labels_.shape == (n,)
    assert model.means_.shape == (k, d)
    assert model.weights_.sum().item() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    ("covariance_type", "n", "d", "k"),
    [
        ("diag", 512, 5, 3),
        ("tied", 384, 4, 3),
        ("full", 2048, 4, 3),
    ],
)
def test_largen_cuda_training_matches_soft_em_torch_path(covariance_type, n, d, k):
    """CUDA large-N training must preserve the existing soft-EM statistics,
    not switch to hard classification-EM updates."""
    from gmmxx.large_n import batch_gmm_largeN_cpu

    torch.manual_seed(99)
    x_cpu = torch.randn(1, n, d, dtype=torch.float32)
    kwargs = dict(
        covariance_type=covariance_type,
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        max_iters=3,
        tol=0.0,
        init_params="random",
        kmeans_use_triton=False,
        # Disable Triton on the torch reference so both paths use cuBLAS-class
        # arithmetic. Triton's tl.dot(input_precision="tf32x3") and cuBLAS TF32
        # are different fp modes; under the strict 1e-4 tolerance their drift
        # over 3 iterations (~0.04 on full D=4) breaks an otherwise-correct
        # apples-to-apples comparison.
        gmm_use_triton=False,
        chunk_size_N=257,
        chunk_size_K=2,
    )

    torch.manual_seed(123)
    labels_cuda, means_cuda, cov_cuda, weights_cuda, info_cuda = batch_gmm_largeN_cpu(
        x_cpu, k, backend="cuda", **kwargs
    )
    torch.manual_seed(123)
    labels_torch, means_torch, cov_torch, weights_torch, info_torch = batch_gmm_largeN_cpu(
        x_cpu, k, backend="torch", **kwargs
    )

    assert info_cuda["backend_breakdown"]["cuda"] == info_cuda["n_iter"]
    assert info_torch.get("backend_breakdown", {}).get("cuda", 0) == 0
    assert torch.allclose(means_cuda, means_torch, atol=1e-4, rtol=1e-4)
    assert torch.allclose(cov_cuda, cov_torch, atol=1e-4, rtol=1e-4)
    assert torch.allclose(weights_cuda, weights_torch, atol=1e-4, rtol=1e-4)
    assert abs(info_cuda["lower_bound"] - info_torch["lower_bound"]) < 1e-4
    assert torch.equal(labels_cuda, labels_torch)


def test_largen_cuda_training_validates_before_dispatch():
    from gmmxx.large_n import batch_gmm_largeN_cpu

    x_cpu = torch.randn(1, 128, 4)
    with pytest.raises(ValueError, match="chunk_size_N"):
        batch_gmm_largeN_cpu(
            x_cpu,
            3,
            covariance_type="diag",
            device=torch.device("cuda:0"),
            dtype=torch.float32,
            backend="cuda",
            chunk_size_N=0,
        )


def test_largen_cuda_training_approx_topk_equal_k_still_uses_exact_cuda():
    from gmmxx.large_n import batch_gmm_largeN_cpu

    torch.manual_seed(5)
    k = 4
    _, _, _, _, info = batch_gmm_largeN_cpu(
        torch.randn(1, 512, 6),
        k,
        covariance_type="diag",
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        backend="cuda",
        max_iters=2,
        tol=0.0,
        init_params="random",
        kmeans_use_triton=False,
        chunk_size_N=129,
        chunk_size_K=2,
        approx_top_k=k,
    )

    assert info["backend_breakdown"]["cuda"] == info["n_iter"]
    assert info["approx_top_k"] is None


def test_largen_cuda_training_auto_falls_back_after_runtime_error(monkeypatch):
    import gmmxx.large_n as large_n

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic cuda failure")

    monkeypatch.setattr(large_n, "_largen_covariance_cuda", boom)
    torch.manual_seed(13)
    _, _, _, _, info = large_n.batch_gmm_largeN_cpu(
        torch.randn(1, 256, 4),
        3,
        covariance_type="diag",
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        backend="auto",
        max_iters=2,
        tol=0.0,
        init_params="random",
        kmeans_use_triton=False,
        chunk_size_N=97,
        chunk_size_K=2,
        gmm_use_triton=False,
    )

    assert info.get("backend_breakdown", {}).get("cuda", 0) == 0
    assert "synthetic cuda failure" in info["fallback_reason"]


@pytest.mark.parametrize(
    ("covariance_type", "expected_cov_shape"),
    [
        ("diag", (4096, 128)),
        ("tied", (128, 128)),
    ],
)
def test_gmmxx_flash_sized_diag_tied_in_memory_uses_streamed_cuda(
    covariance_type, expected_cov_shape
):
    from gmmxx import GMMXX
    from gmmxx._runtime import cuda_diag_streamed_supported, cuda_tied_streamed_supported

    assert cuda_diag_streamed_supported(128, 8192, torch.float16)
    assert cuda_tied_streamed_supported(128, 8192, torch.float16)

    torch.manual_seed(29)
    x = torch.randn(256, 128, device="cuda", dtype=torch.float16)
    model = GMMXX(
        n_components=4096,
        covariance_type=covariance_type,
        backend="cuda",
        max_iter=1,
        tol=0.0,
        random_state=29,
        init_params="random",
        dtype=torch.float16,
        device="cuda",
        compute_labels_on_fit=False,
        chunk_size_data=128,
        chunk_size_centroids=512,
        matmul_precision="high",
    ).fit(x)

    assert model.last_backend_used_ == "cuda"
    assert model.fit_info_["backend_breakdown"] == {"cuda": model.n_iter_}
    assert model.fit_info_["cuda_tensor_streamed_enabled"] is True
    assert model.cuda_estep_enabled_ is True
    assert math.isfinite(model.lower_bound_)
    assert model.labels_ is None
    assert model.means_.shape == (4096, 128)
    assert model.covariances_.shape == expected_cov_shape


def test_gmmxx_full_feasible_streamed_cuda_backend():
    from gmmxx import GMMXX
    from gmmxx._runtime import cuda_full_streamed_supported

    assert cuda_full_streamed_supported(64, 128, torch.float16)
    assert not cuda_full_streamed_supported(128, 8192, torch.float16)

    torch.manual_seed(31)
    x = torch.randn(256, 64, device="cuda", dtype=torch.float16)
    model = GMMXX(
        n_components=128,
        covariance_type="full",
        backend="cuda",
        max_iter=1,
        tol=0.0,
        random_state=31,
        init_params="random",
        dtype=torch.float16,
        device="cuda",
        compute_labels_on_fit=False,
        chunk_size_data=128,
        chunk_size_centroids=64,
        matmul_precision="high",
    ).fit(x)

    assert model.last_backend_used_ == "cuda"
    assert model.fit_info_["backend_breakdown"] == {"cuda": model.n_iter_}
    assert model.fit_info_["cuda_tensor_streamed_enabled"] is True
    assert model.labels_ is None
    assert math.isfinite(model.lower_bound_)
    assert model.means_.shape == (128, 64)
    assert model.covariances_.shape == (128, 64, 64)


@pytest.mark.parametrize("covariance_type", ["diag", "tied"])
def test_gmmxx_mid_k_diag_tied_prefers_streamed_cuda(covariance_type):
    from gmmxx import GMMXX

    torch.manual_seed(37)
    x = torch.randn(256, 64, device="cuda", dtype=torch.float16)
    model = GMMXX(
        n_components=128,
        covariance_type=covariance_type,
        backend="cuda",
        max_iter=1,
        tol=0.0,
        random_state=37,
        init_params="random",
        dtype=torch.float16,
        device="cuda",
        compute_labels_on_fit=False,
        chunk_size_data=128,
        chunk_size_centroids=64,
        matmul_precision="high",
    ).fit(x)

    assert model.last_backend_used_ == "cuda"
    assert model.fit_info_["cuda_tensor_streamed_enabled"] is True
    assert model.fit_info_["backend_breakdown"] == {"cuda": model.n_iter_}
    assert model.means_.shape == (128, 64)


@pytest.mark.parametrize("covariance_type", ["diag", "tied"])
def test_gmmxx_large_diag_tied_streamed_cuda_uses_autotuned_chunks(covariance_type):
    from gmmxx import GMMXX

    torch.manual_seed(41)
    x = torch.randn(256, 128, device="cuda", dtype=torch.float16)
    model = GMMXX(
        n_components=1024,
        covariance_type=covariance_type,
        backend="cuda",
        max_iter=1,
        tol=0.0,
        random_state=41,
        init_params="random",
        dtype=torch.float16,
        device="cuda",
        compute_labels_on_fit=False,
        matmul_precision="high",
    ).fit(x)

    assert model.last_backend_used_ == "cuda"
    assert model.fit_info_["cuda_tensor_streamed_enabled"] is True
    assert model.fit_info_["cuda_tensor_streamed_chunk_size_N"] == 16384
    assert model.fit_info_["cuda_tensor_streamed_chunk_size_K"] == 512
