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


def _dataset(d: int, n: int = 192) -> torch.Tensor:
    torch.manual_seed(41 + d)
    x0 = torch.randn(n // 2, d, device="cuda") * 0.35
    x1 = torch.randn(n - n // 2, d, device="cuda") * 0.45 + 2.5
    return torch.cat([x0, x1], dim=0)


@pytest.mark.parametrize(
    "covariance_type,d,k,hard_update_name,assign_name",
    [
        ("diag", 5, 3, "blocked_update_diag", "diag_assign"),
        ("tied", 4, 3, "blocked_update_spherical", "tied_assign"),
        ("full", 4, 3, "full_blocked_update", "full_assign"),
    ],
)
def test_non_spherical_cuda_training_uses_soft_mstep(
    monkeypatch, covariance_type, d, k, hard_update_name, assign_name
):
    """Training should not call the old hard-assign M-step path.

    compute_labels_on_fit=False also means assign should not run as a final
    labeling pass, so this catches accidental reintroduction of hard EM.
    """
    from gmmxx import GMMXX, _cuda

    def fail(*args, **kwargs):  # pragma: no cover - called only on regression
        del args, kwargs
        raise AssertionError("hard-assignment path was called")

    monkeypatch.setattr(_cuda, hard_update_name, fail)
    monkeypatch.setattr(_cuda, assign_name, fail)

    model = GMMXX(
        d=d,
        k=k,
        niter=2,
        tol=0.0,
        seed=11,
        init_params="random",
        covariance_type=covariance_type,
        backend="cuda",
        device=torch.device("cuda"),
        compute_labels_on_fit=False,
    ).fit(_dataset(d))

    assert model.last_backend_used_ == "cuda"
    assert model.labels_ is None
    assert torch.isfinite(model.means_b).all()
    assert torch.isfinite(model.covariances_b).all()
    assert torch.isfinite(model.weights_b).all()
    assert torch.allclose(
        model.weights_b.sum(dim=-1),
        torch.ones(1, device="cuda", dtype=model.weights_b.dtype),
        atol=1e-5,
    )


def test_non_spherical_cuda_training_still_computes_labels_when_requested():
    from gmmxx import GMMXX

    model = GMMXX(
        d=5,
        k=3,
        niter=1,
        tol=0.0,
        seed=12,
        init_params="random",
        covariance_type="diag",
        backend="cuda",
        device=torch.device("cuda"),
        compute_labels_on_fit=True,
    ).fit(_dataset(5, n=128))

    assert model.labels_.shape == (128,)
    assert model.labels_.dtype == torch.int32


def test_full_finalize_accepts_fractional_soft_counts():
    """Soft counts below 1.0 are active, not empty clusters."""
    from gmmxx import _cuda

    means_target = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    counts = torch.tensor([[0.5, 1.5]], device="cuda", dtype=torch.float32)
    sums = counts.unsqueeze(-1) * means_target

    eye = torch.eye(2, device="cuda", dtype=torch.float32).view(1, 1, 2, 2)
    cov = 0.25 * eye.expand(1, 2, 2, 2)
    outer_sums = counts[:, :, None, None] * (
        cov + means_target.unsqueeze(-1) * means_target.unsqueeze(-2)
    )

    old_means = torch.full_like(means_target, 99.0)
    old_L = eye.expand(1, 2, 2, 2).contiguous()

    means_new, L_new, weights_new = _cuda.full_finalize(
        sums,
        outer_sums,
        counts,
        old_means,
        old_L,
        total_n=2,
        reg_covar=1e-6,
    )

    assert torch.allclose(means_new, means_target, atol=1e-5)
    assert torch.isfinite(L_new).all()
    assert torch.allclose(
        weights_new,
        torch.tensor([[0.25, 0.75]], device="cuda"),
        atol=1e-6,
    )
