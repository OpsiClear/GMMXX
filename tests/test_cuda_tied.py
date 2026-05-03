"""Tied CUDA path correctness tests."""

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


def _setup(B=1, N=256, D=8, K=4, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    means = torch.randn(B, K, D, device=device, dtype=dtype)
    # Random lower-triangular L with positive diagonal.
    L = torch.tril(torch.randn(B, D, D, device=device, dtype=dtype))
    # Ensure positive diagonal for Cholesky validity.
    diag_idx = torch.arange(D, device=device)
    L_diag = L[:, diag_idx, diag_idx].abs() + 1.0
    L = L.clone()
    L[:, diag_idx, diag_idx] = L_diag
    log_w = torch.log_softmax(torch.randn(B, K, device=device), dim=-1).float()
    return x, means, L, log_w


def _ref_tied_logits(x, means, L, log_w):
    """Reference tied logits computed via host."""
    B, N, D = x.shape
    K = means.shape[1]
    # Project x and means.
    x_t = x.float().transpose(-1, -2)  # (B, D, N)
    y = torch.linalg.solve_triangular(L.float(), x_t, upper=False).transpose(-1, -2)  # (B, N, D)
    means_t = means.float().transpose(-1, -2)  # (B, D, K)
    nu = torch.linalg.solve_triangular(L.float(), means_t, upper=False).transpose(-1, -2)  # (B, K, D)
    # log|L|
    log_det_L = L.float().diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)  # (B,)
    # log p_k(x_n) = log_w_k - 0.5*D*log(2π) - log|L| - 0.5*||y_n - ν_k||²
    diff = y.unsqueeze(2) - nu.unsqueeze(1)  # (B, N, K, D)
    dist = diff.pow(2).sum(-1)  # (B, N, K)
    log_norm_const = 0.5 * D * math.log(2 * math.pi)
    return log_w.unsqueeze(1) - log_norm_const - log_det_L.view(B, 1, 1) - 0.5 * dist


class TestTiedProject:
    def test_round_trip(self):
        from gmmxx import _cuda
        x, _, L, _ = _setup(D=16, K=4, dtype=torch.float32)
        y = _cuda.tied_project(x, L)
        # Reconstruct: L @ y^T should give x^T.
        x_recon = (L.float() @ y.float().transpose(-1, -2)).transpose(-1, -2)
        assert torch.allclose(x, x_recon, rtol=1e-4, atol=1e-4)

    def test_log_det(self):
        from gmmxx import _cuda
        _, _, L, _ = _setup(D=8)
        cuda_log_det = _cuda.tied_log_det(L)
        ref_log_det = L.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
        assert torch.allclose(cuda_log_det, ref_log_det, atol=1e-5)


class TestTiedAssign:
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_argmax_matches_reference(self, dtype):
        from gmmxx import _cuda
        x, means, L, log_w = _setup(N=256, D=16, K=8, dtype=dtype)
        cuda_ids = _cuda.tied_assign(x, means, L, log_w)
        ref_ids = _ref_tied_logits(x, means, L, log_w).argmax(-1).int()
        agree = (cuda_ids == ref_ids).float().mean().item()
        assert agree >= 0.99, f"only {agree:.3f} agreement"


class TestTiedLogsumexp:
    def test_matches_reference(self):
        from gmmxx import _cuda
        x, means, L, log_w = _setup(N=256, D=16, K=8, dtype=torch.float32)
        cuda_lse = _cuda.tied_logsumexp(x, means, L, log_w)
        ref_lse = _ref_tied_logits(x, means, L, log_w).logsumexp(-1)
        assert torch.allclose(cuda_lse, ref_lse, rtol=1e-4, atol=1e-4)


class TestTiedResp:
    def test_resp_sums_to_one(self):
        from gmmxx import _cuda
        x, means, L, log_w = _setup()
        lse = _cuda.tied_logsumexp(x, means, L, log_w)
        r = _cuda.tied_resp(x, means, L, log_w, lse)
        assert torch.allclose(r.sum(-1), torch.ones_like(r.sum(-1)), atol=1e-4)


class TestTiedFinalize:
    def test_finalize_outputs_valid_cholesky(self):
        from gmmxx import _cuda
        torch.manual_seed(0)
        B, N, D, K = 1, 256, 8, 4
        x = torch.randn(B, N, D, device="cuda")
        ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
        sums, _, counts = _cuda.blocked_update_spherical(x, ids, K)
        xx_total = x.float().transpose(-1, -2) @ x.float()
        means_new, L_new, weights_new = _cuda.tied_finalize(
            sums, xx_total, counts, N, 1e-6
        )
        assert means_new.shape == (B, K, D)
        assert L_new.shape == (B, D, D)
        assert weights_new.shape == (B, K)
        # Lower triangular
        assert torch.allclose(L_new.triu(1), torch.zeros_like(L_new.triu(1)))
        # Positive diagonal (positive definite Σ)
        diag = L_new.diagonal(dim1=-2, dim2=-1)
        assert (diag > 0).all()
        # Weights sum to 1
        assert abs(weights_new.sum().item() - 1.0) < 1e-4


class TestTiedTrainLoopConverges:
    def test_em_loop_converges(self):
        from gmmxx import GMMXX
        torch.manual_seed(0)
        x = torch.randn(2048, 16, device="cuda")
        gmm = GMMXX(n_components=6, max_iter=15, tol=1e-4, random_state=0,
                    covariance_type="tied", backend="cuda")
        gmm.fit(x)
        assert math.isfinite(gmm.lower_bound_)
        assert gmm.last_backend_used_ == "cuda"
        assert gmm.cuda_estep_enabled_ is True
        assert gmm.means_.shape == (6, 16)
        assert gmm.covariances_.shape == (16, 16)
        # Symmetric
        assert torch.allclose(gmm.covariances_, gmm.covariances_.transpose(-1, -2), atol=1e-4)


class TestTiedInference:
    def test_predict_predict_proba_consistency(self):
        from gmmxx import GMMXX
        torch.manual_seed(0)
        x_train = torch.randn(2048, 16, device="cuda")
        x_test = torch.randn(256, 16, device="cuda")
        gmm = GMMXX(n_components=6, max_iter=10, tol=1e-4, random_state=0,
                    covariance_type="tied", backend="cuda")
        gmm.fit(x_train)
        labels = gmm.predict(x_test)
        proba = gmm.predict_proba(x_test)
        assert labels.shape == (256,)
        assert proba.shape == (256, 6)
        assert torch.allclose(proba.sum(-1), torch.ones(256, device="cuda"), atol=1e-4)
        agree = (labels == proba.argmax(-1).long()).float().mean().item()
        assert agree >= 0.99

    def test_score_samples_finite(self):
        from gmmxx import GMMXX
        torch.manual_seed(0)
        x = torch.randn(2048, 16, device="cuda")
        gmm = GMMXX(n_components=6, max_iter=10, tol=1e-4, random_state=0,
                    covariance_type="tied", backend="cuda")
        gmm.fit(x)
        ll = gmm.score_samples(x[:512])
        assert ll.shape == (512,)
        assert torch.isfinite(ll).all()
