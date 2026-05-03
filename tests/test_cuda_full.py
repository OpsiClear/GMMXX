"""Full covariance CUDA path correctness tests."""

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
    """Make a random fit-state for full covariance with PD Σ_k."""
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    means = torch.randn(B, K, D, device=device, dtype=dtype)
    # Build per-cluster lower-triangular L_k with positive diagonal.
    L = torch.tril(torch.randn(B, K, D, D, device=device, dtype=dtype))
    diag_idx = torch.arange(D, device=device)
    L_diag = L[:, :, diag_idx, diag_idx].abs() + 1.0
    L = L.clone()
    L[:, :, diag_idx, diag_idx] = L_diag
    log_w = torch.log_softmax(torch.randn(B, K, device=device), dim=-1).float()
    return x, means, L, log_w


def _ref_full_logits(x, means, L, log_w):
    """Reference: log p_k(x_n) = log_w_k - 0.5*D*log(2π) - log|L_k| - 0.5*||L_k⁻¹(x_n - μ_k)||²"""
    B, N, D = x.shape
    K = means.shape[1]
    x_f = x.float().unsqueeze(2)               # (B, N, 1, D)
    means_f = means.float().unsqueeze(1)       # (B, 1, K, D)
    diff = x_f - means_f                        # (B, N, K, D)
    diff_t = diff.permute(0, 2, 3, 1).contiguous()  # (B, K, D, N)
    z = torch.linalg.solve_triangular(L.float(), diff_t, upper=False)  # (B, K, D, N)
    dist = z.pow(2).sum(2).permute(0, 2, 1)    # (B, N, K)
    log_det_L = L.float().diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)  # (B, K)
    log_norm_const = 0.5 * D * math.log(2 * math.pi)
    return log_w.unsqueeze(1) - log_norm_const - log_det_L.unsqueeze(1) - 0.5 * dist


class TestFullAssign:
    def test_argmax_matches_reference(self):
        from gmmxx import _cuda
        x, means, L, log_w = _setup(N=256, D=8, K=4, dtype=torch.float32)
        cuda_ids = _cuda.full_assign(x, means, L, log_w)
        ref_ids = _ref_full_logits(x, means, L, log_w).argmax(-1).int()
        agree = (cuda_ids == ref_ids).float().mean().item()
        assert agree >= 0.99

    def test_returns_int32_BN(self):
        from gmmxx import _cuda
        x, means, L, log_w = _setup(B=2, N=64, D=8, K=4)
        ids = _cuda.full_assign(x, means, L, log_w)
        assert ids.shape == (2, 64) and ids.dtype == torch.int32


class TestFullLogsumexp:
    def test_matches_reference(self):
        from gmmxx import _cuda
        x, means, L, log_w = _setup(N=256, D=8, K=4)
        cuda_lse = _cuda.full_logsumexp(x, means, L, log_w)
        ref_lse = _ref_full_logits(x, means, L, log_w).logsumexp(-1)
        assert torch.allclose(cuda_lse, ref_lse, rtol=1e-4, atol=1e-4)


class TestFullResp:
    def test_resp_sums_to_one(self):
        from gmmxx import _cuda
        x, means, L, log_w = _setup()
        lse = _cuda.full_logsumexp(x, means, L, log_w)
        r = _cuda.full_resp(x, means, L, log_w, lse)
        assert torch.allclose(r.sum(-1), torch.ones_like(r.sum(-1)), atol=1e-4)


class TestFullBlockedUpdate:
    def test_counts_sum_to_N(self):
        from gmmxx import _cuda
        torch.manual_seed(0)
        x = torch.randn(1, 256, 8, device="cuda")
        ids = torch.randint(0, 4, (1, 256), device="cuda", dtype=torch.int32)
        sums, outer_sums, counts = _cuda.full_blocked_update(x, ids, 4)
        assert counts.sum().item() == 256

    def test_outer_sums_match_groupby(self):
        from gmmxx import _cuda
        torch.manual_seed(0)
        B, N, D, K = 1, 64, 4, 3
        x = torch.randn(B, N, D, device="cuda")
        ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
        sums, outer_sums, counts = _cuda.full_blocked_update(x, ids, K)

        for b in range(B):
            for k in range(K):
                mask = (ids[b] == k)
                if mask.sum() == 0:
                    assert counts[b, k].item() == 0
                    assert torch.allclose(outer_sums[b, k], torch.zeros(D, D, device="cuda"))
                    continue
                ref_sum = x[b][mask].sum(0)  # (D,)
                xx = x[b][mask].unsqueeze(-1) * x[b][mask].unsqueeze(-2)  # (n_k, D, D)
                ref_outer = xx.sum(0)  # (D, D)
                assert torch.allclose(sums[b, k], ref_sum, atol=1e-4)
                assert torch.allclose(outer_sums[b, k], ref_outer, atol=1e-3)


class TestFullFinalize:
    def test_outputs_lower_triangular(self):
        from gmmxx import _cuda
        torch.manual_seed(0)
        B, N, D, K = 1, 256, 8, 4
        x = torch.randn(B, N, D, device="cuda")
        ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
        sums, outer_sums, counts = _cuda.full_blocked_update(x, ids, K)
        old_means = torch.zeros(B, K, D, device="cuda")
        old_L = torch.eye(D, device="cuda").unsqueeze(0).unsqueeze(0).expand(B, K, D, D).contiguous()
        new_means, new_L, new_w = _cuda.full_finalize(
            sums, outer_sums, counts, old_means, old_L, N, 1e-6
        )
        assert new_means.shape == (B, K, D)
        assert new_L.shape == (B, K, D, D)
        # Lower-triangular per cluster.
        assert torch.allclose(new_L.triu(1), torch.zeros_like(new_L.triu(1)))
        # Positive diagonal per cluster.
        diag = new_L.diagonal(dim1=-2, dim2=-1)
        assert (diag > 0).all()
        # Weights sum to 1.
        assert abs(new_w.sum().item() - 1.0) < 1e-4

    def test_empty_cluster_preserves_old_L(self):
        from gmmxx import _cuda
        B, K, D = 1, 3, 4
        # Cluster 1 has zero count.
        sums = torch.tensor([[[1.0]*D, [0.0]*D, [2.0]*D]], device="cuda")
        outer_sums = torch.zeros(B, K, D, D, device="cuda")
        # Make outer_sums for cluster 0 and 2 PD.
        outer_sums[0, 0] = torch.eye(D, device="cuda") * 5
        outer_sums[0, 2] = torch.eye(D, device="cuda") * 5
        counts = torch.tensor([[1, 0, 1]], device="cuda", dtype=torch.int32)
        old_means = torch.tensor([[[0.0]*D, [99.0]*D, [0.0]*D]], device="cuda")
        old_L = torch.eye(D, device="cuda").unsqueeze(0).unsqueeze(0).expand(B, K, D, D).contiguous()
        # Cluster 1's old_L: 42 * I.
        old_L = old_L.clone()
        old_L[0, 1] *= 42.0
        new_means, new_L, new_w = _cuda.full_finalize(
            sums, outer_sums, counts, old_means, old_L, 2, 1e-6
        )
        # Cluster 1 should keep old means and old_L.
        assert torch.allclose(new_means[0, 1], old_means[0, 1])
        assert torch.allclose(new_L[0, 1], old_L[0, 1])
        # Cluster 1 weight = 0.
        assert new_w[0, 1].item() == 0.0
        # Other clusters should have updated values.
        assert not torch.allclose(new_means[0, 0], old_means[0, 0])


class TestFullTrainLoop:
    def test_full_train_converges(self):
        from gmmxx import GMMXX
        torch.manual_seed(0)
        x = torch.randn(2048, 8, device="cuda")
        gmm = GMMXX(n_components=4, max_iter=15, tol=1e-4, random_state=0,
                    covariance_type="full", backend="cuda")
        gmm.fit(x)
        assert math.isfinite(gmm.lower_bound_)
        assert gmm.last_backend_used_ == "cuda"
        assert gmm.cuda_estep_enabled_ is True
        assert gmm.means_.shape == (4, 8)
        # Full covariance is per-cluster D×D.
        assert gmm.covariances_.shape == (4, 8, 8)
        # Each cluster's covariance is symmetric.
        assert torch.allclose(
            gmm.covariances_, gmm.covariances_.transpose(-1, -2), atol=1e-4
        )


class TestFullInference:
    def test_predict_predict_proba_consistency(self):
        from gmmxx import GMMXX
        torch.manual_seed(0)
        x_train = torch.randn(2048, 8, device="cuda")
        x_test = torch.randn(256, 8, device="cuda")
        gmm = GMMXX(n_components=4, max_iter=10, tol=1e-4, random_state=0,
                    covariance_type="full", backend="cuda")
        gmm.fit(x_train)
        labels = gmm.predict(x_test)
        proba = gmm.predict_proba(x_test)
        assert labels.shape == (256,)
        assert proba.shape == (256, 4)
        assert torch.allclose(proba.sum(-1), torch.ones(256, device="cuda"), atol=1e-4)
        agree = (labels == proba.argmax(-1).long()).float().mean().item()
        assert agree >= 0.99

    def test_score_samples_finite(self):
        from gmmxx import GMMXX
        torch.manual_seed(0)
        x = torch.randn(2048, 8, device="cuda")
        gmm = GMMXX(n_components=4, max_iter=10, tol=1e-4, random_state=0,
                    covariance_type="full", backend="cuda")
        gmm.fit(x)
        ll = gmm.score_samples(x[:512])
        assert ll.shape == (512,)
        assert torch.isfinite(ll).all()
        s = gmm.score(x[:512])
        assert math.isfinite(s)
