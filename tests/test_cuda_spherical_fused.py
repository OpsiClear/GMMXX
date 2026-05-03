"""Correctness tests for the fused single-tile spherical E/M kernel."""

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


def _setup(B=1, N=256, D=32, K=16, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    means = torch.randn(B, K, D, device="cuda", dtype=dtype)
    var = torch.rand(B, K, device="cuda").clamp_min(0.5).float()
    log_w = torch.log_softmax(torch.randn(B, K, device="cuda"), dim=-1).float()
    return x, means, var, log_w


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D,K", [(8, 16), (16, 32), (32, 64), (64, 128)])
def test_fused_matches_unfused_pipeline(dtype, D, K):
    """Fused output should match the unfused assign+blocked+finalize sequence
    within atomic ULP drift (different reduction order)."""
    from gmmxx import _cuda
    x, means, var, log_w = _setup(N=256, D=D, K=K, dtype=dtype)
    reg_covar = 1e-6
    N = x.shape[1]

    # Unfused path: SOFT EM via spherical_resp + manual accumulation. The
    # fused kernel does soft EM (using resp), so for an apples-to-apples
    # comparison we compare against the soft-EM unfused pipeline below.
    lse_u = _cuda.spherical_logsumexp(x, means, var, log_w)
    resp_u = _cuda.spherical_resp(x, means, var, log_w, lse_u)  # (B, N, K)

    # Soft sufficient stats from resp.
    n_k = resp_u.sum(dim=1)                                 # (B, K)
    sum_x = (resp_u.unsqueeze(-1) * x.float().unsqueeze(2)).sum(dim=1)  # (B, K, D)
    sum_xx = (resp_u * x.float().pow(2).sum(-1, keepdim=True)).sum(dim=1)  # (B, K)

    means_u = sum_x / n_k.unsqueeze(-1).clamp_min(1e-30)
    var_u = ((sum_xx / n_k.clamp_min(1e-30)) - means_u.pow(2).sum(-1)) / D
    var_u = var_u.clamp_min(reg_covar)
    weights_u = n_k / N

    # Fused path
    new_means_f, new_var_f, new_weights_f, lse_f, ids_f = _cuda.fused_spherical(
        x, means, var, log_w, reg_covar
    )

    rtol, atol = (1e-4, 1e-3) if dtype == torch.float32 else (5e-3, 5e-3)
    assert torch.allclose(new_weights_f, weights_u, rtol=rtol, atol=atol), (
        f"weights diff: {(new_weights_f - weights_u).abs().max().item()}"
    )
    assert torch.allclose(lse_f, lse_u, rtol=rtol, atol=atol), (
        f"lse diff: {(lse_f - lse_u).abs().max().item()}"
    )
    # means and var are derived from sums divided by counts; tolerate looser bound.
    means_atol = 1e-3 if dtype == torch.float32 else 5e-2
    assert torch.allclose(new_means_f.float(), means_u.to(new_means_f.dtype).float(),
                          rtol=rtol, atol=means_atol), (
        f"means diff: {(new_means_f.float() - means_u.float()).abs().max().item()}"
    )


def test_fused_zero_N():
    """Empty N must not crash; outputs preserved from initial values."""
    from gmmxx import _cuda
    x = torch.empty(1, 0, 16, device="cuda")
    means = torch.randn(1, 8, 16, device="cuda")
    var = torch.ones(1, 8, device="cuda")
    log_w = torch.zeros(1, 8, device="cuda")
    nm, nv, nw, lse, ids = _cuda.fused_spherical(x, means, var, log_w, 1e-6)
    assert nm.shape == (1, 8, 16)
    assert torch.allclose(nm, means)
    assert (nw == 0).all()


@pytest.mark.parametrize("D,K", [(8, 16), (32, 64), (64, 128)])
def test_fused_train_loop_converges(D, K):
    """A few iterations of fused should produce monotone-non-decreasing ELBO."""
    from gmmxx import _cuda
    torch.manual_seed(0)
    x = torch.randn(1, 4096, D, device="cuda")
    means = torch.randn(1, K, D, device="cuda")
    var = torch.ones(1, K, device="cuda")
    log_w = torch.full((1, K), -math.log(K), device="cuda")

    last_elbo = None
    for it in range(5):
        means, var, weights, lse, ids = _cuda.fused_spherical(
            x, means, var, log_w, 1e-6
        )
        log_w = torch.log(weights.clamp_min(1e-30))
        elbo = lse.mean().item()
        assert math.isfinite(elbo), f"ELBO went non-finite at iter {it}"
        if last_elbo is not None:
            # EM is monotone non-decreasing; allow tiny ULP fluctuation.
            assert elbo >= last_elbo - 1e-3, (
                f"ELBO decreased: {last_elbo} -> {elbo} at iter {it}"
            )
        last_elbo = elbo


def test_fused_labels_argmax_consistency():
    """labels output should equal argmax of resp (= argmax of logits)."""
    from gmmxx import _cuda
    torch.manual_seed(0)
    x = torch.randn(1, 256, 16, device="cuda")
    means = torch.randn(1, 8, 16, device="cuda")
    var = torch.ones(1, 8, device="cuda")
    log_w = torch.zeros(1, 8, device="cuda")

    nm, nv, nw, lse, ids = _cuda.fused_spherical(x, means, var, log_w, 1e-6)
    # Compare to the unfused assign output (argmax of the same logits).
    ids_unfused = _cuda.spherical_assign(x, means, var, log_w)
    assert torch.equal(ids, ids_unfused), "fused labels must match unfused assign"
