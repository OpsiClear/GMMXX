# Tied Fused Projected Update Design Note

Status: design spike for Plan 12. This records the intended interface and
oracle before any native tied fused CUDA kernel is attempted.

## Current Path

Tied covariance uses a shared Cholesky factor `L` and projects data into
Euclidean coordinates:

```text
y_n = L^-1 x_n
nu_k = L^-1 mean_k
||L^-1 (x_n - mean_k)||^2 = ||y_n - nu_k||^2
```

The Python CUDA path calls projected spherical kernels for assign/logsumexp/resp
and then accumulates original-space soft stats.

## Proposed Native Fused Update

Interface:

```python
tied_fused_projected_update(x, means, L, log_w, reg_covar)
    -> (nk, sum_x, log_likelihood_sum)
```

The kernel should:

- load/compute projected logits in `L^-1` space;
- normalize responsibilities in fp32;
- accumulate `nk` and original-space `sum_x`;
- leave covariance finalization to `_cuda.tied_finalize`, which already uses
  `xx_total - Σ_k nk_k mean_k mean_k^T`.

## Shape Gate

- `D <= 64`, `K <= 128`, `B <= 8`.
- dtype: fp32/fp16/bf16.
- fallback: current Python orchestration.

## Correctness Oracle

Reference stats:

```python
lse = _cuda.tied_logsumexp(x, means, L, log_w)
resp = _cuda.tied_resp(x, means, L, log_w, lse)
nk = resp.sum(dim=1)
sum_x = torch.bmm(resp.transpose(1, 2), x.float())
ll = lse.sum()
```

Tests must compare `nk`, `sum_x`, and log-likelihood before checking final model
parameters. This avoids hiding projection/stat mismatches behind Cholesky
regularization.

## Risk

The main risk is mixing projected-space sufficient statistics with
original-space finalization. Logits are projected; accumulated `sum_x` must stay
in original coordinates.
