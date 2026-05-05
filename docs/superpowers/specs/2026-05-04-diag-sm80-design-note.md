# Diag sm80 E-step Design Note

Status: design spike for Plan 12. Do not implement until benchmark data shows
the safe diag E-step is the bottleneck on supported shapes.

## Formulation

For diagonal covariance, the per-component logit can be rewritten as:

```text
precision_kd = 1 / var_kd
weighted_mean_kd = mean_kd * precision_kd
mean_precision_mean_k = sum_d mean_kd^2 * precision_kd
logdet_k = sum_d log(var_kd)

logit_nk = log_w_k - 0.5 * (
    sum_d x_nd^2 * precision_kd
    - 2 * sum_d x_nd * weighted_mean_kd
    + mean_precision_mean_k
    + D * log(2π)
    + logdet_k
)
```

The two reductions

- `x @ weighted_mean.T`
- `x_sq @ precision.T`

are both matrix products and are the sm80 target. The epilogue is scalar fp32
math per `(n, k)`.

## Proposed Gate

- dtype: fp16 or bf16 input.
- arch: sm80+.
- shape: `D % 16 == 0`, `D <= 64`, `K <= 512`.
- fallback: current safe diag kernels for fp32, non-multiple-of-16 D, or runtime
  failure.

## Correctness Tests

- Compare assign/logsumexp/resp against safe diag kernels for fp16/bf16.
- Include near-tied logits and assert probability row sums.
- Verify dispatcher fallback when `D % 16 != 0`.

## Risk

The sm80 path introduces different accumulation order and half/bfloat input
rounding. Keep fp32 accumulators and safe fallback. Do not couple this to the
Plan 9 soft-EM correction; it is an E-step optimization only.
