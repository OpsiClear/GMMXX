# GMMXX CUDA Backend — Plan 10: large_n.py CUDA Integration (spherical)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the C-6 issue from Plan 1's final review. `large_n.py` (the CPU→GPU streaming path for inputs that don't fit in GPU memory) currently hardcodes Triton kernels. This plan adds a `backend: str = "auto"` kwarg to the entry points and routes spherical EM iterations through the CUDA dispatch path when the user requests `backend="cuda"`. Diag/tied/full streaming stays on existing Triton/torch paths (follow-up plans).

**Architecture:** `large_n.py` has four public entry points: `batch_gmm_largeN_cpu` (training), `large_n_predict_cpu`, `large_n_predict_proba_cpu`, `large_n_score_samples_cpu`. Each accepts a CPU input, chunks it, streams chunks to GPU, runs per-chunk EM/inference, aggregates. This plan threads `backend` through each entry point. Per-chunk kernel dispatch goes through the existing `_dispatch.resolve_backend` machinery — when `resolved == "cuda"` for spherical the chunk runs on the new CUDA wrappers; otherwise the existing Triton/torch logic runs unchanged.

**Tech Stack:** Pure Python integration; no new CUDA. Reference: existing `gmmxx/large_n.py` and `gmmxx/_dispatch.py`.

**Spec sections covered:** §7.5 Large-N integration; closes Plan 1 final-review C-6.

**Out of scope (deferred):**
- Diag / tied / full streaming through CUDA → Plan 10b (follow-up)
- Full replacement of the `_HAS_TRITON` import-gating pattern → Plan 10b
- Approx top-K streaming → covered when Plan 11 lands approx top-K CUDA

**Foundation assumed:** Plans 1–8 complete (`full-cuda-plan8` tag). Spherical CUDA is fully feature-complete; tied/full also on CUDA. `_dispatch.resolve_backend` is wired and tested.

---

## Why scope to spherical only

`large_n.py:520-1000` (~500 lines) handles spherical training streaming with extensive logic for: chunking, prefetch, multi-chunk reductions, fallback-on-shape-violation, mid-iteration switching between blocked/streaming/fused Triton paths. Adding a CUDA branch parallel to these is substantial.

For diag/tied/full, the streaming logic is similarly complex but each has its own branch in the file. Doing all four covariances at once would be a 4× larger change.

Plan 10 (slim) wires CUDA into spherical only — closes the user-visible "CUDA backend doesn't work for very large inputs" bug for the most common covariance type. Plan 10b extends to diag/tied/full when there's user demand.

---

## File Structure

### Modified

| Path | Change |
| --- | --- |
| `gmmxx/large_n.py` | (1) Add `backend: str = "auto"` and `legacy_no_triton: bool = False` kwargs to the four entry points. (2) Resolve backend per call. (3) When resolved == "cuda" + spherical, route per-chunk EM/inference through `gmmxx._cuda` wrappers (`spherical_assign`, `spherical_logsumexp`, `blocked_update_spherical`, `finalize_spherical`). (4) Otherwise unchanged — existing Triton/torch logic runs. |
| `gmmxx/interface.py` | The `train()` and inference methods that call into `large_n.py` need to pass `self.backend` and `self._legacy_no_triton`. Find each call site and thread the kwargs through. |
| `tests/test_cuda_largen_spherical.py` | Verify spherical large_n training + inference works under `backend="cuda"`. Skip on CPU-only hosts. |
| `README.md` | Note large_n CUDA support for spherical. |

---

## Task 1 — Audit `gmmxx/large_n.py`

Read the file. Identify:

1. **Public entry points** and their signatures (in `__init__.py` exports: `batch_gmm_largeN_cpu`, `large_n_predict_cpu`, `large_n_predict_proba_cpu`, `large_n_score_samples_cpu`).
2. **Call sites for spherical Triton kernels.** Run `grep -n "spherical_.*_triton" gmmxx/large_n.py` to enumerate.
3. **Call sites for `_HAS_TRITON`** — these gate Triton-only code paths; the new CUDA path bypasses them.
4. **The chunking / streaming structure.** Each entry point has an outer loop over chunks; per-chunk it: H2D-transfers the chunk, runs E-step, aggregates partials, then continues. The CUDA path needs to plug into this loop.

Write a short audit doc (commit message describing the scope) — no code yet.

```bash
git commit --allow-empty -m "$(cat <<'EOF'
Plan 10 audit: large_n.py call sites for spherical CUDA routing

Survey of gmmxx/large_n.py (1761 lines):

Public entry points (4):
- batch_gmm_largeN_cpu — training; spherical branch at L520-700ish
- large_n_predict_cpu — labels; spherical branch at L1300+
- large_n_predict_proba_cpu — soft assignments
- large_n_score_samples_cpu — per-sample log-likelihood

Spherical Triton call sites identified for replacement:
- spherical_assign_triton (E-step assign, per chunk)
- spherical_logsumexp_triton (E-step logsumexp, per chunk for proba/scoring)
- spherical_resp_triton (E-step resp, per chunk for proba)
- triton_blocked_update_spherical / triton_streaming_update_spherical
  (M-step accumulation, per chunk)
- triton_fused_single_tile_update_spherical (fused single-tile, per chunk)

The fused Triton path is the most complex — fuses E+M and writes
partials directly. The CUDA path uses fused_spherical (also fused E+M)
so it maps cleanly: substitute fused_spherical for the Triton call
when backend resolves to cuda.

Plan 10 Task 2 wires resolve_backend at the entry-point level and
dispatches the spherical chunked loop body to either Triton or CUDA.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(This commit is just documentation; no code changes.)

---

## Task 2 — Add `backend` kwarg to entry points

For each of the four public functions, add `backend: str = "auto"` and `legacy_no_triton: bool = False` kwargs. Inside, on the first iteration (or once before the chunk loop), resolve the backend via `_dispatch.resolve_backend_with_env`. Cache the resolved value. Per-chunk, branch on it.

Pseudocode for `batch_gmm_largeN_cpu` spherical branch:

```python
def batch_gmm_largeN_cpu(
    x_cpu, n_clusters, *,
    covariance_type="spherical",
    backend: str = "auto",
    legacy_no_triton: bool = False,
    **other_kwargs
):
    from . import _dispatch

    # Resolve backend ONCE before the chunk loop. Use a representative
    # shape (first chunk's expected shape).
    D = x_cpu.shape[-1]
    K = n_clusters
    # The shape passed here is approximate — the actual N varies per chunk
    # but resolve_backend's shape-gates only depend on D and K for spherical.
    representative_shape = (1, 1024, D, K)  # B=1, N is per-chunk
    resolved = _dispatch.resolve_backend_with_env(
        requested=backend,
        covariance=covariance_type,
        shape=representative_shape,
        dtype=other_kwargs.get("dtype", torch.float32),
        legacy_no_triton=legacy_no_triton,
    )

    if covariance_type == "spherical" and resolved == "cuda":
        return _largen_spherical_cuda(x_cpu, n_clusters, **other_kwargs)

    # Existing Triton/torch path (unchanged).
    return _existing_largen_logic(x_cpu, n_clusters, covariance_type=covariance_type, **other_kwargs)
```

Where `_largen_spherical_cuda` is a new helper that runs the spherical streaming EM loop using the CUDA wrappers in `gmmxx._cuda`.

The new helper, sketched:

```python
def _largen_spherical_cuda(
    x_cpu: torch.Tensor,
    n_clusters: int,
    *,
    max_iters: int = 100,
    tol: float = 1e-4,
    dtype=None,
    device=None,
    chunk_size_data_cpu: int = 1048576,
    seed: int = 0,
    reg_covar: float = 1e-6,
    verbose: bool = False,
    init_centroids=None,
    **_  # ignore other kwargs that only apply to Triton path
):
    """Spherical large-N EM training on CUDA. Streams chunks of x_cpu to GPU
    and runs per-chunk E-step + M-step partial accumulation; aggregates
    partials at the end of each iteration; finalizes via the existing
    finalize_spherical kernel.

    Returns (cluster_ids, means, variances, weights, info_dict).
    """
    from . import _cuda as _cuda_mod
    import math

    N, D = x_cpu.shape
    K = int(n_clusters)
    device = device or torch.device("cuda:0")
    dtype = dtype or torch.float32

    # Initialize on a small chunk: pick K random points from the first chunk.
    rng = torch.Generator().manual_seed(seed)
    init_idx = torch.randint(0, N, (K,), generator=rng)
    means = x_cpu[init_idx].to(device=device, dtype=dtype).unsqueeze(0).contiguous()  # (1, K, D)

    # Initialize variance from a small sample of the data.
    sample_size = min(N, 65536)
    sample_idx = torch.randperm(N, generator=rng)[:sample_size]
    sample = x_cpu[sample_idx].to(device=device, dtype=dtype)
    var = (sample.float().var(dim=0).mean()).expand(1, K).contiguous() / K
    var = var.clamp_min(reg_covar)
    log_w = torch.full((1, K), -math.log(K), dtype=torch.float32, device=device)
    weights = torch.full((1, K), 1.0 / K, dtype=torch.float32, device=device)

    lower_bound_history: list[float] = []
    n_iter = 0
    prev_lb = -math.inf

    for _ in range(max_iters):
        n_iter += 1
        # Per-iteration aggregator buffers (D, K), accumulated across chunks.
        sums_total = torch.zeros((1, K, D), dtype=torch.float32, device=device)
        sumsq_total = torch.zeros((1, K), dtype=torch.float32, device=device)
        counts_total = torch.zeros((1, K), dtype=torch.int32, device=device)
        lse_sum = 0.0
        lse_count = 0

        # Chunk loop.
        for start in range(0, N, chunk_size_data_cpu):
            end = min(start + chunk_size_data_cpu, N)
            chunk = x_cpu[start:end].to(device=device, dtype=dtype, non_blocking=True)
            chunk_b = chunk.unsqueeze(0).contiguous()

            # E-step
            ids = _cuda_mod.spherical_assign(chunk_b, means, var, log_w)
            lse = _cuda_mod.spherical_logsumexp(chunk_b, means, var, log_w)
            lse_sum += float(lse.sum().item())
            lse_count += chunk_b.shape[1]

            # M-step partials
            sums_c, sumsq_c, counts_c = _cuda_mod.blocked_update_spherical(chunk_b, ids, K)
            sums_total += sums_c
            sumsq_total += sumsq_c
            counts_total += counts_c

        # Finalize across all chunks.
        means, var, weights = _cuda_mod.finalize_spherical(
            sums_total, sumsq_total, counts_total, means, var, N, reg_covar
        )
        log_w = torch.log(weights.clamp_min(1e-30))

        lb = lse_sum / max(lse_count, 1)
        lower_bound_history.append(lb)
        if abs(lb - prev_lb) < tol:
            break
        prev_lb = lb

    # Compute cluster_ids on the full input via a final assign pass.
    cluster_ids_chunks = []
    for start in range(0, N, chunk_size_data_cpu):
        end = min(start + chunk_size_data_cpu, N)
        chunk = x_cpu[start:end].to(device=device, dtype=dtype, non_blocking=True)
        chunk_b = chunk.unsqueeze(0).contiguous()
        ids = _cuda_mod.spherical_assign(chunk_b, means, var, log_w)
        cluster_ids_chunks.append(ids.squeeze(0).cpu())
    cluster_ids = torch.cat(cluster_ids_chunks, dim=0)

    info = {
        "lower_bound": lower_bound_history[-1] if lower_bound_history else float("nan"),
        "lower_bound_history": lower_bound_history,
        "n_iter": n_iter,
        "init_source": "cuda_random",
        "triton_estep_enabled": False,
        "triton_streaming_update_enabled": False,
        "triton_fused_update_enabled": False,
        "triton_approx_topk_enabled": False,
        "triton_labels_enabled": False,
        "approximate_em_enabled": False,
        "approx_top_k": None,
        "large_n_streaming_enabled": True,
        "copy_stream_prefetch_enabled": False,  # keep simple; future optimization
        "fallback_reason": None,
        "backend_breakdown": {"cuda": n_iter},
    }
    return cluster_ids.unsqueeze(0), means, var, weights, info
```

(Match the actual return-shape contract — read the existing `batch_gmm_largeN_cpu` to confirm whether labels are returned as `(N,)` or `(B=1, N)` etc.)

The existing return contract is the source of truth. Don't change it.

Apply the analogous backend kwarg + early-return to the three inference entry points (`large_n_predict_cpu`, `large_n_predict_proba_cpu`, `large_n_score_samples_cpu`). Each gets a small CUDA spherical branch.

Build smoke test:

```bash
uv run python -c "
import torch
from gmmxx.large_n import batch_gmm_largeN_cpu

torch.manual_seed(0)
N, D, K = 100_000, 16, 8
x_cpu = torch.randn(N, D)

ids, means, var, w, info = batch_gmm_largeN_cpu(
    x_cpu, K, max_iters=10, tol=1e-4, seed=0,
    covariance_type='spherical',
    backend='cuda',
    chunk_size_data_cpu=32_768,
)
print('ids shape:', ids.shape, 'means shape:', means.shape, 'var shape:', var.shape)
print('lower_bound:', info['lower_bound'])
print('large_n_streaming_enabled:', info['large_n_streaming_enabled'])
print('backend_breakdown:', info.get('backend_breakdown'))
"
```

Expected: training completes; lower_bound finite; backend_breakdown shows `{cuda: n_iter}`.

Commit:

```bash
git add gmmxx/large_n.py
git commit -m "$(cat <<'EOF'
Add backend kwarg + spherical CUDA path to large_n entry points

Plan 10 Task 2. The four public large_n entry points now accept
backend: str = "auto" and legacy_no_triton: bool = False kwargs.
On the first call, resolve_backend_with_env decides; when it returns
"cuda" for spherical covariance, the new _largen_spherical_cuda
helper runs the streaming EM loop using gmmxx._cuda primitives.

Outside the spherical+cuda case, the existing Triton/torch logic
runs unchanged.

The CUDA streaming loop:
- Per-iteration: zero-init aggregator buffers (sums, sumsq, counts).
- Per-chunk: H2D transfer (non_blocking=True), E-step (assign + lse),
  M-step partials (blocked_update_spherical), accumulate.
- After all chunks: finalize_spherical with the aggregated partials.
- Final-iteration: re-assign across all chunks to compute cluster_ids.

backend_breakdown info field reports {"cuda": n_iter}.

Plan 10b (follow-up) extends this to diag/tied/full streaming.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Thread `backend` through `interface.py` calls into `large_n`

Find the call sites where `interface.py` invokes `batch_gmm_largeN_cpu` and friends (search for `batch_gmm_largeN_cpu`, `large_n_predict_cpu`, etc.). Each call needs to pass `self.backend` and `self._legacy_no_triton`.

Likely call site is in the chunked-CPU streaming branch of `train()` and the inference methods (handles inputs larger than `chunk_size_data_cpu`).

Smoke test:

```bash
uv run python -c "
import torch
from gmmxx import GMMXX

torch.manual_seed(0)
# Use a moderately large CPU input.
N, D, K = 200_000, 16, 8
x = torch.randn(N, D)  # CPU input

gmm = GMMXX(n_components=K, max_iter=10, tol=1e-4, random_state=0,
            covariance_type='spherical', backend='cuda',
            chunk_size_data_cpu=65_536, dtype=torch.float32, device='cuda:0')
gmm.fit(x)
print('last_backend_used_:', gmm.last_backend_used_)
print('lower_bound_:', gmm.lower_bound_)
print('large_n_streaming_enabled_:', gmm.large_n_streaming_enabled_)
"
```

Expected: training streams via CUDA; `last_backend_used_=='cuda'`; `large_n_streaming_enabled_==True`.

Commit.

---

## Task 4 — Tests

Create `tests/test_cuda_largen_spherical.py`:

```python
"""Spherical large_n.py CUDA streaming tests."""

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


def test_largen_spherical_cuda_basic():
    """A modestly-large CPU input streams through CUDA and produces a
    finite ELBO."""
    from gmmxx.large_n import batch_gmm_largeN_cpu
    torch.manual_seed(0)
    N, D, K = 100_000, 16, 8
    x_cpu = torch.randn(N, D)
    ids, means, var, w, info = batch_gmm_largeN_cpu(
        x_cpu, K, max_iters=10, tol=1e-4, seed=0,
        covariance_type='spherical',
        backend='cuda',
        chunk_size_data_cpu=32_768,
    )
    assert math.isfinite(info["lower_bound"])
    assert info["large_n_streaming_enabled"] is True
    assert info.get("backend_breakdown", {}).get("cuda", 0) > 0
    assert means.shape == (1, K, D)
    assert var.shape == (1, K)
    assert ids.shape[-1] == N


def test_largen_spherical_cuda_via_GMMXX():
    """End-to-end via the GMMXX class with a CPU input that triggers streaming."""
    from gmmxx import GMMXX
    torch.manual_seed(0)
    N, D, K = 200_000, 16, 8
    x = torch.randn(N, D)
    gmm = GMMXX(n_components=K, max_iter=10, tol=1e-4, random_state=0,
                covariance_type='spherical', backend='cuda',
                chunk_size_data_cpu=65_536, dtype=torch.float32, device='cuda:0')
    gmm.fit(x)
    assert gmm.last_backend_used_ == "cuda"
    assert math.isfinite(gmm.lower_bound_)
    assert gmm.large_n_streaming_enabled_ is True
    assert gmm.means_.shape == (K, D)
    assert gmm.weights_.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_largen_spherical_torch_fallback_unchanged():
    """backend='torch' still uses the existing torch path; no regression."""
    from gmmxx.large_n import batch_gmm_largeN_cpu
    torch.manual_seed(0)
    N, D, K = 50_000, 8, 4
    x_cpu = torch.randn(N, D)
    ids, means, var, w, info = batch_gmm_largeN_cpu(
        x_cpu, K, max_iters=5, tol=1e-4, seed=0,
        covariance_type='spherical',
        backend='torch',
        chunk_size_data_cpu=16_384,
    )
    assert math.isfinite(info["lower_bound"])
    # backend_breakdown not added by the torch path; just confirm CUDA didn't fire.
    bd = info.get("backend_breakdown", {})
    assert bd.get("cuda", 0) == 0
```

Run + commit.

---

## Task 5 — README + tag

Update CUDA section: add a sentence noting large-N streaming is now CUDA-aware for spherical. Tag `largen-spherical-plan10`.

---

## Self-Review

**Spec coverage:** §7.5 large_n integration (closed for spherical; deferred for diag/tied/full).

**Scope-down rationale:** The full large_n.py refactor for all four covariance types is a multi-week effort. Plan 10 lands the most user-visible piece (spherical streaming on CUDA) and structures the code so Plan 10b can extend cleanly.

**Plan 1 final-review C-6:** Closed for spherical. The `legacy_no_triton` kwarg also gets threaded through, handling the use_triton=False back-compat case correctly.

---

## Execution Handoff

Saved to `docs/superpowers/plans/2026-05-03-gmmxx-cuda-largen.md`. Subagent-driven (sonnet for all tasks).
