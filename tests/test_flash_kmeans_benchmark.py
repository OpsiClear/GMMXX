from __future__ import annotations

from benchmarks.benchmark_flash_kmeans_sizes import _full_skip_reason


def test_flash_kmeans_benchmark_guards_large_full_covariance():
    note = _full_skip_reason(
        "full",
        128,
        8192,
        allow_large_full=False,
        max_full_cov_elements=2_000_000,
    )

    assert note is not None
    assert "skipped_full_state_KD2" in note
    assert _full_skip_reason(
        "diag",
        128,
        8192,
        allow_large_full=False,
        max_full_cov_elements=2_000_000,
    ) is None
    assert _full_skip_reason(
        "full",
        128,
        8192,
        allow_large_full=True,
        max_full_cov_elements=2_000_000,
    ) is None
