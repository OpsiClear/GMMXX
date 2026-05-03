"""Tests for gmmxx._dispatch.resolve_backend."""

import os
from unittest.mock import patch

import pytest
import torch

from gmmxx import _dispatch


def _set_env(**kw):
    """Helper context manager to set/unset env vars per test."""
    return patch.dict(os.environ, kw, clear=False)


# ---------------------------------------------------------------------------
# Truth table: resolve_backend(requested, covariance, shape, dtype, legacy_no_triton)
# ---------------------------------------------------------------------------

class TestResolveBackend:
    """Plan 1: cuda_*_supported stubs all return False, so 'auto' will land on
    torch on CPU hosts and on triton when triton is present and supports the shape."""

    def test_explicit_torch_always_returns_torch(self):
        result = _dispatch.resolve_backend(
            requested="torch",
            covariance="spherical",
            shape=(1, 1024, 32),
            dtype=None,
        )
        assert result == "torch"

    def test_explicit_triton_returns_triton_when_supported(self):
        # spherical d=32, k=64 is inside TRITON_SPHERICAL_MAX_*.
        result = _dispatch.resolve_backend(
            requested="triton",
            covariance="spherical",
            shape=(1, 1024, 32, 64),  # (B, N, D, K)
            dtype=None,
        )
        assert result == "triton"

    def test_explicit_triton_falls_through_to_torch_when_unsupported(self):
        # Spherical d=200 > TRITON_SPHERICAL_MAX_D.
        result = _dispatch.resolve_backend(
            requested="triton",
            covariance="spherical",
            shape=(1, 1024, 200, 64),
            dtype=None,
        )
        assert result == "torch"

    def test_explicit_cuda_returns_cuda_when_spherical_supported(self):
        # Plan 2: cuda_spherical_supported now returns True for d=32 k=64 fp32.
        # If the host has CUDA available, this resolves to "cuda".
        # On CPU-only hosts, _cuda.has_cuda() is False, so it falls through to "torch".
        result = _dispatch.resolve_backend(
            requested="cuda",
            covariance="spherical",
            shape=(1, 1024, 32, 64),
            dtype=torch.float32,
        )
        # Either outcome is correct; depends on host:
        from gmmxx import _cuda as _cuda_mod
        if _cuda_mod.has_cuda():
            assert result == "cuda"
        else:
            assert result == "torch"

    def test_auto_picks_triton_on_supported_shape_when_cuda_stub_false(self):
        # cuda stub False → fallback to triton when supported.
        result = _dispatch.resolve_backend(
            requested="auto",
            covariance="spherical",
            shape=(1, 1024, 32, 64),
            dtype=None,
        )
        assert result == "triton"

    def test_auto_picks_torch_when_neither_supported(self):
        # Spherical d=200 → no triton; cuda stub False → torch.
        result = _dispatch.resolve_backend(
            requested="auto",
            covariance="spherical",
            shape=(1, 1024, 200, 64),
            dtype=None,
        )
        assert result == "torch"

    def test_legacy_no_triton_filters_triton_in_auto(self):
        # use_triton=False → legacy_no_triton=True → triton is removed from
        # the chain. Plan 1's cuda stub is False, so we land on torch.
        result = _dispatch.resolve_backend(
            requested="auto",
            covariance="spherical",
            shape=(1, 1024, 32, 64),
            dtype=None,
            legacy_no_triton=True,
        )
        assert result == "torch"

    def test_legacy_no_triton_with_explicit_triton_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            _dispatch.resolve_backend(
                requested="triton",
                covariance="spherical",
                shape=(1, 1024, 32, 64),
                dtype=None,
                legacy_no_triton=True,
            )

    def test_invalid_requested_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            _dispatch.resolve_backend(
                requested="bogus",
                covariance="spherical",
                shape=(1, 1024, 32, 64),
                dtype=None,
            )


class TestEnvVarOverride:
    """GMMXX_BACKEND env var overrides the kwarg only when kwarg is 'auto'."""

    def test_env_var_overrides_auto(self):
        with _set_env(GMMXX_BACKEND="torch"):
            result = _dispatch.resolve_backend_with_env(
                requested="auto",
                covariance="spherical",
                shape=(1, 1024, 32, 64),
                dtype=None,
            )
            assert result == "torch"

    def test_env_var_ignored_when_kwarg_explicit(self):
        with _set_env(GMMXX_BACKEND="cuda"):
            result = _dispatch.resolve_backend_with_env(
                requested="torch",
                covariance="spherical",
                shape=(1, 1024, 32, 64),
                dtype=None,
            )
            assert result == "torch"

    def test_invalid_env_var_value_is_ignored(self):
        with _set_env(GMMXX_BACKEND="bogus"):
            # Must not raise; just behave as if env var unset.
            result = _dispatch.resolve_backend_with_env(
                requested="auto",
                covariance="spherical",
                shape=(1, 1024, 32, 64),
                dtype=None,
            )
            # Plan 1: cuda stub False, triton supported → triton
            # (or torch if triton not installed). Just assert it's a valid value.
            assert result in {"cuda", "triton", "torch"}


class TestDispatchKernel:
    def test_cuda_op_resolves_to_cuda_module(self):
        if not _dispatch._cuda.has_cuda():
            pytest.skip("requires CUDA")
        import torch
        x = torch.randn(1, 8, 4, device="cuda")
        means = torch.randn(1, 3, 4, device="cuda")
        var = torch.ones(1, 3, device="cuda")
        log_w = torch.zeros(1, 3, device="cuda")
        out = _dispatch.dispatch_kernel(
            "spherical_assign", "cuda", x, means, var, log_w
        )
        assert out.shape == (1, 8) and out.dtype == torch.int32

    def test_triton_op_path_resolves(self):
        """Verify the _TRITON_OPS_BY_NAME map looks up the right callable."""
        try:
            fn = _dispatch._resolve_callable("spherical_assign", "triton")
            assert callable(fn)
        except (ImportError, KeyError):
            pytest.skip("Triton not installed")

    def test_unknown_op_in_cuda_raises_attribute_error(self):
        if not _dispatch._cuda.has_cuda():
            pytest.skip("requires CUDA")
        with pytest.raises(AttributeError):
            _dispatch._resolve_callable("nonexistent_op", "cuda")

    def test_unknown_op_in_triton_raises_key_error(self):
        with pytest.raises(KeyError):
            _dispatch._resolve_callable("nonexistent_op", "triton")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            _dispatch._resolve_callable("spherical_assign", "bogus")
