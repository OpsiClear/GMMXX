"""End-to-end smoke tests for the Plan 1 foundation.

These do not exercise any real GMM kernels; they verify the public surface
behaves as documented:

  - GMMXX(backend="auto") constructs and the dispatcher resolves to one of
    the three valid values.
  - last_backend_used_ starts None and is settable.
  - cuda_ops module imports and has the expected attributes.
"""

import os

import pytest


def _make_kwargs(**overrides):
    base = {"n_components": 4}
    base.update(overrides)
    return base


def test_construct_with_default_backend():
    from gmmxx import GMMXX
    m = GMMXX(**_make_kwargs())
    assert m.backend == "auto"
    assert m.last_backend_used_ is None
    assert m.cuda_estep_enabled_ is None


def test_construct_with_each_backend():
    from gmmxx import GMMXX
    for b in ("auto", "cuda", "triton", "torch"):
        m = GMMXX(**_make_kwargs(backend=b))
        assert m.backend == b


def test_dispatch_resolves_to_valid_value():
    from gmmxx import _dispatch
    result = _dispatch.resolve_backend(
        requested="auto",
        covariance="spherical",
        shape=(1, 1024, 32, 64),
        dtype=None,
    )
    assert result in {"cuda", "triton", "torch"}


def test_cuda_ops_exposes_documented_surface():
    from gmmxx import cuda_ops
    assert callable(cuda_ops.has_cuda)
    assert callable(cuda_ops.require_cuda)
    assert hasattr(cuda_ops, "CudaBackendUnavailable")
    assert hasattr(cuda_ops, "CudaRuntimeFallback")
    # canary always exists in cuda_ops, but only callable when _C is built.
    assert hasattr(cuda_ops, "canary_add_offset")


def test_env_var_override():
    from gmmxx import _dispatch
    saved = os.environ.pop("GMMXX_BACKEND", None)
    try:
        os.environ["GMMXX_BACKEND"] = "torch"
        result = _dispatch.resolve_backend_with_env(
            requested="auto",
            covariance="spherical",
            shape=(1, 1024, 32, 64),
            dtype=None,
        )
        assert result == "torch"
    finally:
        if saved is None:
            os.environ.pop("GMMXX_BACKEND", None)
        else:
            os.environ["GMMXX_BACKEND"] = saved
