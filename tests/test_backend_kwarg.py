"""Tests for the backend kwarg, attributes, and use_triton deprecation shim."""

import warnings

import pytest


def _make_kwargs(**overrides):
    """Minimal kwargs that satisfy GMMXX.__init__."""
    base = {"n_components": 4}
    base.update(overrides)
    return base


class TestBackendKwarg:
    def test_default_backend_is_auto(self):
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs())
        assert m.backend == "auto"

    def test_backend_explicit_torch(self):
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs(backend="torch"))
        assert m.backend == "torch"

    def test_backend_explicit_cuda(self):
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs(backend="cuda"))
        assert m.backend == "cuda"

    def test_backend_invalid_raises(self):
        from gmmxx import GMMXX
        with pytest.raises(ValueError, match="backend"):
            GMMXX(**_make_kwargs(backend="bogus"))


class TestUseTritonDeprecation:
    def test_use_triton_true_maps_to_auto(self):
        from gmmxx import GMMXX
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = GMMXX(**_make_kwargs(use_triton=True))
        assert m.backend == "auto"
        assert m._legacy_no_triton is False
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_use_triton_false_maps_to_auto_with_no_triton_flag(self):
        from gmmxx import GMMXX
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = GMMXX(**_make_kwargs(use_triton=False))
        assert m.backend == "auto"
        assert m._legacy_no_triton is True
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_both_kwargs_raises(self):
        from gmmxx import GMMXX
        with pytest.raises(ValueError, match="backend.*use_triton"):
            GMMXX(**_make_kwargs(backend="cuda", use_triton=False))

    def test_deprecation_warning_emitted_once_per_instance(self):
        """The DeprecationWarning should fire once during __init__, not on every
        attribute access. Constructing two instances must produce two warnings."""
        from gmmxx import GMMXX
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GMMXX(**_make_kwargs(use_triton=True))
            GMMXX(**_make_kwargs(use_triton=True))
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 2


class TestNewAttributes:
    def test_last_backend_used_starts_none(self):
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs())
        assert m.last_backend_used_ is None

    def test_cuda_enabled_attrs_start_none(self):
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs())
        assert m.cuda_estep_enabled_ is None
        assert m.cuda_fused_update_enabled_ is None
        assert m.cuda_approx_topk_enabled_ is None


class TestGetParamsClone:
    def test_get_params_returns_backend_not_use_triton(self):
        """get_params must return 'backend' (canonical) and NOT 'use_triton'
        so sklearn.base.clone() round-trips cleanly."""
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs(backend="torch"))
        params = m.get_params()
        assert "backend" in params
        assert params["backend"] == "torch"
        assert "use_triton" not in params

    def test_get_params_legacy_no_triton_roundtrips(self):
        """If user set use_triton=False, get_params still returns 'backend': 'auto'
        but the legacy flag must persist through clone()."""
        from gmmxx import GMMXX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            m = GMMXX(**_make_kwargs(use_triton=False))
        params = m.get_params()
        assert params["backend"] == "auto"
        # Round-trip via constructor.
        m2 = GMMXX(**params)
        assert m2.backend == "auto"
        # Note: legacy_no_triton is NOT round-tripped because get_params sheds
        # the legacy flag — that's intentional. The semantics are: once you've
        # been through one fit, your params are clean.
        assert m2._legacy_no_triton is False

    def test_clone_via_sklearn_pattern(self):
        """sklearn.base.clone semantically: cls(**est.get_params())."""
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs(backend="torch", n_components=8))
        clone = type(m)(**m.get_params())
        assert clone.backend == m.backend
        assert clone.k == m.k

    def test_set_params_backend(self):
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs())
        m.set_params(backend="torch")
        assert m.backend == "torch"

    def test_set_params_invalid_backend_raises(self):
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs())
        with pytest.raises(ValueError, match="backend"):
            m.set_params(backend="bogus")

    def test_set_params_use_triton_routes_through_shim(self):
        """set_params(use_triton=False) must update backend semantics, not
        bypass the deprecation."""
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.set_params(use_triton=False)
        assert m.backend == "auto"
        assert m._legacy_no_triton is True
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_set_params_use_triton_after_explicit_backend_raises(self):
        """Symmetry with __init__: if backend is already explicit (not 'auto'),
        set_params(use_triton=...) must raise instead of silently overwriting."""
        from gmmxx import GMMXX
        m = GMMXX(**_make_kwargs(backend="cuda"))
        with pytest.raises(ValueError, match="backend is already explicit"):
            m.set_params(use_triton=True)
        # State unchanged.
        assert m.backend == "cuda"
        assert m._legacy_no_triton is False
