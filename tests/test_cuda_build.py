"""Smoke tests for the CUDA build."""

import os
import subprocess
import sys

import pytest


def test_import_gmmxx_succeeds():
    """import gmmxx must always succeed, with or without the CUDA extension."""
    import gmmxx
    assert hasattr(gmmxx, "GMMXX")


def test_cuda_state_attribute_exists():
    """gmmxx._cuda exposes _HAS_CUDA and has_cuda()."""
    from gmmxx import _cuda
    assert hasattr(_cuda, "_HAS_CUDA")
    assert hasattr(_cuda, "has_cuda")
    # Both are valid: False on CPU-only hosts, True on CUDA hosts with the
    # extension built. Just assert it's a bool.
    assert isinstance(_cuda._HAS_CUDA, bool)
    assert isinstance(_cuda.has_cuda(), bool)


@pytest.mark.skipif(
    not (__import__("torch").cuda.is_available()),
    reason="requires CUDA",
)
def test_canary_kernel_runs_on_cuda():
    """If the extension is built and CUDA is available, the canary works."""
    import torch
    from gmmxx import _cuda

    if not _cuda._HAS_CUDA:
        pytest.skip("gmmxx._C not built")

    x = torch.arange(16, dtype=torch.int32, device="cuda")
    y = _cuda.canary_add_offset(x, 7)
    expected = torch.arange(7, 23, dtype=torch.int32, device="cuda")
    assert torch.equal(y, expected)


def test_skip_cuda_env_var_subprocess():
    """A child process with GMMXX_SKIP_CUDA=1 imports gmmxx without _C.

    Subprocess pattern (mirrors flash-kmeans-cuda's test_persistent.py) because
    GMMXX_SKIP_CUDA is read at install time, not import time. We can't actually
    rebuild here — but we CAN verify that gmmxx._cuda._HAS_CUDA is a bool and
    the require_cuda path raises a clean error when _C is unavailable.
    """
    code = (
        "from gmmxx._cuda import _HAS_CUDA, require_cuda, CudaBackendUnavailable\n"
        "if not _HAS_CUDA:\n"
        "    try:\n"
        "        require_cuda()\n"
        "        print('FAIL: should have raised')\n"
        "    except CudaBackendUnavailable as exc:\n"
        "        print('OK')\n"
        "else:\n"
        "    print('SKIP: _C is built; cannot test missing-extension path here')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert result.stdout.strip() in {"OK", "SKIP: _C is built; cannot test missing-extension path here"}, (
        f"stdout: {result.stdout!r}"
    )
