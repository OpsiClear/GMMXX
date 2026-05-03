"""Build script for GMMXX's CUDA backend.

Compiles a single torch CUDAExtension named ``gmmxx._C`` containing the
hand-written CUDA E-step / M-step / fused / approx kernels for Gaussian
Mixture Model training. Mirrors flash-kmeans-cuda's build pattern:

- nanobind for Python <-> C++ bindings (custom at::Tensor caster in nb_torch.h).
- Single nb_combined.cpp translation unit shipped with nanobind itself.
- Multi-arch fat binary: sm_80 86 89 90 100 120 (Blackwell archs gated on
  nvcc >= 12.8 at build time).
- `-arch=native` is intentionally NOT used so wheels are portable.

Skip the entire CUDA build via the ``GMMXX_SKIP_CUDA=1`` environment variable
for users on hosts without an nvcc toolchain.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "gmmxx" / "csrc"

SKIP_CUDA = os.environ.get("GMMXX_SKIP_CUDA", "").lower() in {"1", "true", "yes"}


def _detect_nvcc_version() -> tuple[int, int] | None:
    """Returns (major, minor) for the nvcc on PATH, or None if not found."""
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _gencode_flags() -> list[str]:
    """Per-architecture flags. Honors TORCH_CUDA_ARCH_LIST by deferring to torch."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []  # let torch.utils.cpp_extension parse the env var
    archs = ["80", "86", "89", "90"]
    blackwell_archs = ["100", "120"]
    if os.environ.get("GMMXX_BUILD_BLACKWELL", "1").lower() in {"1", "true", "yes"}:
        nvcc_version = _detect_nvcc_version()
        if nvcc_version is not None and nvcc_version >= (12, 8):
            archs.extend(blackwell_archs)
    flags: list[str] = []
    for a in archs:
        flags += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    return flags


def _common_nvcc_flags() -> list[str]:
    flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
    ]
    if os.name == "nt":
        # CCCL in CUDA 13 requires the conforming MSVC preprocessor.
        # /DNOMINMAX — prevent windows.h from defining min/max macros.
        # /Usmall — rpcndr.h (pulled in by cuda_runtime.h on Windows) defines
        #   `small` as `char`; ATen/c10 use `small` as a parameter name so
        #   the substitution mangles e.g. `bool small` -> `bool char`.
        #   Passing -Usmall via -Xcompiler undefines it at the MSVC level so it
        #   never reaches ATen headers, regardless of include order.
        flags += ["-Xcompiler=/Zc:preprocessor,/DNOMINMAX,/Usmall"]
    else:
        flags.append("-Xcompiler=-fPIC")
    return flags


def _common_cxx_flags() -> list[str]:
    if os.name == "nt":
        # /bigobj — nanobind's combined TU produces large COFFs.
        # /Zc:preprocessor — required by CCCL.
        # /utf-8 — nanobind ships UTF-8 string literals; default MSVC source charset is CP1252.
        # /Zc:__cplusplus — make __cplusplus report 201703L instead of 199711L.
        return ["/O2", "/std:c++17", "/EHsc", "/bigobj", "/Zc:preprocessor", "/utf-8", "/Zc:__cplusplus"]
    return ["-O3", "-std=c++17", "-fPIC", "-fvisibility=hidden"]


def _build_extension():
    """Construct the CUDAExtension. Imports torch / nanobind lazily so
    `GMMXX_SKIP_CUDA=1` users can install without those build deps."""
    import nanobind
    from torch.utils.cpp_extension import CUDAExtension

    nb_pkg = Path(nanobind.__file__).resolve().parent
    nb_include = nanobind.include_dir()
    nb_combined = nb_pkg / "src" / "nb_combined.cpp"
    nb_robin_include = nb_pkg / "ext" / "robin_map" / "include"
    if not nb_combined.exists():
        raise FileNotFoundError(
            f"Expected nanobind combined source at {nb_combined}; install nanobind>=2.0."
        )
    if not nb_robin_include.exists():
        raise FileNotFoundError(
            f"Expected nanobind robin_map include at {nb_robin_include}; "
            "your nanobind install may be missing bundled deps."
        )

    sources = [
        str(CSRC / "bindings.cpp"),
        str(CSRC / "canary" / "canary.cu"),
        str(CSRC / "estep" / "spherical_safe.cu"),
        str(CSRC / "estep" / "spherical_sm80.cu"),
        str(CSRC / "estep" / "spherical_dispatch.cu"),
        str(CSRC / "mstep" / "blocked_spherical.cu"),
        str(CSRC / "mstep" / "blocked_spherical_sorted.cu"),
        str(CSRC / "mstep" / "finalize_spherical.cu"),
        str(CSRC / "fused" / "spherical_fused.cu"),
        str(nb_combined),
    ]
    include_dirs = [
        str(CSRC),
        str(CSRC / "common"),
        str(CSRC / "canary"),
        str(CSRC / "estep"),
        str(CSRC / "mstep"),
        str(CSRC / "fused"),
        nb_include,
        str(nb_robin_include),
    ]
    return CUDAExtension(
        name="gmmxx._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": _common_cxx_flags(),
            "nvcc": _common_nvcc_flags() + _gencode_flags(),
        },
    )


if SKIP_CUDA:
    print("GMMXX_SKIP_CUDA=1 set; skipping CUDA extension build.", flush=True)
    setup(name="GMMXX")
else:
    from torch.utils.cpp_extension import BuildExtension

    setup(
        name="GMMXX",
        ext_modules=[_build_extension()],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
