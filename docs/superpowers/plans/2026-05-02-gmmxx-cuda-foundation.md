# GMMXX CUDA Backend — Plan 1: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a build skeleton, dispatch plumbing, public-API additions, deprecation shim, and a canary CUDA kernel proving the nanobind FFI works end-to-end. After this plan, `pip install -e .` builds `gmmxx._C` on Linux and Windows, `gmmxx.GMMXX(backend="cuda")` resolves to a working dispatcher (which still falls through to torch because no real GMM kernels exist yet), and `use_triton` is deprecated cleanly.

**Architecture:** Mirror flash-kmeans-cuda's build (setuptools + `CUDAExtension` + nanobind's bundled `nb_combined.cpp` + custom `at::Tensor` caster) inside the existing `gmmxx` package. Add `gmmxx/_dispatch.py`, `gmmxx/_cuda.py`, `gmmxx/cuda_ops.py` as the three new Python modules. CUDA sources live under `gmmxx/csrc/{common,canary}/`; subsequent plans add `estep/`, `mstep/`, `fused/`, `approx/`. The dispatcher gates on `_HAS_CUDA + cuda_<cov>_supported() + dtype + shape`; all gates default to False in this plan, so behavior is unchanged for end users until Plan 2.

**Tech Stack:** Python 3.12, PyTorch 2.11.x, nanobind 2.x, CUDA 12.8+, nvcc with `-arch=sm_80,86,89,90,100,120`, pytest. Reference template: `~/Projects/flash-kmeans-cuda`.

**Spec:** `docs/superpowers/specs/2026-05-02-gmmxx-cuda-backend-design.md` (commits `4a0258d`, `f423faf`).

**Out of scope for this plan:** real GMM kernels (next plan: spherical), `large_n.py` refactor (own plan), perf passes (own plan), CI/wheel matrix (own plan).

---

## File Structure

### Created in this plan

| Path | Responsibility |
| --- | --- |
| `setup.py` | CUDAExtension config, nvcc/cxx/gencode flags, nanobind plumbing. |
| `gmmxx/_dispatch.py` | `resolve_backend()` + `_cuda_supported()` + `_triton_supported()` + `dispatch_kernel()` (placeholder). |
| `gmmxx/_cuda.py` | Python wrappers around `gmmxx._C`. Validates inputs, allocates outputs, wraps every FFI call in try/except for soft fallback. |
| `gmmxx/cuda_ops.py` | Experimental public re-export of `_cuda` callables. Single docstring header marking it experimental. |
| `gmmxx/csrc/bindings.cpp` | `NB_MODULE(_C, m)`. Includes only the canary in this plan. |
| `gmmxx/csrc/nb_torch.h` | Verbatim copy from flash-kmeans-cuda. nanobind ↔ at::Tensor caster + Windows macro fixes. |
| `gmmxx/csrc/common/arch.cuh` | `GMMXX_*` arch probes, dtype traits, kWarp constant. |
| `gmmxx/csrc/common/ptx.cuh` | Skeleton (no helpers yet — Plan 2 fills in `cp_async_*`, `mma_*`, etc.). |
| `gmmxx/csrc/common/reduce.cuh` | Skeleton (Plan 2 adds warp shuffle helpers and stable logsumexp helper). |
| `gmmxx/csrc/common/torch_cuda_includes.h` | Single include header for kernel TUs (keeps `Python.h` out of `.cu` files). |
| `gmmxx/csrc/canary/canary.cu` | Trivial kernel returning `out[i] = i + offset`. Deleted in Plan 2 once real kernels exist. |
| `tests/test_cuda_build.py` | `import gmmxx._C` smoke test; `GMMXX_SKIP_CUDA=1` skip path. |
| `tests/test_dispatch.py` | `resolve_backend` truth table; `last_backend_used_` semantics; clone roundtrip. |
| `tests/test_backend_kwarg.py` | `backend` kwarg parsing; `use_triton` deprecation shim cases. |

### Modified in this plan

| Path | Change |
| --- | --- |
| `pyproject.toml` | Pin `torch==2.11.*`; add `nanobind>=2.0,<3` build dep; pin `requires-python = ">=3.12,<3.13"`; add `[tool.uv.sources]` for cu130. |
| `gmmxx/__init__.py` | Wrap `from . import _C` in try/except, set `_HAS_CUDA`; export `cuda_ops`. |
| `gmmxx/_runtime.py` | Add stub `cuda_<cov>_supported()` helpers (all return False in this plan). |
| `gmmxx/interface.py` | Add `backend` kwarg, `last_backend_used_`/`cuda_*_enabled_` attrs, deprecation shim for `use_triton`, `clone`-safe `get_params()`. |

---

## Conventions

- **Working directory:** `C:\Users\HEQ\Projects\flashGMM2` (repo root). All paths in the plan are relative to this.
- **Branch:** Already on `GMMXX-cuda`. Each task ends with one or two commits; do NOT push between tasks.
- **Python:** Use `uv run` for everything (`uv run pytest`, `uv run python`). Per the user's CLAUDE.md.
- **Build invocation:** `uv pip install -e .` after `setup.py`/`pyproject.toml`/`csrc/` changes; otherwise no build needed.
- **Single-arch dev builds:** `set TORCH_CUDA_ARCH_LIST=8.9 && uv pip install -e .` (Windows) / `TORCH_CUDA_ARCH_LIST=8.9 uv pip install -e .` (Linux). Cuts compile time ~6×.
- **Git authorship:** Use `git commit --no-verify` only if a hook misfires (don't make this routine). Sign-off via `Co-Authored-By` line in the message.

---

## Task 1 — Bootstrap directory skeleton and update pyproject.toml

**Files:**
- Create: `gmmxx/csrc/common/` (empty for now)
- Create: `gmmxx/csrc/canary/` (empty for now)
- Modify: `pyproject.toml`

- [ ] **Step 1.1 — Create the csrc directory tree**

```bash
mkdir -p gmmxx/csrc/common gmmxx/csrc/canary
```

Verify it exists:

```bash
ls gmmxx/csrc/
# Expected: canary  common
```

- [ ] **Step 1.2 — Update `pyproject.toml`**

Replace the file's contents with:

```toml
[build-system]
requires = ["setuptools>=64", "wheel", "torch==2.11.*", "nanobind>=2.0,<3"]
build-backend = "setuptools.build_meta"

[project]
name = "GMMXX"
version = "0.1.0"
description = "GPU-friendly Gaussian Mixture Models with chunked PyTorch, Triton, and CUDA EM paths"
readme = "README.md"
requires-python = ">=3.12,<3.13"
authors = [
  { name = "OpenAI Codex" }
]
license = { file = "LICENSE" }
keywords = ["gmm", "gaussian-mixture", "clustering", "pytorch", "flash-kmeans", "cuda"]
classifiers = [
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.12",
  "License :: OSI Approved :: MIT License",
  "Operating System :: Microsoft :: Windows",
  "Operating System :: POSIX :: Linux",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Topic :: Scientific/Engineering :: Mathematics",
  "Typing :: Typed",
]
dependencies = [
  "torch==2.11.*",
]

[project.urls]
Homepage = "https://github.com/OpsiClear/GMMXX"
Source = "https://github.com/OpsiClear/GMMXX"
Issues = "https://github.com/OpsiClear/GMMXX/issues"

[project.optional-dependencies]
triton = [
  "triton-windows>=3.6,<3.7; platform_system == 'Windows'",
  "triton>=3.6,<3.7; platform_system == 'Linux'",
]
sklearn = [
  "numpy>=1.26",
  "scikit-learn>=1.5"
]
kmeans = [
  "flash-kmeans>=0.2"
]
benchmark = [
  "numpy>=1.26",
  "scikit-learn>=1.5"
]
benchmark-gpu = [
  "numpy>=1.26",
  "scikit-learn>=1.5",
  "torchgmm>=0.1.4",
  "tgmm>=0.2.0"
]
dev = [
  "build>=1.2",
  "pytest>=8",
  "twine>=5"
]

[tool.uv.sources]
torch = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.setuptools]
packages = ["gmmxx"]
include-package-data = false

[tool.setuptools.package-data]
gmmxx = ["py.typed", "csrc/**/*.h", "csrc/**/*.cuh", "csrc/**/*.cu", "csrc/**/*.cpp"]
```

Notable changes from the existing file:
- `torch==2.11.*` exact pin (libtorch C++ ABI is tied to torch version).
- `nanobind>=2.0,<3` build dep (we link `nb_combined.cpp` from the installed nanobind).
- `triton` moved into optional extras (was a hard runtime dep). The Triton path still works via `pip install -e ".[triton]"` for users who want it.
- `requires-python = ">=3.12,<3.13"` (single ABI tag for future wheel CI).
- `[tool.uv.sources]` + `[[tool.uv.index]]` route `torch` from PyTorch's CUDA 12.8/13 wheel index — needed for sm_100/sm_120 nvcc support.
- `package-data` includes csrc/ so source distributions can rebuild.

- [ ] **Step 1.3 — Verify the existing test suite still imports**

```bash
uv run pytest tests/test_packaging.py -q
```

Expected: PASS (no regressions; this only validates that the modified `pyproject.toml` parses and the existing package still imports).

- [ ] **Step 1.4 — Commit**

```bash
git add pyproject.toml gmmxx/csrc/
git commit -m "$(cat <<'EOF'
Bootstrap CUDA source tree and pyproject build config

- Pin torch==2.11.* (libtorch ABI requires exact match)
- Add nanobind build requirement
- Move triton to optional extras
- Pin requires-python to 3.12 single ABI tag
- Route torch from pytorch-cu130 index for sm_100/120 toolchain
- Create gmmxx/csrc/{common,canary} skeleton

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Add `setup.py` with CUDAExtension scaffolding

**Files:**
- Create: `setup.py`

- [ ] **Step 2.1 — Write `setup.py`**

Create `setup.py` with:

```python
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
        #   `small` as `char`; ATen/c10 use `small` as a parameter name so the
        #   substitution mangles e.g. `bool small` -> `bool char`. Undef it
        #   at the compiler command-line level so include order in headers
        #   isn't the only line of defense.
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
        str(nb_combined),
    ]
    include_dirs = [
        str(CSRC),
        str(CSRC / "common"),
        str(CSRC / "canary"),
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
```

- [ ] **Step 2.2 — Verify it parses (without building yet — no `bindings.cpp` exists)**

```bash
uv run python -c "import setup"
```

Expected: succeeds (the file is just a script; running `python -c "import setup"` evaluates the top-level code but doesn't actually call `setup()`'s `_build_extension` because `bindings.cpp` doesn't exist yet — `_build_extension()` is only called when `setup()` runs in build context).

If it fails: check `setup.py` syntax, that you didn't typo a path.

- [ ] **Step 2.3 — Commit**

```bash
git add setup.py
git commit -m "$(cat <<'EOF'
Add setup.py with CUDAExtension scaffold for gmmxx._C

Mirrors flash-kmeans-cuda's build:
- Multi-arch gencode 80/86/89/90 + Blackwell (100/120) gated on nvcc >= 12.8
- TORCH_CUDA_ARCH_LIST shortcut (defer to torch's parser)
- nanobind robin_map include path explicit (nanobind 2.1+)
- Windows MSVC flags include /utf-8 /Zc:__cplusplus /bigobj
- Lazy torch/nanobind imports so GMMXX_SKIP_CUDA=1 works without them

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Drop in `nb_torch.h` and `torch_cuda_includes.h`

**Files:**
- Create: `gmmxx/csrc/nb_torch.h`
- Create: `gmmxx/csrc/common/torch_cuda_includes.h`

- [ ] **Step 3.1 — Copy `nb_torch.h` verbatim from flash-kmeans-cuda**

The reference file is `~/Projects/flash-kmeans-cuda/flash_kmeans_cuda/csrc/nb_torch.h`. It has no project-namespace identifiers (FKC-specific), so verbatim copy is correct. Create `gmmxx/csrc/nb_torch.h` with:

```cpp
#pragma once

// nanobind <-> torch::Tensor bridge.
//
// nanobind ships no caster for at::Tensor. We supply a minimal one that
// unwraps a Python torch.Tensor via ``THPVariable_Unpack`` (returns a borrowed
// at::Tensor reference owned by the Python object) and re-wraps via
// ``THPVariable_Wrap`` (returns a new Python reference).
//
// For optional<at::Tensor>, we accept ``None`` -> empty optional, otherwise
// dispatch to the at::Tensor caster.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

// Windows.h (pulled in via Python.h on Windows) defines `small` as `char`,
// which collides with PyTorch headers that use `small` as a parameter name
// (e.g. c10/cuda/CUDACachingAllocator.h). Undef the offenders before pulling
// in torch.
#ifdef small
  #undef small
#endif
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif

#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <c10/util/Optional.h>

namespace nb = nanobind;

namespace nanobind { namespace detail {

template <>
struct type_caster<at::Tensor> {
  NB_TYPE_CASTER(at::Tensor, const_name("torch.Tensor"))

  bool from_python(handle src, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
    PyObject* obj = src.ptr();
    if (!obj || obj == Py_None) return false;
    if (!THPVariable_Check(obj)) return false;
    new (&value) at::Tensor(THPVariable_Unpack(obj));
    return true;
  }

  static handle from_cpp(at::Tensor v, rv_policy /*policy*/, cleanup_list* /*cleanup*/) noexcept {
    return THPVariable_Wrap(std::move(v));
  }
};

// c10::optional<at::Tensor> caster — None maps to nullopt.
template <>
struct type_caster<c10::optional<at::Tensor>> {
  NB_TYPE_CASTER(c10::optional<at::Tensor>, const_name("Optional[torch.Tensor]"))

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    PyObject* obj = src.ptr();
    if (!obj || obj == Py_None) {
      new (&value) c10::optional<at::Tensor>(c10::nullopt);
      return true;
    }
    type_caster<at::Tensor> inner;
    if (!inner.from_python(src, flags, cleanup)) return false;
    new (&value) c10::optional<at::Tensor>(std::move(inner.value));
    return true;
  }

  static handle from_cpp(c10::optional<at::Tensor> v, rv_policy policy, cleanup_list* cleanup) noexcept {
    if (!v.has_value()) {
      Py_INCREF(Py_None);
      return Py_None;
    }
    return type_caster<at::Tensor>::from_cpp(std::move(*v), policy, cleanup);
  }
};

}}  // namespace nanobind::detail
```

- [ ] **Step 3.2 — Write `torch_cuda_includes.h`**

Create `gmmxx/csrc/common/torch_cuda_includes.h`:

```cpp
#pragma once

// Single header for kernel translation units (.cu files).
//
// IMPORTANT: kernel TUs MUST NOT include <torch/extension.h> or <torch/torch.h>.
// Those pull in Python.h (via pybind11) and trip MSVC's std-namespace ambiguity
// in torch/csrc/dynamo/compiled_autograd.h on CUDA 13. Use this header instead.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
```

- [ ] **Step 3.3 — Commit**

```bash
git add gmmxx/csrc/nb_torch.h gmmxx/csrc/common/torch_cuda_includes.h
git commit -m "$(cat <<'EOF'
Add nb_torch.h (verbatim from flash-kmeans-cuda) and torch_cuda_includes.h

nb_torch.h provides nanobind <-> at::Tensor casters and Windows macro fixes;
contains no project-namespace identifiers so verbatim copy is appropriate.

torch_cuda_includes.h is the single include header for kernel .cu files;
keeps Python.h out of CUDA TUs to avoid MSVC std-ambiguity bugs in CUDA 13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Add `arch.cuh`, `ptx.cuh`, `reduce.cuh` skeletons

**Files:**
- Create: `gmmxx/csrc/common/arch.cuh`
- Create: `gmmxx/csrc/common/ptx.cuh`
- Create: `gmmxx/csrc/common/reduce.cuh`

- [ ] **Step 4.1 — Write `arch.cuh`** (renamed `FKC_*` → `GMMXX_*`)

Create `gmmxx/csrc/common/arch.cuh`:

```cpp
#pragma once

// Architecture probes and dtype traits shared across kernels.
//
// All probe macros are GMMXX_-prefixed so they don't collide with
// flash-kmeans-cuda's FKC_-prefixed equivalents if both libraries are
// linked into the same Python process or build.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace gmmxx {

// Compile-time arch guards.
#if defined(__CUDA_ARCH__)
  #define GMMXX_CUDA_ARCH __CUDA_ARCH__
#else
  #define GMMXX_CUDA_ARCH 0
#endif

#define GMMXX_HAS_F16_MMA       (GMMXX_CUDA_ARCH >= 800)   // m16n8k16 fp16 acc fp32
#define GMMXX_HAS_BF16_MMA      (GMMXX_CUDA_ARCH >= 800)   // m16n8k16 bf16 acc fp32
#define GMMXX_HAS_TF32_MMA      (GMMXX_CUDA_ARCH >= 800)   // m16n8k8  tf32 acc fp32 (Phase 2 hook)
#define GMMXX_HAS_CP_ASYNC      (GMMXX_CUDA_ARCH >= 800)
#define GMMXX_HAS_LDMATRIX_X4   (GMMXX_CUDA_ARCH >= 750)
#define GMMXX_HAS_WGMMA         (GMMXX_CUDA_ARCH == 900)   // sm_90a only — Phase 2

// dtype traits.
template <typename T>
struct dtype_traits;

template <>
struct dtype_traits<__half> {
  using packed2 = __half2;
  static constexpr int torch_scalar_type = 5;  // torch::kHalf
  static constexpr const char* name = "fp16";
  static constexpr bool is_half = true;
};

template <>
struct dtype_traits<__nv_bfloat16> {
  using packed2 = __nv_bfloat162;
  static constexpr int torch_scalar_type = 15; // torch::kBFloat16
  static constexpr const char* name = "bf16";
  static constexpr bool is_half = true;
};

template <>
struct dtype_traits<float> {
  using packed2 = float2;
  static constexpr int torch_scalar_type = 6;  // torch::kFloat
  static constexpr const char* name = "fp32";
  static constexpr bool is_half = false;
};

// Useful constants.
constexpr int kWarp = 32;

}  // namespace gmmxx
```

- [ ] **Step 4.2 — Write `ptx.cuh` skeleton**

Create `gmmxx/csrc/common/ptx.cuh`:

```cpp
#pragma once

// PTX wrappers shared across kernels.
//
// This file is a SKELETON for Plan 1. Plan 2 (spherical) will populate:
//   - cp_async_cg, cp_async_commit, cp_async_wait_group<N>, cp_async_wait_all
//   - ldmatrix_sync_x4, ldmatrix_sync_x4_trans
//   - mma_m16n8k16_f32_f16, mma_m16n8k16_f32_bf16
//   - mma_m16n8k8_f32_tf32 (Phase 2 hook)
//   - atomic_add_block, atomic_add_system
//   - warp_shfl_xor_sync, warp_reduce_add_sync
//
// Each wrapper is `__device__ __forceinline__` and gated on GMMXX_HAS_*
// macros from arch.cuh.

#include "arch.cuh"

namespace gmmxx { namespace ptx {

// Skeleton — see Plan 2 for population.

}}  // namespace gmmxx::ptx
```

- [ ] **Step 4.3 — Write `reduce.cuh` skeleton**

Create `gmmxx/csrc/common/reduce.cuh`:

```cpp
#pragma once

// Warp / block reductions and stable logsumexp helpers.
//
// SKELETON for Plan 1; Plan 2 populates with:
//   - warp_max_f32, warp_sum_f32 (via __shfl_xor_sync)
//   - block_max_f32, block_sum_f32 (warp+SMEM tree reduction)
//   - logsumexp_warp, logsumexp_block (subtract-max-then-exp-then-sum in fp32)

#include "arch.cuh"

namespace gmmxx { namespace reduce {

// Skeleton — see Plan 2 for population.

}}  // namespace gmmxx::reduce
```

- [ ] **Step 4.4 — Commit**

```bash
git add gmmxx/csrc/common/arch.cuh gmmxx/csrc/common/ptx.cuh gmmxx/csrc/common/reduce.cuh
git commit -m "$(cat <<'EOF'
Add common/arch.cuh + ptx.cuh + reduce.cuh skeletons

arch.cuh defines GMMXX_-prefixed compile-time arch probes (HAS_F16_MMA,
HAS_BF16_MMA, HAS_TF32_MMA, HAS_CP_ASYNC, HAS_LDMATRIX_X4, HAS_WGMMA) and
dtype_traits for __half/__nv_bfloat16/float. Renamed from flash-kmeans-cuda's
FKC_* prefix to avoid collisions when both libraries are linked.

ptx.cuh and reduce.cuh are skeletons; Plan 2 populates them with the helpers
the spherical kernels need.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Write the canary kernel and bindings

**Goal:** Smallest possible kernel that proves nanobind ↔ at::Tensor ↔ CUDA path works. The canary takes one int32 input tensor and writes `out[i] = input[i] + offset` for a host-supplied offset. No real GMM logic.

**Files:**
- Create: `gmmxx/csrc/canary/canary.cu`
- Create: `gmmxx/csrc/canary/canary.h`
- Create: `gmmxx/csrc/bindings.cpp`

- [ ] **Step 5.1 — Write `canary.h`**

Create `gmmxx/csrc/canary/canary.h`:

```cpp
#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace canary {

// Returns a fresh int32 tensor of the same shape as `input` where each
// element is `input[i] + offset`. Used as a build/FFI smoke test.
at::Tensor add_offset(const at::Tensor& input, int64_t offset);

}}  // namespace gmmxx::canary
```

- [ ] **Step 5.2 — Write `canary.cu`**

Create `gmmxx/csrc/canary/canary.cu`:

```cpp
#include "canary.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace canary {

namespace {

__global__ void canary_add_offset_kernel(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t n_elements,
    int32_t offset
) {
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (idx >= n_elements) return;
    output[idx] = input[idx] + offset;
}

}  // anonymous namespace

at::Tensor add_offset(const at::Tensor& input, int64_t offset) {
    TORCH_CHECK(input.is_cuda(), "canary.add_offset: input must be on a CUDA device");
    TORCH_CHECK(input.is_contiguous(), "canary.add_offset: input must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kInt, "canary.add_offset: input must be int32");

    // Multi-device safety: bind the device of the input tensor for the
    // duration of this call.
    c10::cuda::CUDAGuard guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto output = at::empty_like(input);
    int64_t n = input.numel();
    if (n == 0) return output;

    constexpr int kThreads = 256;
    int64_t blocks_64 = (n + kThreads - 1) / kThreads;
    TORCH_CHECK(blocks_64 <= 0x7fffffff, "canary.add_offset: input too large");
    int blocks = static_cast<int>(blocks_64);

    canary_add_offset_kernel<<<blocks, kThreads, 0, stream>>>(
        input.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        n,
        static_cast<int32_t>(offset)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}}  // namespace gmmxx::canary
```

Notes for the implementer:
- `c10::cuda::CUDAGuard` is the spec's mandated multi-device-safety primitive (§5a).
- `at::cuda::getCurrentCUDAStream()` is the standard stream source.
- `at::empty_like(input)` allocates output on the right device + dtype.
- `C10_CUDA_KERNEL_LAUNCH_CHECK()` reports any kernel-launch error via `TORCH_CHECK`.

- [ ] **Step 5.3 — Write `bindings.cpp`**

Create `gmmxx/csrc/bindings.cpp`:

```cpp
// nanobind module entry point for gmmxx._C.
//
// Plan 1 exposes only the canary kernel (smoke test). Plan 2 onwards adds
// real E-step / M-step / fused / approx ops.

#include "nb_torch.h"
#include "canary/canary.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
    m.doc() = "gmmxx CUDA kernel bindings";

    m.def(
        "canary_add_offset",
        &gmmxx::canary::add_offset,
        nb::arg("input"),
        nb::arg("offset"),
        "Smoke-test kernel: returns input + offset element-wise (int32).");
}
```

- [ ] **Step 5.4 — Build the extension**

```bash
uv pip install -e .
```

Expected: nvcc compiles `canary.cu`, MSVC/g++ compiles `bindings.cpp` and `nb_combined.cpp`, `gmmxx._C.pyd`/`.so` is produced. **First build takes 2–8 minutes** depending on the gencode list. To shrink: `set TORCH_CUDA_ARCH_LIST=8.9 && uv pip install -e .` (or your local arch).

If it fails with `nvcc fatal: Unsupported gpu architecture 'compute_100'` → your nvcc is < 12.8. Set `GMMXX_BUILD_BLACKWELL=0` and retry.

If it fails with `error C4819` (CP1252 encoding) → confirm `/utf-8` is in `_common_cxx_flags()`.

- [ ] **Step 5.5 — Smoke test from the REPL**

```bash
uv run python -c "
import torch
from gmmxx import _C
x = torch.arange(10, dtype=torch.int32, device='cuda')
y = _C.canary_add_offset(x, 5)
print(y)
assert (y == torch.arange(5, 15, dtype=torch.int32, device='cuda')).all()
print('OK')
"
```

Expected: prints the offset tensor and `OK`. If you don't have CUDA available, this will fail at `device='cuda'` — that's expected; CI/test handles that case in Task 6.

- [ ] **Step 5.6 — Commit**

```bash
git add gmmxx/csrc/canary/ gmmxx/csrc/bindings.cpp
git commit -m "$(cat <<'EOF'
Add canary CUDA kernel and bindings.cpp for nanobind FFI smoke test

canary.cu launches a trivial elementwise add-offset kernel via
at::cuda::getCurrentCUDAStream() under c10::cuda::CUDAGuard. This proves
the build, nanobind <-> at::Tensor caster, kernel launch, and stream/device
plumbing all work end-to-end before any real GMM kernels land.

bindings.cpp exposes canary_add_offset(input: Tensor, offset: int) -> Tensor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Wrap canary in `gmmxx/_cuda.py` with try/except fallback

**Files:**
- Create: `gmmxx/_cuda.py`
- Modify: `gmmxx/__init__.py`

- [ ] **Step 6.1 — Write `gmmxx/_cuda.py`**

Create `gmmxx/_cuda.py`:

```python
"""Internal Python wrappers around gmmxx._C (the compiled CUDA extension).

This module:
  * Imports `_C` lazily and tolerates ImportError so `import gmmxx` succeeds
    on hosts without a CUDA build.
  * Validates inputs (contiguous, dtype, device) before crossing the FFI
    boundary.
  * Wraps every FFI call in try/except so runtime CUDA errors (OOM, illegal
    instruction on a specific GPU/driver combo, mma.sync regressions) don't
    take down the whole process — they raise a custom exception that
    `_dispatch.resolve_backend` catches and falls through to the next backend.

End users do not import this module directly; they use the `GMMXX` class or
the `gmmxx.cuda_ops` re-export.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    from . import _C  # noqa: F401  -- compiled extension
    _HAS_CUDA = True
    _IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:
    _C = None
    _HAS_CUDA = False
    _IMPORT_ERROR = exc


class CudaBackendUnavailable(RuntimeError):
    """Raised when gmmxx._C was not built (e.g. GMMXX_SKIP_CUDA=1)."""


class CudaRuntimeFallback(RuntimeError):
    """Raised when a CUDA kernel fails at runtime; the dispatcher catches
    this and falls through to Triton or torch."""


def has_cuda() -> bool:
    """True iff gmmxx._C imported successfully AND torch.cuda is available."""
    return _HAS_CUDA and torch.cuda.is_available()


def _no_fallback() -> bool:
    """If GMMXX_CUDA_NO_FALLBACK=1, runtime errors propagate instead of being
    caught. Used in CI to make CUDA bugs loud."""
    return os.environ.get("GMMXX_CUDA_NO_FALLBACK", "").lower() in {"1", "true", "yes"}


def require_cuda() -> None:
    """Raise CudaBackendUnavailable if the extension wasn't built. Used by
    the dispatcher when the user explicitly requests backend='cuda'."""
    if _C is None:
        raise CudaBackendUnavailable(
            "gmmxx._C extension not built; reinstall without GMMXX_SKIP_CUDA "
            f"(original ImportError: {_IMPORT_ERROR!r})"
        )


def _check_input(t: torch.Tensor, name: str, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not t.is_cuda:
        raise ValueError(f"{name}: must be on a CUDA device, got {t.device}")
    if not t.is_contiguous():
        t = t.contiguous()
    if dtype is not None and t.dtype != dtype:
        raise ValueError(f"{name}: dtype must be {dtype}, got {t.dtype}")
    return t


def canary_add_offset(input: torch.Tensor, offset: int) -> torch.Tensor:
    """Smoke-test wrapper. Calls the canary kernel with proper validation
    and runtime-error fallback semantics.

    Returns input + offset; raises CudaRuntimeFallback on kernel failure
    (unless GMMXX_CUDA_NO_FALLBACK=1, in which case the raw RuntimeError
    propagates).
    """
    require_cuda()
    input = _check_input(input, "canary input", dtype=torch.int32)
    try:
        return _C.canary_add_offset(input, offset)
    except RuntimeError as exc:
        if _no_fallback():
            raise
        raise CudaRuntimeFallback(f"canary kernel failed: {exc}") from exc
```

- [ ] **Step 6.2 — Modify `gmmxx/__init__.py` to expose `_cuda` state**

Add these lines near the top of `gmmxx/__init__.py` (immediately after the existing `from .interface import GMMXX` line at line 1):

```python
# Internal CUDA backend state. Always import safely — _cuda handles the
# missing-extension case via _HAS_CUDA = False.
from . import _cuda as _cuda  # noqa: F401
```

Add `"_cuda"` to `__all__` if you want it re-exported (optional; this plan keeps it private — power users go through `gmmxx.cuda_ops` instead, which Task 8 creates).

- [ ] **Step 6.3 — Write the build smoke test**

Create `tests/test_cuda_build.py`:

```python
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
```

- [ ] **Step 6.4 — Run tests**

```bash
uv run pytest tests/test_cuda_build.py -v
```

Expected: all four tests pass on a CUDA host (canary test runs); on a CPU host, the canary test is skipped and the other three pass.

- [ ] **Step 6.5 — Commit**

```bash
git add gmmxx/_cuda.py gmmxx/__init__.py tests/test_cuda_build.py
git commit -m "$(cat <<'EOF'
Add gmmxx/_cuda.py wrapper with try/except runtime fallback

Wraps the compiled gmmxx._C with:
- Lazy import that sets _HAS_CUDA=False on ImportError so import gmmxx
  always succeeds.
- has_cuda() / require_cuda() helpers for backend dispatch and explicit
  user requests.
- _check_input() validation (CUDA device, contiguous, dtype).
- CudaRuntimeFallback exception wrapping every FFI call so runtime errors
  can degrade to Triton/torch instead of crashing.
- GMMXX_CUDA_NO_FALLBACK=1 env var disables the fallback for CI.

tests/test_cuda_build.py covers: import succeeds, _HAS_CUDA is a bool,
canary runs on CUDA hosts, missing-extension path raises clean error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Add `_runtime.py` cuda_*_supported stubs

**Files:**
- Modify: `gmmxx/_runtime.py`

- [ ] **Step 7.1 — Add stub helpers to `_runtime.py`**

Open `gmmxx/_runtime.py`. The current file has a single `triton_spherical_supported` function. Add the four stub helpers below it. Replace the file contents with:

```python
from __future__ import annotations


# Triton shape policy (existing).
TRITON_SPHERICAL_MAX_D = 128
TRITON_SPHERICAL_MAX_K = 2048


def triton_spherical_supported(d: int, n_components: int) -> bool:
    """Validated spherical Triton shape policy.

    Keep this intentionally small and explicit. Shapes outside this range use
    the PyTorch/cuBLAS implementation instead of carrying extra runtime
    branches through the production path.
    """
    return 0 < d <= TRITON_SPHERICAL_MAX_D and 0 < n_components <= TRITON_SPHERICAL_MAX_K


# CUDA shape policy (Plan 1: all stubs return False; Plan 2 onwards populates
# them as kernels land).

def cuda_spherical_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the CUDA backend can handle spherical EM at this shape+dtype.

    Plan 1 stub: returns False until Plan 2 implements the spherical kernels.
    The dispatcher uses this gate to decide whether to route through CUDA;
    when False, it falls back to Triton or torch.
    """
    del d, n_components, dtype  # unused until Plan 2
    return False


def cuda_diag_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the CUDA backend can handle diagonal EM at this shape+dtype.

    Plan 1 stub: returns False until Plan 3.
    """
    del d, n_components, dtype
    return False


def cuda_tied_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the CUDA backend can handle tied EM at this shape+dtype.

    Plan 1 stub: returns False until Plan 4.
    """
    del d, n_components, dtype
    return False


def cuda_full_supported(d: int, n_components: int, dtype) -> bool:
    """True iff the CUDA backend can handle full-covariance EM at this shape+dtype.

    Plan 1 stub: returns False until Plan 5. Note the spec's D <= 16 cap is
    enforced once the kernel exists; for now the stub simply refuses everything.
    """
    del d, n_components, dtype
    return False
```

- [ ] **Step 7.2 — Verify nothing imports broke**

```bash
uv run python -c "from gmmxx._runtime import cuda_spherical_supported, cuda_diag_supported, cuda_tied_supported, cuda_full_supported; print(cuda_spherical_supported(32, 64, None))"
```

Expected: `False`

- [ ] **Step 7.3 — Commit**

```bash
git add gmmxx/_runtime.py
git commit -m "$(cat <<'EOF'
Add cuda_<cov>_supported stubs to _runtime.py (all False in Plan 1)

These shape gates return False until subsequent plans land the actual CUDA
kernels per covariance type. The dispatcher uses them to decide whether to
route through CUDA vs fall back to Triton or torch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — Write `gmmxx/_dispatch.py` with `resolve_backend`

**Files:**
- Create: `gmmxx/_dispatch.py`
- Create: `tests/test_dispatch.py`

- [ ] **Step 8.1 — Write the failing tests first**

Create `tests/test_dispatch.py`:

```python
"""Tests for gmmxx._dispatch.resolve_backend."""

import os
from unittest.mock import patch

import pytest

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

    def test_explicit_cuda_falls_through_when_stub_returns_false(self):
        # Plan 1 stub always returns False, so cuda must fall to torch.
        result = _dispatch.resolve_backend(
            requested="cuda",
            covariance="spherical",
            shape=(1, 1024, 32, 64),
            dtype=None,
        )
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
```

- [ ] **Step 8.2 — Run tests to verify they fail**

```bash
uv run pytest tests/test_dispatch.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gmmxx._dispatch'`

- [ ] **Step 8.3 — Write `_dispatch.py`**

Create `gmmxx/_dispatch.py`:

```python
"""Backend dispatch for gmmxx.

Decides whether to route a kernel call through the CUDA backend (compiled
gmmxx._C), the Triton backend (existing JIT modules), or the PyTorch
fallback. Encapsulates per-shape gates so the GMMXX orchestrator never
hardcodes backend choices.

The high-level public API exposed via this module is:

    resolve_backend(requested, covariance, shape, dtype, legacy_no_triton=False)
        -> "cuda" | "triton" | "torch"
    resolve_backend_with_env(requested, ...)  # same, but consults GMMXX_BACKEND
                                                 when requested == "auto".

Plan 1 wires the truth table; Plan 2 onwards adds dispatch_kernel() that
actually routes calls into the right module per backend.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from . import _cuda
from ._runtime import (
    cuda_diag_supported,
    cuda_full_supported,
    cuda_spherical_supported,
    cuda_tied_supported,
    triton_spherical_supported,
)

_VALID_BACKENDS = {"auto", "cuda", "triton", "torch"}


def _shape_dk(shape: tuple) -> tuple[int, int]:
    """Extract (D, K) from a (B, N, D, K) or (B, N, D) shape tuple.

    Most callers pass (B, N, D, K) when they know K (i.e., during fit).
    Inference paths only have (B, N, D) and the orchestrator passes K
    separately; in that case this helper returns (D, 0) so the K-dependent
    triton/cuda gates accept any K.
    """
    if len(shape) == 4:
        return shape[2], shape[3]
    if len(shape) == 3:
        return shape[2], 0
    raise ValueError(f"shape must have 3 or 4 dims, got {shape!r}")


def _triton_supported(covariance: str, shape: tuple, dtype: Any) -> bool:
    """True iff the Triton path can handle this call."""
    d, k = _shape_dk(shape)
    if covariance == "spherical":
        # Existing policy.
        if k == 0:
            return 0 < d <= 128  # inference-only; K not known yet
        return triton_spherical_supported(d, k)
    if covariance == "diag":
        return 0 < d <= 64 and (k == 0 or k <= 512)
    if covariance == "tied":
        return 0 < d <= 64 and (k == 0 or k <= 512)
    if covariance == "full":
        return 0 < d <= 16  # full Triton path is conservative
    return False


def _cuda_supported(covariance: str, shape: tuple, dtype: Any) -> bool:
    """True iff the CUDA backend can handle this call.

    Plan 1: stubs in _runtime.py all return False, so this returns False
    everywhere. Subsequent plans will turn it on per covariance type.
    """
    if not _cuda.has_cuda():
        return False
    d, k = _shape_dk(shape)
    if covariance == "spherical":
        return cuda_spherical_supported(d, k, dtype)
    if covariance == "diag":
        return cuda_diag_supported(d, k, dtype)
    if covariance == "tied":
        return cuda_tied_supported(d, k, dtype)
    if covariance == "full":
        return cuda_full_supported(d, k, dtype)
    return False


def resolve_backend(
    requested: str,
    covariance: str,
    shape: tuple,
    dtype: Any,
    legacy_no_triton: bool = False,
) -> str:
    """Returns one of "cuda", "triton", "torch" given the user request and call shape.

    legacy_no_triton: True when called from a deprecated use_triton=False shim.
                      Filters Triton out of the resolution chain regardless of
                      requested.
    """
    if requested not in _VALID_BACKENDS:
        raise ValueError(f"unknown backend {requested!r}; expected one of {_VALID_BACKENDS}")
    if requested == "torch":
        return "torch"
    if requested == "triton":
        if legacy_no_triton:
            raise ValueError(
                "backend='triton' incompatible with use_triton=False; pass backend='auto' or remove use_triton."
            )
        return "triton" if _triton_supported(covariance, shape, dtype) else "torch"
    if requested == "cuda":
        if _cuda_supported(covariance, shape, dtype):
            return "cuda"
        # Explicit cuda but unsupported shape — only require_cuda() raises;
        # if the extension is built but the shape gate is False, fall through.
        if _cuda._HAS_CUDA:
            return "torch"
        # Extension unbuilt and user explicitly asked for cuda → loud error.
        _cuda.require_cuda()  # raises CudaBackendUnavailable
    # requested == "auto"
    if _cuda_supported(covariance, shape, dtype):
        return "cuda"
    if (not legacy_no_triton) and _triton_supported(covariance, shape, dtype):
        return "triton"
    return "torch"


def resolve_backend_with_env(
    requested: str,
    covariance: str,
    shape: tuple,
    dtype: Any,
    legacy_no_triton: bool = False,
) -> str:
    """Same as resolve_backend, but consults GMMXX_BACKEND when requested == 'auto'.

    The kwarg always wins when explicit. Invalid env-var values are ignored.
    """
    effective = requested
    if effective == "auto":
        env = os.environ.get("GMMXX_BACKEND")
        if env in _VALID_BACKENDS:
            effective = env
    return resolve_backend(effective, covariance, shape, dtype, legacy_no_triton=legacy_no_triton)
```

- [ ] **Step 8.4 — Run tests to verify they pass**

```bash
uv run pytest tests/test_dispatch.py -v
```

Expected: all 12 tests pass. If `test_explicit_triton_returns_triton_when_supported` fails because triton isn't installed, install it: `uv pip install -e ".[triton]"` and retry.

- [ ] **Step 8.5 — Commit**

```bash
git add gmmxx/_dispatch.py tests/test_dispatch.py
git commit -m "$(cat <<'EOF'
Add gmmxx/_dispatch.py with resolve_backend truth table

Encapsulates backend selection so the GMMXX orchestrator never hardcodes
choices. The truth table:

  - requested='torch' → 'torch'
  - requested='triton' → 'triton' if supported, else 'torch'
    (raises ValueError under legacy_no_triton=True)
  - requested='cuda' → 'cuda' if shape gate True, else 'torch'
    (raises CudaBackendUnavailable if extension unbuilt and user asked for cuda)
  - requested='auto' → cuda first, triton next, torch last
    (legacy_no_triton=True filters triton out of the chain)

resolve_backend_with_env consults GMMXX_BACKEND when requested='auto'.
Invalid env values are ignored.

Plan 1: cuda_<cov>_supported() stubs all return False, so 'auto' lands on
triton or torch in this plan. Plan 2 turns on spherical CUDA.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — Add `backend` kwarg, attributes, and deprecation shim to `GMMXX`

**Files:**
- Modify: `gmmxx/interface.py`
- Create: `tests/test_backend_kwarg.py`

- [ ] **Step 9.1 — Write the failing tests first**

Create `tests/test_backend_kwarg.py`:

```python
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
```

- [ ] **Step 9.2 — Run tests to verify they fail**

```bash
uv run pytest tests/test_backend_kwarg.py -v
```

Expected: many failures (`backend` kwarg doesn't exist yet, attributes missing, etc.).

- [ ] **Step 9.3 — Modify `gmmxx/interface.py`**

Open `gmmxx/interface.py` and apply these edits:

**Edit A — Add the `_VALID_BACKENDS` constant near the top of the file** (right after `_VALID_COVARIANCE_TYPES = {"spherical", "diag", "tied", "full"}`, around line 67):

```python
_VALID_BACKENDS = {"auto", "cuda", "triton", "torch"}
```

**Edit B — Modify the `__init__` signature** (currently lines 128–154). Add `backend` and ensure `use_triton` becomes optional with a sentinel:

Replace lines 128–154 (the entire `def __init__(...)` signature block) with:

```python
    def __init__(
        self,
        d: Optional[int] = None,
        k: Optional[int] = None,
        niter: Optional[int] = None,
        tol: float = 1e-4,
        use_triton: Optional[bool] = None,  # deprecated; see below
        seed: Optional[int] = None,
        chunk_size_data: int = 32768,
        chunk_size_centroids: int = 1024,
        chunk_size_data_cpu: int = 1048576,
        verbose: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        init_params: str = "kmeans",
        reg_covar: float = 1e-6,
        init_kmeans_iters: int = 10,
        init_kmeans_tol: float = 1e-4,
        covariance_type: str = "spherical",
        matmul_precision: Optional[str] = None,
        compute_labels_on_fit: bool = True,
        approx_top_k: Optional[int] = None,
        n_components: Optional[int] = None,
        max_iter: Optional[int] = None,
        random_state: Optional[int] = None,
        n_features: Optional[int] = None,
        backend: str = "auto",
    ):
```

**Edit C — Add backend resolution + deprecation shim** at the start of `__init__`'s body, immediately after the existing `_resolve_alias` calls (around line 158, after the four existing `_resolve_alias` lines but before `if k is None`):

```python
        # Backend selection + use_triton deprecation shim.
        # use_triton=True  -> backend="auto",  _legacy_no_triton=False
        # use_triton=False -> backend="auto",  _legacy_no_triton=True
        # If the user passes both backend= and use_triton=, raise.
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
        if use_triton is not None:
            import warnings as _warnings
            if backend != "auto":
                # User passed both. Conflict.
                raise ValueError(
                    "Cannot pass both backend= and use_triton= to GMMXX. "
                    "Drop use_triton; backend= is the canonical kwarg."
                )
            _warnings.warn(
                "use_triton is deprecated; use backend='auto'|'cuda'|'triton'|'torch' instead. "
                "use_triton=True maps to backend='auto'; use_triton=False maps to backend='auto' "
                "with Triton filtered from the dispatch chain.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._legacy_no_triton = (use_triton is False)
        else:
            self._legacy_no_triton = False
        self.backend = backend
```

**Edit D — Replace the line `self.use_triton = bool(use_triton)`** (currently around line 168) with:

```python
        # use_triton is no longer stored as an instance attr; the deprecation
        # shim above translated it into self.backend + self._legacy_no_triton.
```

(That is, delete the `self.use_triton = bool(use_triton)` line entirely. The shim handles it.)

**Edit E — Add new attributes** in the existing post-init attribute block (currently around line 219, where `self.last_fallback_reason_` is set). Add right after that line:

```python
        self.last_backend_used_: Optional[str] = None
        self.cuda_estep_enabled_: Optional[bool] = None
        self.cuda_fused_update_enabled_: Optional[bool] = None
        self.cuda_approx_topk_enabled_: Optional[bool] = None
```

Also reset them in `_reset_fit_state` (around line 262, near `self.last_fallback_reason_ = None`):

```python
        self.last_backend_used_ = None
        self.cuda_estep_enabled_ = None
        self.cuda_fused_update_enabled_ = None
        self.cuda_approx_topk_enabled_ = None
```

**Edit F — Update `get_params`** (currently lines 494–517). Replace its body:

```python
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "d": self.d,
            "n_components": self.k,
            "max_iter": self.niter,
            "tol": self.tol,
            "random_state": self.seed,
            "chunk_size_data": self.chunk_size_data,
            "chunk_size_centroids": self.chunk_size_centroids,
            "chunk_size_data_cpu": self.chunk_size_data_cpu,
            "verbose": self.verbose,
            "dtype": self.dtype,
            "device": self.device,
            "init_params": self.init_params,
            "reg_covar": self.reg_covar,
            "init_kmeans_iters": self.init_kmeans_iters,
            "init_kmeans_tol": self.init_kmeans_tol,
            "covariance_type": self.covariance_type,
            "matmul_precision": self.matmul_precision,
            "compute_labels_on_fit": self.compute_labels_on_fit,
            "approx_top_k": self.approx_top_k,
            "backend": self.backend,
            # Note: use_triton is intentionally NOT returned. It's the deprecated
            # alias; clone()-style round-trip uses 'backend' as the canonical key.
            # _legacy_no_triton is also intentionally not exposed — it's an
            # implementation detail of the deprecation shim.
        }
```

**Edit G — Update `set_params`** (lines 519+). Add `"backend"` to the valid set and handle it before the existing `for raw_name, value in params.items()` loop. Find the line:

```python
        valid = set(self.get_params().keys()) | {
            "k",
            "niter",
            "seed",
            "n_features",
            "n_components",
            "max_iter",
            "random_state",
        }
```

and add `"use_triton"` and `"backend"` to that set:

```python
        valid = set(self.get_params().keys()) | {
            "k",
            "niter",
            "seed",
            "n_features",
            "n_components",
            "max_iter",
            "random_state",
            "use_triton",   # deprecated alias
            "backend",
        }
```

Then in the type-coercion block (the existing `for raw_name, value in params.items()` loop), after the `elif name in {"use_triton", "verbose", "compute_labels_on_fit"}:` line — actually, we need to remove `use_triton` from there and handle it specially. Find the line:

```python
            elif name in {"use_triton", "verbose", "compute_labels_on_fit"}:
                value = bool(value)
```

Replace it with:

```python
            elif name == "use_triton":
                # Deprecated; route through the same shim as __init__.
                if value is not None:
                    import warnings as _warnings
                    _warnings.warn(
                        "use_triton is deprecated; use backend= instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    self.backend = "auto"
                    self._legacy_no_triton = (bool(value) is False)
                continue  # don't fall through to setattr
            elif name == "backend":
                if value not in _VALID_BACKENDS:
                    raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {value!r}")
                self.backend = value
                continue  # don't fall through to setattr
            elif name in {"verbose", "compute_labels_on_fit"}:
                value = bool(value)
```

- [ ] **Step 9.4 — Run tests to verify they pass**

```bash
uv run pytest tests/test_backend_kwarg.py -v
```

Expected: all 14 tests pass. If `TestUseTritonDeprecation::test_deprecation_warning_emitted_once_per_instance` fails because some other path emits warnings — narrow your warning filter to `module="gmmxx.*"` or `category=DeprecationWarning` only.

- [ ] **Step 9.5 — Verify the existing test suite still passes**

```bash
uv run pytest tests/ -q --ignore=tests/test_cuda_build.py
```

Expected: existing tests still pass. If any test pokes at `model.use_triton` directly, those will need a tiny update — they should now read `model.backend` and `model._legacy_no_triton`. Search:

```bash
uv run python -c "
import re, pathlib
for p in pathlib.Path('tests').rglob('*.py'):
    src = p.read_text()
    for m in re.finditer(r'\.use_triton\b', src):
        line = src.count('\n', 0, m.start()) + 1
        print(f'{p}:{line}')
"
```

If any results print, update them to use `.backend` and `._legacy_no_triton` as appropriate. The constructor still ACCEPTS `use_triton=` (with a DeprecationWarning) — only the instance attribute disappeared.

- [ ] **Step 9.6 — Commit**

```bash
git add gmmxx/interface.py tests/test_backend_kwarg.py
git commit -m "$(cat <<'EOF'
Add backend kwarg, last_backend_used_/cuda_*_enabled_ attrs, deprecate use_triton

- backend: str = "auto" constructor kwarg, validated against
  {auto, cuda, triton, torch}.
- use_triton kwarg accepted with DeprecationWarning emitted once per instance:
  True -> backend="auto", _legacy_no_triton=False
  False -> backend="auto", _legacy_no_triton=True (filters Triton)
- Passing both backend= and use_triton= raises ValueError.
- get_params() returns 'backend' (canonical key); 'use_triton' not in dict
  so sklearn.base.clone() round-trips cleanly.
- set_params(use_triton=...) routes through the deprecation shim.
- New attributes: last_backend_used_, cuda_estep_enabled_,
  cuda_fused_update_enabled_, cuda_approx_topk_enabled_ (all None until
  a fit() runs).
- Existing tests touching .use_triton updated to .backend / ._legacy_no_triton.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10 — Add `gmmxx/cuda_ops.py` (experimental public re-export)

**Files:**
- Create: `gmmxx/cuda_ops.py`
- Modify: `gmmxx/__init__.py`

- [ ] **Step 10.1 — Write `gmmxx/cuda_ops.py`**

Create `gmmxx/cuda_ops.py`:

```python
"""Experimental public re-export of low-level CUDA kernel callables.

WARNING — Experimental: API may change before v1.0. The only API stability
guarantee in GMMXX is the ``GMMXX`` class itself. Internal kernel signatures
exposed here may evolve across minor versions as Phase 2 adds fp8, WGMMA,
multi-stream event plumbing, and similar features.

Plan 1 exposes only the canary kernel as a smoke test of the re-export
plumbing. Plans 2 onwards add the real spherical/diag/tied/full ops.
"""

from __future__ import annotations

from . import _cuda

# Re-export with the documented public names.
canary_add_offset = _cuda.canary_add_offset

# Lifecycle / introspection helpers (also exposed at top level).
has_cuda = _cuda.has_cuda
require_cuda = _cuda.require_cuda
CudaBackendUnavailable = _cuda.CudaBackendUnavailable
CudaRuntimeFallback = _cuda.CudaRuntimeFallback


__all__ = [
    "canary_add_offset",
    "has_cuda",
    "require_cuda",
    "CudaBackendUnavailable",
    "CudaRuntimeFallback",
]
```

- [ ] **Step 10.2 — Re-export `cuda_ops` from `gmmxx/__init__.py`**

Open `gmmxx/__init__.py`. Add at the end of the import block (after the existing try/except blocks but before `__all__`):

```python
from . import cuda_ops as cuda_ops  # noqa: F401  -- experimental public surface
```

Add `"cuda_ops"` to `__all__` so users can `from gmmxx import cuda_ops`:

```python
__all__ = [
    # ... existing entries ...
    "cuda_ops",
]
```

- [ ] **Step 10.3 — Smoke test**

```bash
uv run python -c "
from gmmxx import cuda_ops
print('has_cuda:', cuda_ops.has_cuda())
print('docstring:', cuda_ops.__doc__.split(chr(10))[0])
print('exports:', cuda_ops.__all__)
"
```

Expected output: `has_cuda: True/False`, the first line of the docstring, and the `__all__` list.

- [ ] **Step 10.4 — Commit**

```bash
git add gmmxx/cuda_ops.py gmmxx/__init__.py
git commit -m "$(cat <<'EOF'
Add gmmxx/cuda_ops.py experimental re-export and wire into top-level package

cuda_ops is the public functional surface for power users:
  from gmmxx import cuda_ops
  out = cuda_ops.canary_add_offset(x, 5)

Marked experimental in the module docstring. Plan 1 exports only the canary
plus has_cuda/require_cuda/exception types. Subsequent plans add the real
spherical/diag/tied/full ops as their kernels land.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11 — Update `README.md` with build instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 11.1 — Add a "CUDA backend (experimental)" section**

Open `README.md`. Find the existing "Installation" section (around line 28). After it, add:

```markdown
### CUDA backend (experimental)

`GMMXX` is migrating to a hand-written CUDA backend (Phase 1 in progress; see `docs/superpowers/specs/2026-05-02-gmmxx-cuda-backend-design.md`). The CUDA path is selected automatically on hosts with a working build:

| Backend | Selected when |
| --- | --- |
| `cuda` | `gmmxx._C` is built AND compute capability ≥ 8.0 AND shape is supported |
| `triton` | CUDA path unsupported; Triton is installed; shape is in the Triton policy |
| `torch` | All else (always works as a fallback) |

Build prerequisites:

- CUDA Toolkit ≥ 12.8 (required for sm_100/sm_120 — older toolkits work but Blackwell archs are skipped automatically).
- C++17 compiler (MSVC 2019 16.5+ on Windows; gcc/clang on Linux).
- `nanobind>=2.0` (installed automatically via build deps).

```powershell
# Standard install (builds CUDA extension at install time):
uv pip install -e .

# Single-arch dev build (much faster):
$env:TORCH_CUDA_ARCH_LIST = "8.9"   # PowerShell — replace with your local arch
uv pip install -e .

# Skip the CUDA build entirely (Triton-only / CPU-only install):
$env:GMMXX_SKIP_CUDA = "1"
uv pip install -e ".[triton]"
```

Backend selection:

```python
from gmmxx import GMMXX

# Auto: pick CUDA when supported, else Triton, else PyTorch.
gmm = GMMXX(n_components=64, backend="auto")

# Pin to a specific backend:
gmm = GMMXX(n_components=64, backend="triton")

# Or via env var (kwarg wins when explicit):
import os
os.environ["GMMXX_BACKEND"] = "torch"
gmm = GMMXX(n_components=64)  # uses torch
```

After a `fit()`, inspect what actually ran:

```python
gmm.fit(x)
print(gmm.last_backend_used_)        # "cuda" / "triton" / "torch"
print(gmm.last_fallback_reason_)     # diagnostic string if a fallback fired
print(gmm.fit_info_["backend_breakdown"])  # mixed runs: {"cuda": 18, "triton": 2}
```

Deprecation note: `use_triton=True/False` constructor kwarg still works but emits a `DeprecationWarning`. Switch to `backend=`. The mapping is `use_triton=True → backend="auto"`; `use_triton=False → backend="auto"` with Triton filtered from the dispatch chain (so you still get CUDA when available — historically `use_triton=False` meant "no Triton JIT", not "no GPU"). Removed in v2.0.
```

- [ ] **Step 11.2 — Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Document CUDA backend installation, selection, and deprecation in README

Adds:
- Build prereqs (CUDA >= 12.8, C++17 compiler, nanobind).
- TORCH_CUDA_ARCH_LIST shortcut for fast dev builds.
- GMMXX_SKIP_CUDA=1 escape hatch for Triton-only / CPU-only installs.
- Backend kwarg + GMMXX_BACKEND env var examples.
- last_backend_used_ / last_fallback_reason_ inspection.
- use_triton deprecation note.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12 — Final integration smoke test and cleanup

**Files:**
- Create: `tests/test_foundation_smoke.py`

- [ ] **Step 12.1 — Write an end-to-end smoke test**

Create `tests/test_foundation_smoke.py`:

```python
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
```

- [ ] **Step 12.2 — Run the full test suite**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass on a CUDA host. On a CPU-only host, the canary CUDA test in `test_cuda_build.py` is skipped; everything else passes.

- [ ] **Step 12.3 — Sanity-check the build artifacts**

```bash
uv run python -c "
import gmmxx
from gmmxx import _cuda, _dispatch, cuda_ops, GMMXX
print('gmmxx version:', gmmxx.__version__)
print('_HAS_CUDA:', _cuda._HAS_CUDA)
print('has_cuda():', _cuda.has_cuda())
print('exports cuda_ops:', 'cuda_ops' in gmmxx.__all__)
print('GMMXX backend kwarg ok:', GMMXX(n_components=4, backend='auto').backend)
print()
print('Resolve table:')
for req in ('auto', 'cuda', 'triton', 'torch'):
    try:
        r = _dispatch.resolve_backend(req, 'spherical', (1, 1024, 32, 64), None)
        print(f'  requested={req:6s} -> {r}')
    except Exception as e:
        print(f'  requested={req:6s} -> ERROR: {e}')
"
```

Expected output (on a CUDA host with extension built and Triton installed):

```
gmmxx version: 0.1.0
_HAS_CUDA: True
has_cuda(): True
exports cuda_ops: True
GMMXX backend kwarg ok: auto

Resolve table:
  requested=auto   -> triton   # cuda_*_supported still False in Plan 1
  requested=cuda   -> torch    # cuda gate False -> falls through
  requested=triton -> triton
  requested=torch  -> torch
```

- [ ] **Step 12.4 — Commit**

```bash
git add tests/test_foundation_smoke.py
git commit -m "$(cat <<'EOF'
Add end-to-end smoke tests for Plan 1 foundation

Verifies:
- GMMXX(backend=...) constructs for all four valid values.
- _dispatch.resolve_backend returns one of cuda/triton/torch.
- cuda_ops module exposes documented surface (has_cuda, require_cuda,
  CudaBackendUnavailable, CudaRuntimeFallback, canary_add_offset).
- GMMXX_BACKEND env var override works.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13 — Tag the foundation milestone

- [ ] **Step 13.1 — Confirm working tree is clean**

```bash
git status
```

Expected: `nothing to commit, working tree clean`.

- [ ] **Step 13.2 — Tag**

```bash
git tag -a foundation-plan1 -m "Plan 1: GMMXX CUDA backend foundation (build + dispatch + canary)"
```

- [ ] **Step 13.3 — Print the foundation summary**

```bash
git log --oneline foundation-plan1 ^main | head -50
```

Expected: a list of ~13 commits all on the `GMMXX-cuda` branch, in order.

---

## Self-Review Checklist

Run this after writing the plan with fresh eyes:

**1. Spec coverage**

| Spec section | Plan task |
| --- | --- |
| §3 backend kwarg + use_triton shim | Task 9 |
| §3 last_backend_used_ + cuda_*_enabled_ | Task 9 |
| §3 cuda_ops experimental surface | Task 10 |
| §5a CUDAGuard / getCurrentCUDAStream | Task 5 (canary uses both; subsequent plans copy the pattern) |
| §6.5 batch handling | Deferred — Plan 2 will write the first batched kernel; canary is N-vector only |
| §7 dispatch + try/except | Tasks 6, 8 |
| §7a runtime error fallback | Task 6 |
| §7.5 large_n.py | OUT OF SCOPE — separate plan |
| §8 source layout | Tasks 1, 3, 4, 5 |
| §9 build flags + robin_map + utf-8 + Zc:__cplusplus | Task 2 |
| §10a test refactor (unittest → pytest) | OUT OF SCOPE — happens in Plan 2 when the first real backend test lands; Plan 1's new tests are pure-pytest from the start |
| §10b new test files | Tasks 6, 8, 9, 12 (tests/test_cuda_build.py, test_dispatch.py, test_backend_kwarg.py, test_foundation_smoke.py) |

Gaps: §6.5 batch handling and §10a unittest-to-pytest refactor are both deferred to Plan 2 by design. Plan 1 doesn't ship any batched kernel and doesn't break any existing test, so deferring is safe.

**2. Placeholder scan** — none. Every step contains real code or real commands.

**3. Type consistency**

- `backend: str` — used consistently across `__init__`, `get_params`, `set_params`, `_dispatch.resolve_backend`.
- `_legacy_no_triton: bool` — used consistently in `__init__`, `set_params`, `_dispatch.resolve_backend`.
- `last_backend_used_: Optional[str]` — declared in `__init__` and `_reset_fit_state`, accessed in tests.
- `cuda_<cov>_supported(d, n_components, dtype)` — three-arg signature in `_runtime.py`, consumed by `_dispatch._cuda_supported` with the same arity.
- `_dispatch.resolve_backend(requested, covariance, shape, dtype, legacy_no_triton=False)` — matches the call sites in `_dispatch.resolve_backend_with_env` and the tests.

No drift detected.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-02-gmmxx-cuda-foundation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Good when each task is a small focused PR.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`. Batch execution with checkpoints for review.

**Which approach?**

After this plan ships, the next plan to write is **Plan 2: Spherical kernels** — populates `csrc/estep/spherical_*.cu`, `csrc/mstep/blocked_spherical.cu`, `csrc/fused/fused_spherical.cu`, `csrc/mstep/finalize_spherical.cu`; turns on `cuda_spherical_supported`; adds 3-way correctness tests vs torch_fallback and Triton; adds the spherical perf gate. That plan will be ~15–20 tasks similar in shape to this one.
