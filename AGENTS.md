# Repository Guidelines

## Project Structure & Module Organization

`gmmxx/` contains the importable Python package. Core estimator and dispatch logic live in `interface.py`, `_dispatch.py`, `_runtime.py`, and backend modules such as `_cuda.py`, `torch_fallback.py`, and `*_triton.py`. Native CUDA/C++ sources are under `gmmxx/csrc/`, grouped by kernel area (`estep/`, `mstep/`, `fused/`, `common/`). Tests live in `tests/` and follow backend or feature names. Benchmark and validation entry points live in `benchmarks/`; root-level `bench_*.py`, `profile_*.py`, and `trace_*.py` are ad hoc performance tools. Design notes and implementation plans are in `docs/`.

## Build, Test, and Development Commands

- `uv pip install -e .`: install the package in editable mode and build the CUDA extension when available.
- `$env:GMMXX_SKIP_CUDA = "1"; uv pip install -e ".[triton]"`: install without compiling the CUDA extension.
- `python -m pip install -e ".[dev]"`: install pytest/build/twine development extras.
- `python -m pytest tests -q`: run the full test suite; CUDA tests skip automatically when requirements are missing.
- `python benchmarks\validate_equivalence.py --device cuda`: compare optimized paths against internal references.
- `python benchmarks\validate_size_sweep.py --device cuda --profile standard`: run the standard shape regression sweep.
- `python -m pip wheel . --no-deps -w dist`: build a wheel for distribution checks.

## Coding Style & Naming Conventions

Use Python 3.12 syntax and typed public APIs where practical; the package ships `py.typed`. Follow existing 4-space Python indentation and sklearn-style estimator naming: constructor parameters use plain names, learned attributes end in `_` (`means_`, `weights_`, `last_backend_used_`). Backend files should keep explicit names such as `assign_spherical_triton.py` or `test_cuda_diag_safe.py`. CUDA/C++ code should preserve the current directory split and C++17 build assumptions.

## Testing Guidelines

Use pytest. Place tests in `tests/test_*.py`, use `pytest.mark.parametrize` for shape/dtype/backend matrices, and gate hardware-specific coverage with `pytest.mark.skipif` or `pytest.skip`. For numerical changes, compare CUDA/Triton paths to the PyTorch reference and include tolerances suitable for `float32`, `float16`, and `bfloat16`.

## Commit & Pull Request Guidelines

Recent history uses concise imperative subjects and experiment tags, for example `Exp53: chunked N processing for L2-resident logits/resp at large N` and `bench: add --shape-grid {small,large,xlarge} preset for stress tests`. Keep PRs focused, describe the affected backend or covariance type, list validation commands run, and call out any CUDA architecture, Triton, or fallback-behavior changes.

## Security & Configuration Tips

Do not commit local caches, build products, logs, or benchmark outputs. Prefer environment switches already used by the project, especially `GMMXX_SKIP_CUDA`, `GMMXX_BACKEND`, and `TORCH_CUDA_ARCH_LIST`, rather than hard-coding machine-specific paths or GPU capabilities.
