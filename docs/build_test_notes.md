# Build and Test Notes

Last verified: 2026-05-06 on Windows, RTX 4090 (`sm_89`), Python 3.12.10,
PyTorch `2.11.0+cu130`, CUDA toolkit 13.2, MSVC 14.44.

## Working Build Recipe (PowerShell)

```powershell
# 1. Strip msys64 / Git Unix bin paths so Git's `link.exe` does not shadow
#    MSVC's. Prepend the MSVC Hostx64\x64 bin so cl/link resolve correctly.
$msvcBin = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
$cleanedParts = $env:PATH -split ';' | Where-Object {
    $_ -and ($_ -notmatch 'Git\\usr\\bin') -and ($_ -notmatch 'Git\\mingw') `
         -and ($_ -notmatch 'msys64\\usr\\bin') -and ($_ -notmatch 'msys64\\ucrt64\\bin')
}
$env:PATH = "$msvcBin;" + ($cleanedParts -join ';')

# 2. Tell setuptools to trust the active VS environment, lock to local arch,
#    cap parallel jobs to keep memory in check, skip Blackwell.
$env:DISTUTILS_USE_SDK     = "1"
$env:TORCH_CUDA_ARCH_LIST  = "8.9"
$env:MAX_JOBS              = "2"
$env:GMMXX_BUILD_BLACKWELL = "0"

python setup.py build_ext --inplace
```

Cold full build observed at ~225 s on this machine: 218 s for 16 nvcc
translation units with `MAX_JOBS=2`, ~5 s for the link step. For release
wheels, drop `TORCH_CUDA_ARCH_LIST` so `setup.py` emits the portable arch
set defined in `setup.py`.

CUDA-skip mode also works for pure-Python iteration:

```powershell
$env:GMMXX_SKIP_CUDA = "1"
python setup.py build_ext --inplace
```

## Working Test Command

```powershell
python -m pytest tests\test_cuda_tied.py tests\test_cuda_full.py `
  tests\test_cuda_soft_em_non_spherical.py `
  tests\test_cuda_largen_training_covariances.py `
  tests\test_flash_kmeans_benchmark.py `
  tests\test_cuda_approx_topk_spherical.py `
  tests\test_cuda_diag_safe.py `
  tests\test_dispatch.py `
  tests\test_gmmxx.py::GMMXXTests::test_runtime_gate_policy -q
```

Result: 92 passed in ~3 s against a freshly built
`gmmxx/_C.cp312-win_amd64.pyd`.

## Root Causes Tracked Down This Session

### 1. Apparent "nvcc hangs" were actually `data_ptr<T>()` parse errors

Symptom: `python setup.py build_ext --inplace` appeared to stall after
generating `build.ninja` and emitting only `nb_combined.obj`. Direct ninja
runs of `canary.cu` looked the same at first.

Diagnosis: running the exact `nvcc` command from `build.ninja` directly
finished in seconds with errors of the form

```
canary.cu(41): error: type name is not allowed
        input.data_ptr<int32_t>(),
                       ^
canary.cu(41): error: expected an expression
        input.data_ptr<int32_t>(),
                                ^
```

That is the classic symptom of `at::Tensor` being only forward-declared at
the point `.data_ptr<T>()` is parsed.

Cause: `gmmxx/csrc/common/torch_cuda_includes.h` no longer included
`<ATen/ATen.h>`. Earlier in the session the shim had been pruned to match
flash-kmeans-cuda's minimal include list, but flash-kmeans-cuda's
per-module headers (e.g. `assign.h`) include `<ATen/ATen.h>` themselves.
GMMXX's design uses `torch_cuda_includes.h` as the single shim for kernel
TUs, so removing `<ATen/ATen.h>` from the shim leaves every kernel TU
without the full `at::Tensor` definition.

Fix: restored `<ATen/ATen.h>` in `gmmxx/csrc/common/torch_cuda_includes.h`,
between the Windows macro undefs and the CUDA helper headers.

### 2. Link step picks up Git's `link.exe`

Symptom: after the 16 CUDA TUs all compile cleanly, link fails with

```
"C:\Program Files\Git\usr\bin\link.exe" /nologo /INCREMENTAL:NO /LTCG /DLL ...
/usr/bin/link: extra operand '/LTCG'
```

Cause: `C:\Program Files\Git\usr\bin` (Git's coreutils `link`, which is the
GNU "make a hard link" tool) appears earlier in `PATH` than MSVC's
`Hostx64\x64`. `where.exe link.exe` returns the Git one first.

Fix: prepend the MSVC `Hostx64\x64` directory to `PATH`, and strip
`Git\usr\bin`, `Git\mingw*\bin`, `msys64\usr\bin`, and
`msys64\ucrt64\bin` out of `PATH` before launching `setup.py`. The
PowerShell snippet above does this. Sourcing `vcvars64.bat` is *not*
sufficient on this machine because the user-level `PATH` puts Git
earlier.

This matches the pattern in
`flash-kmeans-cuda/scripts/windows/_setup_env.bat`, which strips the same
Git directories with `set "PATH=%PATH:...;=%"`.

## Environment Notes

- Use the global Python:
  `C:\Users\HEQ\AppData\Local\Programs\Python\Python312\python.exe`. The
  repo's `.venv` is missing `nanobind` and so cannot drive the build.
- `uv run` currently fails with "access denied" on
  `C:\Users\HEQ\AppData\Local\uv\cache\sdists-v6\.git`, so the uv-managed
  Python path is unusable until that cache permission is fixed.
- `pip install -e . --no-build-isolation` hits a Windows pip build-tracker
  permission error even with `TMP`/`TEMP` redirected, so the direct
  `python setup.py build_ext --inplace` invocation is the most reliable
  development build path.
- The CUDA-version warning ("detected CUDA version (13.2) has a minor
  version mismatch with the version that was used to compile PyTorch
  (13.0)") is benign on this setup.

Reference: PyTorch `torch.utils.cpp_extension` documents `CUDAExtension`,
`BuildExtension`, `TORCH_CUDA_ARCH_LIST`, ninja builds, and `MAX_JOBS`:
https://docs.pytorch.org/docs/main/cpp_extension.html
