#pragma once

// Single header for kernel translation units (.cu files).
//
// IMPORTANT: kernel TUs MUST NOT include <torch/extension.h> or <torch/torch.h>.
// Those pull in Python.h (via pybind11) and trip MSVC's std-namespace ambiguity
// in torch/csrc/dynamo/compiled_autograd.h on CUDA 13. Use this header instead.

// Windows compatibility note (MSVC + CUDA 13 / /Zc:preprocessor):
// some Windows headers can define `small`, `min`, and `max` as macros. These
// collide with c10 parameter names and std functions under the conforming MSVC
// preprocessor. Undef them before the CUDA helper headers, matching the local
// flash-kmeans-cuda build pattern.
#ifdef small
  #undef small
#endif
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif

#include <ATen/ATen.h>

// Re-undef in case ATen's Windows include chain reintroduced them.
#ifdef small
  #undef small
#endif
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
