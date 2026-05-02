#pragma once

// Single header for kernel translation units (.cu files).
//
// IMPORTANT: kernel TUs MUST NOT include <torch/extension.h> or <torch/torch.h>.
// Those pull in Python.h (via pybind11) and trip MSVC's std-namespace ambiguity
// in torch/csrc/dynamo/compiled_autograd.h on CUDA 13. Use this header instead.

// Windows compatibility note (MSVC + CUDA 13 / /Zc:preprocessor):
//
// rpcndr.h (pulled in transitively by ATen/ATen.h on Windows) defines
// `#define small char`. c10/cuda/CUDACachingAllocator.h uses `small` as a
// bool parameter name, so if CUDACachingAllocator.h is included while `small`
// is a live macro the preprocessor mangles `bool small` → `bool char` and
// the compiler emits "invalid combination of type specifiers".
//
// Resolution: include ATen/ATen.h first (which triggers the Windows include
// chain and sets all the #pragma once guards), then undef the offending
// macros, then include the CUDA-guard / stream headers that transitively pull
// in CUDACachingAllocator.h. This mirrors the pattern in flash-kmeans-cuda.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

// Re-undef Windows macros that rpcndr.h (via ATen's Windows include chain)
// may have re-introduced. CUDAGuard.h → CUDAGuardImpl.h → CUDACachingAllocator.h
// uses `small` as a parameter name and must see it undefined.
#ifdef small
  #undef small
#endif
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
