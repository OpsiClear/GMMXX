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
