#pragma once

// Warp-level PTX wrappers shared across kernels.
//
// Plan 2 populates only what the spherical safe path needs: warp shuffle
// reductions in fp32. Plan 3 will add cp_async_*, ldmatrix_sync_x4,
// mma_m16n8k16_*. All wrappers are __device__ __forceinline__ and gated
// on GMMXX_HAS_* macros from arch.cuh.

#include "arch.cuh"
#include <cuda_runtime.h>

namespace gmmxx { namespace ptx {

// Full-warp xor shuffle. Available on all CUDA-capable arches via
// __shfl_xor_sync; we wrap it for symmetry with the named-helper style
// used by the kernels.
__device__ __forceinline__ float warp_shfl_xor_f32(float v, int laneMask) {
    return __shfl_xor_sync(0xffffffffu, v, laneMask, kWarp);
}

// Full-warp sum reduction. Returns the same value on every lane.
__device__ __forceinline__ float warp_reduce_add_f32(float v) {
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset, kWarp);
    }
    return v;
}

// Full-warp max reduction. Returns the same value on every lane.
__device__ __forceinline__ float warp_reduce_max_f32(float v) {
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, v, offset, kWarp);
        v = fmaxf(v, other);
    }
    return v;
}

}}  // namespace gmmxx::ptx
