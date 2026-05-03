#pragma once

// Block-level reductions and stable logsumexp helpers.
//
// All operate in fp32 regardless of input dtype. Each helper assumes the
// CTA layout is 1D (blockDim.x = nthreads, blockDim.y = blockDim.z = 1).

#include "arch.cuh"
#include "ptx.cuh"

namespace gmmxx { namespace reduce {

// Block-wide max over fp32 values, one per thread. Uses warp shuffle within
// each warp, then a small SMEM tree across warps.
//
// Caller MUST provide a shared-memory pointer of at least
// (blockDim.x / kWarp) fp32 entries.
__device__ __forceinline__ float block_max_f32(float v, float* smem) {
    int tid = threadIdx.x;
    int lane = tid & (kWarp - 1);
    int warp_id = tid / kWarp;
    int n_warps = (blockDim.x + kWarp - 1) / kWarp;

    // 1. Per-warp max.
    v = ptx::warp_reduce_max_f32(v);
    if (lane == 0) smem[warp_id] = v;
    __syncthreads();

    // 2. Cross-warp tree using the first warp.
    if (warp_id == 0) {
        v = (lane < n_warps) ? smem[lane] : -INFINITY;
        v = ptx::warp_reduce_max_f32(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// Block-wide sum over fp32 values, one per thread. Same SMEM contract as
// block_max_f32.
__device__ __forceinline__ float block_sum_f32(float v, float* smem) {
    int tid = threadIdx.x;
    int lane = tid & (kWarp - 1);
    int warp_id = tid / kWarp;
    int n_warps = (blockDim.x + kWarp - 1) / kWarp;

    v = ptx::warp_reduce_add_f32(v);
    if (lane == 0) smem[warp_id] = v;
    __syncthreads();

    if (warp_id == 0) {
        v = (lane < n_warps) ? smem[lane] : 0.0f;
        v = ptx::warp_reduce_add_f32(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// Stable logsumexp of K values per row, computed in fp32.
// `logits[k]` is the per-thread input (k = thread id; assumes K == blockDim.x
// or that out-of-range threads pass v = -INFINITY).
// Returns the same value on every thread.
__device__ __forceinline__ float logsumexp_block_f32(float v, float* smem) {
    float m = block_max_f32(v, smem);
    if (isinf(m) && m < 0.0f) {
        // All inputs are -inf; logsumexp is -inf and exp((−inf) − (−inf)) = nan.
        // Return -inf to match torch_fallback semantics.
        return -INFINITY;
    }
    float e = expf(v - m);
    float s = block_sum_f32(e, smem);
    return m + logf(s);
}

}}  // namespace gmmxx::reduce
