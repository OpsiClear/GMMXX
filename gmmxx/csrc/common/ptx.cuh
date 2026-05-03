#pragma once

// Warp-level PTX wrappers shared across kernels.
//
// Plan 2: warp shuffle reductions in fp32.
// Plan 3: cp_async_*, ldmatrix_sync_x4, mma_m16n8k16_* (sm_80+ helpers).
// All wrappers are __device__ __forceinline__ and gated on GMMXX_HAS_*
// macros from arch.cuh.

#include "arch.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace gmmxx { namespace ptx {

// ---- Plan 2: warp shuffle helpers -------------------------------------------

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

// ---- Plan 3: sm_80+ helpers (verbatim from flash-kmeans-cuda, FKC_->GMMXX_) -

// Generic-to-shared address conversion.
__device__ __forceinline__ unsigned int cvta_to_shared(const void* p) {
  return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

// ---- cp.async (Ampere+) -----------------------------------------------------
// Copy 16 bytes from gmem to smem; issues asynchronously.
// ``smem`` is a shared-memory address (cvta_to_shared output).
__device__ __forceinline__ void cp_async_16B(unsigned int smem, const void* gmem,
                                             bool predicate = true) {
#if GMMXX_HAS_CP_ASYNC
  // cp.async.cg: cache-global, bypass L1.
  // The optional predicate uses ``cp-size 16`` for true and ``cp-size 0`` for
  // false (zero-fills, simpler than a conditional branch).
  int src_size = predicate ? 16 : 0;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
      :
      : "r"(smem), "l"(gmem), "r"(src_size));
#else
  // Compile-time fallback: synchronous 16-byte copy. Should not be hit in
  // practice because the host dispatch only routes to cp.async-using kernels
  // on sm_80+.
  uint4 v = *reinterpret_cast<const uint4*>(gmem);
  *reinterpret_cast<uint4*>(__cvta_shared_to_generic(smem)) =
      predicate ? v : uint4{0, 0, 0, 0};
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if GMMXX_HAS_CP_ASYNC
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if GMMXX_HAS_CP_ASYNC
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if GMMXX_HAS_CP_ASYNC
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

// ---- ldmatrix (sm_75+) ------------------------------------------------------
// Loads 4x (8x8) fp16 tiles into 4x 32-bit registers per thread.
// ``smem`` points to the start of the 8x8 tile this lane is responsible for —
// the standard pattern is lane%16 picks the row, lane/16 picks the column-half.
__device__ __forceinline__ void ldmatrix_x4(uint32_t& d0, uint32_t& d1,
                                            uint32_t& d2, uint32_t& d3,
                                            unsigned int smem) {
#if GMMXX_HAS_LDMATRIX_X4
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(smem));
#else
  d0 = d1 = d2 = d3 = 0;
#endif
}

// Transposed variant — used when the operand layout in SMEM is column-major.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& d0, uint32_t& d1,
                                                  uint32_t& d2, uint32_t& d3,
                                                  unsigned int smem) {
#if GMMXX_HAS_LDMATRIX_X4
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(smem));
#else
  d0 = d1 = d2 = d3 = 0;
#endif
}

// ---- mma.sync m16n8k16 fp16/bf16 -> fp32 ------------------------------------
// Computes D = A * B + C with A: 16x16 fp16, B: 16x8 fp16, C/D: 16x8 fp32.
// Per-thread inputs:
//   a0..a3 : two halves each of 8 fp16 values from A (4 regs total)
//   b0..b1 : two halves each of 4 fp16 values from B (2 regs total)
//   c0..c3 : 4 fp32 accumulator regs (per-thread output is 4x fp32)
__device__ __forceinline__ void mma_m16n8k16_fp16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
#if GMMXX_HAS_F16_MMA
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
      "  {%0, %1, %2, %3},\n"
      "  {%4, %5, %6, %7},\n"
      "  {%8, %9},\n"
      "  {%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
  d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
#if GMMXX_HAS_BF16_MMA
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
      "  {%0, %1, %2, %3},\n"
      "  {%4, %5, %6, %7},\n"
      "  {%8, %9},\n"
      "  {%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
  d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

// fp16-accumulator variant: 2x throughput on Ada (8 cycles vs 16 for fp32 acc).
// Per thread: D = 2 u32 regs each holding 2 fp16 values.
//   d0_packed: D[m=lane/4, n=2*(lane%4)..+1]
//   d1_packed: D[m=lane/4+8, n=2*(lane%4)..+1]
__device__ __forceinline__ void mma_m16n8k16_fp16_acc_fp16(
    uint32_t& d0, uint32_t& d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1) {
#if GMMXX_HAS_F16_MMA
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16\n"
      "  {%0, %1},\n"
      "  {%2, %3, %4, %5},\n"
      "  {%6, %7},\n"
      "  {%8, %9};\n"
      : "=r"(d0), "=r"(d1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "r"(c0), "r"(c1));
#else
  d0 = c0; d1 = c1;
#endif
}

}}  // namespace gmmxx::ptx
