#pragma once

// Architecture probes and dtype traits shared across kernels.
//
// All probe macros are GMMXX_-prefixed so they don't collide with
// flash-kmeans-cuda's FKC_-prefixed equivalents if both libraries are
// linked into the same Python process or build.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <c10/core/ScalarType.h>

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
  static constexpr c10::ScalarType torch_scalar_type = c10::ScalarType::Half;
  static constexpr const char* name = "fp16";
  static constexpr bool is_half = true;
};

template <>
struct dtype_traits<__nv_bfloat16> {
  using packed2 = __nv_bfloat162;
  static constexpr c10::ScalarType torch_scalar_type = c10::ScalarType::BFloat16;
  static constexpr const char* name = "bf16";
  static constexpr bool is_half = true;
};

template <>
struct dtype_traits<float> {
  using packed2 = float2;
  static constexpr c10::ScalarType torch_scalar_type = c10::ScalarType::Float;
  static constexpr const char* name = "fp32";
  static constexpr bool is_half = false;
};

// Useful constants.
constexpr int kWarp = 32;

}  // namespace gmmxx
