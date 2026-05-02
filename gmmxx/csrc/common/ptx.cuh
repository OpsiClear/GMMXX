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
