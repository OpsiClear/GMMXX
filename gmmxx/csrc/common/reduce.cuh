#pragma once

// Warp / block reductions and stable logsumexp helpers.
//
// SKELETON for Plan 1; Plan 2 populates with:
//   - warp_max_f32, warp_sum_f32 (via __shfl_xor_sync)
//   - block_max_f32, block_sum_f32 (warp+SMEM tree reduction)
//   - logsumexp_warp, logsumexp_block (subtract-max-then-exp-then-sum in fp32)

#include "arch.cuh"

namespace gmmxx { namespace reduce {

// Skeleton — see Plan 2 for population.

}}  // namespace gmmxx::reduce
