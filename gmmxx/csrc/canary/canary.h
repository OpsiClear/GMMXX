#pragma once

#include "../common/torch_cuda_includes.h"

namespace gmmxx { namespace canary {

// Returns a fresh int32 tensor of the same shape as `input` where each
// element is `input[i] + offset`. Used as a build/FFI smoke test.
at::Tensor add_offset(const at::Tensor& input, int64_t offset);

}}  // namespace gmmxx::canary
