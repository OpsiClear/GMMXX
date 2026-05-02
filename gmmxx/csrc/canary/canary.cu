#include "canary.h"
#include "../common/arch.cuh"

namespace gmmxx { namespace canary {

namespace {

__global__ void canary_add_offset_kernel(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t n_elements,
    int32_t offset
) {
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (idx >= n_elements) return;
    output[idx] = input[idx] + offset;
}

}  // anonymous namespace

at::Tensor add_offset(const at::Tensor& input, int64_t offset) {
    TORCH_CHECK(input.is_cuda(), "canary.add_offset: input must be on a CUDA device");
    TORCH_CHECK(input.is_contiguous(), "canary.add_offset: input must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kInt, "canary.add_offset: input must be int32");

    // Multi-device safety: bind the device of the input tensor for the
    // duration of this call.
    c10::cuda::CUDAGuard guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto output = at::empty_like(input);
    int64_t n = input.numel();
    if (n == 0) return output;

    constexpr int kThreads = 256;
    int64_t blocks_64 = (n + kThreads - 1) / kThreads;
    TORCH_CHECK(blocks_64 <= 0x7fffffff, "canary.add_offset: input too large");
    int blocks = static_cast<int>(blocks_64);

    canary_add_offset_kernel<<<blocks, kThreads, 0, stream>>>(
        input.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        n,
        static_cast<int32_t>(offset)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}}  // namespace gmmxx::canary
