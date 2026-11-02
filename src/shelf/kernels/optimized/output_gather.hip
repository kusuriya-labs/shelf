#include <shelf/kernels/optimized/output_gather.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

using bf16 = __hip_bfloat16;

constexpr uint32_t kGatherThreads = 256;

template <typename T>
__global__ __launch_bounds__(kGatherThreads)
void output_gather_kernel(
    const T* __restrict__ input, uint32_t input_row_stride,
    T* __restrict__ output, uint32_t output_row_stride,
    const ::ps::runtime::DeviceBatchContext* __restrict__ ctx,
    uint32_t features) {
    const uint32_t i = blockIdx.y;
    const uint32_t src = ctx->output_rows[i];
    const size_t src_base = static_cast<size_t>(src) * input_row_stride;
    const size_t dst_base = static_cast<size_t>(i) * output_row_stride;
    for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < features;
         c += blockDim.x * gridDim.x) {
        output[dst_base + c] = input[src_base + c];
    }
}

}  // namespace

hipError_t launch_output_gather(
    const void* input, uint32_t input_row_stride,
    void* output, uint32_t output_row_stride,
    const ::ps::runtime::DeviceBatchContext* ctx,
    uint32_t num_outputs,
    uint32_t features, ::ps::runtime::ValueDType dtype, hipStream_t stream) {
    if (input == nullptr || output == nullptr || ctx == nullptr || features == 0u)
        return hipErrorInvalidValue;
    if (num_outputs == 0u)
        return hipErrorInvalidValue;
    const uint32_t blocks_x = (features + kGatherThreads - 1u) / kGatherThreads;
    dim3 grid(blocks_x, num_outputs);
    switch (dtype) {
        case ::ps::runtime::ValueDType::BF16:
            output_gather_kernel<bf16><<<grid, kGatherThreads, 0, stream>>>(
                reinterpret_cast<const bf16*>(input), input_row_stride,
                reinterpret_cast<bf16*>(output), output_row_stride, ctx, features);
            break;
        case ::ps::runtime::ValueDType::F32:
            output_gather_kernel<float><<<grid, kGatherThreads, 0, stream>>>(
                reinterpret_cast<const float*>(input), input_row_stride,
                reinterpret_cast<float*>(output), output_row_stride, ctx, features);
            break;
        case ::ps::runtime::ValueDType::I32:
            output_gather_kernel<int32_t><<<grid, kGatherThreads, 0, stream>>>(
                reinterpret_cast<const int32_t*>(input), input_row_stride,
                reinterpret_cast<int32_t*>(output), output_row_stride, ctx, features);
            break;
        default:
            return hipErrorNotSupported;
    }
    return hipGetLastError();
}

}  // namespace ps::kernel
