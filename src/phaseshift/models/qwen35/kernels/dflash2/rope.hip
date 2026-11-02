#include <phaseshift/models/qwen35/kernels/dflash2/rope.h>
#include "detail/rope_device.h"
#include <phaseshift/kernels/optimized/detail/vector_io.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace ps::kernel {

using namespace ::ps::kernel::detail;

namespace {

__global__ void dflash2_rope_kernel(
    const bf16* input,
    uint32_t input_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    uint32_t rows,
    uint32_t head_dim,
    uint32_t position_start,
    float theta) {
    const uint32_t row = blockIdx.y;
    const uint32_t head = blockIdx.x;
    const uint32_t pair = threadIdx.x;
    const uint32_t half = head_dim / 2u;
    if (row >= rows || pair >= half) return;

    const uint32_t base = head * head_dim;
    const bf16* src = input + static_cast<std::size_t>(row) * input_row_stride + base;
    bf16* dst = output + static_cast<std::size_t>(row) * output_row_stride + base;

    detail::dflash2_rope_pair_body(src, dst, pair, head_dim, position_start + row, theta);
}

__global__ void dflash2_rope_positions_kernel(
    const bf16* input,
    uint32_t input_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    const DFlash2RequestDescriptor* requests,
    uint32_t head_dim,
    float theta) {
    const DFlash2RequestDescriptor& request = requests[blockIdx.z];
    const uint32_t row = blockIdx.y;
    const uint32_t head = blockIdx.x;
    const uint32_t pair = threadIdx.x;
    const uint32_t half = head_dim / 2u;
    if (row >= request.block_rows || pair >= half) return;

    const uint32_t scratch_row = request.row_begin + row;
    const uint32_t base = head * head_dim;
    const bf16* src = input + static_cast<std::size_t>(scratch_row) * input_row_stride + base;
    bf16* dst = output + static_cast<std::size_t>(scratch_row) * output_row_stride + base;

    detail::dflash2_rope_pair_body(src, dst, pair, head_dim,
                                   request.block_position_start + row, theta);
}

}  // namespace

hipError_t launch_dflash2_rope_bf16(
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    uint32_t features,
    uint32_t head_dim,
    uint32_t position_start,
    float theta,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (head_dim == 0u || head_dim % 2u != 0u || features == 0u ||
        features % head_dim != 0u) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u) {
        return hipSuccess;
    }
    if (input_row_stride < features || output_row_stride < features) {
        return hipErrorInvalidValue;
    }
    const uint32_t heads = features / head_dim;
    dim3 grid(heads, rows);
    dflash2_rope_kernel<<<grid, head_dim / 2u, 0, stream>>>(
        reinterpret_cast<const bf16*>(input), input_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride, rows, head_dim, position_start,
        theta);
    return hipGetLastError();
}

hipError_t launch_dflash2_rope_positions_bf16(
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    const DFlash2RequestDescriptor* requests,
    uint32_t request_count,
    uint32_t max_block_rows,
    uint32_t features,
    uint32_t head_dim,
    float theta,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr || requests == nullptr) {
        return hipErrorInvalidValue;
    }
    if (head_dim == 0u || head_dim % 2u != 0u || features == 0u ||
        features % head_dim != 0u) {
        return hipErrorInvalidValue;
    }
    if (request_count == 0u || max_block_rows == 0u) {
        return hipErrorInvalidValue;
    }
    if (input_row_stride < features || output_row_stride < features) {
        return hipErrorInvalidValue;
    }
    const uint32_t heads = features / head_dim;
    dim3 grid(heads, max_block_rows, request_count);
    dflash2_rope_positions_kernel<<<grid, head_dim / 2u, 0, stream>>>(
        reinterpret_cast<const bf16*>(input), input_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride, requests, head_dim, theta);
    return hipGetLastError();
}

}
