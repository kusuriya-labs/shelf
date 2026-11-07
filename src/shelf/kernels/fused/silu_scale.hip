#include <shelf/kernels/fused/silu_scale.h>

#include <shelf/kernels/optimized/elementwise.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {
namespace {

__global__ void silu_scale_bf16_kernel(ElementwiseArgs args) {
    const uint32_t row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kElementwiseVec;
    if (col0 >= args.features) return;
    const auto* in_row = reinterpret_cast<const __hip_bfloat16*>(args.in0) +
                         static_cast<std::uint64_t>(row) * args.in0_row_stride;
    auto* out_row = reinterpret_cast<__hip_bfloat16*>(args.out0) +
                    static_cast<std::uint64_t>(row) * args.out0_row_stride;
    for (uint32_t i = 0u; i < kElementwiseVec; ++i) {
        const uint32_t c = col0 + i;
        if (c >= args.features) break;
        const float x = __bfloat162float(in_row[c]) * args.scalar_a;
        out_row[c] = __float2bfloat16(x / (1.0f + __expf(-x)));
    }
}

}  // namespace

hipError_t launch_silu_bf16(
    const bf16_t* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, float scale, hipStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0u || features == 0u)
        return hipErrorInvalidValue;
    ElementwiseArgs args;
    args.in0 = input;
    args.in0_row_stride = input_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.rows = rows;
    args.scalar_a = scale;
    silu_scale_bf16_kernel<<<elementwise_grid(features, rows), kElementwiseBlock, 0,
                             stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
