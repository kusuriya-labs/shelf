#include <shelf/models/qwen35/kernels/dflash2/swiglu.h>
#include <shelf/kernels/optimized/detail/vector_io.h>
#include "detail/swiglu_device.h"
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

using namespace ::shelf::kernel::detail;
constexpr uint32_t kDflash2SwigluThreads = 256u;

namespace {

__global__ void dflash2_swiglu_kernel(
    const bf16* gate,
    uint32_t gate_row_stride,
    const bf16* up,
    uint32_t up_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    uint32_t rows,
    uint32_t features) {
    const uint32_t total = rows * features;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        const uint32_t row = i / features;
        const uint32_t col = i % features;
        output[static_cast<std::size_t>(row) * output_row_stride + col] =
            f32_to_bf16(dflash2_swiglu_rounded_f32(
                bf16_to_f32(gate[static_cast<std::size_t>(row) * gate_row_stride + col]),
                bf16_to_f32(up[static_cast<std::size_t>(row) * up_row_stride + col])));
    }
}

}  // namespace

hipError_t launch_dflash2_swiglu_bf16(
    const bf16_t* gate,
    uint32_t gate_row_stride,
    const bf16_t* up,
    uint32_t up_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    uint32_t features,
    hipStream_t stream) {
    if (gate == nullptr || up == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || features == 0u) {
        return hipSuccess;
    }
    if (gate_row_stride < features || up_row_stride < features ||
        output_row_stride < features) {
        return hipErrorInvalidValue;
    }
    const uint32_t total = rows * features;
    const uint32_t blocks = (total + kDflash2SwigluThreads - 1u) / kDflash2SwigluThreads;
    dflash2_swiglu_kernel<<<blocks, kDflash2SwigluThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(gate), gate_row_stride,
        reinterpret_cast<const bf16*>(up), up_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride, rows, features);
    return hipGetLastError();
}

}
