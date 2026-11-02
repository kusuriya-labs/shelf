#include <phaseshift/models/qwen35/kernels/dflash2/grouped_dynamic_conv.h>
#include <phaseshift/kernels/optimized/detail/vector_io.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

using namespace ::ps::kernel::detail;
constexpr uint32_t kDynamicConvThreads = 256u;

namespace {

__device__ __forceinline__ float round_to_bf16_f32(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        u = ((u & 0x007FFFFFu) != 0u) ? 0x7FC00000u : (u & 0xFF800000u);
        return __uint_as_float(u);
    }
    u += 0x7FFFu + ((u >> 16u) & 1u);
    return __uint_as_float(u & 0xFFFF0000u);
}

__global__ void grouped_dynamic_conv_kernel(
    const bf16* input,
    const bf16* dynamic,
    const bf16* base_kernel,
    bf16* output,
    const uint32_t* row_has_prev,
    uint32_t rows,
    uint32_t hidden,
    uint32_t groups,
    uint32_t group_size,
    uint32_t phase) {
    const uint32_t total = rows * hidden;
    const uint32_t phase_stride = kDFlash2ConvKernelSize * groups;
    const uint32_t row_stride = kDFlash2ConvKernelSize * phase_stride;
    const bf16* base_phase = base_kernel + static_cast<std::size_t>(phase) * 2u * hidden;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        const uint32_t row = i / hidden;
        const uint32_t f = i % hidden;
        const uint32_t group = f / group_size;
        const bf16* dyn = dynamic + static_cast<std::size_t>(row) * row_stride +
                          static_cast<std::size_t>(phase) * phase_stride + group;

        const bf16 v0 = input[static_cast<std::size_t>(row) * hidden + f];
        const float v0f = bf16_to_f32(v0);
        const float t0 = round_to_bf16_f32(bf16_to_f32(base_phase[f]) * v0f);
        const float o0 = round_to_bf16_f32(t0 + bf16_to_f32(dyn[0]) * v0f);
        float acc = o0;
        const bool has_prev = row_has_prev != nullptr ? (row_has_prev[row] != 0u) : (row >= 1u);
        if (has_prev) {
            const bf16 v1 = input[static_cast<std::size_t>(row - 1u) * hidden + f];
            const float v1f = bf16_to_f32(v1);
            const float t1 = round_to_bf16_f32(bf16_to_f32(base_phase[hidden + f]) * v1f);
            const float o1 = round_to_bf16_f32(acc + t1);
            acc = round_to_bf16_f32(o1 + bf16_to_f32(dyn[groups]) * v1f);
        }
        output[i] = f32_to_bf16(acc);
    }
}

}  // namespace

hipError_t launch_dflash2_grouped_dynamic_conv(
    const bf16_t* input,
    const bf16_t* dynamic,
    const bf16_t* base_kernel,
    bf16_t* output,
    uint32_t rows,
    uint32_t hidden_size,
    uint32_t group_size,
    uint32_t phase,
    hipStream_t stream) {
    return launch_dflash2_grouped_dynamic_conv_rows(input, dynamic, base_kernel, output, rows,
                                                    hidden_size, group_size, phase, nullptr,
                                                    stream);
}

hipError_t launch_dflash2_grouped_dynamic_conv_rows(
    const bf16_t* input,
    const bf16_t* dynamic,
    const bf16_t* base_kernel,
    bf16_t* output,
    uint32_t rows,
    uint32_t hidden_size,
    uint32_t group_size,
    uint32_t phase,
    const uint32_t* row_has_prev,
    hipStream_t stream) {
    if (input == nullptr || dynamic == nullptr || base_kernel == nullptr ||
        output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (phase >= kDFlash2ConvKernelSize) {
        return hipErrorInvalidValue;
    }
    if (hidden_size == 0u || group_size == 0u || hidden_size % group_size != 0u) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u) {
        return hipSuccess;
    }
    const uint32_t groups = hidden_size / group_size;
    const uint32_t total = rows * hidden_size;
    const uint32_t blocks = (total + kDynamicConvThreads - 1u) / kDynamicConvThreads;
    grouped_dynamic_conv_kernel<<<blocks, kDynamicConvThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(input),
        reinterpret_cast<const bf16*>(dynamic),
        reinterpret_cast<const bf16*>(base_kernel),
        reinterpret_cast<bf16*>(output),
        row_has_prev,
        rows,
        hidden_size,
        groups,
        group_size,
        phase);
    return hipGetLastError();
}

}
