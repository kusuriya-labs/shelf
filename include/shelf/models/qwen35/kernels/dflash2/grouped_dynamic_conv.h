#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

constexpr uint32_t kDFlash2ConvKernelSize = 2u;

hipError_t launch_dflash2_grouped_dynamic_conv(
    const bf16_t* input,
    const bf16_t* dynamic,
    const bf16_t* base_kernel,
    bf16_t* output,
    uint32_t rows,
    uint32_t hidden_size,
    uint32_t group_size,
    uint32_t phase,
    hipStream_t stream);

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
    hipStream_t stream);

}
