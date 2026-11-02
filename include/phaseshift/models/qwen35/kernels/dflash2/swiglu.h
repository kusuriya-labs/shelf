#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

hipError_t launch_dflash2_swiglu_bf16(
    const bf16_t* gate,
    uint32_t gate_row_stride,
    const bf16_t* up,
    uint32_t up_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    uint32_t features,
    hipStream_t stream);

}
