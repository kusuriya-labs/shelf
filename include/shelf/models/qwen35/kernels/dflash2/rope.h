#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/kernels/dflash2/batch.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

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
    hipStream_t stream);

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
    hipStream_t stream);

}
