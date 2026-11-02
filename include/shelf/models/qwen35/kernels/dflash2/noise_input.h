#pragma once

#include <shelf/models/qwen35/kernels/dflash2/batch.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

hipError_t launch_dflash2_noise_token_ids(
    int32_t anchor_token,
    int32_t mask_token,
    int32_t* output,
    uint32_t block_rows,
    hipStream_t stream);

hipError_t launch_dflash2_noise_token_ids_batch(
    const DFlash2RequestDescriptor* requests,
    uint32_t request_count,
    uint32_t max_block_rows,
    int32_t mask_token,
    int32_t* token_ids,
    uint32_t* row_has_prev,
    hipStream_t stream);

}  // namespace shelf::kernel
