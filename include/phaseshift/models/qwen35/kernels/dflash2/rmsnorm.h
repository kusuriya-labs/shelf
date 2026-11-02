#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

hipError_t launch_dflash2_rmsnorm_direct_bf16(
    const bf16_t* input,
    const bf16_t* weight,
    bf16_t* output,
    uint32_t rows,
    uint32_t features,
    uint32_t group_size,
    float eps,
    hipStream_t stream);

}
