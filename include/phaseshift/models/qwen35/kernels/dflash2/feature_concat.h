#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

constexpr uint32_t kDFlash2TargetTaps = 5u;

hipError_t launch_dflash2_feature_concat(
    const bf16_t* tap0,
    const bf16_t* tap1,
    const bf16_t* tap2,
    const bf16_t* tap3,
    const bf16_t* tap4,
    uint32_t tap_features,
    uint32_t rows,
    bf16_t* output,
    uint32_t output_row_stride,
    hipStream_t stream);

}
