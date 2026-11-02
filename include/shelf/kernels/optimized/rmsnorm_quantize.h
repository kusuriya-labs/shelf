#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

struct FusedRmsNormQuantizeArgs {
    const void* input = nullptr;
    const void* weight = nullptr;
    void* output = nullptr;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
    float eps = 0.0f;
};

hipError_t launch_fused_rmsnorm_quantize(const FusedRmsNormQuantizeArgs& args,
                                         hipStream_t stream);

}
