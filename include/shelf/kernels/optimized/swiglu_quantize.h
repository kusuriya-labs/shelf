#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

struct FusedSwiGluQuantizeArgs {
    const void* gate = nullptr;
    const void* up = nullptr;
    void* output = nullptr;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t gate_row_stride = 0;
    uint32_t up_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
};

hipError_t launch_fused_swiglu_quantize(const FusedSwiGluQuantizeArgs& args,
                                        hipStream_t stream);

}
