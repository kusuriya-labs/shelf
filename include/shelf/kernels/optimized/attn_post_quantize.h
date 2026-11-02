#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

struct FusedAttnPostQuantizeArgs {
    const ::ps::bf16_t* gate = nullptr;
    uint32_t gate_row_stride = 0;
    const float* ctx = nullptr;
    uint32_t ctx_row_stride = 0;
    ::ps::bf16_t* gated_out = nullptr;
    uint32_t gated_row_stride = 0;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t ctx_first = 1u;
};

hipError_t launch_fused_attn_post_quantize(const FusedAttnPostQuantizeArgs& args,
                                           hipStream_t stream);

}
