#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

struct FusedGdnPostArgs {
    const float* rec = nullptr;
    uint32_t rec_row_stride = 0;
    const ::ps::bf16_t* norm_weight = nullptr;
    const ::ps::bf16_t* z = nullptr;
    uint32_t z_row_stride = 0;
    float* rms_out = nullptr;
    uint32_t rms_row_stride = 0;
    float* gated_z_out = nullptr;
    uint32_t gated_z_row_stride = 0;
    ::ps::bf16_t* gated_out = nullptr;
    uint32_t gated_row_stride = 0;
    float eps = 0.0f;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
};

hipError_t launch_fused_gdn_post(const FusedGdnPostArgs& args, hipStream_t stream);

}
