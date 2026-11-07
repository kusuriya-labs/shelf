#pragma once

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// out[r, h * hidden + i] = in[r, i] for every hc stream.
struct RepeatStreamsArgs {
    const bf16_t* input = nullptr;
    bf16_t* output = nullptr;
    uint32_t rows = 0;
    uint32_t hc = 0;
    uint32_t hidden = 0;
    uint32_t input_stride = 0;
    uint32_t output_stride = 0;
};

hipError_t launch_repeat_streams(const RepeatStreamsArgs& args, hipStream_t stream);

}  // namespace shelf::kernel
