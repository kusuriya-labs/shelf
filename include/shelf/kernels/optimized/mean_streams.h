#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// Mean one hidden row over hc hyper-connection streams.
struct MeanStreamsArgs {
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t hc = 0;
    uint32_t hidden = 0;
    uint32_t input_stride = 0;
    uint32_t output_stride = 0;
};

hipError_t launch_mean_streams_bf16(const MeanStreamsArgs& args, hipStream_t stream);

}  // namespace shelf::kernel
