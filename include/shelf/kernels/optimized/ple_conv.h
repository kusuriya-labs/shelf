#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// PLE short depthwise convolution: kernel 4, dilation 3, causal, followed by SiLU.
// State is [max_sequences][9][channels] BF16 and carries the last nine inputs per
// sequence. The kernel iterates a request's rows in order and writes the state after the
// last row.
struct PleConvArgs {
    const shelf::bf16_t* input = nullptr;
    shelf::bf16_t* output = nullptr;
    const shelf::bf16_t* weight = nullptr;
    shelf::bf16_t* state = nullptr;
    const ::shelf::runtime::DeviceRequestDescriptor* requests = nullptr;

    uint32_t num_requests = 0;
    uint32_t actual_rows = 0;
    uint32_t channels = 0;
    uint32_t max_sequences = 0;
    uint32_t state_index = 0;

    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;

    uint64_t state_slot_stride = 0;
    uint64_t state_layer_stride = 0;
    uint64_t state_history_stride = 0;
};

inline constexpr uint32_t kPleConvThreads = 256u;
inline constexpr uint32_t kPleConvHistory = 9u;

constexpr uint32_t ple_conv_blocks(uint32_t channels) {
    return (channels + kPleConvThreads - 1u) / kPleConvThreads;
}

hipError_t launch_ple_conv(const PleConvArgs& args, hipStream_t stream);

}  // namespace shelf::kernel
