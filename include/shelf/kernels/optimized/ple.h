#pragma once

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// PLE gate shaping: sign(x) * sqrt(max(|x|, 1e-6)).
struct PleGateArgs {
    const bf16_t* input = nullptr;
    bf16_t* output = nullptr;
    uint32_t rows = 0;
    uint32_t hidden = 0;
    uint32_t input_stride = 0;
    uint32_t output_stride = 0;
};

// PLE broadcast multiply: out[r, h * hidden + i] = sigmoid(gate[r, i]) * value[r, i].
struct PleBroadcastMulArgs {
    const bf16_t* gate = nullptr;
    const bf16_t* value = nullptr;
    bf16_t* output = nullptr;
    uint32_t rows = 0;
    uint32_t hc = 0;
    uint32_t hidden = 0;
    uint32_t gate_stride = 0;
    uint32_t value_stride = 0;
    uint32_t output_stride = 0;
};

hipError_t launch_ple_gate(const PleGateArgs& args, hipStream_t stream);
hipError_t launch_ple_broadcast_mul(const PleBroadcastMulArgs& args, hipStream_t stream);

}  // namespace shelf::kernel
