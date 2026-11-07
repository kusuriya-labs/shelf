#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// Qwen4Exp gated residual: the gate-weighted mean of the hyper streams that feeds a
// sublayer. gate and normed are [rows, hc * hidden] BF16; output is [rows, hidden].
// When sigmoid_gate is non-zero the gate is sigmoid(gate) in the same pass.
struct HyperMixArgs {
    const void* gate = nullptr;
    const void* normed = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t hc = 0;
    uint32_t hidden = 0;
    uint32_t gate_stride = 0;
    uint32_t normed_stride = 0;
    uint32_t output_stride = 0;
    uint32_t sigmoid_gate = 1;
};

// Qwen4Exp injection gate: out[r, h] = scale / (1 + exp(-in[r, h] / hc)).
struct HyperInjectArgs {
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t hc = 0;
    uint32_t input_stride = 0;
    uint32_t output_stride = 0;
    float scale = 1.0f;
    float inverse_hc = 1.0f;
};

hipError_t launch_hyper_mix(const HyperMixArgs& args, hipStream_t stream);
hipError_t launch_hyper_inject(const HyperInjectArgs& args, hipStream_t stream);

}  // namespace shelf::kernel
