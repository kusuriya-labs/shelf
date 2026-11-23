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

struct HyperApplyArgs {
    const void* raw = nullptr;
    const void* y = nullptr;
    const void* injection = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t hc = 0;
    uint32_t hidden = 0;
    uint32_t raw_stride = 0;
    uint32_t y_stride = 0;
    uint32_t injection_stride = 0;
    uint32_t output_stride = 0;
};

inline constexpr uint32_t kHyperUpMixMaxHc = 8u;

// Fused up projection + gated mean. up_weight is [hc * hidden, k] BF16 and input is
// [rows, k] BF16. Each hidden index computes its hc dot products, rounds each to BF16
// (the value the unfused up GEMM would have stored), then applies the same sigmoid gate
// and 1/hc mean as HyperMixArgs.
struct HyperUpMixArgs {
    const void* up_weight = nullptr;
    const void* input = nullptr;
    const void* normed = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t hc = 0;
    uint32_t hidden = 0;
    uint32_t k = 0;
    uint32_t input_stride = 0;
    uint32_t normed_stride = 0;
    uint32_t output_stride = 0;
    uint32_t sigmoid_gate = 1;
};

hipError_t launch_hyper_up_mix(const HyperUpMixArgs& args, hipStream_t stream);
hipError_t launch_hyper_mix(const HyperMixArgs& args, hipStream_t stream);
hipError_t launch_hyper_inject(const HyperInjectArgs& args, hipStream_t stream);
hipError_t launch_hyper_apply(const HyperApplyArgs& args, hipStream_t stream);

}  // namespace shelf::kernel
