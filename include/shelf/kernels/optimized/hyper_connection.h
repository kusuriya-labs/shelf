#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// Qwen4Exp gated residual: the gate-weighted mean of the hyper streams that feeds a
// sublayer. gate and normed are [rows, hc * hidden] BF16; output is [rows, hidden].
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
};

// Qwen4Exp gated residual injection: x[h] += y * injection[h] for each hyper stream.
// raw and output are [rows, hc * hidden]; y is [rows, hidden]; injection is [rows, hc].
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

hipError_t launch_hyper_mix(const HyperMixArgs& args, hipStream_t stream);
hipError_t launch_hyper_apply(const HyperApplyArgs& args, hipStream_t stream);

}  // namespace shelf::kernel
