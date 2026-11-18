#pragma once

#include <cstdint>

namespace shelf::quantization::mxfp4 {

// One MXFP4 expert plane in the GPU-native (16-row x 32-K block) layout, read by the
// CPU. `slot` selects the expert: the plane base is advanced by the batch strides.
struct CpuExpertPlane {
    const std::uint8_t* codes = nullptr;
    const std::uint8_t* scales = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t k_padded = 0;
    std::uint32_t blocks = 0;
    std::uint32_t codes_batch_stride = 0;
    std::uint32_t scales_batch_stride = 0;
    std::uint32_t scale_row_stride = 0;
};

// A W4A8 activation row: int8 codes with one f32 scale per 32 codes.
struct CpuActivationPlane {
    const std::int8_t* codes = nullptr;
    const float* scales = nullptr;
    std::uint32_t code_stride = 0;
    std::uint32_t scale_stride = 0;
};

// Computes one output row (out row `slot_row`, written at `out + slot_row * out_stride`)
// for the expert `slot` and the activation `row`. `out_stride` is in floats.
void gemv_row(const CpuExpertPlane& weights, std::uint32_t slot,
              const CpuActivationPlane& activation, std::uint32_t row, float* out,
              std::uint32_t out_stride);

// Same as gemv_row for a contiguous run of output rows.
void gemv_rows(const CpuExpertPlane& weights, std::uint32_t slot,
               const CpuActivationPlane& activation, std::uint32_t row, float* out,
               std::uint32_t out_stride, std::uint32_t begin, std::uint32_t end);

inline float silu_mul(float gate, float up) {
    return (gate / (1.0f + __builtin_expf(-gate))) * up;
}

}  // namespace shelf::quantization::mxfp4
