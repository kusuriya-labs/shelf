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

// One 16-row activation tile as the MXFP4 W4A8 producer writes it: E4M3 codes in the
// gfx1201 fragment layout with every 128-K group's E8M0 exponent folded into the code
// relative to the row's maximum group exponent, and the row reference 2^ref in the row's
// first scale slot. The grouped kernel applies 2^ref once in its epilogue, so the CPU path
// multiplies `row_scale` the same way.
struct CpuActivationTile {
    const std::uint8_t* codes = nullptr;
    const float* row_scale = nullptr;
    std::uint32_t k_padded = 0;
};

// Computes output rows [begin, end) for the expert `slot` and the activation row `lane` of
// the tile. `out_stride` is in floats.
void gemv_rows(const CpuExpertPlane& weights, std::uint32_t slot,
               const CpuActivationTile& activation, std::uint32_t lane, float* out,
               std::uint32_t out_stride, std::uint32_t begin, std::uint32_t end);

// Same as gemv_rows for every output row of the expert.
void gemv_row(const CpuExpertPlane& weights, std::uint32_t slot,
              const CpuActivationTile& activation, std::uint32_t lane, float* out,
              std::uint32_t out_stride);

inline float silu_mul(float gate, float up) {
    return (gate / (1.0f + __builtin_expf(-gate))) * up;
}

}  // namespace shelf::quantization::mxfp4
