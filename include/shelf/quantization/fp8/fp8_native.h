#pragma once

#include <shelf/quantization/fp8/fp8.h>
#include <cstdint>
#include <memory>
#include <span>

namespace ps::quantization::fp8 {

// GPU-native (WMMA fragment-order) payload for a single FP8 K128N128
// weight matrix. Codes are tiled in the exact 16-output x 64-K fragment-block
// order the gfx1201 FP8 WMMA consumer reads (see fp8_frag.h); the FP32 scales
// stay row-major [ceil(N/128)][Kp/128]. Padded output rows are zero-filled so
// the guarded consumer can load full 16-row tiles.
struct Fp8NativeView {
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t codes_row_stride_bytes = 0;
    uint64_t scale_n = 0;
    uint64_t scale_k = 0;
    const uint8_t* codes = nullptr;
    uint64_t codes_bytes = 0;
    const float* scales = nullptr;
    uint64_t scales_bytes = 0;
};

struct Fp8NativeHost {
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t codes_row_stride_bytes = 0;
    uint64_t scale_n = 0;
    uint64_t scale_k = 0;
    std::unique_ptr<uint8_t[]> codes;
    uint64_t codes_bytes = 0;
    uint64_t codes_capacity = 0;
    std::unique_ptr<float[]> scales;
    uint64_t scale_count = 0;
    uint64_t scales_capacity = 0;

    std::span<const uint8_t> codes_span() const noexcept;
    std::span<const float> scales_span() const noexcept;
    Fp8NativeView view() const;
};

// Batched canonical matrices are not supported: 128x128 scale tiles must not
// straddle a matrix boundary, and the consumer addresses one matrix at a time.
// Input contract: the canonical view is the whole logical tensor owned by the
// caller; this function never partitions and never sees tp_rank / tp_size.
bool preshuffle_fp8_native(const Fp8BlockCanonicalView& v,
                                    Fp8NativeHost& out);

}  // namespace ps::quantization::fp8
