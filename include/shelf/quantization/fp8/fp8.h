#pragma once

#include <shelf/quantization/quant_format.h>
#include <cstdint>
#include <vector>

namespace ps::quantization::fp8 {

// FP8 E4M3 weights with FP32 scales on a 128 (N) x 128 (K) tile.
inline constexpr uint32_t kBlockN = 128;
inline constexpr uint32_t kBlockK = 128;
inline constexpr float kE4m3Max = 448.0f;

// Quantize a matrix `n` x `k` (row-major, `src_stride` floats per row; columns
// >= k are treated as zero) into codes[n][k_padded] (one E4M3 byte per element)
// and scales[ceil(n/128)][k_padded/128] (FP32 per tile).
void quantize_fp8_matrix(
    const float* src,
    uint64_t src_stride,
    uint64_t n,
    uint64_t k,
    uint64_t k_padded,
    uint8_t* codes,
    float* scales,
    uint64_t scale_block_k = kBlockK);

void dequantize_fp8_matrix(
    const uint8_t* codes,
    const float* scales,
    uint64_t n,
    uint64_t k_padded,
    float* dst,
    uint64_t scale_block_k = kBlockK);

// Canonical on-disk payload for one or more stacked matrices:
// codes[..., N, Kp] U8 and scales[..., ceil(N/128), Kp/128] F32. Batch tiles
// never straddle a matrix boundary.
struct Fp8BlockCanonicalView {
    uint64_t batch = 1;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    const uint8_t* codes = nullptr;
    uint64_t codes_bytes = 0;
    const float* scales = nullptr;
    uint64_t scales_bytes = 0;
    uint64_t scale_n = 0;
    uint64_t scale_k = 0;
    uint64_t scale_block_k = 0;

    bool validate() const;
    uint64_t effective_scale_block_k() const {
        return scale_block_k != 0 ? scale_block_k : kBlockK;
    }
};

struct Fp8BlockCanonicalStore {
    uint64_t batch = 1;
    uint64_t n = 0;
    uint64_t logical_k = 0;
    uint64_t padded_k = 0;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    uint64_t scale_block_k = 0;

    void init(uint64_t batch, uint64_t n, uint64_t logical_k, uint64_t padded_k);
    uint64_t effective_scale_block_k() const {
        return scale_block_k != 0 ? scale_block_k : kBlockK;
    }
    Fp8BlockCanonicalView view() const;
    bool quantize_matrix(uint64_t matrix_index, const float* src, uint64_t src_stride);
};

void dequantize_fp8_canonical(const Fp8BlockCanonicalView& v, std::vector<float>& out);
void dequantize_fp8_canonical_matrix(
    const Fp8BlockCanonicalView& v, uint64_t matrix_index, float* dst, uint64_t dst_stride);

}  // namespace ps::quantization::fp8
