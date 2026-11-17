#pragma once
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

// Decode-shaped FP8 E4M3 lm_head. Weights are row-major [vocab][k] codes with one
// FP16 scale per 32-element block ([vocab][k/32]); the hidden rows are BF16 and the
// logits are F32. The weight is streamed once per call, so `rows` up to 16 adds no
// weight traffic: each vocab row accumulates every hidden row from one read.
hipError_t launch_lm_head_fp8_w32(
    const bf16_t* hidden, uint32_t hidden_row_stride,
    const uint8_t* weight_codes, const fp16_t* weight_scales,
    float* logits, uint32_t logits_row_stride,
    uint32_t rows, uint32_t vocab, uint32_t k, hipStream_t stream);

// Quantizes a BF16 [rows][k] matrix into E4M3 codes plus one FP16 scale per
// 32-element block. Writes `rows * k` code bytes and `rows * k / 32` scales.
hipError_t launch_quantize_bf16_to_fp8_w32(
    const bf16_t* src, uint32_t src_row_stride,
    uint8_t* codes, fp16_t* scales,
    uint32_t rows, uint32_t k, hipStream_t stream);

}  // namespace shelf::kernel
