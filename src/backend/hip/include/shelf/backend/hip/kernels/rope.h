// vendored from PhaseShift (own work); namespace/type adapted only.
#pragma once

#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk {

using shelf::bf16_t;

inline constexpr uint32_t kRopePairBlock = 128u;
inline constexpr uint32_t kRopePairHeadDim = 256u;
inline constexpr uint32_t kRopePairRotary = 64u;
inline constexpr uint32_t kRopeInvFreqTable = 32u;

inline constexpr const char* kRopeF32Bf16PairSymbol =
    "phaseshift_rope_f32_bf16_pair";

struct RopeF32Bf16PairArgs {
    const float* input = nullptr;
    uint32_t input_row_stride = 0;
    bf16_t* output = nullptr;
    uint32_t output_row_stride = 0;
    const uint32_t* positions = nullptr;
    const double* inv_freq = nullptr;
    uint32_t features = 0;
    float attention_factor = 1.0f;
};

static_assert(sizeof(RopeF32Bf16PairArgs) == 56);
static_assert(alignof(RopeF32Bf16PairArgs) == 8);
static_assert(offsetof(RopeF32Bf16PairArgs, input) == 0);
static_assert(offsetof(RopeF32Bf16PairArgs, input_row_stride) == 8);
static_assert(offsetof(RopeF32Bf16PairArgs, output) == 16);
static_assert(offsetof(RopeF32Bf16PairArgs, output_row_stride) == 24);
static_assert(offsetof(RopeF32Bf16PairArgs, positions) == 32);
static_assert(offsetof(RopeF32Bf16PairArgs, inv_freq) == 40);
static_assert(offsetof(RopeF32Bf16PairArgs, features) == 48);
static_assert(offsetof(RopeF32Bf16PairArgs, attention_factor) == 52);

bool rope_pair_shape_supported(
    const float* input, bf16_t* output,
    uint32_t input_row_stride, uint32_t output_row_stride,
    uint32_t features, uint32_t head_dim, uint32_t rotary_dim);

hipError_t launch_rope_inv_freq_init(
    double* inv_freq, uint32_t half, float theta, hipStream_t stream);

hipError_t launch_rope_inv_freq_yarn_init(
    double* inv_freq, uint32_t half, float theta, float factor,
    uint32_t original_max_position_embeddings, float beta_fast, float beta_slow,
    bool truncate, hipStream_t stream);

hipError_t launch_rope_f32_bf16_generic(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    hipStream_t stream);

hipError_t launch_rope_f32_bf16_pair(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    hipStream_t stream);

hipError_t launch_rope_f32_bf16(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    hipStream_t stream);

}  // namespace shelf::hipk
