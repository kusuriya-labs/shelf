// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/attention/rope.h.)
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/backend/sycl/sycl_api.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

using shelf::bf16_t;

inline constexpr uint32_t kRopePairBlock = 128u;
inline constexpr uint32_t kRopePairHeadDim = 256u;
inline constexpr uint32_t kRopePairRotary = 64u;
inline constexpr uint32_t kRopeInvFreqTable = 32u;

::shelf::Status launch_rope_inv_freq_init(
    double* inv_freq, uint32_t half, float theta, const ::shelf::sycl::Stream& stream);

::shelf::Status launch_rope_inv_freq_yarn_init(
    double* inv_freq, uint32_t half, float theta, float factor,
    uint32_t original_max_position_embeddings, float beta_fast, float beta_slow,
    bool truncate, const ::shelf::sycl::Stream& stream);

::shelf::Status launch_rope_f32_bf16_generic(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    const ::shelf::sycl::Stream& stream);

::shelf::Status launch_rope_f32_bf16_pair(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    const ::shelf::sycl::Stream& stream);

::shelf::Status launch_rope_f32_bf16(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    const ::shelf::sycl::Stream& stream);

} // namespace shelf::syclk
