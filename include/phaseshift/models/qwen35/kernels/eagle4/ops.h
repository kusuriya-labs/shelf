#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

constexpr uint32_t kEagle4OpsThreads = 128u;

hipError_t launch_eagle4_tap_norm_concat_bf16(
    const bf16_t* tap0,
    const bf16_t* tap1,
    const bf16_t* tap2,
    const bf16_t* weight0,
    const bf16_t* weight1,
    const bf16_t* weight2,
    uint32_t tap_features,
    uint32_t rows,
    bf16_t* output,
    float eps,
    hipStream_t stream);

hipError_t launch_eagle4_layer_input_norm_bf16(
    const bf16_t* embed,
    const bf16_t* hidden,
    const bf16_t* embed_weight,
    const bf16_t* hidden_weight,
    uint32_t features,
    uint32_t rows,
    bf16_t* output,
    float eps,
    hipStream_t stream);

hipError_t launch_eagle4_add_bf16(
    const bf16_t* lhs,
    const bf16_t* rhs,
    bf16_t* output,
    std::size_t count,
    hipStream_t stream);

hipError_t launch_eagle4_gemm_fallback_bf16(
    const bf16_t* weight,
    uint32_t out_features,
    uint32_t k,
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    hipStream_t stream);

hipError_t launch_eagle4_argmax_bf16(
    const bf16_t* logits,
    uint32_t rows,
    uint32_t vocab,
    int32_t* out_ids,
    hipStream_t stream);

hipError_t launch_eagle4_rope_uniform_bf16(
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    uint32_t head_dim,
    uint32_t position,
    float theta,
    hipStream_t stream);

hipError_t launch_eagle4_round_kv_append_bf16(
    const bf16_t* k_src,
    const bf16_t* v_src,
    bf16_t* k_round,
    bf16_t* v_round,
    uint32_t rows,
    uint32_t kv_features,
    uint32_t round_capacity,
    const int32_t* row_nodes,
    hipStream_t stream);

hipError_t launch_eagle4_gather_rows_bf16(
    const bf16_t* src,
    uint32_t src_row_stride,
    bf16_t* dst,
    uint32_t dst_row_stride,
    const int32_t* src_rows,
    uint32_t rows,
    uint32_t features,
    hipStream_t stream);

}  // namespace ps::kernel
