#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/kernels/dflash2/batch.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

hipError_t launch_dflash2_attention_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ctx,
    const bf16_t* v_ctx,
    uint32_t context_rows,
    uint32_t context_row_stride,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t block_rows,
    uint32_t noise_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t context_position_start,
    uint32_t block_position_start,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream);

hipError_t launch_dflash2_attention_ring_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ring,
    const bf16_t* v_ring,
    uint32_t ring_capacity,
    uint32_t context_rows,
    uint32_t context_start_position,
    uint32_t context_row_stride,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t block_rows,
    uint32_t noise_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t block_position_start,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream);

hipError_t launch_dflash2_attention_ring_gqa_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ring,
    const bf16_t* v_ring,
    uint32_t ring_capacity,
    uint32_t context_rows,
    uint32_t context_start_position,
    uint32_t context_row_stride,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t block_rows,
    uint32_t noise_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t block_position_start,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream);

hipError_t launch_dflash2_attention_ring_requests_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const DFlash2RequestDescriptor* requests,
    uint32_t request_count,
    uint32_t max_block_rows,
    uint32_t max_keys,
    uint32_t ring_capacity,
    uint64_t ring_layer_offset,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t kv_features,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream);

hipError_t launch_dflash2_attention_ring_gqa_requests_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const DFlash2RequestDescriptor* requests,
    uint32_t request_count,
    uint32_t max_block_rows,
    uint32_t max_keys,
    uint32_t ring_capacity,
    uint64_t ring_layer_offset,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t kv_features,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream);

}
