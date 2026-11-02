#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kEagle4AttentionThreads = 128u;
inline constexpr uint32_t kEagle4AttentionHeadDim = 128u;

hipError_t launch_eagle4_ring_attention_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ring,
    const bf16_t* v_ring,
    uint32_t ring_capacity,
    uint32_t kv_features,
    uint32_t block_position_start,
    uint32_t rows,
    uint32_t q_heads,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    bf16_t* output,
    uint32_t output_row_stride,
    hipStream_t stream);

hipError_t launch_eagle4_tree_attention_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ring,
    const bf16_t* v_ring,
    uint32_t ring_capacity,
    uint32_t kv_features,
    const bf16_t* k_round,
    const bf16_t* v_round,
    uint32_t round_capacity,
    uint32_t committed_end,
    uint32_t row_position,
    uint32_t rows,
    uint32_t q_heads,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    const int32_t* ancestor_offsets,
    const int32_t* ancestor_nodes,
    bf16_t* output,
    uint32_t output_row_stride,
    hipStream_t stream);

}
