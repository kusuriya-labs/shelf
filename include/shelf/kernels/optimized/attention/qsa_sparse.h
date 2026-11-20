#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

inline constexpr uint32_t kQsaSparseHeadDim = 256u;
inline constexpr uint32_t kQsaSparseBlockTokens = 4u;
inline constexpr uint32_t kQsaSparseQPerKv = 6u;
inline constexpr uint32_t kQsaSparseQPerKvWide = 12u;

uint64_t qsa_sparse_partial_bytes(uint32_t rows, uint32_t kv_heads, uint32_t splits,
                                  uint32_t q_per_kv = kQsaSparseQPerKv);

uint32_t qsa_sparse_splits(uint32_t rows, uint32_t kv_heads, uint64_t workspace_bytes,
                           uint32_t q_per_kv = kQsaSparseQPerKv);

hipError_t launch_qsa_sparse_attention(const PagedAttentionCommonArgs& common,
                                       const bf16_t* k_pool, const bf16_t* v_pool,
                                       float* partials, uint32_t splits,
                                       hipStream_t stream,
                                       uint32_t q_per_kv = kQsaSparseQPerKv);

hipError_t launch_qsa_sparse_attention_fp8(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool, const uint8_t* v_pool,
    const float* k_scales, const float* v_scales,
    float* partials, uint32_t splits, hipStream_t stream,
    uint32_t q_per_kv = kQsaSparseQPerKv);

hipError_t launch_qsa_sparse_attention_octave(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool, const uint8_t* v_pool,
    const float* levels, OctaveKVCacheFormat format,
    float* partials, uint32_t splits, hipStream_t stream,
    uint32_t q_per_kv = kQsaSparseQPerKv);

hipError_t launch_qsa_sparse_reduce(const PagedAttentionCommonArgs& common,
                                    const float* partials, uint32_t splits,
                                    hipStream_t stream,
                                    uint32_t q_per_kv = kQsaSparseQPerKv);

}  // namespace shelf::kernel
