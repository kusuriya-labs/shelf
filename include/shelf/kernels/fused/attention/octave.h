#pragma once

#include <shelf/kernels/kv_types.h>
#include <shelf/kernels/optimized/attention/paged_attention.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_attention_paged_octave_split(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool,
    const uint8_t* v_pool,
    const float* levels,
    OctaveKVCacheFormat format,
    float* partials,
    uint32_t splits,
    hipStream_t stream);

hipError_t launch_attention_paged_octave_qsa_split(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool,
    const uint8_t* v_pool,
    const float* levels,
    OctaveKVCacheFormat format,
    float* partials,
    uint32_t splits,
    hipStream_t stream);

hipError_t launch_attention_paged_octave_reduce(
    const PagedAttentionCommonArgs& common,
    const float* partials,
    uint32_t splits,
    hipStream_t stream);

}  // namespace shelf::kernel
