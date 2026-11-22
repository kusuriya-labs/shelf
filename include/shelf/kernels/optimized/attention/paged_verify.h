#pragma once
#include <shelf/kernels/optimized/attention/paged_attention.h>

namespace shelf::kernel {

inline constexpr uint32_t kPagedVerifySplits = 64u;
inline constexpr uint32_t kPagedVerifyPartialFloats = 258u;

inline bool paged_verify_geometry(const PagedAttentionCommonArgs& a) {
    return a.rows >= 2u && a.rows <= 16u && a.q_heads == 24u && a.kv_heads == 4u && a.head_dim == 256u &&
           a.page_tokens == 16u && a.output_dtype == PagedAttentionOutputDType::F32 &&
           a.tree_anc_offsets == nullptr && a.tree_anc_positions == nullptr && a.block_mask == nullptr;
}

inline uint64_t paged_verify_workspace_bytes(uint32_t rows, uint32_t heads, uint32_t splits,
                                             KVCacheDType dtype) {
    const uint64_t jobs = uint64_t(rows) * heads;
    return jobs * splits * kPagedVerifyPartialFloats * sizeof(float) +
           (dtype == KVCacheDType::OCTAVE ? jobs * 256u * sizeof(uint16_t) : 0u);
}

struct PagedVerifyCache {
    const void* k;
    const void* v;
    const float* k_scale;
    const float* v_scale;
    const float* levels;
    OctaveKVCacheFormat format;
};

hipError_t launch_paged_verify_split(const PagedAttentionCommonArgs& args, KVCacheDType dtype,
                                     PagedVerifyCache cache, float* partials, uint32_t splits,
                                     hipStream_t stream);

hipError_t launch_paged_verify_reduce(const PagedAttentionCommonArgs& args, KVCacheDType dtype,
                                      const float* partials, uint32_t splits, hipStream_t stream);

}
