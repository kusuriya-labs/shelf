#pragma once
#include <shelf/runtime/graph/value_type.h>
#include <shelf/kernels/kv_types.h>
#include <cstdint>

namespace shelf::execution {

enum class PagedAttentionImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct PagedAttentionSelectorInput {
    ::shelf::kernel::KVCacheDType kv_dtype = ::shelf::kernel::KVCacheDType::BF16;
    ::shelf::runtime::ValueDType output_dtype = ::shelf::runtime::ValueDType::BF16;
    uint32_t rows = 0;
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t page_tokens = 0;
    uint32_t max_visible_tokens = 0;
};

PagedAttentionImplementation
select_paged_attention_implementation(const PagedAttentionSelectorInput& input);

}  // namespace shelf::execution
