#pragma once
#include <phaseshift/runtime/graph/value_type.h>
#include <phaseshift/kernels/kv_types.h>
#include <cstdint>

namespace ps::execution {

enum class PagedAttentionImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct PagedAttentionSelectorInput {
    ::ps::kernel::KVCacheDType kv_dtype = ::ps::kernel::KVCacheDType::BF16;
    ::ps::runtime::ValueDType output_dtype = ::ps::runtime::ValueDType::BF16;
    uint32_t rows = 0;
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t page_tokens = 0;
    uint32_t max_visible_tokens = 0;
};

PagedAttentionImplementation
select_paged_attention_implementation(const PagedAttentionSelectorInput& input);

}  // namespace ps::execution
