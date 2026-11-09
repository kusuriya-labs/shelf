#include <shelf/inference/execution/selectors/paged_attention_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>

namespace shelf::execution {
namespace {

struct PagedAttentionRule {
    ::shelf::kernel::KVCacheDType kv_dtype;
    ::shelf::runtime::ValueDType output_dtype;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t page_tokens;
    uint32_t min_rows;
    uint32_t max_rows;
    uint32_t min_visible_tokens;
    uint32_t max_visible_tokens;
};

constexpr uint32_t kMeasuredMaxRows = kOptimizedMaxRows;
constexpr uint32_t kMeasuredMaxVisible = 32768;
constexpr uint32_t kMeasuredMaxVisible24 = 524288;
// (q_heads, kv_heads) = (16, 2) is the MoE local geometry (q_per_kv 8). Both KV dtypes are
// measured through the model's max sequence length, so there is no cliff inside the range.
constexpr uint32_t kMeasuredMaxVisible16x2 = 262144;

constexpr PagedAttentionRule kPagedAttentionRules[] = {
    { ::shelf::kernel::KVCacheDType::OCTAVE, ::shelf::runtime::ValueDType::F32,
      24, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::shelf::kernel::KVCacheDType::BF16, ::shelf::runtime::ValueDType::F32,
      16, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible },
    { ::shelf::kernel::KVCacheDType::FP8_E4M3, ::shelf::runtime::ValueDType::F32,
      16, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible },
    { ::shelf::kernel::KVCacheDType::BF16, ::shelf::runtime::ValueDType::F32,
      24, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::shelf::kernel::KVCacheDType::BF16, ::shelf::runtime::ValueDType::F32,
      12, 2, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::shelf::kernel::KVCacheDType::FP8_E4M3, ::shelf::runtime::ValueDType::F32,
      24, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::shelf::kernel::KVCacheDType::FP8_E4M3, ::shelf::runtime::ValueDType::F32,
      12, 2, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::shelf::kernel::KVCacheDType::BF16, ::shelf::runtime::ValueDType::F32,
      16, 2, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible16x2 },
    { ::shelf::kernel::KVCacheDType::FP8_E4M3, ::shelf::runtime::ValueDType::F32,
      16, 2, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible16x2 },
    // (q_heads, kv_heads) = (8, 1) is the same model at tp_size 2: the same q_per_kv 8 and half
    // the heads. The visible range is the model's max sequence length, as for (16, 2).
    // Qwen4Exp's QSA geometry is 24 query heads and 2 KV heads, replicated on both ranks.
    { ::shelf::kernel::KVCacheDType::BF16, ::shelf::runtime::ValueDType::F32,
      24, 2, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::shelf::kernel::KVCacheDType::FP8_E4M3, ::shelf::runtime::ValueDType::F32,
      24, 2, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::shelf::kernel::KVCacheDType::BF16, ::shelf::runtime::ValueDType::F32,
      8, 1, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible16x2 },
    { ::shelf::kernel::KVCacheDType::FP8_E4M3, ::shelf::runtime::ValueDType::F32,
      8, 1, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible16x2 },
    { ::shelf::kernel::KVCacheDType::BF16, ::shelf::runtime::ValueDType::F32,
      12, 1, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::shelf::kernel::KVCacheDType::FP8_E4M3, ::shelf::runtime::ValueDType::F32,
      12, 1, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
};

}  // namespace

PagedAttentionImplementation
select_paged_attention_implementation(const PagedAttentionSelectorInput& input) {
    if (input.rows == 0u)
        return PagedAttentionImplementation::Correctness;

    for (const auto& rule : kPagedAttentionRules) {
        if (rule.kv_dtype != input.kv_dtype
            || rule.output_dtype != input.output_dtype
            || rule.q_heads != input.q_heads
            || rule.kv_heads != input.kv_heads
            || rule.head_dim != input.head_dim
            || rule.page_tokens != input.page_tokens) {
            continue;
        }
        if (input.rows >= rule.min_rows && input.rows <= rule.max_rows
            && input.max_visible_tokens >= rule.min_visible_tokens
            && input.max_visible_tokens <= rule.max_visible_tokens) {
            return PagedAttentionImplementation::Optimized;
        }
        return PagedAttentionImplementation::Correctness;
    }

    return PagedAttentionImplementation::Correctness;
}

}  // namespace shelf::execution
