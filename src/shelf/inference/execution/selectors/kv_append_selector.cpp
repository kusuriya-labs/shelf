#include <shelf/inference/execution/selectors/kv_append_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>

namespace shelf::execution {
namespace {

struct KvAppendRule {
    ::shelf::kernel::KVCacheDType kv_dtype;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t page_tokens;
    uint32_t min_rows;
    uint32_t max_rows;
};

constexpr uint32_t kMeasuredMaxRows = kOptimizedMaxRows;

constexpr KvAppendRule kKvAppendRules[] = {
    { ::shelf::kernel::KVCacheDType::BF16, 4, 256, 16, 1, kMeasuredMaxRows },
    { ::shelf::kernel::KVCacheDType::FP8_E4M3, 4, 256, 16, 1, kMeasuredMaxRows },
    { ::shelf::kernel::KVCacheDType::BF16, 2, 256, 16, 1, kMeasuredMaxRows },
    { ::shelf::kernel::KVCacheDType::FP8_E4M3, 2, 256, 16, 1, kMeasuredMaxRows },
};

}  // namespace

KvAppendImplementation select_kv_append_implementation(const KvAppendSelectorInput& input) {
    if (input.rows == 0u)
        return KvAppendImplementation::Correctness;

    for (const auto& rule : kKvAppendRules) {
        if (rule.kv_dtype != input.kv_dtype
            || rule.kv_heads != input.kv_heads
            || rule.head_dim != input.head_dim
            || rule.page_tokens != input.page_tokens) {
            continue;
        }
        if (input.rows >= rule.min_rows && input.rows <= rule.max_rows) {
            return KvAppendImplementation::Optimized;
        }
        return KvAppendImplementation::Correctness;
    }

    return KvAppendImplementation::Correctness;
}

}  // namespace shelf::execution
