#include "phaseshift/inference/execution/selectors/gdn_recurrence_selector.h"
#include <phaseshift/inference/execution/selectors/selector_limits.h>

namespace ps::execution {
namespace {
struct Rule {
    uint32_t key_heads;
    uint32_t num_v_heads;
    uint32_t head_k;
    uint32_t head_v;
    uint32_t min_rows;
    uint32_t max_rows;
    uint32_t min_requests;
    uint32_t max_requests;
    uint32_t min_max_request_rows;
    uint32_t max_max_request_rows;
};
}  // namespace

GdnRecurrenceImplementation
select_gdn_recurrence_implementation(const GdnRecurrenceSelectorInput& in) {
    constexpr Rule kRule_4b{16u, 32u, 128u, 128u, 1u, kOptimizedMaxRows, 1u, kOptimizedMaxRows, 1u, kOptimizedMaxRows};
    constexpr Rule kRule_27b{16u, 48u, 128u, 128u, 1u, kOptimizedMaxRows, 1u, kOptimizedMaxRows, 1u, kOptimizedMaxRows};
    constexpr Rule kRule_27b_tp2{8u, 24u, 128u, 128u, 1u, kOptimizedMaxRows, 1u, kOptimizedMaxRows, 1u, kOptimizedMaxRows};
    const Rule rules[] = {kRule_4b, kRule_27b, kRule_27b_tp2};
    for (const auto& r : rules) {
        if (in.key_heads != r.key_heads || in.num_v_heads != r.num_v_heads ||
            in.head_k != r.head_k || in.head_v != r.head_v) {
            continue;
        }
        if (in.rows < r.min_rows || in.rows > r.max_rows) continue;
        if (in.num_requests < r.min_requests || in.num_requests > r.max_requests) continue;
        if (in.max_request_rows < r.min_max_request_rows ||
            in.max_request_rows > r.max_max_request_rows) continue;
        return GdnRecurrenceImplementation::Optimized;
    }
    return GdnRecurrenceImplementation::Correctness;
}

}  // namespace ps::execution
