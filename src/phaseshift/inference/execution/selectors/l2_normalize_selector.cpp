#include <phaseshift/inference/execution/selectors/l2_normalize_selector.h>
#include <phaseshift/inference/execution/selectors/selector_limits.h>

namespace ps::execution {
namespace {

struct L2NormalizeRule {
    ::ps::runtime::ValueDType input_dtype;
    ::ps::runtime::ValueDType output_dtype;
    uint32_t features;
    uint32_t group_size;
    uint32_t min_rows;
    uint32_t max_rows;
};

constexpr L2NormalizeRule kRules[] = {
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      2048u, 128u, 1u, kOptimizedMaxRows },
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      1024u, 128u, 1u, kOptimizedMaxRows },
};

}  // namespace

L2NormalizeImplementation select_l2_normalize_implementation(
    const L2NormalizeSelectorInput& in) {
    if (in.rows == 0u)
        return L2NormalizeImplementation::Correctness;
    for (const auto& r : kRules) {
        if (r.input_dtype != in.input_dtype || r.output_dtype != in.output_dtype ||
            r.features != in.features || r.group_size != in.group_size) {
            continue;
        }
        if (in.rows >= r.min_rows && in.rows <= r.max_rows)
            return L2NormalizeImplementation::Optimized;
        return L2NormalizeImplementation::Correctness;
    }
    return L2NormalizeImplementation::Correctness;
}

}  // namespace ps::execution
