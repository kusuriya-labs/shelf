#include <shelf/inference/execution/selectors/rope_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>

namespace ps::execution {
namespace {

struct RopeRule {
    ::ps::runtime::ValueDType input_dtype;
    ::ps::runtime::ValueDType output_dtype;
    uint32_t features;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t min_rows;
    uint32_t max_rows;
};

constexpr uint32_t kMeasuredMaxRows = kOptimizedMaxRows;

constexpr RopeRule kRules[] = {
    { ::ps::runtime::ValueDType::F32, ::ps::runtime::ValueDType::BF16,
      1024u, 256u, 64u, 1u, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::F32, ::ps::runtime::ValueDType::BF16,
      4096u, 256u, 64u, 1u, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::F32, ::ps::runtime::ValueDType::BF16,
      6144u, 256u, 64u, 1u, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::F32, ::ps::runtime::ValueDType::BF16,
      3072u, 256u, 64u, 1u, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::F32, ::ps::runtime::ValueDType::BF16,
      512u, 256u, 64u, 1u, kMeasuredMaxRows },
};

}  // namespace

RopeImplementation select_rope_implementation(const RopeSelectorInput& in) {
    if (in.rows == 0u)
        return RopeImplementation::Correctness;
    for (const auto& r : kRules) {
        if (r.input_dtype != in.input_dtype || r.output_dtype != in.output_dtype ||
            r.features != in.features || r.head_dim != in.head_dim ||
            r.rotary_dim != in.rotary_dim) {
            continue;
        }
        if (in.rows >= r.min_rows && in.rows <= r.max_rows)
            return RopeImplementation::Optimized;
        return RopeImplementation::Correctness;
    }
    return RopeImplementation::Correctness;
}

}  // namespace ps::execution
