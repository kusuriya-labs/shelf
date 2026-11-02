#include <phaseshift/inference/execution/selectors/gdn_conv_selector.h>
#include <phaseshift/inference/execution/selectors/selector_limits.h>

namespace ps::execution {
namespace {

struct GdnConvRule {
    ::ps::runtime::ValueDType input_dtype;
    ::ps::runtime::ValueDType output_dtype;
    uint32_t conv_dim;
    uint32_t history;
    uint32_t min_rows;
    uint32_t max_rows;
    uint32_t max_requests;
    uint32_t max_request_rows;
};

constexpr uint32_t kMeasuredMaxRows = kOptimizedMaxRows;
constexpr uint32_t kMeasuredMaxRequests = kOptimizedMaxRows;

constexpr GdnConvRule kRules[] = {
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      8192u, 3u, 1u, kMeasuredMaxRows, kMeasuredMaxRequests, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      10240u, 3u, 1u, kMeasuredMaxRows, kMeasuredMaxRequests, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      5120u, 3u, 1u, kMeasuredMaxRows, kMeasuredMaxRequests, kMeasuredMaxRows },
};

}  // namespace

GdnConvImplementation select_gdn_conv_implementation(const GdnConvSelectorInput& in) {
    if (in.rows == 0u || in.num_requests == 0u || in.max_request_rows == 0u)
        return GdnConvImplementation::Correctness;
    for (const auto& r : kRules) {
        if (r.input_dtype != in.input_dtype || r.output_dtype != in.output_dtype ||
            r.conv_dim != in.conv_dim || r.history != in.history) {
            continue;
        }
        if (in.rows < r.min_rows || in.rows > r.max_rows) return GdnConvImplementation::Correctness;
        if (in.num_requests > r.max_requests) return GdnConvImplementation::Correctness;
        if (in.max_request_rows > r.max_request_rows) return GdnConvImplementation::Correctness;
        return GdnConvImplementation::Optimized;
    }
    return GdnConvImplementation::Correctness;
}

}  // namespace ps::execution
