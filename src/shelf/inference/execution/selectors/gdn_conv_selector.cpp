#include <shelf/inference/execution/selectors/gdn_conv_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>

namespace shelf::execution {
namespace {

struct GdnConvRule {
    ::shelf::runtime::ValueDType input_dtype;
    ::shelf::runtime::ValueDType output_dtype;
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
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      8192u, 3u, 1u, kMeasuredMaxRows, kMeasuredMaxRequests, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      10240u, 3u, 1u, kMeasuredMaxRows, kMeasuredMaxRequests, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      5120u, 3u, 1u, kMeasuredMaxRows, kMeasuredMaxRequests, kMeasuredMaxRows },
    // Qwen3.6-35B-A3B MoE at tp_size 2: local conv features (Q and K each halved, V halved).
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      4096u, 3u, 1u, kMeasuredMaxRows, kMeasuredMaxRequests, kMeasuredMaxRows },
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

}  // namespace shelf::execution
