#include <shelf/inference/execution/selectors/embedding_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>

#include <array>

namespace shelf::execution {
namespace {

const std::array<EmbeddingRule, 2> kRules = {{
    {EmbeddingStorage::Bf16, 2560u, 1u, kOptimizedMaxRows},
    {EmbeddingStorage::Bf16, 5120u, 1u, kOptimizedMaxRows},
}};

}  // namespace

EmbeddingImplementation select_embedding_implementation(const EmbeddingSelectorInput& in) {
    if (in.rows == 0u || in.hidden_size == 0u) return EmbeddingImplementation::Correctness;
    for (const auto& r : kRules) {
        if (r.storage == in.storage && r.hidden_size == in.hidden_size &&
            in.rows >= r.min_rows && in.rows <= r.max_rows)
            return EmbeddingImplementation::Optimized;
    }
    return EmbeddingImplementation::Correctness;
}

}  // namespace shelf::execution
