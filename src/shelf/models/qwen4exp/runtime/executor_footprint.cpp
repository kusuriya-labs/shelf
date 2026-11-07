#include <shelf/models/qwen4exp/runtime/executor_footprint.h>

#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/runtime/graph/grouped_row_shape.h>
#include <shelf/runtime/program/moe_plan_scratch.h>

#include <algorithm>

namespace shelf {
namespace qwen4exp {

namespace {

std::size_t padded_rows(std::size_t rows) noexcept {
    auto bucket = ::shelf::runtime::resolve_row_bucket(
        static_cast<std::uint32_t>(rows));
    if (bucket.ok()) {
        const std::uint32_t limit =
            ::shelf::runtime::row_bucket_limit(bucket.value());
        if (limit > rows) rows = limit;
    }
    return rows;
}

}  // namespace

std::size_t mtp_grouped_executor_bytes(const Qwen4ExpTextConfig& config,
                                       std::size_t max_batch_rows,
                                       std::uint32_t max_context_tokens) noexcept {
    const std::uint32_t experts = static_cast<std::uint32_t>(config.ffn.num_experts);
    const std::uint32_t top_k =
        static_cast<std::uint32_t>(config.ffn.num_experts_per_tok);
    const std::uint32_t inter =
        static_cast<std::uint32_t>(config.ffn.moe_intermediate_size);
    const std::uint32_t hidden = static_cast<std::uint32_t>(config.hidden_size);
    if (experts == 0u || top_k == 0u || inter == 0u || hidden == 0u ||
        max_batch_rows == 0u) {
        return 0u;
    }
    const std::uint32_t tokens = static_cast<std::uint32_t>(max_batch_rows);
    const std::uint32_t group_rows = ::shelf::runtime::grouped_row_capacity(
        tokens, ::shelf::runtime::GroupedRowShape{experts, top_k});
    std::size_t total = static_cast<std::size_t>(group_rows) * inter * 2u * 2u;
    total += static_cast<std::size_t>(group_rows) * hidden * 2u;

    const std::uint32_t ratio = config.indexer.compress_ratio != 0u
                                    ? static_cast<std::uint32_t>(config.indexer.compress_ratio)
                                    : 1u;
    const std::size_t blocks =
        (static_cast<std::size_t>(max_context_tokens) + ratio - 1u) / ratio;
    const std::size_t mask_bytes = ((blocks + 31u) / 32u) * 4u;
    const std::size_t rows = padded_rows(tokens);
    total += rows * ((mask_bytes + 15u) / 16u) * 16u;
    total += ::shelf::runtime::MoePlanScratch::make(tokens, experts, top_k).bytes();
    return total;
}

}
}
