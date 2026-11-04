#include <shelf/models/qwen35moe/runtime/executor_footprint.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/runtime/graph/grouped_row_shape.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/runtime/program/moe_plan_scratch.h>

#include <algorithm>

namespace shelf::qwen35moe {

namespace {

constexpr std::size_t kAlignBytes = 256u;

std::size_t align_up(std::size_t bytes) noexcept {
    return ((bytes + kAlignBytes - 1u) / kAlignBytes) * kAlignBytes;
}

std::size_t bf16_plane(std::size_t rows, std::size_t features) noexcept {
    return align_up(rows * features * sizeof(std::uint16_t));
}

}  // namespace

std::size_t moe_padded_row_count(std::size_t max_batch_rows) noexcept {
    std::size_t rows = max_batch_rows;
    auto bucket = ::shelf::runtime::resolve_row_bucket(static_cast<std::uint32_t>(rows));
    if (bucket.ok()) {
        const std::uint32_t limit = ::shelf::runtime::row_bucket_limit(bucket.value());
        if (limit > rows) rows = limit;
    }
    return rows;
}

std::size_t moe_executor_grouped_bytes(const Qwen35MoeTextConfig& config,
                                       std::size_t max_batch_rows) noexcept {
    const std::size_t tokens = moe_padded_row_count(max_batch_rows);
    if (tokens == 0u || config.ffn.num_experts == 0u || config.ffn.num_experts_per_tok == 0u)
        return 0u;

    auto plan_result = build_moe_expert_group_plan(config, 0);
    if (!plan_result.ok()) return 0u;
    const MoeExpertGroupPlan& plan = plan_result.value();

    const ::shelf::runtime::GroupedRowShape shape{
        static_cast<std::uint32_t>(config.ffn.num_experts),
        static_cast<std::uint32_t>(config.ffn.num_experts_per_tok)};
    const std::size_t group_rows = ::shelf::runtime::grouped_row_capacity(
        static_cast<std::uint32_t>(tokens), shape);

    std::size_t total = 0u;
    total += bf16_plane(group_rows, plan.w13_rows);
    total += bf16_plane(group_rows, plan.w2_cols);
    total += bf16_plane(group_rows, plan.w2_rows);
    {
        const ::shelf::runtime::Int8ActivationWorkspaceLayout fused =
            ::shelf::runtime::Int8ActivationWorkspaceLayout::make(plan.w13_k_padded,
                                                                  group_rows);
        total += align_up(fused.total_bytes);
        const ::shelf::runtime::Int8ActivationWorkspaceLayout down =
            ::shelf::runtime::Int8ActivationWorkspaceLayout::make(plan.w2_k_padded, group_rows);
        total += align_up(down.total_bytes);
    }
    total += align_up(::shelf::runtime::MoePlanScratch::make(
                          static_cast<std::uint32_t>(tokens), shape.experts, shape.top_k)
                          .bytes());
    return total;
}

}  // namespace shelf::qwen35moe
