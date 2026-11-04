#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>

#include <shelf/models/common/qwen_hybrid/weight_loader.h>

namespace shelf::qwen35moe {

namespace {

constexpr uint32_t kMxScaleBlockK = 32u;
constexpr uint32_t kMxPaddedKAlignment = 64u;
constexpr uint32_t kNativeOutputTile = 16u;

uint32_t align_up_u32(uint32_t value, uint32_t alignment) noexcept {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

void add_source(MoeExpertGroupPlan& plan, uint32_t expert, MoeExpertProjection projection,
                const char* projection_name, uint32_t row_offset, uint32_t rows) {
    MoeExpertTensorSource src;
    src.name = moe_expert_tensor_name(plan.layer, expert, projection_name, false);
    src.expert = expert;
    src.projection = projection;
    src.row_offset = row_offset;
    src.rows = rows;
    plan.sources.push_back(std::move(src));
}

}  // namespace

std::size_t moe_expert_tensor_count(const Qwen35MoeTextConfig& config) {
    if (config.ffn.num_experts == 0)
        return 0;
    return config.num_hidden_layers * config.ffn.num_experts * 3u * 2u;
}

std::string moe_mlp_tensor_name(std::size_t layer, const std::string& suffix) {
    return shelf::qwen_hybrid::layer_tensor_prefix(layer) + "mlp." + suffix;
}

std::string moe_expert_tensor_name(std::size_t layer, uint32_t expert,
                                   const char* projection, bool scale) {
    return moe_mlp_tensor_name(layer, "experts." + std::to_string(expert) + "." + projection +
                                      ".weight" + (scale ? "_scale" : ""));
}

Result<MoeExpertGroupPlan> build_moe_expert_group_plan(const Qwen35MoeTextConfig& config,
                                                        std::size_t layer) {
    if (config.ffn.num_experts == 0)
        return Status::unsupported(
            "build_moe_expert_group_plan: num_experts is 0 (not a MoE config)",
            __FILE__, __LINE__);
    if (layer >= config.num_hidden_layers)
        return Status::invalid_argument(
            "build_moe_expert_group_plan: layer out of range", __FILE__, __LINE__);

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t inter = static_cast<uint32_t>(config.ffn.moe_intermediate_size);
    if (inter == 0 || hidden == 0)
        return Status::invalid_argument(
            "build_moe_expert_group_plan: moe_intermediate_size and hidden_size are required",
            __FILE__, __LINE__);

    MoeExpertGroupPlan plan;
    plan.layer = static_cast<uint32_t>(layer);
    plan.experts = static_cast<uint32_t>(config.ffn.num_experts);
    plan.scale_block_k = kMxScaleBlockK;

    plan.w13_rows = 2u * inter;
    plan.w13_cols = hidden;
    plan.w13_gate_rows = inter;
    plan.w13_up_rows = inter;
    plan.w13_k_padded = align_up_u32(plan.w13_cols, kMxPaddedKAlignment);

    plan.w2_rows = hidden;
    plan.w2_cols = inter;
    plan.w2_k_padded = align_up_u32(plan.w2_cols, kMxPaddedKAlignment);

    if ((plan.w13_rows % kNativeOutputTile) != 0u || (plan.w2_rows % kNativeOutputTile) != 0u)
        return Status::unsupported(
            "build_moe_expert_group_plan: expert output rows must be a multiple of 16",
            __FILE__, __LINE__);

    const uint32_t w13_blocks = plan.w13_k_padded / kMxScaleBlockK;
    const uint32_t w2_blocks = plan.w2_k_padded / kMxScaleBlockK;
    plan.w13_codes_row_stride_bytes = w13_blocks * 256u;
    plan.w13_scales_row_stride_bytes = w13_blocks;
    plan.w2_codes_row_stride_bytes = w2_blocks * 256u;
    plan.w2_scales_row_stride_bytes = w2_blocks;

    plan.w13_codes_batch_stride_bytes =
        (static_cast<uint64_t>(plan.w13_rows) / kNativeOutputTile) *
        plan.w13_codes_row_stride_bytes;
    plan.w13_scales_batch_stride_bytes =
        static_cast<uint64_t>(plan.w13_rows) * plan.w13_scales_row_stride_bytes;
    plan.w2_codes_batch_stride_bytes =
        (static_cast<uint64_t>(plan.w2_rows) / kNativeOutputTile) *
        plan.w2_codes_row_stride_bytes;
    plan.w2_scales_batch_stride_bytes =
        static_cast<uint64_t>(plan.w2_rows) * plan.w2_scales_row_stride_bytes;

    plan.w13_codes_bytes = plan.w13_codes_batch_stride_bytes * plan.experts;
    plan.w13_scales_bytes = plan.w13_scales_batch_stride_bytes * plan.experts;
    plan.w2_codes_bytes = plan.w2_codes_batch_stride_bytes * plan.experts;
    plan.w2_scales_bytes = plan.w2_scales_batch_stride_bytes * plan.experts;

    plan.w13_row_ref_bytes_per_expert = plan.w13_rows;
    plan.w2_row_ref_bytes_per_expert = plan.w2_rows;

    plan.sources.reserve(static_cast<std::size_t>(plan.experts) * 3u);
    for (uint32_t e = 0; e < plan.experts; ++e) {
        add_source(plan, e, MoeExpertProjection::Down, "down_proj", 0u, plan.w2_rows);
        add_source(plan, e, MoeExpertProjection::Gate, "gate_proj", 0u, inter);
        add_source(plan, e, MoeExpertProjection::Up, "up_proj", inter, inter);
    }
    return plan;
}

}  // namespace shelf::qwen35moe
