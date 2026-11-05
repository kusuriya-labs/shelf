#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>

#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>

namespace shelf::qwen35moe {

namespace {

namespace QMX = ::shelf::quantization::mxfp4;

constexpr uint32_t kMxScaleBlockK = 32u;
constexpr uint32_t kMxPaddedKAlignment = 64u;
constexpr uint32_t kNativeOutputTile = QMX::mxfp4_native_output_tile;
constexpr uint32_t kNativeKGroupAlignment = kNativeOutputTile * 8u;

uint32_t align_up_u32(uint32_t value, uint32_t alignment) noexcept {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

void add_source(MoeExpertGroupPlan& plan, uint32_t expert, MoeExpertProjection projection,
                const char* projection_name, uint32_t row_offset, uint32_t rows) {
    MoeExpertTensorSource src;
    src.name = moe_expert_tensor_name(plan.mlp_prefix, expert, projection_name, false);
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

std::string moe_mlp_prefix(std::size_t layer) {
    return shelf::qwen_hybrid::layer_tensor_prefix(layer) + "mlp.";
}

std::string moe_mtp_layer_prefix() {
    return "mtp.layers.0.";
}

std::string moe_mtp_mlp_prefix() {
    return moe_mtp_layer_prefix() + "mlp.";
}

std::string moe_mlp_tensor_name(std::size_t layer, const std::string& suffix) {
    return moe_mlp_prefix(layer) + suffix;
}

std::string moe_expert_tensor_name(const std::string& mlp, uint32_t expert,
                                   const char* projection, bool scale) {
    return mlp + "experts." + std::to_string(expert) + "." + projection + ".weight" +
           (scale ? "_scale" : "");
}

std::string moe_expert_tensor_name(std::size_t layer, uint32_t expert,
                                   const char* projection, bool scale) {
    return moe_expert_tensor_name(moe_mlp_prefix(layer), expert, projection, scale);
}

Result<MoeExpertGroupPlan> build_moe_expert_group_plan(const Qwen35MoeTextConfig& config,
                                                        std::size_t layer,
                                                        uint32_t tp_size,
                                                        uint32_t tp_rank) {
    if (config.ffn.num_experts == 0)
        return Status::unsupported(
            "build_moe_expert_group_plan: num_experts is 0 (not a MoE config)",
            __FILE__, __LINE__);
    if (layer >= config.num_hidden_layers)
        return Status::invalid_argument(
            "build_moe_expert_group_plan: layer out of range", __FILE__, __LINE__);
    if (tp_size == 0 || tp_rank >= tp_size)
        return Status::invalid_argument(
            "build_moe_expert_group_plan: tp_size / tp_rank out of range", __FILE__, __LINE__);
    const uint32_t global_experts = static_cast<uint32_t>(config.ffn.num_experts);
    if (global_experts % tp_size != 0u)
        return Status::unsupported(
            "build_moe_expert_group_plan: num_experts must divide by tp_size", __FILE__,
            __LINE__);

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t inter = static_cast<uint32_t>(config.ffn.moe_intermediate_size);
    if (inter == 0 || hidden == 0)
        return Status::invalid_argument(
            "build_moe_expert_group_plan: moe_intermediate_size and hidden_size are required",
            __FILE__, __LINE__);

    MoeExpertGroupPlan plan;
    plan.layer = static_cast<uint32_t>(layer);
    plan.mlp_prefix = moe_mlp_prefix(layer);
    plan.experts = global_experts / tp_size;
    plan.expert_begin = tp_rank * plan.experts;
    plan.global_experts = global_experts;
    plan.tp_size = tp_size;
    plan.tp_rank = tp_rank;
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
    if ((plan.w13_k_padded % kNativeKGroupAlignment) != 0u ||
        (plan.w2_k_padded % kNativeKGroupAlignment) != 0u)
        return Status::unsupported(
            "build_moe_expert_group_plan: expert k_padded must be a multiple of 128",
            __FILE__, __LINE__);

    const uint32_t w13_blocks = plan.w13_k_padded / kMxScaleBlockK;
    const uint32_t w2_blocks = plan.w2_k_padded / kMxScaleBlockK;
    plan.w13_codes_row_stride_bytes = w13_blocks * QMX::mxfp4_native_block_bytes;
    plan.w13_scales_row_stride_bytes = w13_blocks;
    plan.w2_codes_row_stride_bytes = w2_blocks * QMX::mxfp4_native_block_bytes;
    plan.w2_scales_row_stride_bytes = w2_blocks;

    plan.w13_codes_batch_stride_bytes = QMX::
        mxfp4_native_codes_batch_stride_bytes(plan.w13_rows, plan.w13_k_padded);
    plan.w13_scales_batch_stride_bytes = QMX::
        mxfp4_native_scale_batch_stride_bytes(plan.w13_rows, plan.w13_k_padded);
    plan.w2_codes_batch_stride_bytes = QMX::
        mxfp4_native_codes_batch_stride_bytes(plan.w2_rows, plan.w2_k_padded);
    plan.w2_scales_batch_stride_bytes = QMX::
        mxfp4_native_scale_batch_stride_bytes(plan.w2_rows, plan.w2_k_padded);

    plan.w13_codes_bytes = plan.w13_codes_batch_stride_bytes * plan.experts;
    plan.w13_scales_bytes = plan.w13_scales_batch_stride_bytes * plan.experts;
    plan.w2_codes_bytes = plan.w2_codes_batch_stride_bytes * plan.experts;
    plan.w2_scales_bytes = plan.w2_scales_batch_stride_bytes * plan.experts;

    plan.w13_row_ref_bytes_per_expert = plan.w13_rows;
    plan.w2_row_ref_bytes_per_expert = plan.w2_rows;

    plan.sources.reserve(static_cast<std::size_t>(plan.experts) * 3u);
    for (uint32_t e = plan.expert_begin; e < plan.expert_begin + plan.experts; ++e) {
        add_source(plan, e, MoeExpertProjection::Down, "down_proj", 0u, plan.w2_rows);
        add_source(plan, e, MoeExpertProjection::Gate, "gate_proj", 0u, inter);
        add_source(plan, e, MoeExpertProjection::Up, "up_proj", inter, inter);
    }
    return plan;
}

Result<MoeExpertGroupPlan> build_moe_expert_group_plan(const Qwen35MoeTextConfig& config,
                                                        std::size_t layer) {
    return build_moe_expert_group_plan(config, layer, 1u, 0u);
}

Result<MoeExpertGroupPlan> build_moe_mtp_expert_group_plan(const Qwen35MoeTextConfig& config,
                                                            uint32_t tp_size,
                                                            uint32_t tp_rank) {
    if (config.num_hidden_layers == 0u)
        return Status::invalid_argument(
            "build_moe_mtp_expert_group_plan: the config has no layer geometry", __FILE__,
            __LINE__);
    auto plan = build_moe_expert_group_plan(config, 0u, tp_size, tp_rank);
    if (!plan.ok()) return plan;
    MoeExpertGroupPlan& out = plan.value();
    out.layer = 0u;
    out.fused_rank3 = true;
    out.mlp_prefix = moe_mtp_mlp_prefix();
    out.w13_name = out.mlp_prefix + "experts.gate_up_proj";
    out.w2_name = out.mlp_prefix + "experts.down_proj";
    out.sources.clear();
    return plan;
}

Result<MoeExpertGroupPlan> build_moe_mtp_expert_group_plan(const Qwen35MoeTextConfig& config) {
    return build_moe_mtp_expert_group_plan(config, 1u, 0u);
}

}  // namespace shelf::qwen35moe
