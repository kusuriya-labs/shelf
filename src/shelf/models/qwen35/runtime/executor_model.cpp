#include <shelf/models/qwen35/runtime/executor.h>

#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/qwen35_model.h>

#include <string>

namespace shelf {
namespace qwen35 {
namespace runtime {
namespace {

Status validate_geometry(const shelf::qwen_hybrid::QwenHybridConfig& tc,
                         const ::shelf::runtime::TensorParallelContext* tp,
                         const ::shelf::qwen_hybrid::PipelineStage& stage,
                         const GdnStatePoolLayout& actual_gdn,
                         const PagedKVPool& kv_pool) {
    const GdnStatePoolLayout expected_storage = make_qwen35_gdn_state_layout_local(
        tc, qwen35_stage_gdn_layers(tc, stage),
        tp != nullptr ? tp->local_gdn_key_heads
                      : static_cast<std::uint32_t>(tc.linear_num_key_heads),
        tp != nullptr ? tp->local_gdn_value_heads
                      : static_cast<std::uint32_t>(tc.linear_num_value_heads));
    const GdnStatePoolLayout& expected_gdn = expected_storage;
    const bool empty_gdn =
        actual_gdn.num_gdn_states == 0u && qwen35_stage_gdn_layers(tc, stage) == 0u;
    if (!empty_gdn &&
        (actual_gdn.num_gdn_states != expected_gdn.num_gdn_states ||
         actual_gdn.conv_dim != expected_gdn.conv_dim ||
         actual_gdn.conv_history != expected_gdn.conv_history ||
         actual_gdn.num_v_heads != expected_gdn.num_v_heads ||
         actual_gdn.head_k != expected_gdn.head_k ||
         actual_gdn.head_v != expected_gdn.head_v)) {
        std::string msg =
            "GDN state pool geometry does not match model config: actual("
            + std::to_string(actual_gdn.num_gdn_states) + ","
            + std::to_string(actual_gdn.conv_dim) + ","
            + std::to_string(actual_gdn.conv_history) + ","
            + std::to_string(actual_gdn.num_v_heads) + ","
            + std::to_string(actual_gdn.head_k) + ","
            + std::to_string(actual_gdn.head_v) + ") expected("
            + std::to_string(expected_gdn.num_gdn_states) + ","
            + std::to_string(expected_gdn.conv_dim) + ","
            + std::to_string(expected_gdn.conv_history) + ","
            + std::to_string(expected_gdn.num_v_heads) + ","
            + std::to_string(expected_gdn.head_k) + ","
            + std::to_string(expected_gdn.head_v) + ")";
        return Status::invalid_argument(msg.c_str(), __FILE__, __LINE__);
    }
    const uint32_t expected_attn_layers = qwen35_stage_attention_layers(tc, stage);
    const uint32_t expected_kv_heads = (tp != nullptr && tp->tp_size > 1)
        ? tp->local_key_value_heads
        : static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t expected_head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    if (kv_pool.num_attention_layers() != expected_attn_layers ||
        kv_pool.kv_heads() != expected_kv_heads ||
        kv_pool.head_dim() != expected_head_dim) {
        std::string msg =
            "KV pool geometry does not match model config: actual("
            + std::to_string(kv_pool.num_attention_layers()) + ","
            + std::to_string(kv_pool.kv_heads()) + ","
            + std::to_string(kv_pool.head_dim()) + ") expected("
            + std::to_string(expected_attn_layers) + ","
            + std::to_string(expected_kv_heads) + ","
            + std::to_string(expected_head_dim) + ")";
        return Status::invalid_argument(msg.c_str(), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Result<Executor> create_model_executor(
    const Qwen35Model& model,
    SequenceSlotPool& sequence_pool,
    GdnStatePool& gdn_state_pool,
    PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const ExecutorConfig& config) {

    std::vector<LayerOverrideSource> overrides =
        read_layer_override_sources(static_cast<uint32_t>(model.text_config().hidden_size));
    std::vector<uint32_t> override_layers;
    override_layers.reserve(overrides.size());
    for (const auto& ov : overrides) override_layers.push_back(ov.layer);

    Qwen35LowerOptions lower_options;
    lower_options.rope_profile = config.rope_profile;
    lower_options.stage = config.stage;
    lower_options.emit_normed_token_hidden = config.emit_normed_token_hidden;
    lower_options.hidden_taps = std::span<const uint32_t>(
        config.target_hidden_taps.data(), config.target_hidden_tap_count);
    lower_options.tp = config.tp;
    lower_options.tp_combine_f32 = config.tp_combine_f32 != 0;
    lower_options.override_layers =
        std::span<const uint32_t>(override_layers.data(), override_layers.size());

    auto lower_res =
        lower_qwen35_to_primitives(model.text_config(), model.weights(), lower_options);
    if (!lower_res.ok()) return lower_res.status();

    ExecutorModelInput input;
    input.model.config = &model.text_config();
    input.model.embed_tokens = &model.weights().embed_tokens;
    input.lowered = lower_res.release();
    input.layer_overrides = std::move(overrides);

    ExecutorConfig exec_config = config;
    exec_config.validate_geometry = validate_geometry;
    return create_executor(std::move(input), sequence_pool, gdn_state_pool, kv_pool, arena,
                           exec_config);
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace shelf
