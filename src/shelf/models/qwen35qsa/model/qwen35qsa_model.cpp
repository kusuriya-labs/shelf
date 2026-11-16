#include <shelf/models/qwen35qsa/model/qwen35qsa_model.h>

#include <shelf/inference/state/qsa_state_pool.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/qwen35qsa/model/lower_to_primitives.h>

#include <utility>

namespace shelf {
namespace qwen35qsa {

namespace {

std::uint32_t indexer_blocks(const Qwen35QsaTextConfig& config,
                             std::uint32_t max_context_tokens) {
    const auto ratio = static_cast<std::uint32_t>(config.indexer.block_size);
    if (ratio == 0u || max_context_tokens == 0u) return 0u;
    return (max_context_tokens + ratio - 1u) / ratio;
}

::shelf::state::QsaKeyStatePoolLayout qsa_layout(const Qwen35QsaTextConfig& config,
                                                 std::uint32_t max_context_tokens) {
    const auto counts = shelf::qwen_hybrid::count_layer_topology(config);
    return ::shelf::state::QsaKeyStatePoolLayout::make(
        static_cast<std::uint32_t>(counts.attention_layers),
        static_cast<std::uint32_t>(config.indexer.head_dim),
        static_cast<std::uint32_t>(config.indexer.block_size),
        indexer_blocks(config, max_context_tokens));
}

}  // namespace

Qwen35QsaModel::Qwen35QsaModel(Qwen35QsaModelWeights&& weights,
                               Qwen35QsaTextConfig text_config)
    : weights_(std::move(weights)), text_config_(std::move(text_config)) {}

Result<Qwen35QsaModel> Qwen35QsaModel::load_from_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35QsaLoadOptions& options) {
    auto config = read_qwen35qsa_text_config(model_dir);
    if (!config.ok()) return config.status();
    auto weights = load_qwen35qsa_weights(model_dir, arena, stream, options);
    if (!weights.ok()) return weights.status();
    Status valid = validate_qwen35qsa_weights(weights.value(), config.value());
    if (!valid.ok()) return valid;
    return Qwen35QsaModel(weights.release(), config.release());
}

Result<Qwen35QsaModel> Qwen35QsaModel::load_tensor_parallel_from_safetensors(
    const std::string& model_dir,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35QsaLoadOptions& options) {
    auto config = read_qwen35qsa_text_config(model_dir);
    if (!config.ok()) return config.status();
    auto weights = load_qwen35qsa_weights_tensor_parallel_rank(
        model_dir, tp_size, tp_rank, arena, stream, options);
    if (!weights.ok()) return weights.status();
    Status valid = validate_qwen35qsa_weights(weights.value(), config.value());
    if (!valid.ok()) return valid;
    return Qwen35QsaModel(weights.release(), config.release());
}

Result<Qwen35QsaModel> Qwen35QsaModel::adopt(Qwen35QsaModelWeights&& weights,
                                             Qwen35QsaTextConfig text_config) {
    Status valid = validate_qwen35qsa_weights(weights, text_config);
    if (!valid.ok()) return valid;
    return Qwen35QsaModel(std::move(weights), std::move(text_config));
}

Result<shelf::qwen35::Executor> create_qwen35qsa_model_executor(
    const Qwen35QsaModel& model,
    shelf::qwen35::SequenceSlotPool& sequence_pool,
    shelf::qwen35::GdnStatePool& gdn_state_pool,
    shelf::qwen35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const shelf::qwen35::ExecutorConfig& config) {
    Qwen35QsaLowerOptions lower_options;
    lower_options.backbone.rope_profile = config.rope_profile;
    lower_options.backbone.emit_normed_token_hidden = config.emit_normed_token_hidden;
    lower_options.backbone.hidden_taps = std::span<const std::uint32_t>(
        config.target_hidden_taps.data(), config.target_hidden_tap_count);
    lower_options.backbone.tp = config.tp;
    lower_options.backbone.tp_combine_f32 = config.tp_combine_f32 != 0;
    lower_options.max_context_tokens = config.max_context_tokens;

    auto lowered =
        lower_qwen35qsa_to_primitives(model.text_config(), model.weights(), lower_options);
    if (!lowered.ok()) return lowered.status();

    shelf::qwen35::ExecutorModelInput input;
    input.model.config = &model.text_config();
    input.model.embed_tokens = &model.weights().backbone.embed_tokens;
    input.lowered = lowered.release();

    // The pool is only needed when selection can prune, which the lowering decides from
    // the same context length.
    const bool qsa_active = config.max_context_tokens > model.text_config().indexer.budget;
    if (qsa_active) {
        const auto layout = qsa_layout(model.text_config(), config.max_context_tokens);
        input.uses_qsa_state = true;
        input.qsa_layers = layout.num_qsa_states;
        input.qsa_head_dim = layout.head_dim;
        input.qsa_compress_ratio = layout.compress_ratio;
        input.qsa_max_blocks = layout.max_blocks;
    }

    return shelf::qwen35::create_executor(std::move(input), sequence_pool, gdn_state_pool,
                                          kv_pool, arena, config);
}

std::size_t qwen35qsa_executor_reserve_bytes(
    const Qwen35QsaTextConfig& config,
    std::size_t ffn_intermediate_size,
    const shelf::qwen35::ExecutorConfig& exec_config,
    std::size_t tp_size,
    bool exl3_linear_workspace,
    std::size_t max_sequences) {
    const bool qsa_active = exec_config.max_context_tokens > config.indexer.budget;
    std::size_t total = shelf::qwen35::runtime::qwen35_executor_reserve_bytes(
        config, ffn_intermediate_size, exec_config, tp_size, exl3_linear_workspace,
        qsa_active);
    if (qsa_active) {
        const auto layout = qsa_layout(config, exec_config.max_context_tokens);
        total += ::shelf::state::qsa_key_state_pool_bytes(
                     max_sequences == 0u ? 1u : max_sequences, layout)
                     .total();
    }
    return total;
}

}
}
