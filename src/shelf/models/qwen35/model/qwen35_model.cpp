#include <shelf/models/qwen35/model/qwen35_model.h>

#include <cstddef>
#include <iterator>
#include <utility>

namespace shelf::qwen35 {

Status validate_qwen35_weights(
    const Qwen35ModelWeights& weights,
    const Qwen35TextConfig& config)
{
    if (weights.embed_tokens.rows == 0 || weights.embed_tokens.cols == 0) {
        return Status::invalid_argument(
            "Qwen35Model: embed_tokens dimensions must be > 0",
            __FILE__, __LINE__);
    }
    if (weights.lm_head.rows == 0 || weights.lm_head.cols == 0) {
        return Status::invalid_argument(
            "Qwen35Model: lm_head dimensions must be > 0",
            __FILE__, __LINE__);
    }
    if (weights.final_norm_weight.ndim() != 1) {
        return Status::invalid_argument(
            "Qwen35Model: final_norm_weight must be 1D",
            __FILE__, __LINE__);
    }

    const std::size_t hidden_size = weights.embed_tokens.cols;
    if (weights.final_norm_weight.dim(0) != hidden_size) {
        return Status::invalid_argument(
            "Qwen35Model: final_norm hidden size mismatch",
            __FILE__, __LINE__);
    }

    if (weights.embed_tokens.rows != config.vocab_size) {
        return Status::invalid_argument(
            "Qwen35Model: embed_tokens vocab size mismatch",
            __FILE__, __LINE__);
    }
    if (weights.lm_head.rows != config.vocab_size) {
        const bool vocab_sharded =
            weights.lm_head.partition.has_value() &&
            weights.lm_head.partition->axis == 0 &&
            weights.lm_head.partition->count > 1 &&
            static_cast<std::size_t>(weights.lm_head.rows) *
                    weights.lm_head.partition->count ==
                config.vocab_size;
        if (!vocab_sharded) {
            return Status::invalid_argument(
                "Qwen35Model: lm_head vocab size mismatch",
                __FILE__, __LINE__);
        }
    }
    if (hidden_size != config.hidden_size) {
        return Status::invalid_argument(
            "Qwen35Model: hidden size mismatch",
            __FILE__, __LINE__);
    }
    if (weights.layers.size() != config.num_hidden_layers) {
        return Status::invalid_argument(
            "Qwen35Model: layer count mismatch",
            __FILE__, __LINE__);
    }

    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const bool expected_gdn =
            (config.full_attention_interval == 0)
            ? (layer % 4 < 3)
            : (layer % config.full_attention_interval <
               config.full_attention_interval - 1);
        if (weights.layers[layer].is_gdn != expected_gdn) {
            return Status::invalid_argument(
                "Qwen35Model: unexpected layer topology",
                __FILE__, __LINE__);
        }
    }

    return Status::make_ok();
}

Result<Qwen35Model> Qwen35Model::load_from_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    auto config_result = read_qwen35_text_config(model_dir);
    if (!config_result.ok()) {
        return config_result.status();
    }
    Qwen35TextConfig text_config = config_result.release();

    auto weights_result =
        load_qwen35_weights_from_safetensors(model_dir, arena, stream, options);
    if (!weights_result.ok()) {
        return weights_result.status();
    }
    Qwen35ModelWeights weights = weights_result.release();

    auto validation = validate_qwen35_weights(weights, text_config);
    if (!validation.ok()) {
        return validation;
    }

    return Qwen35Model(std::move(weights), text_config);
}

Result<Qwen35Model> Qwen35Model::load_tensor_parallel_from_safetensors(
    const std::string& model_dir,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    auto config_result = read_qwen35_text_config(model_dir);
    if (!config_result.ok()) {
        return config_result.status();
    }
    Qwen35TextConfig text_config = config_result.release();

    auto weights_result = load_qwen35_weights_tensor_parallel_rank(
        model_dir, tp_size, tp_rank, arena, stream, options);
    if (!weights_result.ok()) {
        return weights_result.status();
    }
    Qwen35ModelWeights weights = weights_result.release();

    auto validation = validate_qwen35_weights(weights, text_config);
    if (!validation.ok()) {
        return validation;
    }

    return Qwen35Model(std::move(weights), text_config);
}

Result<Qwen35Model> Qwen35Model::adopt(
    Qwen35ModelWeights&& weights,
    Qwen35TextConfig text_config)
{
    auto validation = validate_qwen35_weights(weights, text_config);
    if (!validation.ok()) {
        return validation;
    }
    return Qwen35Model(std::move(weights), text_config);
}

Qwen35Model::Qwen35Model(
    Qwen35ModelWeights&& weights,
    Qwen35TextConfig text_config)
    : weights_(std::move(weights)),
      text_config_(std::move(text_config))
{
}

} // namespace shelf::qwen35
