#include <shelf/models/qwen35moe/model/qwen35moe_model.h>

#include <utility>

namespace shelf::qwen35moe {

Result<Qwen35MoeModel> Qwen35MoeModel::load_from_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35MoeLoadOptions& options) {
    auto config_result = read_qwen35_moe_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    Qwen35MoeTextConfig text_config = config_result.release();

    auto weights_result = load_qwen35moe_weights(model_dir, arena, stream, options);
    if (!weights_result.ok()) return weights_result.status();
    Qwen35MoeModelWeights weights = weights_result.release();

    // A rank of an expert-axis split carries its own range, so the invariants are checked
    // against that range. The plan is the only statement of it.
    const shelf::weights::TensorPartitionPlan* plan = options.weights.partition_plan;
    Status validation = plan != nullptr
        ? validate_qwen35moe_weights(weights, text_config, plan->tp_size, plan->tp_rank)
        : validate_qwen35moe_weights(weights, text_config);
    if (!validation.ok()) return validation;

    return Qwen35MoeModel(std::move(weights), std::move(text_config));
}

Result<Qwen35MoeModel> Qwen35MoeModel::adopt(
    Qwen35MoeModelWeights&& weights,
    Qwen35MoeTextConfig text_config) {
    Status validation = validate_qwen35moe_weights(weights, text_config);
    if (!validation.ok()) return validation;
    return Qwen35MoeModel(std::move(weights), std::move(text_config));
}

}  // namespace shelf::qwen35moe
