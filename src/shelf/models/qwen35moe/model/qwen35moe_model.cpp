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
    const std::uint32_t matrix_size = plan != nullptr ? plan->tp_size : 1u;
    const std::uint32_t matrix_rank = plan != nullptr ? plan->tp_rank : 0u;
    const std::uint32_t expert_size = options.ep_size == 0u ? 1u : options.ep_size;
    const std::uint32_t expert_rank = options.ep_size == 0u ? 0u : options.ep_rank;
    Status validation = validate_qwen35moe_weights(
        weights, text_config, expert_size, expert_rank, matrix_size, matrix_rank,
        options.stage);
    if (!validation.ok()) return validation;

    return Qwen35MoeModel(std::move(weights), std::move(text_config), options.stage);
}

Result<Qwen35MoeModel> Qwen35MoeModel::adopt(
    Qwen35MoeModelWeights&& weights,
    Qwen35MoeTextConfig text_config,
    const shelf::qwen_hybrid::PipelineStage& stage) {
    Status validation = validate_qwen35moe_weights(weights, text_config, 1u, 0u, stage);
    if (!validation.ok()) return validation;
    return Qwen35MoeModel(std::move(weights), std::move(text_config), stage);
}

}  // namespace shelf::qwen35moe
