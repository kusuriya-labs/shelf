#include <shelf/models/qwen4exp/model/qwen4exp_model.h>

#include <utility>

namespace shelf {
namespace qwen4exp {

Qwen4ExpModel::Qwen4ExpModel(Qwen4ExpRankWeights&& weights, Qwen4ExpTextConfig config)
    : weights_(std::move(weights)), config_(std::move(config)) {}

Result<Qwen4ExpModel> Qwen4ExpModel::load_from_safetensors(
    const std::string& model_dir, std::uint32_t tp_size, std::uint32_t tp_rank,
    gpu::GpuArena& arena, hipStream_t stream, const Qwen4ExpLoadOptions& options_in) {
    auto config = read_qwen4exp_text_config(model_dir);
    if (!config.ok()) return config.status();

    Qwen4ExpLoadOptions options = options_in;
    options.tp_size = tp_size;
    options.tp_rank = tp_rank;
    auto weights = load_qwen4exp_rank(model_dir, arena, stream, options);
    if (!weights.ok()) return weights.status();

    Status valid = validate_qwen4exp_rank(weights.value(), config.value(), tp_size, tp_rank);
    if (!valid.ok()) return valid;
    return Qwen4ExpModel(weights.release(), config.release());
}

Result<Qwen4ExpModel> Qwen4ExpModel::adopt(Qwen4ExpRankWeights&& weights,
                                           Qwen4ExpTextConfig config) {
    return Qwen4ExpModel(std::move(weights), std::move(config));
}

}  // namespace qwen4exp
}  // namespace shelf
