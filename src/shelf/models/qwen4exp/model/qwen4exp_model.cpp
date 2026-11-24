#include <shelf/models/qwen4exp/model/qwen4exp_model.h>
#include <shelf/models/qwen4exp/weights/ple_table.h>

#include <utility>

namespace shelf {
namespace qwen4exp {

Qwen4ExpModel::Qwen4ExpModel(Qwen4ExpRankWeights&& weights, Qwen4ExpTextConfig config)
    : weights_(std::move(weights)), config_(std::move(config)) {}

Result<Qwen4ExpModel> Qwen4ExpModel::load_from_safetensors(
    const std::string& model_dir, std::uint32_t tp_size, std::uint32_t tp_rank,
    gpu::GpuArena& arena, hipStream_t stream, const Qwen4ExpLoadOptions& options_in,
    std::uint32_t ep_size, std::uint32_t ep_rank) {
    auto config = read_qwen4exp_text_config(model_dir);
    if (!config.ok()) return config.status();

    Qwen4ExpLoadOptions options = options_in;
    options.tp_size = tp_size;
    options.tp_rank = tp_rank;
    options.ep_size = ep_size == 0u ? options_in.ep_size : ep_size;
    options.ep_rank = ep_rank;
    auto weights = load_qwen4exp_rank(model_dir, arena, stream, options);
    if (!weights.ok()) return weights.status();

    const std::uint32_t backbone_tp_size =
        options.backbone_tp_size != 0u ? options.backbone_tp_size : tp_size;
    const std::uint32_t backbone_tp_rank =
        options.backbone_tp_size != 0u ? options.backbone_tp_rank : tp_rank;
    Status valid = options.stage.enabled
        ? validate_qwen4exp_stage(weights.value(), config.value(), options.stage)
        : validate_qwen4exp_rank(weights.value(), config.value(), backbone_tp_size,
                                 backbone_tp_rank, options.ep_size, options.ep_rank,
                                 tp_size);
    if (!valid.ok()) return valid;
    return adopt(weights.release(), config.release(), model_dir);
}

Result<Qwen4ExpModel> Qwen4ExpModel::adopt(Qwen4ExpRankWeights&& weights,
                                           Qwen4ExpTextConfig config,
                                           const std::string& model_dir) {
    Qwen4ExpModel model(std::move(weights), std::move(config));
    if (!model.config_.ple.layer_ids.empty()) {
        if (model_dir.empty()) {
            return Status::invalid_argument("PLE layers require a model directory", __FILE__, __LINE__);
        }
        auto table = PleTable::load_shared(model_dir, model.config_);
        if (!table.ok()) return table.status();
        model.ple_table_ = table.release();
    }
    return model;
}

}  // namespace qwen4exp
}  // namespace shelf
