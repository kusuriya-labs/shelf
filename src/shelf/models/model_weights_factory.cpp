#include <shelf/models/model_weights_factory.h>

#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

namespace shelf::models {

ModelFamily model_family(const LoadedModelWeights& loaded) noexcept {
    return std::holds_alternative<shelf::qwen35moe::Qwen35MoeModelWeights>(loaded)
        ? ModelFamily::Moe
        : ModelFamily::Dense;
}

Result<ModelFamily> probe_model_dir(const std::string& model_dir) {
    auto experts = shelf::qwen35moe::read_declared_num_experts(model_dir);
    if (!experts.ok()) return experts.status();
    return experts.value() > 0u ? ModelFamily::Moe : ModelFamily::Dense;
}

Result<LoadedModelWeights> load_model_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::qwen35::Qwen35LoadOptions& options) {
    auto family = probe_model_dir(model_dir);
    if (!family.ok()) return family.status();

    if (family.value() == ModelFamily::Moe) {
        if (options.load_mtp_layers) {
            return Status::unsupported(
                "qwen35moe does not load MTP layers: speculative decode uses the DFlash2 "
                "drafter",
                __FILE__, __LINE__);
        }
        auto config = shelf::qwen35moe::read_qwen35_moe_text_config(model_dir);
        if (!config.ok()) return config.status();
        auto weights = shelf::qwen35moe::load_qwen35moe_weights(
            model_dir, arena, stream,
            static_cast<const shelf::qwen_hybrid::HybridLoadOptions&>(options));
        if (!weights.ok()) return weights.status();
        Status valid = shelf::qwen35moe::validate_qwen35moe_weights(weights.value(),
                                                                    config.value());
        if (!valid.ok()) return valid;
        return LoadedModelWeights{weights.release()};
    }

    auto config = shelf::qwen35::read_qwen35_text_config(model_dir);
    if (!config.ok()) return config.status();
    auto weights =
        shelf::qwen35::load_qwen35_weights_from_safetensors(model_dir, arena, stream, options);
    if (!weights.ok()) return weights.status();
    Status valid = shelf::qwen35::validate_qwen35_weights(weights.value(), config.value());
    if (!valid.ok()) return valid;
    return LoadedModelWeights{weights.release()};
}

}  // namespace shelf::models
