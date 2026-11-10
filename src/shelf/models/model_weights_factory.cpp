#include <shelf/models/model_weights_factory.h>

#include <shelf/models/plamo3/model/config.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

namespace shelf::models {

ModelFamily model_family(const LoadedModelWeights& loaded) noexcept {
    if (std::holds_alternative<shelf::plamo3::ModelWeights>(loaded)) return ModelFamily::Plamo3;
    return std::holds_alternative<shelf::qwen35moe::Qwen35MoeModelWeights>(loaded)
        ? ModelFamily::Moe
        : ModelFamily::Dense;
}

Result<ModelFamily> probe_model_dir(const std::string& model_dir) {
    auto plamo = shelf::plamo3::is_plamo3_config(model_dir);
    if (!plamo.ok()) return plamo.status();
    if (plamo.value()) {
        auto config = shelf::plamo3::read_config(model_dir);
        if (!config.ok()) return config.status();
        return ModelFamily::Plamo3;
    }
    auto experts = shelf::qwen35moe::read_declared_num_experts(model_dir);
    if (!experts.ok()) return experts.status();
    auto ple = shelf::qwen_hybrid::has_text_config_key(model_dir, "ple_layer_ids");
    if (!ple.ok()) return ple.status();
    auto hc = shelf::qwen_hybrid::has_text_config_key(model_dir, "hc_count");
    if (!hc.ok()) return hc.status();
    if (ple.value() && hc.value()) return ModelFamily::Qwen4Exp;
    return experts.value() > 0u ? ModelFamily::Moe : ModelFamily::Dense;
}

Result<LoadedModelWeights> load_model_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::qwen35::Qwen35LoadOptions& options) {
    auto family = probe_model_dir(model_dir);
    if (!family.ok()) return family.status();

    if (family.value() == ModelFamily::Plamo3) {
        if (options.stage.enabled || options.load_mtp_layers || options.weights.partition_plan ||
            !options.load_embedding || !options.load_lm_head || !options.load_final_norm)
            return Status::unsupported("plamo3: pipeline, MTP and tensor parallel loading are not supported", __FILE__, __LINE__);
        auto config = shelf::plamo3::read_config(model_dir);
        if (!config.ok()) return config.status();
        auto weights = shelf::plamo3::load_weights(model_dir, config.value(), arena, stream);
        if (!weights.ok()) return weights.status();
        return LoadedModelWeights{weights.release()};
    }
    if (family.value() == ModelFamily::Qwen4Exp) {
        return Status::unsupported(
            "Qwen4Exp checkpoints use the qwen4exp load path, not the compute "
            "LoadedModel family",
            __FILE__, __LINE__);
    }

    if (family.value() == ModelFamily::Moe) {
        if (options.stage.enabled) {
            return Status::unsupported(
                "a model with grouped experts has no pipeline stage weight load", __FILE__,
                __LINE__);
        }
        shelf::qwen35moe::Qwen35MoeLoadOptions moe_options;
        static_cast<shelf::qwen_hybrid::HybridLoadOptions&>(moe_options) =
            static_cast<const shelf::qwen_hybrid::HybridLoadOptions&>(options);
        moe_options.load_mtp_layers = options.load_mtp_layers;
        auto config = shelf::qwen35moe::read_qwen35_moe_text_config(model_dir);
        if (!config.ok()) return config.status();
        auto weights = shelf::qwen35moe::load_qwen35moe_weights(model_dir, arena, stream,
                                                                moe_options);
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
