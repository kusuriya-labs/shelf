#include <shelf/models/model_weights_factory.h>

#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

namespace shelf::models {

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

    LoadedModelWeights out;
    out.family = family.value();

    if (out.family == ModelFamily::Moe) {
        auto config = shelf::qwen35moe::read_qwen35_moe_text_config(model_dir);
        if (!config.ok()) return config.status();
        auto weights = shelf::qwen35moe::load_qwen35moe_weights(model_dir, arena, stream, options);
        if (!weights.ok()) return weights.status();
        out.moe = weights.release();
        Status valid =
            shelf::qwen35::validate_qwen35_weights(out.moe.body, config.value());
        if (!valid.ok()) return valid;
        return out;
    }

    auto config = shelf::qwen35::read_qwen35_text_config(model_dir);
    if (!config.ok()) return config.status();
    auto weights =
        shelf::qwen35::load_qwen35_weights_from_safetensors(model_dir, arena, stream, options);
    if (!weights.ok()) return weights.status();
    out.dense = weights.release();
    Status valid = shelf::qwen35::validate_qwen35_weights(out.dense, config.value());
    if (!valid.ok()) return valid;
    return out;
}

}  // namespace shelf::models
