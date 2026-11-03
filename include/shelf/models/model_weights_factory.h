#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

namespace shelf::models {

// Which weight family a model directory needs. The decision is num_experts in
// config.json, never the model_type string (docs/developer/qwen35moe.md sec.2), and it
// lives here because this module is the one place that links both family loaders.
enum class ModelFamily : std::uint8_t {
    Dense,
    Moe,
};

struct LoadedModelWeights {
    ModelFamily family = ModelFamily::Dense;

    // Filled when family == Dense.
    shelf::qwen35::Qwen35ModelWeights dense;

    // Filled when family == Moe. Its body carries embed / lm_head / final norm and each
    // layer's layernorms and attention; layers carries the routed-expert FFN.
    shelf::qwen35moe::MoeModelWeights moe;

    // The qwen35-shaped part of either family: what the attention and norm primitives
    // consume.
    const shelf::qwen35::Qwen35ModelWeights& body() const noexcept {
        return family == ModelFamily::Moe ? moe.body : dense;
    }
};

// num_experts > 0 selects Moe. Fails only when config.json cannot be read.
Result<ModelFamily> probe_model_dir(const std::string& model_dir);

// Reads config.json, loads with the family's loader, and runs
// shelf::qwen35::validate_qwen35_weights on the resulting body.
Result<LoadedModelWeights> load_model_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::qwen35::Qwen35LoadOptions& options = {});

}  // namespace shelf::models
