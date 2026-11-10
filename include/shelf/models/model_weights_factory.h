#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/plamo3/weights/model_weights.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <variant>

namespace shelf::models {

// PLaMo is selected by identity; Qwen families are distinguished by their config topology.
enum class ModelFamily : std::uint8_t {
    Dense,
    Moe,
    Qwen4Exp,
    Plamo3,
};

// A loaded model holds exactly one family's weights. There is no empty member of the
// other family and no dense view of an MoE checkpoint, so an MoE model cannot be read as
// if it had a dense FFN. Access is explicit: std::get_if<Qwen35MoeModelWeights>(...) is
// null for a dense model.
using LoadedModelWeights = std::variant<
    shelf::qwen35::Qwen35ModelWeights,
    shelf::qwen35moe::Qwen35MoeModelWeights,
    shelf::plamo3::ModelWeights>;

ModelFamily model_family(const LoadedModelWeights& loaded) noexcept;

// Selects the checkpoint family from its config.
Result<ModelFamily> probe_model_dir(const std::string& model_dir);

// Reads config.json, loads with that family's loader, and runs that family's validation.
// A dense checkpoint is rejected by the MoE loader and an MoE checkpoint by the dense
// loader, so the probe cannot silently pick the wrong path. The options are the dense
// ones; a pipeline stage is rejected for a model with grouped experts instead of being
// ignored. This module does not decide how a model runs on the GPU.
Result<LoadedModelWeights> load_model_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::qwen35::Qwen35LoadOptions& options = {});

}  // namespace shelf::models
