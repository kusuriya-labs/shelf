#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/plamo3/model/config.h>
#include <shelf/models/plamo3/weights/model_weights.h>

namespace shelf::plamo3 {
Result<shelf::qwen_hybrid::HybridLoweredPrimitives> lower_to_primitives(
    const Config& config, const ModelWeights& weights);
}
