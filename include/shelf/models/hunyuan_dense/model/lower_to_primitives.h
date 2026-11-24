#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/hunyuan_dense/model/config.h>
#include <shelf/models/hunyuan_dense/weights/model_weights.h>

namespace shelf::hunyuan_dense {

Result<shelf::qwen_hybrid::HybridLoweredPrimitives> lower_to_primitives(
    const Config& config, const ModelWeights& weights);

}
