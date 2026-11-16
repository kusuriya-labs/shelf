#pragma once

#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35qsa/model/qwen35qsa_config.h>
#include <shelf/models/qwen35qsa/weights/model_weights.h>

#include <cstdint>

namespace shelf {
namespace qwen35qsa {

struct Qwen35QsaLowerOptions {
    // Everything the dense lowering takes. Pipeline stages and tensor parallel are not
    // implemented for the sparse path and are rejected.
    shelf::qwen35::Qwen35LowerOptions backbone;
    // Cached context the indexer block table is sized from.
    std::uint32_t max_context_tokens = 0;
};

Result<shelf::qwen35::Qwen35LoweredPrimitives> lower_qwen35qsa_to_primitives(
    const Qwen35QsaTextConfig& config,
    const Qwen35QsaModelWeights& weights,
    const Qwen35QsaLowerOptions& options = {});

}
}
