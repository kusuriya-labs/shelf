#pragma once
#include <shelf/core/memory/tensor.h>
#include <shelf/weights/matrix_weight.h>

namespace shelf {
namespace qwen_hybrid {

// The model-level weights of a Qwen hybrid model: the embedding, the LM head (which may
// alias the embedding), and the final norm. Neither FFN appears here.
struct HybridModelLevelWeights {
    shelf::weights::MatrixWeight embed_tokens;
    gpu::Tensor final_norm_weight;

    shelf::weights::MatrixWeight lm_head;
    bool lm_head_tied = true;
};

}  // namespace qwen_hybrid
}  // namespace shelf
