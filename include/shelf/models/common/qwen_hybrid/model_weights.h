#pragma once
#include <shelf/core/memory/tensor.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/pinned_host_arena.h>

namespace shelf {
namespace qwen_hybrid {

// The model-level weights of a Qwen hybrid model: the embedding, the LM head (which may
// alias the embedding), and the final norm. Neither FFN appears here.
struct HybridModelLevelWeights {
    shelf::weights::MatrixWeight embed_tokens;
    // Backs an embedding table that was placed in page-locked host memory.
    shelf::weights::PinnedHostArena pinned_host;
    gpu::Tensor final_norm_weight;

    shelf::weights::MatrixWeight lm_head;
    bool lm_head_tied = true;
};

}  // namespace qwen_hybrid
}  // namespace shelf
