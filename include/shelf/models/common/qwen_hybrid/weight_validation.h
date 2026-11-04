#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/common/qwen_hybrid/layer_weights.h>
#include <shelf/models/common/qwen_hybrid/model_weights.h>
#include <shelf/weights/matrix_weight.h>

#include <cstddef>

namespace shelf {
namespace qwen_hybrid {

// The invariants every Qwen hybrid checkpoint must satisfy, whatever its FFN is. A model
// family adds its own FFN checks on top of these.

// Matrix dimensions as the model states them, not as one rank holds them: a partition
// records the axis it splits and into how many equal parts, so a local rows or cols
// multiplied by that count is the model's dimension.
Status check_matrix_shape(const shelf::weights::MatrixWeight& w,
                          std::size_t rows,
                          std::size_t cols,
                          const char* tag);

// Embedding and LM head vocab (a vocabulary-sharded LM head is accepted), hidden width and
// final norm rank, over the layer range the weights were loaded for. A range that starts at
// layer 0 carries the embedding, a range that ends at the last layer carries the final norm
// and the LM head, and a middle stage carries neither. Row counts are the global counts a
// partition reports, so a rank of a sharded model passes.
Status validate_hybrid_model_level(const HybridModelLevelWeights& model,
                                   const QwenHybridConfig& config,
                                   std::size_t loaded_layer_lo,
                                   std::size_t loaded_layer_hi);

// One layer's backbone for a known topology: the two layernorms are [hidden], every
// tensor of that topology is loaded, and the other topology's tensors are absent, so a
// layer cannot carry both attentions or neither. The head geometry of a loaded
// projection is not re-derived here: lowering takes it from the weight, so a width
// difference is not decided at this boundary.
Status validate_hybrid_layer_body(const HybridLayerWeights& layer,
                                  std::size_t hidden,
                                  bool is_gdn,
                                  const char* tag);

// The same, with the topology read from the config, so the stored is_gdn flag cannot
// disagree with it.
Status validate_hybrid_layer_body(const HybridLayerWeights& layer,
                                  const QwenHybridConfig& config,
                                  std::size_t layer_index);

}  // namespace qwen_hybrid
}  // namespace shelf
