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

// Embedding and LM head vocab (a vocabulary-sharded LM head is accepted), hidden width,
// final norm rank and width, and the layer count.
Status validate_hybrid_model_level(const HybridModelLevelWeights& model,
                                   const QwenHybridConfig& config,
                                   std::size_t layer_count);

// One layer's backbone for a known topology: the two layernorms, and the tensors of that
// topology only. The other topology must stay unloaded, so a layer cannot carry both
// attentions or neither.
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
