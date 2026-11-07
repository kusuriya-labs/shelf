#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/attention_lowering.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/weights/matrix_weight.h>

#include <cstdint>

namespace shelf {
namespace qwen4exp {

// QSA indexer geometry. The indexer is MQA: `index_heads` query heads share one key head.
// `blocks` is the key-state pool capacity, `top_k` is the block budget (indexer_budget /
// compress_ratio), and `mask_bytes` is the per-row width of the block bitmap the select
// writes.
struct Qwen4ExpQsaIndexerGeometry {
    std::uint32_t hidden = 0;
    std::uint32_t index_heads = 0;
    std::uint32_t index_kv_heads = 1;
    std::uint32_t index_head_dim = 0;
    std::uint32_t index_rotary_dim = 0;
    std::uint32_t compress_ratio = 4;
    std::uint32_t blocks = 0;
    std::uint32_t top_k = 0;
    std::uint32_t mask_bytes = 0;
    std::uint32_t state_index = 0;
    float theta = 10000000.0f;
    float position_scale = 1.0f;
    float attention_factor = 1.0f;
    float eps = 1e-6f;
};

struct Qwen4ExpQsaIndexerWeights {
    const ::shelf::weights::MatrixWeight* qk_proj = nullptr;
};

struct Qwen4ExpQsaIndexerIndices {
    std::uint32_t w_qk_proj = 0;
    std::uint32_t p_q_norm = 0;
    std::uint32_t p_k_norm = 0;
};

struct Qwen4ExpQsaIndexerOutputs {
    ::shelf::runtime::ValueId q;        // [rows, index_heads * index_head_dim]
    ::shelf::runtime::ValueId mask;     // [rows, mask_bytes] U8
};

// Geometry implied by the checkpoint config. `blocks` is the key-state capacity, and the
// block bitmap width follows from it.
Qwen4ExpQsaIndexerGeometry make_qwen4exp_qsa_indexer_geometry(
    const Qwen4ExpTextConfig& config, std::uint32_t blocks, std::uint32_t state_index);

// Indexer part of one QSA layer. The main attention uses the existing gated-attention
// lowering; this emits the raw key cache and the block selection the attention consumes.
Result<Qwen4ExpQsaIndexerOutputs> emit_qwen4exp_qsa_indexer(
    ::shelf::qwen_hybrid::Emitter& em,
    const Qwen4ExpQsaIndexerGeometry& geometry,
    const Qwen4ExpQsaIndexerWeights& weights,
    const Qwen4ExpQsaIndexerIndices& indices,
    ::shelf::runtime::ValueId normed,
    const char* debug_prefix);

// One QSA attention site: the indexer emits the block selection, and the gated attention
// consumes it.
struct Qwen4ExpQsaLoweringGeometry {
    ::shelf::qwen_hybrid::GatedAttentionLoweringGeometry attention;
    Qwen4ExpQsaIndexerGeometry indexer;
};

struct Qwen4ExpQsaLoweringWeights {
    ::shelf::qwen_hybrid::GatedAttentionLoweringWeights attention;
    Qwen4ExpQsaIndexerWeights indexer;
};

struct Qwen4ExpQsaLoweringIndices {
    ::shelf::qwen_hybrid::GatedAttentionLoweringIndices attention;
    Qwen4ExpQsaIndexerIndices indexer;
};

Result<::shelf::runtime::ValueId> emit_qwen4exp_qsa(
    ::shelf::qwen_hybrid::Emitter& em,
    const Qwen4ExpQsaLoweringGeometry& geometry,
    const Qwen4ExpQsaLoweringWeights& weights,
    const Qwen4ExpQsaLoweringIndices& indices,
    ::shelf::runtime::ValueId normed,
    const char* debug_prefix);

}  // namespace qwen4exp
}  // namespace shelf
