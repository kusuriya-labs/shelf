#include <shelf/models/qwen4exp/model/qsa_lowering.h>

namespace shelf {
namespace qwen4exp {

namespace RT = ::shelf::runtime;
using ::shelf::qwen_hybrid::Emitter;

Qwen4ExpQsaIndexerGeometry make_qwen4exp_qsa_indexer_geometry(
    const Qwen4ExpTextConfig& config, std::uint32_t blocks, std::uint32_t state_index) {
    const std::uint32_t index_comp =
        static_cast<std::uint32_t>(config.indexer.compress_ratio);
    const std::uint32_t index_budget =
        static_cast<std::uint32_t>(config.indexer.budget);
    const std::uint32_t rotary_dim = static_cast<std::uint32_t>(
        static_cast<double>(config.attention_head_dim) * config.partial_rotary_factor);
    Qwen4ExpQsaIndexerGeometry geometry;
    geometry.hidden = static_cast<std::uint32_t>(config.hidden_size);
    geometry.heads = static_cast<std::uint32_t>(config.indexer.n_heads);
    geometry.head_dim = static_cast<std::uint32_t>(config.indexer.head_dim);
    geometry.rotary_dim = rotary_dim;
    geometry.compress_ratio = index_comp;
    geometry.blocks = blocks;
    geometry.top_k = index_comp != 0u ? index_budget / index_comp : 0u;
    geometry.mask_bytes = ((blocks + 31u) / 32u) * 4u;
    geometry.state_index = state_index;
    geometry.theta = config.rope_theta;
    geometry.eps = config.rms_norm_eps;
    geometry.pre_pool_key_norm = false;
    return geometry;
}

Result<Qwen4ExpQsaIndexerOutputs> emit_qwen4exp_qsa_indexer(
    Emitter& em, const Qwen4ExpQsaIndexerGeometry& geometry,
    const Qwen4ExpQsaIndexerWeights& weights, const Qwen4ExpQsaIndexerIndices& indices,
    RT::ValueId normed, const char* debug_prefix) {
    return ::shelf::qwen_hybrid::emit_qsa_indexer(em, geometry, weights, indices, normed,
                                                  debug_prefix);
}

Result<RT::ValueId> emit_qwen4exp_qsa(
    Emitter& em, const Qwen4ExpQsaLoweringGeometry& geometry,
    const Qwen4ExpQsaLoweringWeights& weights, const Qwen4ExpQsaLoweringIndices& indices,
    RT::ValueId normed, const char* debug_prefix) {
    const std::uint32_t ratio = geometry.indexer.compress_ratio;
    if (ratio == 0u || (ratio & (ratio - 1u)) != 0u) {
        return Status::invalid_argument(
            "Qwen4Exp QSA block size must be a power of two", __FILE__, __LINE__);
    }
    auto indexer = emit_qwen4exp_qsa_indexer(em, geometry.indexer, weights.indexer,
                                             indices.indexer, normed, debug_prefix);
    if (!indexer.ok()) return indexer.status();

    ::shelf::qwen_hybrid::GatedAttentionLoweringIndices attention_indices =
        indices.attention;
    attention_indices.has_block_mask = true;
    attention_indices.block_mask = indexer.value().mask;
    attention_indices.block_size = geometry.indexer.compress_ratio;
    return ::shelf::qwen_hybrid::emit_gated_attention(
        em, geometry.attention, weights.attention, attention_indices, normed,
        debug_prefix);
}

}
}
