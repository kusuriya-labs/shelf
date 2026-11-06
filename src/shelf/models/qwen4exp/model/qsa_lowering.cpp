#include <shelf/models/qwen4exp/model/qsa_lowering.h>

#include <shelf/kernels/optimized/qsa.h>

#include <cmath>
#include <string>

namespace shelf {
namespace qwen4exp {

namespace RT = ::shelf::runtime;
using ::shelf::qwen_hybrid::Emitter;
using ::shelf::weights::MatrixWeight;

namespace {

RT::ValueId emit_unary(Emitter& em, const RT::PrimitiveNode& node, RT::ValueId input,
                       std::uint32_t features,
                       RT::ValueDType dtype = RT::ValueDType::BF16) {
    RT::ValueId out = em.value(features, dtype);
    em.emit(node, {input}, {out});
    return out;
}

}  // namespace

Qwen4ExpQsaIndexerGeometry make_qwen4exp_qsa_indexer_geometry(
    const Qwen4ExpTextConfig& config, std::uint32_t blocks, std::uint32_t state_index) {
    const std::uint32_t index_comp = static_cast<std::uint32_t>(config.indexer.compress_ratio);
    const std::uint32_t index_budget = static_cast<std::uint32_t>(config.indexer.budget);
    const std::uint32_t rope_dim = static_cast<std::uint32_t>(
        static_cast<double>(config.attention_head_dim) * config.partial_rotary_factor);
    Qwen4ExpQsaIndexerGeometry geometry;
    geometry.hidden = static_cast<std::uint32_t>(config.hidden_size);
    geometry.index_heads = static_cast<std::uint32_t>(config.indexer.n_heads);
    geometry.index_kv_heads = static_cast<std::uint32_t>(config.indexer.kv_heads);
    geometry.index_head_dim = static_cast<std::uint32_t>(config.indexer.head_dim);
    geometry.index_rotary_dim = rope_dim;
    geometry.compress_ratio = index_comp;
    geometry.blocks = blocks;
    geometry.top_k = index_comp != 0u ? index_budget / index_comp : 0u;
    geometry.mask_bytes = ((blocks + 31u) / 32u) * 4u;
    geometry.state_index = state_index;
    geometry.theta = config.rope_theta;
    geometry.eps = config.rms_norm_eps;
    return geometry;
}

Result<Qwen4ExpQsaIndexerOutputs> emit_qwen4exp_qsa_indexer(
    Emitter& em, const Qwen4ExpQsaIndexerGeometry& geometry,
    const Qwen4ExpQsaIndexerWeights& weights, const Qwen4ExpQsaIndexerIndices& indices,
    RT::ValueId normed, const char* debug_prefix) {
    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = std::string(debug_prefix) + suffix;
    };

    const std::uint32_t index_width = geometry.index_heads * geometry.index_head_dim;
    const std::uint32_t key_width = geometry.index_kv_heads * geometry.index_head_dim;
    if (geometry.hidden == 0u || geometry.index_heads == 0u ||
        geometry.index_kv_heads == 0u || geometry.index_head_dim != 128u ||
        geometry.index_rotary_dim == 0u ||
        (geometry.index_rotary_dim & 1u) != 0u ||
        geometry.index_rotary_dim > geometry.index_head_dim ||
        geometry.compress_ratio == 0u || geometry.blocks == 0u ||
        geometry.top_k == 0u || geometry.mask_bytes * 8u < geometry.blocks) {
        return Status::invalid_argument("Qwen4Exp QSA indexer geometry is invalid",
                                        __FILE__, __LINE__);
    }
    if (geometry.index_heads > ::shelf::kernel::kQsaSelectMaxHeads ||
        geometry.top_k > ::shelf::kernel::kQsaSelectMaxTopK) {
        return Status::unsupported(
            "Qwen4Exp QSA select exceeds the indexer kernel limits", __FILE__, __LINE__);
    }
    if (pg.value_features(normed) != geometry.hidden) {
        return Status::invalid_argument("Qwen4Exp QSA indexer input feature mismatch",
                                        __FILE__, __LINE__);
    }
    if (weights.qk_proj == nullptr ||
        weights.qk_proj->rows != index_width + key_width ||
        weights.qk_proj->cols != geometry.hidden) {
        return Status::invalid_argument("Qwen4Exp QSA index_qk_proj shape mismatch",
                                        __FILE__, __LINE__);
    }

    const RT::ValueId qk = em.linear(*weights.qk_proj, indices.w_qk_proj, normed);
    name_last("index_qk_proj");

    const RT::ValueId q_raw = pg.alloc_value(index_width, RT::ValueDType::BF16);
    const RT::ValueId k_raw = pg.alloc_value(key_width, RT::ValueDType::BF16);
    pg.views.push_back(RT::PrimitiveValueView{q_raw, qk, 0u, index_width});
    pg.views.push_back(RT::PrimitiveValueView{k_raw, qk, index_width, key_width});

    RT::RmsNormNode q_norm;
    q_norm.shape = RT::RowwiseShapeKey{index_width};
    q_norm.eps = geometry.eps;
    q_norm.group_size = geometry.index_head_dim;
    q_norm.parameter_index = indices.p_q_norm;
    q_norm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId q_normed = emit_unary(em, RT::PrimitiveNode{q_norm}, q_raw,
                                            index_width);
    name_last("index_q_norm");

    const RT::ValueId q = emit_unary(
        em,
        RT::PrimitiveNode{RT::QsaRopeNode{RT::RowwiseShapeKey{index_width},
                                          geometry.index_heads,
                                          geometry.index_head_dim,
                                          geometry.index_rotary_dim, geometry.theta,
                                          geometry.position_scale}},
        q_normed, index_width);
    name_last("index_q_rope");

    const RT::StateId qsa_state =
        pg.alloc_state(RT::PrimitiveStateKind::QSA_KEY_STATE, geometry.state_index);
    RT::QsaKeyCacheNode key_cache;
    key_cache.shape = RT::RowwiseShapeKey{geometry.index_head_dim};
    key_cache.parameter_index = indices.p_k_norm;
    key_cache.state_index = geometry.state_index;
    key_cache.compress_ratio = geometry.compress_ratio;
    key_cache.rotary_dim = geometry.index_rotary_dim;
    key_cache.max_blocks = geometry.blocks;
    key_cache.theta = geometry.theta;
    key_cache.position_scale = geometry.position_scale;
    em.emit(RT::PrimitiveNode{key_cache}, {k_raw}, {}, {}, {qsa_state});
    name_last("index_key_cache");

    RT::QsaSelectNode select;
    select.shape = RT::RowwiseShapeKey{index_width};
    select.state_index = geometry.state_index;
    select.n_heads = geometry.index_heads;
    select.head_dim = geometry.index_head_dim;
    select.blocks = geometry.blocks;
    select.compress_ratio = geometry.compress_ratio;
    select.top_k = geometry.top_k;
    select.mask_bytes = geometry.mask_bytes;
    select.scale = 1.0f / std::sqrt(static_cast<float>(geometry.index_head_dim));
    const RT::ValueId mask = pg.alloc_value(geometry.mask_bytes, RT::ValueDType::U8);
    em.emit(RT::PrimitiveNode{select}, {q}, {mask}, {qsa_state});
    name_last("index_select");

    return Qwen4ExpQsaIndexerOutputs{
        .q = q,
        .mask = mask,
    };
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

}  // namespace qwen4exp
}  // namespace shelf
