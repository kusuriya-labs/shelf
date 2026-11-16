#pragma once

#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/models/qwen35qsa/weights/model_weights.h>
#include <shelf/runtime/graph/primitive_node.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace shelf {
namespace qwen35qsa {

// The indexer stage of one full-attention layer: project the query/key heads from the
// layer's normed input, score the sequence's pooled block keys and emit the block bitmap
// the layer's paged attention keeps. `hidden` is the backbone width the projections read.
struct QsaIndexerLowering {
    std::uint32_t hidden = 0;
    std::uint32_t heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rotary_dim = 0;
    std::uint32_t compress_ratio = 0;
    std::uint32_t blocks = 0;
    std::uint32_t top_k = 0;
    std::uint32_t mask_bytes = 0;
    float theta = 10000000.0f;
    float eps = 1.0e-6f;
    std::uint32_t state_index = 0;

    const QsaIndexerLayerWeights* weights = nullptr;
    std::uint32_t w_q_proj = 0;
    std::uint32_t w_k_proj = 0;
    std::uint32_t p_q_norm = 0;
    std::uint32_t p_k_norm = 0;
    std::uint32_t p_k_block_norm = 0;
};

// Emits the stage between the layer's input norm and its attention. Geometry and weight
// shapes are validated before lowering, so nothing here can fail.
inline ::shelf::runtime::ValueId emit_qsa_block_mask(
    ::shelf::qwen_hybrid::Emitter& em,
    const QsaIndexerLowering& qsa,
    ::shelf::runtime::ValueId normed,
    const std::string& debug_prefix) {
    namespace RT = ::shelf::runtime;
    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = debug_prefix + suffix;
    };

    const std::uint32_t q_features = qsa.heads * qsa.head_dim;
    const std::uint32_t k_features = qsa.head_dim;

    const RT::ValueId q_raw = em.linear(qsa.weights->q_proj, qsa.w_q_proj, normed);
    name_last("index_q_proj");
    const RT::ValueId k_raw = em.linear(qsa.weights->k_proj, qsa.w_k_proj, normed);
    name_last("index_k_proj");

    // One `head_dim` norm vector is shared by every query head: the runtime expresses that
    // as a per-group weight (elements == group_size), not a per-feature one.
    RT::RmsNormNode q_norm;
    q_norm.shape = RT::RowwiseShapeKey{q_features};
    q_norm.eps = qsa.eps;
    q_norm.group_size = qsa.head_dim;
    q_norm.parameter_index = qsa.p_q_norm;
    q_norm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId q_normed = em.unary(RT::PrimitiveNode{q_norm}, q_raw, q_features);
    name_last("index_q_norm");

    const RT::ValueId q_roped = em.unary(
        RT::PrimitiveNode{RT::QsaRopeNode{RT::RowwiseShapeKey{q_features}, qsa.heads,
                                          qsa.head_dim, qsa.rotary_dim, qsa.theta,
                                          1.0f}},
        q_normed, q_features);
    name_last("index_q_rope");

    // The reference normalizes the raw key head before pooling and normalizes the pooled
    // block key again (with its own parameter) before the block's rotary.
    RT::RmsNormNode k_norm;
    k_norm.shape = RT::RowwiseShapeKey{k_features};
    k_norm.eps = qsa.eps;
    k_norm.group_size = qsa.head_dim;
    k_norm.parameter_index = qsa.p_k_norm;
    k_norm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId k_normed = em.unary(RT::PrimitiveNode{k_norm}, k_raw, k_features);
    name_last("index_k_norm");

    const RT::StateId qsa_state =
        pg.alloc_state(RT::PrimitiveStateKind::QSA_KEY_STATE, qsa.state_index);
    RT::QsaKeyCacheNode key_cache;
    key_cache.shape = RT::RowwiseShapeKey{qsa.head_dim};
    key_cache.parameter_index = qsa.p_k_block_norm;
    key_cache.state_index = qsa.state_index;
    key_cache.compress_ratio = qsa.compress_ratio;
    key_cache.rotary_dim = qsa.rotary_dim;
    key_cache.max_blocks = qsa.blocks;
    key_cache.theta = qsa.theta;
    key_cache.position_scale = 1.0f;
    em.emit(RT::PrimitiveNode{key_cache}, {k_normed}, {}, {}, {qsa_state});
    name_last("index_key_cache");

    RT::QsaSelectNode select;
    select.shape = RT::RowwiseShapeKey{q_features};
    select.state_index = qsa.state_index;
    select.n_heads = qsa.heads;
    select.head_dim = qsa.head_dim;
    select.blocks = qsa.blocks;
    select.compress_ratio = qsa.compress_ratio;
    select.top_k = qsa.top_k;
    select.mask_bytes = qsa.mask_bytes;
    select.scale = 1.0f / std::sqrt(static_cast<float>(qsa.head_dim));
    const RT::ValueId mask = pg.alloc_value(qsa.mask_bytes, RT::ValueDType::U8);
    em.emit(RT::PrimitiveNode{select}, {q_roped}, {mask}, {qsa_state});
    name_last("index_select");
    return mask;
}

}
}
