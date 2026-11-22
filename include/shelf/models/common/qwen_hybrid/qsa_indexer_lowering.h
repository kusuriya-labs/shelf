#pragma once

#include <shelf/core/status.h>
#include <shelf/kernels/optimized/qsa.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/weights/matrix_weight.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace shelf {
namespace qwen_hybrid {

struct QsaIndexerGeometry {
    std::uint32_t hidden = 0;
    std::uint32_t heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rotary_dim = 0;
    std::uint32_t compress_ratio = 0;
    std::uint32_t blocks = 0;
    std::uint32_t top_k = 0;
    std::uint32_t mask_bytes = 0;
    std::uint32_t state_index = 0;
    float theta = 10000000.0f;
    float position_scale = 1.0f;
    float eps = 1e-6f;
    bool pre_pool_key_norm = false;
};

struct QsaIndexerWeights {
    const ::shelf::weights::MatrixWeight* qk_proj = nullptr;
    const ::shelf::weights::MatrixWeight* q_proj = nullptr;
    const ::shelf::weights::MatrixWeight* k_proj = nullptr;
};

struct QsaIndexerIndices {
    std::uint32_t w_qk_proj = 0;
    std::uint32_t w_q_proj = 0;
    std::uint32_t w_k_proj = 0;
    std::uint32_t p_q_norm = 0;
    std::uint32_t p_k_norm = 0;
    std::uint32_t p_k_block_norm = 0;
};

struct QsaIndexerOutputs {
    RT::ValueId q;
    RT::ValueId mask;
};

inline Result<QsaIndexerOutputs> emit_qsa_indexer(
    Emitter& em,
    const QsaIndexerGeometry& geometry,
    const QsaIndexerWeights& weights,
    const QsaIndexerIndices& indices,
    RT::ValueId normed,
    const char* debug_prefix) {
    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = std::string(debug_prefix) + suffix;
    };

    const std::uint32_t index_width = geometry.heads * geometry.head_dim;
    const std::uint32_t key_width = geometry.head_dim;
    const bool fused = weights.qk_proj != nullptr;

    if (geometry.hidden == 0u || geometry.heads == 0u ||
        geometry.head_dim != ::shelf::kernel::kQsaHeadDim ||
        geometry.rotary_dim == 0u || (geometry.rotary_dim & 1u) != 0u ||
        geometry.rotary_dim > geometry.head_dim ||
        geometry.compress_ratio == 0u || geometry.blocks == 0u ||
        geometry.top_k == 0u || geometry.mask_bytes * 8u < geometry.blocks) {
        return Status::invalid_argument("QSA indexer geometry is invalid",
                                        __FILE__, __LINE__);
    }
    if (geometry.heads > ::shelf::kernel::kQsaSelectMaxHeads ||
        geometry.top_k > ::shelf::kernel::kQsaSelectMaxTopK) {
        return Status::unsupported(
            "QSA select exceeds the indexer kernel limits", __FILE__, __LINE__);
    }
    if (pg.value_features(normed) != geometry.hidden) {
        return Status::invalid_argument("QSA indexer input feature mismatch",
                                        __FILE__, __LINE__);
    }
    if (fused) {
        if (weights.qk_proj->rows != index_width + key_width ||
            weights.qk_proj->cols != geometry.hidden) {
            return Status::invalid_argument("QSA indexer qk projection shape mismatch",
                                            __FILE__, __LINE__);
        }
    } else if (weights.q_proj == nullptr || weights.k_proj == nullptr ||
               weights.q_proj->rows != index_width ||
               weights.k_proj->rows != key_width ||
               weights.q_proj->cols != geometry.hidden ||
               weights.k_proj->cols != geometry.hidden) {
        return Status::invalid_argument("QSA indexer q/k projection shape mismatch",
                                        __FILE__, __LINE__);
    }

    RT::ValueId q_raw{};
    RT::ValueId k_raw{};
    if (fused) {
        const RT::ValueId qk = em.linear(*weights.qk_proj, indices.w_qk_proj, normed);
        name_last("index_qk_proj");
        q_raw = pg.alloc_value(index_width, RT::ValueDType::BF16);
        k_raw = pg.alloc_value(key_width, RT::ValueDType::BF16);
        pg.views.push_back(RT::PrimitiveValueView{q_raw, qk, 0u, index_width});
        pg.views.push_back(RT::PrimitiveValueView{k_raw, qk, index_width, key_width});
    } else {
        q_raw = em.linear(*weights.q_proj, indices.w_q_proj, normed);
        name_last("index_q_proj");
        k_raw = em.linear(*weights.k_proj, indices.w_k_proj, normed);
        name_last("index_k_proj");
    }

    RT::RmsNormNode q_norm;
    q_norm.shape = RT::RowwiseShapeKey{index_width};
    q_norm.eps = geometry.eps;
    q_norm.group_size = geometry.head_dim;
    q_norm.parameter_index = indices.p_q_norm;
    q_norm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId q_normed =
        em.unary(RT::PrimitiveNode{q_norm}, q_raw, index_width);
    name_last("index_q_norm");

    const RT::ValueId q = em.unary(
        RT::PrimitiveNode{RT::QsaRopeNode{RT::RowwiseShapeKey{index_width},
                                          geometry.heads, geometry.head_dim,
                                          geometry.rotary_dim, geometry.theta,
                                          geometry.position_scale}},
        q_normed, index_width);
    name_last("index_q_rope");

    RT::ValueId key_input = k_raw;
    std::uint32_t key_norm_parameter = indices.p_k_norm;
    if (geometry.pre_pool_key_norm) {
        RT::RmsNormNode k_norm;
        k_norm.shape = RT::RowwiseShapeKey{key_width};
        k_norm.eps = geometry.eps;
        k_norm.group_size = geometry.head_dim;
        k_norm.parameter_index = indices.p_k_norm;
        k_norm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
        key_input = em.unary(RT::PrimitiveNode{k_norm}, k_raw, key_width);
        name_last("index_k_norm");
        key_norm_parameter = indices.p_k_block_norm;
    }

    const RT::StateId qsa_state =
        pg.alloc_state(RT::PrimitiveStateKind::QSA_KEY_STATE, geometry.state_index);
    RT::QsaKeyCacheNode key_cache;
    key_cache.shape = RT::RowwiseShapeKey{geometry.head_dim};
    key_cache.parameter_index = key_norm_parameter;
    key_cache.state_index = geometry.state_index;
    key_cache.compress_ratio = geometry.compress_ratio;
    key_cache.rotary_dim = geometry.rotary_dim;
    key_cache.max_blocks = geometry.blocks;
    key_cache.theta = geometry.theta;
    key_cache.position_scale = geometry.position_scale;
    em.emit(RT::PrimitiveNode{key_cache}, {key_input}, {}, {}, {qsa_state});
    name_last("index_key_cache");

    RT::QsaSelectNode select;
    select.shape = RT::RowwiseShapeKey{index_width};
    select.state_index = geometry.state_index;
    select.n_heads = geometry.heads;
    select.head_dim = geometry.head_dim;
    select.blocks = geometry.blocks;
    select.compress_ratio = geometry.compress_ratio;
    select.top_k = geometry.top_k;
    select.mask_bytes = geometry.mask_bytes;
    select.scale = 1.0f / std::sqrt(static_cast<float>(geometry.head_dim));
    const RT::ValueId mask = pg.alloc_value(geometry.mask_bytes, RT::ValueDType::U8);
    em.emit(RT::PrimitiveNode{select}, {q}, {mask}, {qsa_state});
    name_last("index_select");

    return QsaIndexerOutputs{.q = q, .mask = mask};
}

}
}
