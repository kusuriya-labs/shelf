#pragma once

#include <phaseshift/models/common/qwen_hybrid/emitter.h>
#include <phaseshift/weights/matrix_weight.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace ps {
namespace qwen_hybrid {

struct GatedAttentionLoweringGeometry {
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rotary_dim = 0;
    float rope_theta = 10000000.0f;
    float rope_attention_factor = 1.0f;
    float eps = 1e-6f;
    RT::ValueDType out_dtype = RT::ValueDType::BF16;
};

struct GatedAttentionLoweringWeights {
    const ::ps::weights::MatrixWeight* q = nullptr;
    const ::ps::weights::MatrixWeight* k = nullptr;
    const ::ps::weights::MatrixWeight* v = nullptr;
    const ::ps::weights::MatrixWeight* o = nullptr;
};

struct GatedAttentionLoweringIndices {
    std::uint32_t w_q = 0;
    std::uint32_t w_k = 0;
    std::uint32_t w_v = 0;
    std::uint32_t w_o = 0;
    std::uint32_t p_q_norm = 0;
    std::uint32_t p_k_norm = 0;
    std::uint32_t state_index = 0;
};

inline RT::ValueId emit_gated_attention(
    Emitter& em,
    const GatedAttentionLoweringGeometry& geo,
    const GatedAttentionLoweringWeights& w,
    const GatedAttentionLoweringIndices& idx,
    RT::ValueId normed,
    const std::string& debug_prefix) {

    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = debug_prefix + suffix;
    };

    const uint32_t q_features = geo.query_heads * geo.head_dim;
    const uint32_t kv_features = geo.kv_heads * geo.head_dim;
    const uint32_t q_proj_features = 2 * q_features;
    const RT::AttentionShapeKey attn_key{geo.query_heads, geo.kv_heads,
                                         geo.head_dim, geo.rotary_dim};

    const RT::ValueId v_qproj = em.linear(*w.q, idx.w_q, normed);
    name_last("q_proj");
    const RT::ValueId v_k = em.linear(*w.k, idx.w_k, normed);
    name_last("k_proj");
    const RT::ValueId v_v = em.linear(*w.v, idx.w_v, normed);
    name_last("v_proj");

    const RT::ValueId v_q = em.value(q_features);
    const RT::ValueId v_gate = em.value(q_features);
    {
        RT::SplitNode sn;
        sn.input_shape = RT::RowwiseShapeKey{q_proj_features};
        sn.output_shape_a = RT::RowwiseShapeKey{q_features};
        sn.output_shape_b = RT::RowwiseShapeKey{q_features};
        sn.split_dim = 1;
        sn.layout = RT::SplitLayout::INTERLEAVED_HEADS;
        sn.head_dim = geo.head_dim;
        em.emit(RT::PrimitiveNode{sn}, {v_qproj}, {v_q, v_gate});
        name_last("q_gate_split");
    }

    RT::RmsNormNode q_rms;
    q_rms.shape = RT::RowwiseShapeKey{q_features};
    q_rms.eps = geo.eps;
    q_rms.group_size = geo.head_dim;
    q_rms.parameter_index = idx.p_q_norm;
    q_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId v_qn = em.unary(RT::PrimitiveNode{q_rms}, v_q, q_features,
                                      RT::ValueDType::F32);
    name_last("q_norm");
    const RT::ValueId v_qr = em.unary(
        RT::PrimitiveNode{RT::RoPENode{attn_key, geo.rotary_dim, geo.head_dim,
                                       geo.rope_theta, geo.rope_attention_factor}},
        v_qn, q_features);
    name_last("q_rope");

    RT::RmsNormNode k_rms;
    k_rms.shape = RT::RowwiseShapeKey{kv_features};
    k_rms.eps = geo.eps;
    k_rms.group_size = geo.head_dim;
    k_rms.parameter_index = idx.p_k_norm;
    k_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId v_kn = em.unary(RT::PrimitiveNode{k_rms}, v_k, kv_features,
                                      RT::ValueDType::F32);
    name_last("k_norm");
    const RT::ValueId v_kr = em.unary(
        RT::PrimitiveNode{RT::RoPENode{attn_key, geo.rotary_dim, geo.head_dim,
                                       geo.rope_theta, geo.rope_attention_factor}},
        v_kn, kv_features);
    name_last("k_rope");

    const RT::StateId kv_state = pg.alloc_state(
        RT::PrimitiveStateKind::KV_CACHE, idx.state_index);
    RT::KvAppendNode kan{attn_key, idx.state_index};
    em.emit(RT::PrimitiveNode{kan}, {v_kr, v_v}, {}, {}, {kv_state});
    name_last("kv_append");

    RT::ValueId v_ctx{0};
    {
        RT::PagedAttentionNode pan{
            attn_key,
            geo.head_dim ? 1.0f / sqrtf(static_cast<float>(geo.head_dim)) : 1.0f,
            idx.state_index};
        v_ctx = em.value(q_features, RT::ValueDType::F32);
        auto& pgn = pg.add_node(RT::PrimitiveNode{pan});
        pgn.inputs = {v_qr};
        pgn.outputs = {v_ctx};
        pgn.state_inputs = {kv_state};
        name_last("attn_core");
    }

    const RT::ValueId v_gs = em.unary(
        RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{q_features}}},
        v_gate, q_features, RT::ValueDType::F32);
    name_last("attn_gate");

    RT::ValueId v_att{0};
    {
        RT::MulNode mn{RT::RowwiseShapeKey{q_features}};
        v_att = em.value(q_features);
        em.emit(RT::PrimitiveNode{mn}, {v_ctx, v_gs}, {v_att});
        name_last("attn_gated");
    }
    const RT::ValueId block_out =
        em.linear(*w.o, idx.w_o, v_att, geo.out_dtype);
    name_last("o_proj");
    pg.nodes.back().tp_combine = 1;
    return block_out;
}

}
}
