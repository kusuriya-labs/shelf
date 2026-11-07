#pragma once

#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/weights/matrix_weight.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace shelf {
namespace qwen_hybrid {

struct GdnLoweringGeometry {
    std::uint32_t key_heads = 0;
    std::uint32_t value_heads = 0;
    std::uint32_t key_head_dim = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel = 0;
    float eps = 1e-6f;
    RT::ValueDType out_dtype = RT::ValueDType::BF16;
    // Qwen3.5 hardcodes silu on the gate branch; Qwen3.8-Flash-Next declares sigmoid.
    bool sigmoid_gate = false;
    // True when the out projection is column-split across ranks and its partial sums must
    // be reduced. A family that replicates the GDN weights clears this.
    bool tp_combine = true;
};

struct GdnLoweringWeights {
    const ::shelf::weights::MatrixWeight* qkv = nullptr;
    const ::shelf::weights::MatrixWeight* z = nullptr;
    const ::shelf::weights::MatrixWeight* b = nullptr;
    const ::shelf::weights::MatrixWeight* a = nullptr;
    const ::shelf::weights::MatrixWeight* out = nullptr;
};

struct GdnLoweringIndices {
    std::uint32_t w_qkv = 0;
    std::uint32_t w_z = 0;
    std::uint32_t w_b = 0;
    std::uint32_t w_a = 0;
    std::uint32_t w_out = 0;
    std::uint32_t p_conv = 0;
    std::uint32_t p_dt_bias = 0;
    std::uint32_t p_a_log = 0;
    std::uint32_t p_gdn_norm = 0;
    std::uint32_t state_index = 0;
};

inline RT::ValueId emit_gdn(
    Emitter& em,
    const GdnLoweringGeometry& geo,
    const GdnLoweringWeights& w,
    const GdnLoweringIndices& idx,
    RT::ValueId normed,
    const std::string& debug_prefix) {

    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = debug_prefix + suffix;
    };

    const std::uint32_t qk_features = geo.key_heads * geo.key_head_dim;
    const std::uint32_t value_features = geo.value_heads * geo.value_head_dim;
    const std::uint32_t conv_features = 2 * qk_features + value_features;
    const RT::GdnShapeKey gdn_key{geo.key_heads, geo.value_heads,
                                  geo.key_head_dim, geo.value_head_dim,
                                  geo.conv_kernel};

    const RT::ValueId v_qkv = em.linear(*w.qkv, idx.w_qkv, normed);
    name_last("gdn_qkv");
    const RT::ValueId v_z = em.linear(*w.z, idx.w_z, normed);
    name_last("gdn_z");
    const RT::ValueId v_b = em.linear(*w.b, idx.w_b, normed);
    name_last("gdn_b");
    const RT::ValueId v_a = em.linear(*w.a, idx.w_a, normed);
    name_last("gdn_a");

    const RT::ValueId v_conv = em.value(conv_features, RT::ValueDType::F32);
    const RT::StateId conv_state =
        pg.alloc_state(RT::PrimitiveStateKind::GDN_CONV_STATE, idx.state_index);
    RT::StatefulCausalConv1DNode cn;
    cn.gdn = gdn_key;
    cn.parameter_index = idx.p_conv;
    cn.state_index = idx.state_index;
    em.emit(RT::PrimitiveNode{cn}, {v_qkv}, {v_conv},
            {conv_state}, {conv_state});
    name_last("gdn_conv");

    const RT::ValueId v_act = em.unary(
        RT::PrimitiveNode{RT::SiLUNode{RT::RowwiseShapeKey{conv_features}}},
        v_conv, conv_features);
    name_last("gdn_act");

    const RT::ValueId v_q_view = em.value(qk_features);
    const RT::ValueId v_k_view = em.value(qk_features);
    const RT::ValueId v_v_view = em.value(value_features);
    pg.views.push_back(RT::PrimitiveValueView{v_q_view, v_act, 0, qk_features});
    pg.views.push_back(
        RT::PrimitiveValueView{v_k_view, v_act, qk_features, qk_features});
    pg.views.push_back(
        RT::PrimitiveValueView{v_v_view, v_act, 2 * qk_features, value_features});

    const RT::ValueId v_nq = em.unary(
        RT::PrimitiveNode{RT::L2NormalizeNode{
            gdn_key, RT::RowwiseShapeKey{qk_features}, 1e-6f, geo.key_head_dim}},
        v_q_view, qk_features, RT::ValueDType::F32);
    name_last("gdn_q_normalize");
    const RT::ValueId v_nk = em.unary(
        RT::PrimitiveNode{RT::L2NormalizeNode{
            gdn_key, RT::RowwiseShapeKey{qk_features}, 1e-6f, geo.key_head_dim}},
        v_k_view, qk_features, RT::ValueDType::F32);
    name_last("gdn_k_normalize");
    const float scale = geo.key_head_dim
        ? 1.0f / sqrtf(static_cast<float>(geo.key_head_dim))
        : 1.0f;
    const RT::ValueId v_sq = em.unary(
        RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{qk_features}, scale}},
        v_nq, qk_features, RT::ValueDType::F32);
    name_last("gdn_q_scale");

    const RT::ValueId v_rec = em.value(value_features, RT::ValueDType::F32);
    const RT::StateId rec_state =
        pg.alloc_state(RT::PrimitiveStateKind::GDN_RECURRENCE_STATE,
                       idx.state_index);
    RT::GdnRecurrenceNode rn;
    rn.gdn = gdn_key;
    rn.dt_bias_parameter_index = idx.p_dt_bias;
    rn.a_log_parameter_index = idx.p_a_log;
    rn.state_index = idx.state_index;
    em.emit(RT::PrimitiveNode{rn}, {v_sq, v_nk, v_v_view, v_a, v_b}, {v_rec},
            {rec_state}, {rec_state}, {v_qkv});
    name_last("gdn_recurrence");

    RT::RmsNormNode gdn_rms;
    gdn_rms.shape = RT::RowwiseShapeKey{value_features};
    gdn_rms.eps = geo.eps;
    gdn_rms.group_size = geo.value_head_dim;
    gdn_rms.parameter_index = idx.p_gdn_norm;
    gdn_rms.weight_mode = RT::RmsNormWeightMode::DIRECT;
    const RT::ValueId v_rms = em.unary(RT::PrimitiveNode{gdn_rms}, v_rec,
                                       value_features, RT::ValueDType::F32);
    name_last("gdn_norm");

    const RT::PrimitiveNode gate_node = geo.sigmoid_gate
        ? RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{value_features}}}
        : RT::PrimitiveNode{RT::SiLUNode{RT::RowwiseShapeKey{value_features}}};
    const RT::ValueId v_gated_z = em.unary(gate_node, v_z, value_features,
                                           RT::ValueDType::F32);
    name_last(geo.sigmoid_gate ? "gdn_z_sigmoid" : "gdn_z_gate");

    RT::ValueId v_gated{0};
    {
        RT::MulNode mn{RT::RowwiseShapeKey{value_features}};
        v_gated = em.value(value_features);
        em.emit(RT::PrimitiveNode{mn}, {v_rms, v_gated_z}, {v_gated});
        name_last("gdn_gated");
    }
    const RT::ValueId block_out =
        em.linear(*w.out, idx.w_out, v_gated, geo.out_dtype);
    name_last("gdn_out");
    if (geo.tp_combine) pg.nodes.back().tp_combine = 1;
    return block_out;
}

}
}
