#include <shelf/models/qwen4exp/model/ple_lowering.h>

#include <cmath>
#include <string>

namespace shelf {
namespace qwen4exp {

namespace RT = ::shelf::runtime;
using ::shelf::qwen_hybrid::Emitter;
using ::shelf::weights::MatrixWeight;

namespace {

Status require_matrix(const MatrixWeight* weight, std::uint32_t rows, std::uint32_t cols,
                      const char* what) {
    if (weight == nullptr || weight->rows != rows || weight->cols != cols) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp PLE matrix mismatch: ") + what).c_str(), __FILE__,
            __LINE__);
    }
    return Status::make_ok();
}

RT::ValueId emit_unary(Emitter& em, const RT::PrimitiveNode& node, RT::ValueId input,
                       std::uint32_t features) {
    RT::ValueId out = em.value(features, RT::ValueDType::BF16);
    em.emit(node, {input}, {out});
    return out;
}

}  // namespace

Result<RT::ValueId> emit_qwen4exp_ple(Emitter& em,
                                      const Qwen4ExpPleLoweringGeometry& geometry,
                                      const Qwen4ExpPleLoweringWeights& weights,
                                      const Qwen4ExpPleLoweringIndices& indices,
                                      RT::ValueId hyper,
                                      RT::ValueId embeddings,
                                      const char* debug_prefix) {
    (void)debug_prefix;
    if (geometry.hc == 0u || geometry.hidden == 0u || geometry.embed_dim == 0u) {
        return Status::invalid_argument("Qwen4Exp PLE geometry is empty", __FILE__,
                                        __LINE__);
    }
    const std::uint32_t width = geometry.hc * geometry.hidden;
    if (em.pg.value_features(hyper) != width ||
        em.pg.value_features(embeddings) != geometry.embed_dim) {
        return Status::invalid_argument("Qwen4Exp PLE input feature mismatch", __FILE__,
                                        __LINE__);
    }
    Status st = require_matrix(weights.key_proj, width, geometry.embed_dim, "key_proj");
    if (!st.ok()) return st;
    st = require_matrix(weights.value_proj, geometry.hidden, geometry.embed_dim,
                        "value_proj");
    if (!st.ok()) return st;

    const RT::ValueId key = em.linear(*weights.key_proj, indices.w_key_proj, embeddings);

    RT::RmsNormNode key_norm;
    key_norm.shape = RT::RowwiseShapeKey{width};
    key_norm.eps = geometry.eps;
    key_norm.group_size = geometry.hidden;
    key_norm.parameter_index = geometry.norm_key_parameter_index;
    key_norm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId key_normed =
        emit_unary(em, RT::PrimitiveNode{key_norm}, key, width);

    RT::RmsNormNode query_norm;
    query_norm.shape = RT::RowwiseShapeKey{width};
    query_norm.eps = geometry.eps;
    query_norm.group_size = geometry.hidden;
    query_norm.parameter_index = geometry.norm_query_parameter_index;
    query_norm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId query_normed =
        emit_unary(em, RT::PrimitiveNode{query_norm}, hyper, width);

    RT::HyperMixNode mix;
    mix.shape = RT::RowwiseShapeKey{geometry.hidden};
    mix.hc = geometry.hc;
    mix.hidden = geometry.hidden;
    RT::ValueId gate_raw = em.value(geometry.hidden, RT::ValueDType::BF16);
    em.emit(RT::PrimitiveNode{mix}, {key_normed, query_normed}, {gate_raw});

    const float gate_scale =
        static_cast<float>(geometry.hc) / std::sqrt(static_cast<float>(geometry.hidden));
    const RT::ValueId scaled = emit_unary(
        em, RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{geometry.hidden},
                                            gate_scale}},
        gate_raw, geometry.hidden);
    const RT::ValueId gate = emit_unary(
        em, RT::PrimitiveNode{RT::PleGateNode{RT::RowwiseShapeKey{geometry.hidden}}},
        scaled, geometry.hidden);

    const RT::ValueId value = em.linear(*weights.value_proj, indices.w_value_proj,
                                        embeddings);
    RT::PleBroadcastMulNode broadcast;
    broadcast.shape = RT::RowwiseShapeKey{width};
    broadcast.hc = geometry.hc;
    broadcast.hidden = geometry.hidden;
    RT::ValueId gated = em.value(width, RT::ValueDType::BF16);
    em.emit(RT::PrimitiveNode{broadcast}, {gate, value}, {gated});

    RT::RmsNormNode conv_norm;
    conv_norm.shape = RT::RowwiseShapeKey{width};
    conv_norm.eps = geometry.eps;
    conv_norm.group_size = geometry.hidden;
    conv_norm.parameter_index = geometry.norm_conv_parameter_index;
    conv_norm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId gated_normed =
        emit_unary(em, RT::PrimitiveNode{conv_norm}, gated, width);

    const RT::StateId conv_state =
        em.pg.alloc_state(RT::PrimitiveStateKind::PLE_CONV_STATE, geometry.state_index);
    RT::PleConvNode conv;
    conv.shape = RT::RowwiseShapeKey{width};
    conv.parameter_index = geometry.conv_weight_parameter_index;
    conv.state_index = geometry.state_index;
    RT::ValueId short_conv = em.value(width, RT::ValueDType::BF16);
    em.emit(RT::PrimitiveNode{conv}, {gated_normed}, {short_conv}, {conv_state},
            {conv_state}, {gated});

    RT::ValueId output = em.value(width, RT::ValueDType::BF16);
    em.emit(RT::PrimitiveNode{RT::ResidualAddNode{RT::RowwiseShapeKey{width}}},
            {gated, short_conv}, {output});
    RT::ValueId updated = em.value(width, RT::ValueDType::BF16);
    em.emit(RT::PrimitiveNode{RT::ResidualAddNode{RT::RowwiseShapeKey{width}}},
            {hyper, output}, {updated});
    return updated;
}

}  // namespace qwen4exp
}  // namespace shelf
