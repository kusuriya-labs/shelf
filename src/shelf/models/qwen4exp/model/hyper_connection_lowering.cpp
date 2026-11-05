#include <shelf/models/qwen4exp/model/hyper_connection_lowering.h>

#include <string>

namespace shelf {
namespace qwen4exp {

namespace RT = ::shelf::runtime;
using ::shelf::qwen_hybrid::Emitter;
using ::shelf::weights::MatrixWeight;

namespace {

RT::ValueId emit_unary(Emitter& em, const RT::PrimitiveNode& node, RT::ValueId input,
                       std::uint32_t features, RT::ValueDType dtype) {
    RT::ValueId out = em.value(features, dtype);
    em.emit(node, {input}, {out});
    return out;
}

Status require_matrix(const MatrixWeight* weight, std::uint32_t rows, std::uint32_t cols,
                      const char* what) {
    if (weight == nullptr || weight->rows != rows || weight->cols != cols) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp HyperConnection matrix mismatch: ") + what).c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Result<HyperConnectionMixOutput> emit_hyper_connection_mix(
    Emitter& em,
    const HyperConnectionSiteGeometry& geometry,
    const HyperConnectionSiteWeights& weights,
    const HyperConnectionSiteIndices& indices,
    RT::ValueId hyper,
    bool with_inject,
    const char* debug_prefix) {
    (void)debug_prefix;
    if (geometry.hc == 0u || geometry.hidden == 0u || geometry.lowrank == 0u) {
        return Status::invalid_argument("Qwen4Exp HyperConnection geometry is empty",
                                        __FILE__, __LINE__);
    }
    const std::uint32_t width = geometry.hc * geometry.hidden;
    if (em.pg.value_features(hyper) != width ||
        em.pg.value_dtype(hyper) != RT::ValueDType::BF16) {
        return Status::invalid_argument("Qwen4Exp HyperConnection input is not BF16 width",
                                        __FILE__, __LINE__);
    }
    Status st = require_matrix(weights.down, geometry.lowrank, width, "down");
    if (!st.ok()) return st;
    st = require_matrix(weights.up, width, geometry.lowrank, "up");
    if (!st.ok()) return st;
    if (with_inject) {
        st = require_matrix(weights.inject, geometry.hc, width, "inject");
        if (!st.ok()) return st;
    }

    RT::RmsNormNode norm_node;
    norm_node.shape = RT::RowwiseShapeKey{width};
    norm_node.eps = geometry.eps;
    norm_node.group_size = geometry.hidden;
    norm_node.parameter_index = geometry.norm_parameter_index;
    norm_node.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId normed =
        emit_unary(em, RT::PrimitiveNode{norm_node}, hyper, width, RT::ValueDType::BF16);

    const RT::ValueId down = em.linear(*weights.down, indices.w_down, normed);
    const float inverse_hc = 1.0f / static_cast<float>(geometry.hc);
    const RT::ValueId down_scaled = emit_unary(
        em, RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{geometry.lowrank},
                                            inverse_hc}},
        down, geometry.lowrank, RT::ValueDType::BF16);
    const RT::ValueId down_act = emit_unary(
        em, RT::PrimitiveNode{RT::SiLUNode{RT::RowwiseShapeKey{geometry.lowrank}}},
        down_scaled, geometry.lowrank, RT::ValueDType::BF16);
    const RT::ValueId up = em.linear(*weights.up, indices.w_up, down_act);
    const RT::ValueId gate = emit_unary(
        em, RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{width}}}, up, width,
        RT::ValueDType::BF16);

    RT::HyperMixNode mix_node;
    mix_node.shape = RT::RowwiseShapeKey{geometry.hidden};
    mix_node.hc = geometry.hc;
    mix_node.hidden = geometry.hidden;
    RT::ValueId mixed = em.value(geometry.hidden, RT::ValueDType::BF16);
    em.emit(RT::PrimitiveNode{mix_node}, {gate, normed}, {mixed});

    RT::ValueId injection{0};
    if (with_inject) {
        const RT::ValueId inject = em.linear(*weights.inject, indices.w_inject, normed);
        const RT::ValueId inject_scaled = emit_unary(
            em,
            RT::PrimitiveNode{
                RT::ScaleNode{RT::RowwiseShapeKey{geometry.hc}, inverse_hc}},
            inject, geometry.hc, RT::ValueDType::BF16);
        const RT::ValueId inject_act = emit_unary(
            em, RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{geometry.hc}}},
            inject_scaled, geometry.hc, RT::ValueDType::BF16);
        injection = emit_unary(
            em, RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{geometry.hc}, 2.0f}},
            inject_act, geometry.hc, RT::ValueDType::BF16);
    }
    return HyperConnectionMixOutput{mixed, injection};
}

Result<RT::ValueId> emit_hyper_connection_apply(
    Emitter& em,
    const HyperConnectionSiteGeometry& geometry,
    RT::ValueId hyper,
    RT::ValueId sublayer,
    RT::ValueId injection,
    const char* debug_prefix) {
    (void)debug_prefix;
    const std::uint32_t width = geometry.hc * geometry.hidden;
    if (em.pg.value_features(hyper) != width ||
        em.pg.value_features(sublayer) != geometry.hidden ||
        em.pg.value_features(injection) != geometry.hc) {
        return Status::invalid_argument(
            "Qwen4Exp HyperConnection apply feature counts are invalid", __FILE__,
            __LINE__);
    }
    RT::HyperApplyNode apply_node;
    apply_node.shape = RT::RowwiseShapeKey{width};
    apply_node.hc = geometry.hc;
    apply_node.hidden = geometry.hidden;
    RT::ValueId applied = em.value(width, RT::ValueDType::BF16);
    em.emit(RT::PrimitiveNode{apply_node}, {hyper, sublayer, injection}, {applied});
    return applied;
}

Result<HyperConnectionSiteOutput> emit_hyper_connection_site(
    Emitter& em,
    const HyperConnectionSiteGeometry& geometry,
    const HyperConnectionSiteWeights& weights,
    const HyperConnectionSiteIndices& indices,
    RT::ValueId hyper,
    RT::ValueId sublayer,
    bool with_inject,
    const char* debug_prefix) {
    auto mix = emit_hyper_connection_mix(em, geometry, weights, indices, hyper,
                                         with_inject, debug_prefix);
    if (!mix.ok()) return mix.status();
    if (!with_inject) {
        return HyperConnectionSiteOutput{mix.value().mixed, RT::ValueId{0}};
    }
    auto applied = emit_hyper_connection_apply(em, geometry, hyper, sublayer,
                                               mix.value().injection, debug_prefix);
    if (!applied.ok()) return applied.status();
    return HyperConnectionSiteOutput{mix.value().mixed, applied.value()};
}

}  // namespace qwen4exp
}  // namespace shelf
