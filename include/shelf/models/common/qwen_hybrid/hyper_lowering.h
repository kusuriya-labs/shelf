#pragma once

#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/weights/matrix_weight.h>

#include <cstdint>
#include <string>

namespace ps {
namespace qwen_hybrid {

struct HyperLoweringGeometry {
    std::uint32_t streams = 0;
    std::uint32_t hidden = 0;
    std::uint32_t lowrank = 0;
    float eps = 1e-6f;
};

struct HyperLoweringWeights {
    const ::ps::weights::MatrixWeight* mix_down = nullptr;
    const ::ps::weights::MatrixWeight* mix_up = nullptr;
    const ::ps::weights::MatrixWeight* block_inject = nullptr;
};

struct HyperLoweringIndices {
    std::uint32_t w_mix_down = 0;
    std::uint32_t w_mix_up = 0;
    std::uint32_t w_block_inject = 0;
    std::uint32_t p_hc_norm = 0;
};

struct HyperMixedOutputs {
    RT::ValueId normed{0};
    RT::ValueId mixed{0};
};

struct HyperReadOutputs {
    RT::ValueId mixed{0};
    RT::ValueId inject{0};
};

inline HyperMixedOutputs emit_hyper_mixed(
    Emitter& em,
    const HyperLoweringGeometry& geo,
    const HyperLoweringWeights& w,
    const HyperLoweringIndices& idx,
    RT::ValueId x4,
    const std::string& debug_prefix) {

    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = debug_prefix + suffix;
    };

    const std::uint32_t hc_hidden = geo.streams * geo.hidden;
    const float stream_scale = 1.0f / static_cast<float>(geo.streams);

    RT::RmsNormNode hc_rms;
    hc_rms.shape = RT::RowwiseShapeKey{hc_hidden};
    hc_rms.eps = geo.eps;
    hc_rms.group_size = geo.hidden;
    hc_rms.parameter_index = idx.p_hc_norm;
    hc_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    const RT::ValueId v_normed =
        em.unary(RT::PrimitiveNode{hc_rms}, x4, hc_hidden, RT::ValueDType::F32);
    name_last("hyper_hc_norm");

    const RT::ValueId v_down =
        em.linear(*w.mix_down, idx.w_mix_down, v_normed, RT::ValueDType::F32);
    name_last("hyper_down");
    const RT::ValueId v_down_scaled = em.unary(
        RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{geo.lowrank}, stream_scale}},
        v_down, geo.lowrank, RT::ValueDType::F32);
    name_last("hyper_down_scale");
    const RT::ValueId v_act = em.unary(
        RT::PrimitiveNode{RT::SiLUNode{RT::RowwiseShapeKey{geo.lowrank}}},
        v_down_scaled, geo.lowrank, RT::ValueDType::F32);
    name_last("hyper_act");
    const RT::ValueId v_up =
        em.linear(*w.mix_up, idx.w_mix_up, v_act, RT::ValueDType::F32);
    name_last("hyper_up");
    const RT::ValueId v_gate = em.unary(
        RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{hc_hidden}}},
        v_up, hc_hidden, RT::ValueDType::F32);
    name_last("hyper_gate");

    const RT::ValueId v_gated = em.value(hc_hidden, RT::ValueDType::F32);
    {
        RT::MulNode mn{RT::RowwiseShapeKey{hc_hidden}};
        em.emit(RT::PrimitiveNode{mn}, {v_gate, v_normed}, {v_gated});
        name_last("hyper_gated");
    }

    RT::GroupMeanNode gm;
    gm.shape = RT::RowwiseShapeKey{geo.hidden};
    gm.group_size = geo.streams;
    const RT::ValueId mixed = em.value(geo.hidden, RT::ValueDType::F32);
    em.emit(RT::PrimitiveNode{gm}, {v_gated}, {mixed});
    name_last("hyper_mixed");

    return HyperMixedOutputs{v_normed, mixed};
}

inline HyperReadOutputs emit_hyper_read(
    Emitter& em,
    const HyperLoweringGeometry& geo,
    const HyperLoweringWeights& w,
    const HyperLoweringIndices& idx,
    RT::ValueId x4,
    const std::string& debug_prefix) {

    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = debug_prefix + suffix;
    };

    const float stream_scale = 1.0f / static_cast<float>(geo.streams);

    const HyperMixedOutputs m = emit_hyper_mixed(em, geo, w, idx, x4, debug_prefix);

    const RT::ValueId v_inj_linear =
        em.linear(*w.block_inject, idx.w_block_inject, m.normed, RT::ValueDType::F32);
    name_last("hyper_inject_linear");
    const RT::ValueId v_inj_scaled = em.unary(
        RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{geo.streams}, stream_scale}},
        v_inj_linear, geo.streams, RT::ValueDType::F32);
    name_last("hyper_inject_scale");
    const RT::ValueId v_inj_gate = em.unary(
        RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{geo.streams}}},
        v_inj_scaled, geo.streams, RT::ValueDType::F32);
    name_last("hyper_inject_sigmoid");
    const RT::ValueId inject = em.unary(
        RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{geo.streams}, 2.0f}},
        v_inj_gate, geo.streams, RT::ValueDType::F32);
    name_last("hyper_inject");

    return HyperReadOutputs{m.mixed, inject};
}

inline RT::ValueId emit_hyper_write(
    Emitter& em,
    const HyperLoweringGeometry& geo,
    RT::ValueId x4,
    RT::ValueId sub_out,
    RT::ValueId inject,
    const std::string& debug_prefix) {

    auto& pg = em.pg;
    const std::uint32_t hc_hidden = geo.streams * geo.hidden;

    RT::GroupBroadcastMulNode gm;
    gm.shape = RT::RowwiseShapeKey{geo.hidden};
    gm.group_size = geo.streams;
    const RT::ValueId injected = em.value(hc_hidden, RT::ValueDType::F32);
    em.emit(RT::PrimitiveNode{gm}, {sub_out, inject}, {injected});
    pg.nodes.back().debug_name = debug_prefix + "hyper_inject_expand";

    RT::ResidualAddNode ra{RT::RowwiseShapeKey{hc_hidden}};
    const RT::ValueId out = em.value(hc_hidden, RT::ValueDType::F32);
    em.emit(RT::PrimitiveNode{ra}, {x4, injected}, {out});
    pg.nodes.back().debug_name = debug_prefix + "hyper_write_add";
    return out;
}

inline RT::ValueId emit_hyper_collapse(
    Emitter& em,
    const HyperLoweringGeometry& geo,
    const HyperLoweringWeights& w,
    const HyperLoweringIndices& idx,
    RT::ValueId x4,
    const std::string& debug_prefix) {
    const HyperMixedOutputs m = emit_hyper_mixed(em, geo, w, idx, x4, debug_prefix);
    return m.mixed;
}

}
}
