#include <shelf/models/qwen4exp/model/moe_lowering.h>

#include <string>

namespace shelf {
namespace qwen4exp {

namespace RT = ::shelf::runtime;
using ::shelf::qwen_hybrid::Emitter;
using ::shelf::weights::MatrixWeight;

namespace {

Status require_matrix(const MatrixWeight& weight, std::uint32_t rows, std::uint32_t cols,
                      const char* what) {
    if (weight.rows != rows || weight.cols != cols) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp MoE matrix mismatch: ") + what).c_str(), __FILE__,
            __LINE__);
    }
    return Status::make_ok();
}

Status require_experts(const MatrixWeight& weight, std::uint32_t local_experts,
                       std::uint32_t rows, std::uint32_t cols, const char* what) {
    if (weight.experts != local_experts || weight.rows != rows || weight.cols != cols) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp MoE expert plane mismatch: ") + what).c_str(), __FILE__,
            __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Result<RT::ValueId> emit_qwen4exp_moe(Emitter& em,
                                     const Qwen4ExpMoeLoweringGeometry& geometry,
                                     const Qwen4ExpMoeWeights& weights,
                                     const Qwen4ExpMoeLoweringIndices& indices,
                                     RT::ValueId post_norm,
                                     const char* debug_prefix) {
    (void)debug_prefix;
    if (geometry.global_experts == 0u || geometry.local_experts == 0u ||
        geometry.top_k == 0u || geometry.top_k > geometry.global_experts ||
        geometry.expert_begin + geometry.local_experts > geometry.global_experts ||
        geometry.hidden == 0u || geometry.intermediate == 0u ||
        geometry.shared_intermediate == 0u) {
        return Status::invalid_argument("Qwen4Exp MoE geometry is invalid", __FILE__,
                                        __LINE__);
    }
    if (em.pg.value_features(post_norm) != geometry.hidden) {
        return Status::invalid_argument("Qwen4Exp MoE input is not hidden width", __FILE__,
                                        __LINE__);
    }
    if (em.pg.grouped_row_shape.experts != geometry.local_experts ||
        em.pg.grouped_row_shape.top_k != geometry.top_k) {
        return Status::invalid_argument("Qwen4Exp grouped row shape differs from the plan",
                                        __FILE__, __LINE__);
    }

    Status st = require_matrix(weights.router, geometry.global_experts, geometry.hidden,
                               "router");
    if (!st.ok()) return st;
    st = require_matrix(weights.shared_gate, 1u, geometry.hidden, "shared_gate");
    if (!st.ok()) return st;
    st = require_matrix(weights.shared_gate_proj, geometry.shared_intermediate,
                        geometry.hidden, "shared_gate_proj");
    if (!st.ok()) return st;
    st = require_matrix(weights.shared_up_proj, geometry.shared_intermediate,
                        geometry.hidden, "shared_up_proj");
    if (!st.ok()) return st;
    st = require_matrix(weights.shared_down_proj, geometry.hidden,
                        geometry.shared_intermediate, "shared_down_proj");
    if (!st.ok()) return st;
    st = require_experts(weights.gate_experts, geometry.local_experts,
                         geometry.intermediate, geometry.hidden, "gate_experts");
    if (!st.ok()) return st;
    st = require_experts(weights.up_experts, geometry.local_experts,
                         geometry.intermediate, geometry.hidden, "up_experts");
    if (!st.ok()) return st;
    st = require_experts(weights.down_experts, geometry.local_experts, geometry.hidden,
                         geometry.intermediate, "down_experts");
    if (!st.ok()) return st;

    const RT::ValueId logits = em.linear(weights.router, indices.w_router, post_norm,
                                         RT::ValueDType::F32);
    em.pg.nodes.back().debug_name = std::string(debug_prefix) + "moe_router_logits";
    {
        RT::MoeRouterNode route;
        route.experts = geometry.global_experts;
        route.top_k = geometry.top_k;
        route.local_experts = geometry.local_experts;
        route.expert_begin = geometry.expert_begin;
        em.emit(RT::PrimitiveNode{route}, {logits}, {});
    }

    const RT::ValueId gate = em.pg.alloc_grouped_value(geometry.intermediate);
    {
        RT::MoeGroupedLinearNode n;
        n.shape = RT::MatrixwiseShapeKey{geometry.intermediate, geometry.hidden};
        n.weight_index = indices.w_gate_experts;
        n.out_dtype = RT::ValueDType::BF16;
        em.emit(RT::PrimitiveNode{n}, {post_norm}, {gate});
    }
    const RT::ValueId up = em.pg.alloc_grouped_value(geometry.intermediate);
    {
        RT::MoeGroupedLinearNode n;
        n.shape = RT::MatrixwiseShapeKey{geometry.intermediate, geometry.hidden};
        n.weight_index = indices.w_up_experts;
        n.out_dtype = RT::ValueDType::BF16;
        em.emit(RT::PrimitiveNode{n}, {post_norm}, {up});
    }
    const RT::ValueId act = em.pg.alloc_grouped_value(geometry.intermediate);
    {
        RT::SwiGluNode n{RT::RowwiseShapeKey{geometry.intermediate}};
        em.emit(RT::PrimitiveNode{n}, {gate, up}, {act});
    }
    const RT::ValueId expert_out = em.pg.alloc_grouped_value(geometry.hidden);
    {
        RT::MoeGroupedLinearNode n;
        n.shape = RT::MatrixwiseShapeKey{geometry.hidden, geometry.intermediate};
        n.weight_index = indices.w_down_experts;
        n.out_dtype = RT::ValueDType::BF16;
        em.emit(RT::PrimitiveNode{n}, {act}, {expert_out});
        em.pg.nodes.back().debug_name = std::string(debug_prefix) + "moe_expert_out";
    }

    const RT::ValueId shared_gate =
        em.linear(weights.shared_gate_proj, indices.w_shared_gate_proj, post_norm);
    em.pg.nodes.back().debug_name = std::string(debug_prefix) + "moe_shared_gate";
    const RT::ValueId shared_up =
        em.linear(weights.shared_up_proj, indices.w_shared_up_proj, post_norm);
    em.pg.nodes.back().debug_name = std::string(debug_prefix) + "moe_shared_up";
    const RT::ValueId shared_act = em.value(geometry.shared_intermediate);
    {
        RT::SwiGluNode n{RT::RowwiseShapeKey{geometry.shared_intermediate}};
        em.emit(RT::PrimitiveNode{n}, {shared_gate, shared_up}, {shared_act});
    }
    em.pg.nodes.back().debug_name = std::string(debug_prefix) + "moe_shared_act";
    RT::ValueId shared_out =
        em.linear(weights.shared_down_proj, indices.w_shared_down_proj, shared_act);
    if (geometry.shared_scale != 1.0f) {
        RT::ValueId scaled = em.value(geometry.hidden);
        em.emit(RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{geometry.hidden},
                                                geometry.shared_scale}},
                {shared_out}, {scaled});
        shared_out = scaled;
    }
    em.pg.nodes.back().debug_name = std::string(debug_prefix) + "moe_shared_out";
    const RT::ValueId shared_logit =
        em.linear(weights.shared_gate, indices.w_shared_gate, post_norm,
                  RT::ValueDType::F32);
    const RT::ValueId shared_weight = em.value(1u, RT::ValueDType::F32);
    {
        RT::SigmoidNode n{RT::RowwiseShapeKey{1u}};
        em.emit(RT::PrimitiveNode{n}, {shared_logit}, {shared_weight});
    }

    const RT::ValueId output = em.value(geometry.hidden);
    {
        RT::MoeCombineNode n;
        n.shape = RT::RowwiseShapeKey{geometry.hidden};
        n.top_k = geometry.top_k;
        em.emit(RT::PrimitiveNode{n}, {expert_out, shared_out, shared_weight}, {output});
    }
    // The routed experts are an expert-axis partial sum; the shared expert is already
    // complete on the contributing rank.
    em.pg.nodes.back().debug_name = std::string(debug_prefix) + "moe_combine";
    em.pg.nodes.back().tp_combine = 1;
    return output;
}

}  // namespace qwen4exp
}  // namespace shelf
