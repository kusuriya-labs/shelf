#include <shelf/models/qwen35moe/model/lower_to_primitives.h>
#include <shelf/models/common/qwen_hybrid/block_lowering.h>
#include <shelf/models/common/qwen_hybrid/head_lowering.h>
#include <shelf/models/common/qwen_hybrid/geometry_check.h>
#include <shelf/models/common/qwen_hybrid/weight_slot.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/runtime/graph/primitive_node.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace shelf::qwen35moe {

namespace {

namespace RT = ::shelf::runtime;
namespace QH = ::shelf::qwen_hybrid;

using QH::make_weight_slot;
using MW = ::shelf::weights::MatrixWeight;

struct LayerWeightIndices {
    uint32_t attn_q = 0;
    uint32_t attn_k = 0;
    uint32_t attn_v = 0;
    uint32_t attn_o = 0;
    uint32_t gdn_qkv = 0;
    uint32_t gdn_z = 0;
    uint32_t gdn_b = 0;
    uint32_t gdn_a = 0;
    uint32_t gdn_out = 0;

    uint32_t router = 0;
    uint32_t expert_w13 = 0;
    uint32_t expert_w2 = 0;
    uint32_t shared_gate = 0;
    uint32_t shared_up = 0;
    uint32_t shared_down = 0;
    uint32_t shared_scale = 0;
};

struct LayerParamIndices {
    uint32_t input_norm = UINT32_MAX;
    uint32_t post_norm = UINT32_MAX;
    uint32_t q_norm = UINT32_MAX;
    uint32_t k_norm = UINT32_MAX;
    uint32_t conv = UINT32_MAX;
    uint32_t gdn_norm = UINT32_MAX;
    uint32_t dt_bias = UINT32_MAX;
    uint32_t a_log = UINT32_MAX;
};

Status validate_hidden_taps(const Qwen35MoeLowerOptions& options, std::size_t layers) {
    for (std::size_t t = 0; t < options.hidden_taps.size(); ++t) {
        const std::size_t layer = options.hidden_taps[t];
        if (layer >= layers) {
            return Status::invalid_argument("hidden tap layer index out of range", __FILE__,
                                            __LINE__);
        }
        for (std::size_t u = 0; u < t; ++u) {
            if (static_cast<std::size_t>(options.hidden_taps[u]) == layer) {
                return Status::invalid_argument("duplicate hidden tap layer index", __FILE__,
                                                __LINE__);
            }
        }
    }
    return Status::make_ok();
}

// The one MoE FFN of one layer: a router over the routed experts, the two grouped expert
// projections with the gate / up fusion read as views, a shared expert of its own width,
// and the combine that folds the selected expert rows and the shared expert back into one
// row per token.
struct MoeFfnOutput {
    RT::ValueId layer_output;
};

Status emit_moe_ffn(
    QH::Emitter& em,
    const MoeExpertGroupPlan& plan,
    const LayerWeightIndices& weights,
    const MoeFfnWeights& ffn,
    RT::ValueId post_norm,
    RT::ValueId attention_residual,
    const RT::RowwiseShapeKey& hidden_key,
    uint32_t moe_intermediate,
    uint32_t shared_intermediate,
    uint32_t experts,
    uint32_t top_k,
    const std::string& layer_prefix,
    MoeFfnOutput& out) {

    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = layer_prefix + suffix;
    };

    RT::ValueId logits = em.linear(ffn.router, weights.router, post_norm, RT::ValueDType::F32);
    name_last("router");

    {
        RT::MoeRouterNode route;
        route.experts = experts;
        route.top_k = top_k;
        em.emit(RT::PrimitiveNode{route}, {logits}, {});
        name_last("route");
    }

    const RT::ValueId y13 = pg.alloc_value(plan.w13_rows, RT::ValueDType::BF16,
                                           RT::ValueRowDomain::GROUPED_ROWS);
    {
        RT::MoeGroupedLinearNode n;
        n.shape = RT::MatrixwiseShapeKey{plan.w13_rows, plan.w13_cols};
        n.weight_index = weights.expert_w13;
        n.out_dtype = RT::ValueDType::BF16;
        em.emit(RT::PrimitiveNode{n}, {post_norm}, {y13});
        name_last("expert_gate_up");
    }
    // The fused plane's two halves are read where they stand: gate rows first, then up
    // rows. Nothing is split out of the plane.
    const RT::ValueId gate_rows = pg.alloc_value(moe_intermediate, RT::ValueDType::BF16,
                                                 RT::ValueRowDomain::GROUPED_ROWS);
    const RT::ValueId up_rows = pg.alloc_value(moe_intermediate, RT::ValueDType::BF16,
                                               RT::ValueRowDomain::GROUPED_ROWS);
    pg.views.push_back(RT::PrimitiveValueView{gate_rows, y13, 0u, moe_intermediate});
    pg.views.push_back(RT::PrimitiveValueView{up_rows, y13, moe_intermediate, moe_intermediate});

    const RT::ValueId expert_act = pg.alloc_value(moe_intermediate, RT::ValueDType::BF16,
                                                  RT::ValueRowDomain::GROUPED_ROWS);
    {
        RT::SwiGluNode n{RT::RowwiseShapeKey{moe_intermediate}};
        em.emit(RT::PrimitiveNode{n}, {gate_rows, up_rows}, {expert_act});
        name_last("expert_swiglu");
    }

    const RT::ValueId expert_out = pg.alloc_value(plan.w2_rows, RT::ValueDType::BF16,
                                                  RT::ValueRowDomain::GROUPED_ROWS);
    {
        RT::MoeGroupedLinearNode n;
        n.shape = RT::MatrixwiseShapeKey{plan.w2_rows, plan.w2_cols};
        n.weight_index = weights.expert_w2;
        n.out_dtype = RT::ValueDType::BF16;
        em.emit(RT::PrimitiveNode{n}, {expert_act}, {expert_out});
        name_last("expert_down");
    }

    const RT::ValueId shared_gate = em.linear(ffn.shared_gate_proj, weights.shared_gate,
                                              post_norm);
    name_last("shared_gate");
    const RT::ValueId shared_up = em.linear(ffn.shared_up_proj, weights.shared_up, post_norm);
    name_last("shared_up");
    RT::ValueId shared_act = em.value(shared_intermediate);
    {
        RT::SwiGluNode n{RT::RowwiseShapeKey{shared_intermediate}};
        em.emit(RT::PrimitiveNode{n}, {shared_gate, shared_up}, {shared_act});
        name_last("shared_swiglu");
    }
    const RT::ValueId shared_out = em.linear(ffn.shared_down_proj, weights.shared_down,
                                             shared_act);
    name_last("shared_down");

    const RT::ValueId shared_logit =
        em.linear(ffn.shared_gate, weights.shared_scale, post_norm);
    name_last("shared_scale");
    RT::ValueId shared_weight = em.unary(RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{1u}}},
                                         shared_logit, 1u, RT::ValueDType::F32);
    name_last("shared_gate_act");

    RT::ValueId ffn_out = em.value(static_cast<uint32_t>(hidden_key.features));
    {
        RT::MoeCombineNode n;
        n.shape = hidden_key;
        n.top_k = top_k;
        em.emit(RT::PrimitiveNode{n}, {expert_out, shared_out, shared_weight}, {ffn_out});
        name_last("combine");
    }

    RT::ValueId layer_output = em.value(static_cast<uint32_t>(hidden_key.features));
    {
        RT::ResidualAddNode ra{hidden_key};
        em.emit(RT::PrimitiveNode{ra}, {ffn_out, attention_residual}, {layer_output});
        name_last("output");
    }

    out.layer_output = layer_output;
    return Status::make_ok();
}

}  // namespace

Result<QH::HybridLoweredPrimitives> lower_qwen35_moe_to_primitives(
    const Qwen35MoeTextConfig& config,
    const Qwen35MoeModelWeights& weights,
    const Qwen35MoeLowerOptions& options) {

    Status status = validate_qwen35moe_weights(weights, config);
    if (!status.ok()) return status;
    status = validate_hidden_taps(options, weights.layers.size());
    if (!status.ok()) return status;

    uint32_t experts = 0;
    uint32_t top_k = 0;
    uint32_t moe_intermediate = 0;
    uint32_t shared_intermediate = 0;
    uint32_t hidden_size = 0;
    status = QH::checked_u32(config.ffn.num_experts, "invalid expert count", experts);
    if (!status.ok()) return status;
    status = QH::checked_u32(config.ffn.num_experts_per_tok, "invalid expert selection width",
                             top_k);
    if (!status.ok()) return status;
    status = QH::checked_u32(config.ffn.moe_intermediate_size, "invalid routed expert width",
                             moe_intermediate);
    if (!status.ok()) return status;
    status = QH::checked_u32(config.ffn.shared_expert_intermediate_size,
                             "invalid shared expert width", shared_intermediate);
    if (!status.ok()) return status;
    status = QH::checked_u32(config.hidden_size, "invalid hidden size", hidden_size);
    if (!status.ok()) return status;
    if (top_k > experts) {
        return Status::invalid_argument("more experts are selected than exist", __FILE__,
                                        __LINE__);
    }
    uint32_t fused_rows = 0;
    status = QH::checked_double(moe_intermediate, "invalid fused expert width", fused_rows);
    if (!status.ok()) return status;

    QH::HybridLoweredPrimitives lowered;
    RT::PrimitiveGraph& pg = lowered.graph;
    pg.grouped_row_shape = RT::GroupedRowShape{experts, top_k};

    std::vector<const MW*> weight_pointers;
    auto add_weight = [&](const MW& w) -> uint32_t {
        weight_pointers.push_back(&w);
        lowered.weights.push_back(make_weight_slot(w));
        pg.weight_table.push_back(RT::MatrixwiseShapeKey{w.rows, w.cols});
        return static_cast<uint32_t>(weight_pointers.size() - 1);
    };

    std::vector<MoeExpertGroupPlan> plans(weights.layers.size());
    std::vector<LayerWeightIndices> layer_weights(weights.layers.size());
    const uint32_t embed_index = add_weight(weights.embed_tokens);
    for (std::size_t li = 0; li < weights.layers.size(); ++li) {
        auto plan_result = build_moe_expert_group_plan(config, li);
        if (!plan_result.ok()) return plan_result.status();
        plans[li] = plan_result.release();

        const auto& lw = weights.layers[li];
        auto& idx = layer_weights[li];
        if (lw.is_gdn) {
            idx.gdn_qkv = add_weight(lw.attn_in_proj_qkv);
            idx.gdn_z = add_weight(lw.attn_in_proj_z);
            idx.gdn_b = add_weight(lw.attn_in_proj_b);
            idx.gdn_a = add_weight(lw.attn_in_proj_a);
            idx.gdn_out = add_weight(lw.attn_out_proj);
        } else {
            idx.attn_q = add_weight(lw.attn_q_proj);
            idx.attn_k = add_weight(lw.attn_k_proj);
            idx.attn_v = add_weight(lw.attn_v_proj);
            idx.attn_o = add_weight(lw.attn_o_proj);
        }
        idx.router = add_weight(lw.ffn.router);
        idx.expert_w13 = add_weight(lw.ffn.expert_w13);
        idx.expert_w2 = add_weight(lw.ffn.expert_w2);
        idx.shared_gate = add_weight(lw.ffn.shared_gate_proj);
        idx.shared_up = add_weight(lw.ffn.shared_up_proj);
        idx.shared_down = add_weight(lw.ffn.shared_down_proj);
        idx.shared_scale = add_weight(lw.ffn.shared_gate);
    }
    const bool lm_tied = weights.lm_head_tied;
    const uint32_t lm_index = lm_tied ? embed_index : add_weight(weights.lm_head);

    float eps = config.rms_norm_eps;
    float rope_theta = config.rope_theta;
    float partial = config.partial_rotary_factor;
    if (eps == 0) eps = 1e-6f;
    if (rope_theta == 0) rope_theta = 10000000.0f;
    if (partial == 0) partial = 0.25f;
    const uint32_t head_dim = static_cast<uint32_t>(config.attention_head_dim);
    const uint32_t rotary_dim = static_cast<uint32_t>((double)head_dim * (double)partial);

    auto reg_param = [&](const gpu::Tensor& t) -> uint32_t {
        RT::StaticParameterSlot s{};
        s.device_ptr = t.data<void>();
        std::size_t n = 1;
        for (std::size_t i = 0; i < t.ndim(); ++i) n *= t.dim(i);
        s.elements = static_cast<uint32_t>(n);
        s.dtype = t.element_size() == 2 ? RT::ValueDType::BF16 : RT::ValueDType::F32;
        lowered.parameters.push_back(s);
        return static_cast<uint32_t>(lowered.parameters.size() - 1);
    };
    std::vector<LayerParamIndices> layer_params(weights.layers.size());
    for (std::size_t li = 0; li < weights.layers.size(); ++li) {
        const auto& lw = weights.layers[li];
        auto& lp = layer_params[li];
        if (lw.input_layernorm_weight.ndim() > 0)
            lp.input_norm = reg_param(lw.input_layernorm_weight);
        if (lw.post_attention_layernorm_weight.ndim() > 0)
            lp.post_norm = reg_param(lw.post_attention_layernorm_weight);
        if (lw.is_gdn) {
            if (lw.attn_conv1d_weight.ndim() > 0) lp.conv = reg_param(lw.attn_conv1d_weight);
            if (lw.attn_norm_weight.ndim() > 0) lp.gdn_norm = reg_param(lw.attn_norm_weight);
            if (lw.attn_dt_bias.ndim() > 0) lp.dt_bias = reg_param(lw.attn_dt_bias);
            if (lw.attn_A_log.ndim() > 0) lp.a_log = reg_param(lw.attn_A_log);
        } else {
            if (lw.attn_q_norm_weight.ndim() > 0) lp.q_norm = reg_param(lw.attn_q_norm_weight);
            if (lw.attn_k_norm_weight.ndim() > 0) lp.k_norm = reg_param(lw.attn_k_norm_weight);
        }
    }
    const uint32_t final_norm_param =
        weights.final_norm_weight.ndim() > 0 ? reg_param(weights.final_norm_weight) : UINT32_MAX;

    QH::Emitter em{pg};
    const RT::RowwiseShapeKey hidden_key{hidden_size};

    RT::ValueId tokens = em.value(0, RT::ValueDType::I32);
    RT::ValueId layer_input = em.value(hidden_size);
    {
        RT::EmbeddingLookupNode n;
        n.vocab_size = weights.embed_tokens.rows;
        n.hidden_size = hidden_size;
        n.weight_shape =
            RT::MatrixwiseShapeKey{weights.embed_tokens.rows, weights.embed_tokens.cols};
        n.weight_index = embed_index;
        em.emit(RT::PrimitiveNode{n}, {tokens}, {layer_input});
        pg.nodes.back().debug_name = "embedding";
        pg.external_inputs.push_back(tokens);
    }

    std::vector<RT::ValueId> hidden_tap_values(options.hidden_taps.size());
    std::vector<uint8_t> hidden_tap_found(options.hidden_taps.size(), 0);

    uint32_t next_gdn_state_index = 0;
    uint32_t next_attention_state_index = 0;

    for (std::size_t li = 0; li < weights.layers.size(); ++li) {
        const auto& lw = weights.layers[li];
        const auto& lw_idx = layer_weights[li];
        const auto& lp = layer_params[li];
        const std::string layer_prefix = "L" + std::to_string(li) + ".";

        QH::HybridBlockLowering block;
        block.is_gdn = lw.is_gdn;
        block.eps = eps;
        block.weight_input_norm = lp.input_norm;
        block.weight_post_norm = lp.post_norm;
        if (lw.is_gdn) {
            block.gdn_geometry.key_heads = static_cast<uint32_t>(config.linear_num_key_heads);
            block.gdn_geometry.value_heads =
                static_cast<uint32_t>(config.linear_num_value_heads);
            block.gdn_geometry.key_head_dim =
                static_cast<uint32_t>(config.linear_key_head_dim);
            block.gdn_geometry.value_head_dim =
                static_cast<uint32_t>(config.linear_value_head_dim);
            block.gdn_geometry.conv_kernel =
                static_cast<uint32_t>(config.linear_conv_kernel_dim);
            block.gdn_geometry.eps = eps;
            block.gdn_weights.qkv = weight_pointers[lw_idx.gdn_qkv];
            block.gdn_weights.z = weight_pointers[lw_idx.gdn_z];
            block.gdn_weights.b = weight_pointers[lw_idx.gdn_b];
            block.gdn_weights.a = weight_pointers[lw_idx.gdn_a];
            block.gdn_weights.out = weight_pointers[lw_idx.gdn_out];
            block.gdn_indices.w_qkv = lw_idx.gdn_qkv;
            block.gdn_indices.w_z = lw_idx.gdn_z;
            block.gdn_indices.w_b = lw_idx.gdn_b;
            block.gdn_indices.w_a = lw_idx.gdn_a;
            block.gdn_indices.w_out = lw_idx.gdn_out;
            block.gdn_indices.p_conv = lp.conv;
            block.gdn_indices.p_dt_bias = lp.dt_bias;
            block.gdn_indices.p_a_log = lp.a_log;
            block.gdn_indices.p_gdn_norm = lp.gdn_norm;
            block.gdn_indices.state_index = next_gdn_state_index++;
        } else {
            block.attention_geometry.query_heads =
                static_cast<uint32_t>(config.num_attention_heads);
            block.attention_geometry.kv_heads =
                static_cast<uint32_t>(config.num_key_value_heads);
            block.attention_geometry.head_dim = head_dim;
            block.attention_geometry.rotary_dim = rotary_dim;
            block.attention_geometry.rope_theta = rope_theta;
            block.attention_geometry.rope_attention_factor = options.rope_attention_factor;
            block.attention_geometry.eps = eps;
            block.attention_weights.q = weight_pointers[lw_idx.attn_q];
            block.attention_weights.k = weight_pointers[lw_idx.attn_k];
            block.attention_weights.v = weight_pointers[lw_idx.attn_v];
            block.attention_weights.o = weight_pointers[lw_idx.attn_o];
            block.attention_indices.w_q = lw_idx.attn_q;
            block.attention_indices.w_k = lw_idx.attn_k;
            block.attention_indices.w_v = lw_idx.attn_v;
            block.attention_indices.w_o = lw_idx.attn_o;
            block.attention_indices.p_q_norm = lp.q_norm;
            block.attention_indices.p_k_norm = lp.k_norm;
            block.attention_indices.state_index = next_attention_state_index++;
        }

        for (std::size_t o = 0; o < options.override_layers.size(); ++o) {
            if (static_cast<std::size_t>(options.override_layers[o]) == li)
                lowered.layer_input_values.emplace_back(static_cast<uint32_t>(li),
                                                        layer_input);
        }

        const QH::HybridBlockOutput blk =
            QH::emit_hybrid_block(em, hidden_key, block, layer_input, layer_prefix);

        MoeFfnOutput ffn_out;
        status = emit_moe_ffn(em, plans[li], lw_idx, lw.ffn, blk.post_norm,
                              blk.attention_residual, hidden_key, moe_intermediate,
                              shared_intermediate, experts, top_k, layer_prefix, ffn_out);
        if (!status.ok()) return status;
        layer_input = ffn_out.layer_output;

        for (std::size_t t = 0; t < options.hidden_taps.size(); ++t) {
            if (static_cast<std::size_t>(options.hidden_taps[t]) == li) {
                hidden_tap_values[t] = layer_input;
                hidden_tap_found[t] = 1;
            }
        }
    }

    const MW* lmw = lm_tied ? &weights.embed_tokens : &weights.lm_head;
    if (lmw->rows != config.vocab_size) {
        return Status::unsupported("MoE lowering requires an unsharded lm_head", __FILE__,
                                   __LINE__);
    }
    QH::HybridHeadLowering head;
    head.hidden_features = hidden_size;
    head.eps = eps;
    head.final_norm_param = final_norm_param;
    head.lm_shape = RT::MatrixwiseShapeKey{lmw->rows, lmw->cols};
    head.lm_weight_index = lm_index;
    head.out_vocab = lmw->rows;
    QH::emit_hybrid_head(em, head, layer_input);

    for (std::size_t t = 0; t < hidden_tap_values.size(); ++t) {
        if (hidden_tap_found[t] == 0) {
            return Status::invalid_state("hidden tap layer was not lowered", __FILE__,
                                         __LINE__);
        }
        pg.external_outputs.push_back(hidden_tap_values[t]);
    }

    status = RT::validate_primitive_graph(pg);
    if (!status.ok()) return status;

    return lowered;
}

}  // namespace shelf::qwen35moe
