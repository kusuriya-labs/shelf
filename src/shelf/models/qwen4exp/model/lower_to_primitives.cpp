#include <shelf/models/qwen4exp/model/lower_to_primitives.h>

#include <shelf/models/common/qwen_hybrid/head_lowering.h>
#include <shelf/models/common/qwen_hybrid/weight_slot.h>
#include <shelf/models/qwen4exp/model/hyper_connection_lowering.h>
#include <shelf/models/qwen4exp/model/layer_lowering.h>
#include <shelf/models/qwen4exp/model/qsa_lowering.h>
#include <shelf/runtime/graph/primitive_node.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace shelf {
namespace qwen4exp {

namespace RT = ::shelf::runtime;
namespace QH = ::shelf::qwen_hybrid;
using ::shelf::weights::MatrixWeight;

namespace {

struct LayerWeightIndices {
    std::uint32_t attn_q = 0;
    std::uint32_t attn_k = 0;
    std::uint32_t attn_v = 0;
    std::uint32_t attn_o = 0;
    std::uint32_t index_qk = 0;
    std::uint32_t gdn_qkv = 0;
    std::uint32_t gdn_z = 0;
    std::uint32_t gdn_b = 0;
    std::uint32_t gdn_a = 0;
    std::uint32_t gdn_out = 0;
    std::uint32_t router = 0;
    std::uint32_t shared_gate = 0;
    std::uint32_t shared_gate_proj = 0;
    std::uint32_t shared_up_proj = 0;
    std::uint32_t shared_down_proj = 0;
    std::uint32_t gate_experts = 0;
    std::uint32_t up_experts = 0;
    std::uint32_t down_experts = 0;
    std::uint32_t ple_key = 0;
    std::uint32_t ple_value = 0;
};

struct LayerParamIndices {
    std::uint32_t attn_norm = UINT32_MAX;
    std::uint32_t mlp_norm = UINT32_MAX;
    std::uint32_t conv = UINT32_MAX;
    std::uint32_t gdn_norm = UINT32_MAX;
    std::uint32_t dt_bias = UINT32_MAX;
    std::uint32_t a_log = UINT32_MAX;
    std::uint32_t q_norm = UINT32_MAX;
    std::uint32_t k_norm = UINT32_MAX;
    std::uint32_t index_q_norm = UINT32_MAX;
    std::uint32_t index_k_norm = UINT32_MAX;
    std::uint32_t ple_norm_key = UINT32_MAX;
    std::uint32_t ple_norm_query = UINT32_MAX;
    std::uint32_t ple_norm_conv = UINT32_MAX;
    std::uint32_t ple_conv = UINT32_MAX;
};

std::uint32_t tensor_elements(const gpu::Tensor& t) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < t.ndim(); ++i) n *= t.dim(i);
    return static_cast<std::uint32_t>(n);
}

}  // namespace

Result<QH::HybridLoweredPrimitives> lower_qwen4exp_to_primitives(
    const Qwen4ExpTextConfig& config, const Qwen4ExpRankWeights& weights,
    const Qwen4ExpLowerOptions& options) {
    if (options.max_context_tokens == 0u) {
        return Status::invalid_argument("Qwen4Exp lowering needs a non-zero context",
                                        __FILE__, __LINE__);
    }
    if (weights.layers.size() != config.num_hidden_layers) {
        return Status::invalid_argument("Qwen4Exp lowering layer count mismatch", __FILE__,
                                        __LINE__);
    }

    const std::uint32_t hidden = static_cast<std::uint32_t>(config.hidden_size);
    const std::uint32_t hc = static_cast<std::uint32_t>(config.hc_count);
    const std::uint32_t lowrank = static_cast<std::uint32_t>(config.hc_lowrank);
    const std::uint32_t stream_width = hc * hidden;
    const std::uint32_t vocab = static_cast<std::uint32_t>(config.vocab_size);
    const std::uint32_t global_experts =
        static_cast<std::uint32_t>(config.ffn.num_experts);
    const std::uint32_t top_k =
        static_cast<std::uint32_t>(config.ffn.num_experts_per_tok);
    const std::uint32_t intermediate =
        static_cast<std::uint32_t>(config.ffn.moe_intermediate_size);
    const std::uint32_t shared_intermediate =
        static_cast<std::uint32_t>(config.ffn.shared_expert_intermediate_size);
    const std::uint32_t compress =
        static_cast<std::uint32_t>(config.indexer.compress_ratio);
    const std::uint32_t index_blocks =
        (options.max_context_tokens + compress - 1u) / compress;
    if (hc == 0u || hidden == 0u || lowrank == 0u || global_experts == 0u ||
        top_k == 0u || compress == 0u) {
        return Status::invalid_argument("Qwen4Exp lowering geometry is empty", __FILE__,
                                        __LINE__);
    }

    const std::uint32_t local_experts =
        static_cast<std::uint32_t>(weights.layers.front().mlp.experts.local_experts);
    const std::uint32_t expert_begin =
        static_cast<std::uint32_t>(weights.layers.front().mlp.experts.expert_begin);
    for (const auto& layer : weights.layers) {
        if (layer.mlp.experts.local_experts != local_experts ||
            layer.mlp.experts.expert_begin != expert_begin) {
            return Status::invalid_argument(
                "Qwen4Exp lowering expert range differs between layers", __FILE__,
                __LINE__);
        }
    }

    float eps = config.rms_norm_eps;
    float rope_theta = config.rope_theta;
    float partial = config.partial_rotary_factor;
    if (eps == 0.0f) eps = 1e-6f;
    if (rope_theta == 0.0f) rope_theta = 10000000.0f;
    if (partial == 0.0f) partial = 0.25f;
    const std::uint32_t head_dim = static_cast<std::uint32_t>(config.attention_head_dim);
    const std::uint32_t rotary_dim =
        static_cast<std::uint32_t>(static_cast<double>(head_dim) * partial);

    QH::HybridLoweredPrimitives lowered;
    RT::PrimitiveGraph& pg = lowered.graph;
    pg.grouped_row_shape = RT::GroupedRowShape{local_experts, top_k};

    std::vector<const MatrixWeight*> weight_pointers;
    auto add_weight = [&](const MatrixWeight& w) -> std::uint32_t {
        weight_pointers.push_back(&w);
        lowered.weights.push_back(QH::make_weight_slot(w));
        pg.weight_table.push_back(RT::MatrixwiseShapeKey{w.rows, w.cols});
        return static_cast<std::uint32_t>(weight_pointers.size() - 1);
    };
    auto reg_param = [&](const gpu::Tensor& t) -> std::uint32_t {
        RT::StaticParameterSlot s{};
        s.device_ptr = t.data<void>();
        s.elements = tensor_elements(t);
        s.dtype = t.element_size() == 2 ? RT::ValueDType::BF16 : RT::ValueDType::F32;
        lowered.parameters.push_back(s);
        return static_cast<std::uint32_t>(lowered.parameters.size() - 1);
    };

    const std::uint32_t embed_index = add_weight(weights.embed_tokens);
    const std::uint32_t lm_index = add_weight(weights.lm_head);

    std::vector<LayerWeightIndices> layer_weights(weights.layers.size());
    std::vector<LayerParamIndices> layer_params(weights.layers.size());
    for (std::size_t li = 0; li < weights.layers.size(); ++li) {
        const auto& lw = weights.layers[li];
        auto& idx = layer_weights[li];
        if (lw.attention.is_gdn) {
            idx.gdn_qkv = add_weight(lw.attention.in_proj_qkv);
            idx.gdn_z = add_weight(lw.attention.in_proj_z);
            idx.gdn_b = add_weight(lw.attention.in_proj_b);
            idx.gdn_a = add_weight(lw.attention.in_proj_a);
            idx.gdn_out = add_weight(lw.attention.out_proj);
        } else {
            idx.attn_q = add_weight(lw.attention.q_proj);
            idx.attn_k = add_weight(lw.attention.k_proj);
            idx.attn_v = add_weight(lw.attention.v_proj);
            idx.attn_o = add_weight(lw.attention.o_proj);
            idx.index_qk = add_weight(lw.attention.index_qk_proj);
        }
        idx.router = add_weight(lw.mlp.router);
        idx.shared_gate = add_weight(lw.mlp.shared_gate);
        idx.shared_gate_proj = add_weight(lw.mlp.shared_gate_proj);
        idx.shared_up_proj = add_weight(lw.mlp.shared_up_proj);
        idx.shared_down_proj = add_weight(lw.mlp.shared_down_proj);
        idx.gate_experts = add_weight(lw.mlp.gate_experts);
        idx.up_experts = add_weight(lw.mlp.up_experts);
        idx.down_experts = add_weight(lw.mlp.down_experts);
        if (lw.ple.present) {
            idx.ple_key = add_weight(lw.ple.key_proj);
            idx.ple_value = add_weight(lw.ple.value_proj);
        }

        auto& lp = layer_params[li];
        lp.attn_norm = reg_param(lw.attn_hyper_connection.hc_norm);
        lp.mlp_norm = reg_param(lw.mlp_hyper_connection.hc_norm);
        if (lw.attention.is_gdn) {
            lp.conv = reg_param(lw.attention.conv1d);
            lp.gdn_norm = reg_param(lw.attention.norm);
            lp.dt_bias = reg_param(lw.attention.dt_bias);
            lp.a_log = reg_param(lw.attention.a_log);
        } else {
            lp.q_norm = reg_param(lw.attention.q_norm);
            lp.k_norm = reg_param(lw.attention.k_norm);
            lp.index_q_norm = reg_param(lw.attention.index_q_layernorm);
            lp.index_k_norm = reg_param(lw.attention.index_k_layernorm);
        }
        if (lw.ple.present) {
            lp.ple_norm_key = reg_param(lw.ple.norm_key);
            lp.ple_norm_query = reg_param(lw.ple.norm_query);
            lp.ple_norm_conv = reg_param(lw.ple.norm_conv);
            lp.ple_conv = reg_param(lw.ple.conv1d);
        }
    }

    const std::uint32_t mixer_down = add_weight(weights.mixer.input_mix_down);
    const std::uint32_t mixer_up = add_weight(weights.mixer.input_mix_up);
    const std::uint32_t mixer_norm = reg_param(weights.mixer.hc_norm);

    QH::Emitter em{pg};
    RT::ValueId tokens = em.value(0, RT::ValueDType::I32);
    pg.external_inputs.push_back(tokens);

    RT::ValueId embedded = em.value(hidden);
    {
        RT::EmbeddingLookupNode n;
        n.vocab_size = weights.embed_tokens.rows;
        n.hidden_size = hidden;
        n.weight_shape =
            RT::MatrixwiseShapeKey{weights.embed_tokens.rows, weights.embed_tokens.cols};
        n.weight_index = embed_index;
        em.emit(RT::PrimitiveNode{n}, {tokens}, {embedded});
        pg.nodes.back().debug_name = "embedding";
    }

    RT::ValueId hyper = em.value(stream_width);
    {
        RT::RepeatStreamsNode n;
        n.shape = RT::RowwiseShapeKey{stream_width};
        n.hc = hc;
        n.hidden = hidden;
        em.emit(RT::PrimitiveNode{n}, {embedded}, {hyper});
        pg.nodes.back().debug_name = "hc_init";
    }

    const bool has_ple =
        std::any_of(weights.layers.begin(), weights.layers.end(),
                    [](const Qwen4ExpLayerWeights& lw) { return lw.ple.present; });
    RT::ValueId ple_embeddings{0};
    if (has_ple) {
        ple_embeddings = em.value(static_cast<std::uint32_t>(config.ple.embed_dim));
        pg.external_inputs.push_back(ple_embeddings);
    }

    std::uint32_t next_gdn_state = 0;
    std::uint32_t next_attention_state = 0;
    std::uint32_t next_ple_state = 0;

    for (std::size_t li = 0; li < weights.layers.size(); ++li) {
        const auto& lw = weights.layers[li];
        const auto& idx = layer_weights[li];
        const auto& lp = layer_params[li];
        const std::string prefix = "L" + std::to_string(li) + ".";

        Qwen4ExpLayerLowering layer;
        layer.is_gdn = lw.attention.is_gdn;
        layer.hyper.hc = hc;
        layer.hyper.hidden = hidden;
        layer.hyper.lowrank = lowrank;
        layer.hyper.eps = eps;
        layer.hyper.norm_parameter_index = UINT32_MAX;

        layer.attn_hyper_weights.down = &lw.attn_hyper_connection.input_mix_down;
        layer.attn_hyper_weights.up = &lw.attn_hyper_connection.input_mix_up;
        layer.attn_hyper_weights.inject = &lw.attn_hyper_connection.block_inject;
        layer.attn_hyper_indices = {
            add_weight(lw.attn_hyper_connection.input_mix_down),
            add_weight(lw.attn_hyper_connection.input_mix_up),
            add_weight(lw.attn_hyper_connection.block_inject),
        };
        layer.mlp_hyper_weights.down = &lw.mlp_hyper_connection.input_mix_down;
        layer.mlp_hyper_weights.up = &lw.mlp_hyper_connection.input_mix_up;
        layer.mlp_hyper_weights.inject = &lw.mlp_hyper_connection.block_inject;
        layer.mlp_hyper_indices = {
            add_weight(lw.mlp_hyper_connection.input_mix_down),
            add_weight(lw.mlp_hyper_connection.input_mix_up),
            add_weight(lw.mlp_hyper_connection.block_inject),
        };
        layer.hyper.norm_parameter_index = lp.attn_norm;
        layer.mlp_hyper_norm_parameter_index = lp.mlp_norm;

        if (lw.attention.is_gdn) {
            layer.gdn_geometry.key_heads =
                static_cast<std::uint32_t>(config.linear_num_key_heads);
            layer.gdn_geometry.value_heads =
                static_cast<std::uint32_t>(config.linear_num_value_heads);
            layer.gdn_geometry.key_head_dim =
                static_cast<std::uint32_t>(config.linear_key_head_dim);
            layer.gdn_geometry.value_head_dim =
                static_cast<std::uint32_t>(config.linear_value_head_dim);
            layer.gdn_geometry.conv_kernel =
                static_cast<std::uint32_t>(config.linear_conv_kernel_dim);
            layer.gdn_geometry.sigmoid_gate = true;
            layer.gdn_geometry.eps = eps;
            layer.gdn_weights.qkv = &lw.attention.in_proj_qkv;
            layer.gdn_weights.z = &lw.attention.in_proj_z;
            layer.gdn_weights.b = &lw.attention.in_proj_b;
            layer.gdn_weights.a = &lw.attention.in_proj_a;
            layer.gdn_weights.out = &lw.attention.out_proj;
            layer.gdn_indices = {
                idx.gdn_qkv, idx.gdn_z, idx.gdn_b, idx.gdn_a, idx.gdn_out,
                lp.conv, lp.dt_bias, lp.a_log, lp.gdn_norm, next_gdn_state++,
            };
        } else {
            layer.qsa_geometry.attention.query_heads =
                static_cast<std::uint32_t>(config.num_attention_heads);
            layer.qsa_geometry.attention.kv_heads =
                static_cast<std::uint32_t>(config.num_key_value_heads);
            layer.qsa_geometry.attention.head_dim = head_dim;
            layer.qsa_geometry.attention.rotary_dim = rotary_dim;
            layer.qsa_geometry.attention.rope_theta = rope_theta;
            layer.qsa_geometry.attention.rope_attention_factor =
                options.rope_attention_factor;
            layer.qsa_geometry.attention.eps = eps;
            layer.qsa_geometry.indexer = make_qwen4exp_qsa_indexer_geometry(
                config, index_blocks, options.max_context_tokens, next_attention_state);
            layer.qsa_geometry.indexer.top_k =
                std::min(layer.qsa_geometry.indexer.top_k, index_blocks);
            layer.qsa_weights.attention.q = &lw.attention.q_proj;
            layer.qsa_weights.attention.k = &lw.attention.k_proj;
            layer.qsa_weights.attention.v = &lw.attention.v_proj;
            layer.qsa_weights.attention.o = &lw.attention.o_proj;
            layer.qsa_weights.indexer.qk_proj = &lw.attention.index_qk_proj;
            layer.qsa_indices.attention.w_q = idx.attn_q;
            layer.qsa_indices.attention.w_k = idx.attn_k;
            layer.qsa_indices.attention.w_v = idx.attn_v;
            layer.qsa_indices.attention.w_o = idx.attn_o;
            layer.qsa_indices.attention.p_q_norm = lp.q_norm;
            layer.qsa_indices.attention.p_k_norm = lp.k_norm;
            layer.qsa_indices.attention.state_index = next_attention_state++;
            layer.qsa_indices.indexer.w_qk_proj = idx.index_qk;
            layer.qsa_indices.indexer.p_q_norm = lp.index_q_norm;
            layer.qsa_indices.indexer.p_k_norm = lp.index_k_norm;
        }

        layer.moe_geometry = {
            global_experts,
            local_experts,
            expert_begin,
            top_k,
            hidden,
            intermediate,
            shared_intermediate,
            options.tp_rank == 0u ? 1.0f : 0.0f,
        };
        layer.moe_weights = lw.mlp;
        layer.moe_indices = {
            idx.router,
            idx.shared_gate,
            idx.shared_gate_proj,
            idx.shared_up_proj,
            idx.shared_down_proj,
            idx.gate_experts,
            idx.up_experts,
            idx.down_experts,
        };

        if (lw.ple.present) {
            layer.has_ple = true;
            layer.ple_geometry.hc = hc;
            layer.ple_geometry.hidden = hidden;
            layer.ple_geometry.embed_dim =
                static_cast<std::uint32_t>(config.ple.embed_dim);
            layer.ple_geometry.eps = eps;
            layer.ple_geometry.norm_key_parameter_index = lp.ple_norm_key;
            layer.ple_geometry.norm_query_parameter_index = lp.ple_norm_query;
            layer.ple_geometry.norm_conv_parameter_index = lp.ple_norm_conv;
            layer.ple_geometry.conv_weight_parameter_index = lp.ple_conv;
            layer.ple_geometry.state_index = next_ple_state++;
            layer.ple_weights.key_proj = &lw.ple.key_proj;
            layer.ple_weights.value_proj = &lw.ple.value_proj;
            layer.ple_indices = {idx.ple_key, idx.ple_value};
            layer.ple_embeddings = ple_embeddings;
        }

        auto out = emit_qwen4exp_layer(em, layer, hyper, prefix.c_str());
        if (!out.ok()) return out.status();
        hyper = out.value();
    }

    HyperConnectionSiteGeometry mixer_geometry;
    mixer_geometry.hc = hc;
    mixer_geometry.hidden = hidden;
    mixer_geometry.lowrank = lowrank;
    mixer_geometry.eps = eps;
    mixer_geometry.norm_parameter_index = mixer_norm;
    HyperConnectionSiteWeights mixer_weights;
    mixer_weights.down = &weights.mixer.input_mix_down;
    mixer_weights.up = &weights.mixer.input_mix_up;
    HyperConnectionSiteIndices mixer_indices{mixer_down, mixer_up, 0u};
    auto mixed = emit_hyper_connection_mix(em, mixer_geometry, mixer_weights,
                                           mixer_indices, hyper, false, "mixer.");
    if (!mixed.ok()) return mixed.status();

    QH::HybridHeadLowering head;
    head.hidden_features = hidden;
    head.eps = eps;
    head.final_norm_param = UINT32_MAX;
    head.apply_final_norm = false;
    head.lm_shape = RT::MatrixwiseShapeKey{weights.lm_head.rows, weights.lm_head.cols};
    head.lm_weight_index = lm_index;
    head.out_vocab = vocab;
    QH::emit_hybrid_head(em, head, mixed.value().mixed, "head.");

    Status status = RT::validate_primitive_graph(pg);
    if (!status.ok()) return status;
    return lowered;
}

}  // namespace qwen4exp
}  // namespace shelf
