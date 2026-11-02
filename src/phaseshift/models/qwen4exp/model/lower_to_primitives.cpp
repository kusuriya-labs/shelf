#include <phaseshift/models/qwen4exp/model/lower_to_primitives.h>

#include <phaseshift/models/common/qwen_hybrid/attention_geometry.h>
#include <phaseshift/models/common/qwen_hybrid/attention_lowering.h>
#include <phaseshift/models/common/qwen_hybrid/gdn_geometry.h>
#include <phaseshift/models/common/qwen_hybrid/gdn_lowering.h>
#include <phaseshift/models/common/qwen_hybrid/geometry_check.h>
#include <phaseshift/models/common/qwen_hybrid/hyper_lowering.h>
#include <phaseshift/runtime/graph/primitive_node.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace ps::qwen4exp {

namespace {

namespace RT = ::ps::runtime;
namespace QH = ::ps::qwen_hybrid;

using QH::checked_product;
using QH::checked_u32;
using QH::validate_matrix;
using Emitter = QH::Emitter;
using MW = ::ps::weights::MatrixWeight;

std::size_t tensor_element_count(const gpu::Tensor& t) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < t.ndim(); ++i) {
        n *= t.dim(i);
    }
    return n;
}

RT::WeightSlot make_weight_slot(const MW& w) {
    RT::WeightSlot s{};
    s.rows = w.rows;
    s.cols = w.cols;
    s.k_padded = w.k_padded;
    s.encoding = static_cast<uint8_t>(w.encoding);
    s.compute_spec = static_cast<uint8_t>(w.compute_spec);
    if (w.encoding == ps::weights::MatrixEncoding::Bf16) {
        s.device_ptr = w.bf16.data<void>();
        s.bytes = static_cast<uint64_t>(w.rows) * w.cols * sizeof(uint16_t);
        s.codes = nullptr;
        s.scales = nullptr;
    } else {
        s.device_ptr = nullptr;
        s.bytes = 0;
        s.codes = w.codes.ndim() != 0 ? w.codes.data<void>() : nullptr;
        s.scales = w.scales.ndim() != 0 ? w.scales.data<void>() : nullptr;
        s.sign = w.sign.ndim() != 0 ? w.sign.data<void>() : nullptr;
        s.storage_scale_stride_bytes = w.storage_scale_stride_bytes;
        s.compute_codes = w.compute_codes.ndim() != 0 ? w.compute_codes.data<void>() : nullptr;
        s.compute_scales_bf16 =
            w.compute_scales_bf16.ndim() != 0 ? w.compute_scales_bf16.data<void>() : nullptr;
        s.codes_row_stride_bytes = w.codes_row_stride_bytes;
        s.scale_row_stride_bytes = w.scale_row_stride_bytes;
        s.weight_scale_group = w.weight_scale_group;
        s.preshuffled = w.preshuffled;
    }
    return s;
}

Status validate_hyper_weights(const Qwen4ExpHyperWeights& hyper,
                              std::uint32_t hc_hidden,
                              std::uint32_t lowrank,
                              std::uint32_t streams,
                              bool with_inject,
                              const char* name) {
    auto status = validate_matrix(hyper.mix_down,
                                  RT::MatrixwiseShapeKey{lowrank, hc_hidden},
                                  name);
    if (!status.ok()) return status;
    status = validate_matrix(hyper.mix_up,
                             RT::MatrixwiseShapeKey{hc_hidden, lowrank},
                             name);
    if (!status.ok()) return status;
    if (hyper.hc_norm.ndim() == 0 ||
        tensor_element_count(hyper.hc_norm) != hc_hidden) {
        return Status::invalid_argument(name, __FILE__, __LINE__);
    }
    if (with_inject) {
        if (!hyper.has_block_inject) {
            return Status::invalid_argument(name, __FILE__, __LINE__);
        }
        status = validate_matrix(hyper.block_inject,
                                 RT::MatrixwiseShapeKey{streams, hc_hidden},
                                 name);
        if (!status.ok()) return status;
    }
    return Status::make_ok();
}

Status validate_gdn_layer(const Qwen4ExpTextConfig& config, const Qwen4ExpLayerWeights& lw,
                          std::uint32_t hidden_size) {
    std::uint32_t key_heads = 0;
    std::uint32_t value_heads = 0;
    std::uint32_t key_head_dim = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel = 0;
    auto status = checked_u32(config.linear_num_key_heads, "invalid GDN key heads",
                              key_heads);
    if (!status.ok()) return status;
    status = checked_u32(config.linear_num_value_heads, "invalid GDN value heads",
                         value_heads);
    if (!status.ok()) return status;
    status = checked_u32(config.linear_key_head_dim, "invalid GDN key head dimension",
                         key_head_dim);
    if (!status.ok()) return status;
    status = checked_u32(config.linear_value_head_dim,
                         "invalid GDN value head dimension", value_head_dim);
    if (!status.ok()) return status;
    status = checked_u32(config.linear_conv_kernel_dim, "invalid GDN convolution kernel",
                         conv_kernel);
    if (!status.ok()) return status;

    const QH::GdnGeometry geometry{
        .key_heads = key_heads,
        .value_heads = value_heads,
        .key_head_dim = key_head_dim,
        .value_head_dim = value_head_dim,
        .conv_kernel = conv_kernel,
    };
    const QH::GdnWeightView view{
        .qkv = &lw.gdn.qkv,
        .z = &lw.gdn.z,
        .b = &lw.gdn.b,
        .a = &lw.gdn.a,
        .out = &lw.gdn.out,
    };
    status = QH::validate_gdn_geometry(geometry, hidden_size, view);
    if (!status.ok()) return status;

    QH::GdnFeatureSizes sizes{};
    status = QH::gdn_feature_sizes(geometry, sizes);
    if (!status.ok()) return status;
    const std::uint32_t conv_features = sizes.conv_features;

    if (lw.gdn.conv1d.ndim() == 0 ||
        tensor_element_count(lw.gdn.conv1d) !=
            static_cast<std::size_t>(conv_features) * conv_kernel) {
        return Status::invalid_argument("invalid GDN convolution weight", __FILE__,
                                        __LINE__);
    }
    if (lw.gdn.norm.ndim() == 0 ||
        tensor_element_count(lw.gdn.norm) != value_head_dim) {
        return Status::invalid_argument("invalid GDN norm weight", __FILE__, __LINE__);
    }
    if (lw.gdn.dt_bias.ndim() == 0 ||
        tensor_element_count(lw.gdn.dt_bias) != value_heads) {
        return Status::invalid_argument("invalid GDN dt bias", __FILE__, __LINE__);
    }
    if (lw.gdn.a_log.ndim() == 0 || tensor_element_count(lw.gdn.a_log) != value_heads) {
        return Status::invalid_argument("invalid GDN A log", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status validate_qsa_layer(const Qwen4ExpTextConfig& config, const Qwen4ExpLayerWeights& lw,
                          std::uint32_t hidden_size) {
    float partial = config.partial_rotary_factor;
    if (partial == 0.0f) partial = 0.25f;
    const QH::GatedAttentionGeometry geometry{
        .query_heads = static_cast<std::uint32_t>(config.num_attention_heads),
        .kv_heads = static_cast<std::uint32_t>(config.num_key_value_heads),
        .head_dim = static_cast<std::uint32_t>(config.head_dim),
        .partial_rotary_factor = partial,
    };
    const QH::GatedAttentionWeightView view{
        .q_proj = &lw.qsa.q,
        .k_proj = &lw.qsa.k,
        .v_proj = &lw.qsa.v,
        .o_proj = &lw.qsa.o,
    };
    auto status = QH::validate_gated_attention_geometry(geometry, hidden_size, view);
    if (!status.ok()) return status;

    const std::uint32_t head_dim = static_cast<std::uint32_t>(config.head_dim);
    if (lw.qsa.q_norm.ndim() == 0 ||
        tensor_element_count(lw.qsa.q_norm) != head_dim) {
        return Status::invalid_argument("invalid QSA q norm", __FILE__, __LINE__);
    }
    if (lw.qsa.k_norm.ndim() == 0 ||
        tensor_element_count(lw.qsa.k_norm) != head_dim) {
        return Status::invalid_argument("invalid QSA k norm", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status validate_moe_layer(const Qwen4ExpTextConfig& config, const Qwen4ExpLayerWeights& lw,
                          std::uint32_t hidden_size) {
    std::uint32_t experts = 0;
    std::uint32_t top_k = 0;
    std::uint32_t inter = 0;
    std::uint32_t shared_inter = 0;
    auto status = checked_u32(config.num_experts, "invalid MoE expert count", experts);
    if (!status.ok()) return status;
    status = checked_u32(config.num_experts_per_tok, "invalid MoE top k", top_k);
    if (!status.ok()) return status;
    if (top_k > experts) {
        return Status::invalid_argument("invalid MoE top k", __FILE__, __LINE__);
    }
    status = checked_u32(config.moe_intermediate_size, "invalid MoE intermediate size",
                         inter);
    if (!status.ok()) return status;
    status = checked_u32(config.shared_expert_intermediate_size,
                         "invalid shared expert intermediate size", shared_inter);
    if (!status.ok()) return status;

    if (lw.moe.expert_count != experts) {
        return Status::invalid_argument("MoE expert count mismatch", __FILE__, __LINE__);
    }
    status = validate_matrix(lw.moe.router,
                             RT::MatrixwiseShapeKey{experts, hidden_size},
                             "invalid MoE router weight");
    if (!status.ok()) return status;
    status = validate_matrix(lw.moe.shared.gate,
                             RT::MatrixwiseShapeKey{shared_inter, hidden_size},
                             "invalid shared expert gate projection");
    if (!status.ok()) return status;
    status = validate_matrix(lw.moe.shared.up,
                             RT::MatrixwiseShapeKey{shared_inter, hidden_size},
                             "invalid shared expert up projection");
    if (!status.ok()) return status;
    status = validate_matrix(lw.moe.shared.down,
                             RT::MatrixwiseShapeKey{hidden_size, shared_inter},
                             "invalid shared expert down projection");
    if (!status.ok()) return status;
    if (lw.moe.shared.gate_scale.ndim() == 0 ||
        tensor_element_count(lw.moe.shared.gate_scale) != hidden_size) {
        return Status::invalid_argument("invalid shared expert gate scale",
                                        __FILE__, __LINE__);
    }

    if (lw.moe.expert_gate_up.size() != experts ||
        lw.moe.expert_down.size() != experts) {
        return Status::invalid_argument(
            "expert weights are not loaded (Qwen4ExpLoadOptions::load_expert_weights)",
            __FILE__, __LINE__);
    }
    for (std::uint32_t e = 0; e < experts; ++e) {
        const MW& gate_up = lw.moe.expert_gate_up[e];
        if (gate_up.rows != 2u * inter || gate_up.cols != hidden_size) {
            return Status::invalid_argument("invalid expert gate_up projection",
                                            __FILE__, __LINE__);
        }
        const MW& down = lw.moe.expert_down[e];
        if (down.rows != hidden_size || down.cols != inter) {
            return Status::invalid_argument("invalid expert down projection", __FILE__,
                                            __LINE__);
        }
    }
    return Status::make_ok();
}

Status validate_geometry(const Qwen4ExpTextConfig& config,
                         const Qwen4ExpModelWeights& weights,
                         const Qwen4ExpLowerOptions& options) {
    std::uint32_t hidden_size = 0;
    std::uint32_t streams = 0;
    std::uint32_t lowrank = 0;
    auto status = checked_u32(config.hidden_size, "invalid hidden size", hidden_size);
    if (!status.ok()) return status;
    status = checked_u32(config.hc_count, "invalid hyper connection count", streams);
    if (!status.ok()) return status;
    status = checked_u32(config.hc_lowrank, "invalid hyper connection lowrank", lowrank);
    if (!status.ok()) return status;
    std::uint32_t hc_hidden = 0;
    status = checked_product(streams, hidden_size, "invalid hyper hidden size",
                             hc_hidden);
    if (!status.ok()) return status;

    if (config.num_hidden_layers == 0 ||
        weights.layers.size() != config.num_hidden_layers) {
        return Status::invalid_argument("Qwen4Exp layer metadata count mismatch",
                                        __FILE__, __LINE__);
    }
    if (!config.layer_types.empty() &&
        config.layer_types.size() != config.num_hidden_layers) {
        return Status::invalid_argument("Qwen4Exp layer type count mismatch", __FILE__,
                                        __LINE__);
    }

    status = validate_matrix(
        weights.embed_tokens,
        RT::MatrixwiseShapeKey{weights.embed_tokens.rows, hidden_size},
        "invalid embedding weight");
    if (!status.ok()) return status;
    if (weights.embed_tokens.rows != config.vocab_size) {
        return Status::invalid_argument("invalid embedding vocabulary size", __FILE__,
                                        __LINE__);
    }
    if (!weights.lm_head_tied) {
        if (weights.lm_head.cols != hidden_size ||
            weights.lm_head.rows != config.vocab_size) {
            return Status::invalid_argument("invalid lm head weight", __FILE__,
                                            __LINE__);
        }
    }

    status = validate_hyper_weights(weights.final_hyper, hc_hidden, lowrank, streams,
                                    false, "invalid final hyper connection weights");
    if (!status.ok()) return status;

    if (options.connect_ple) {
        const std::size_t ple_layer = config.ple_layer_index;
        if (ple_layer >= config.num_hidden_layers) {
            return Status::invalid_argument("Qwen4Exp PLE layer index out of range",
                                            __FILE__, __LINE__);
        }
        if (!weights.layers[ple_layer].ple.present) {
            return Status::invalid_argument("Qwen4Exp PLE weights are missing", __FILE__,
                                            __LINE__);
        }
        if (config.ple_embed_dim != hidden_size) {
            return Status::invalid_argument("Qwen4Exp PLE embed dim mismatch", __FILE__,
                                            __LINE__);
        }
    }

    const std::size_t valid_begin =
        std::min<std::size_t>(options.stage_begin, weights.layers.size());
    const std::size_t valid_end =
        options.stage_end == 0
            ? weights.layers.size()
            : std::min<std::size_t>(options.stage_end, weights.layers.size());
    for (std::size_t li = valid_begin; li < valid_end; ++li) {
        const Qwen4ExpLayerWeights& lw = weights.layers[li];
        bool is_gdn = lw.is_gdn;
        if (!config.layer_types.empty()) {
            const bool config_is_qsa = config.layer_is_qsa(li);
            if (config_is_qsa == is_gdn) {
                return Status::invalid_argument(
                    "Qwen4Exp layer type conflicts with weight metadata", __FILE__,
                    __LINE__);
            }
            is_gdn = !config_is_qsa;
        }

        status = validate_hyper_weights(lw.attn_hyper, hc_hidden, lowrank, streams, true,
                                        "invalid attention hyper connection weights");
        if (!status.ok()) return status;
        if (options.connect_moe) {
            status = validate_hyper_weights(lw.mlp_hyper, hc_hidden, lowrank, streams,
                                            true,
                                            "invalid mlp hyper connection weights");
            if (!status.ok()) return status;
        }

        if (is_gdn) {
            status = validate_gdn_layer(config, lw, hidden_size);
        } else {
            status = validate_qsa_layer(config, lw, hidden_size);
        }
        if (!status.ok()) return status;

        if (options.connect_moe) {
            status = validate_moe_layer(config, lw, hidden_size);
            if (!status.ok()) return status;
        }
    }
    return Status::make_ok();
}

struct LayerWeightIndices {
    std::uint32_t attn_mix_down = 0;
    std::uint32_t attn_mix_up = 0;
    std::uint32_t attn_block_inject = 0;
    std::uint32_t mlp_mix_down = 0;
    std::uint32_t mlp_mix_up = 0;
    std::uint32_t mlp_block_inject = 0;
    std::uint32_t gdn_qkv = 0;
    std::uint32_t gdn_z = 0;
    std::uint32_t gdn_b = 0;
    std::uint32_t gdn_a = 0;
    std::uint32_t gdn_out = 0;
    std::uint32_t qsa_q = 0;
    std::uint32_t qsa_k = 0;
    std::uint32_t qsa_v = 0;
    std::uint32_t qsa_o = 0;
    std::uint32_t moe_router = 0;
    std::uint32_t shared_gate = 0;
    std::uint32_t shared_up = 0;
    std::uint32_t shared_down = 0;
    std::uint32_t shared_gate_scale = 0;
    std::uint32_t expert_gate_up = 0;
    std::uint32_t expert_down = 0;
    std::uint32_t p_attn_hc = 0;
    std::uint32_t p_mlp_hc = 0;
    std::uint32_t p_conv = 0;
    std::uint32_t p_gdn_norm = 0;
    std::uint32_t p_dt_bias = 0;
    std::uint32_t p_a_log = 0;
    std::uint32_t p_q_norm = 0;
    std::uint32_t p_k_norm = 0;
};

RT::ValueId emit_moe_sublayer(Emitter& em, const Qwen4ExpTextConfig& config,
                              const Qwen4ExpLayerWeights& lw,
                              const LayerWeightIndices& idx, RT::ValueId mixed,
                              const std::string& prefix, std::uint32_t hidden_size) {
    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = prefix + suffix;
    };

    const std::uint32_t experts = static_cast<std::uint32_t>(config.num_experts);
    const std::uint32_t top_k = static_cast<std::uint32_t>(config.num_experts_per_tok);
    const std::uint32_t inter =
        static_cast<std::uint32_t>(config.moe_intermediate_size);
    const std::uint32_t shared_inter =
        static_cast<std::uint32_t>(config.shared_expert_intermediate_size);

    RT::MoERouterNode router;
    router.weight_shape = RT::MatrixwiseShapeKey{experts, hidden_size};
    router.weight_index = idx.moe_router;
    router.experts = experts;
    router.top_k = top_k;
    router.renorm = config.norm_topk_prob;
    const RT::ValueId top_idx = em.value(top_k, RT::ValueDType::I32);
    const RT::ValueId top_w = em.value(top_k, RT::ValueDType::F32);
    em.emit(RT::PrimitiveNode{router}, {mixed}, {top_idx, top_w});
    name_last("moe_router");

    RT::MoEExpertsNode experts_node;
    experts_node.gate_up_shape = RT::MatrixwiseShapeKey{2u * inter, hidden_size};
    experts_node.down_shape = RT::MatrixwiseShapeKey{hidden_size, inter};
    experts_node.weight_index = idx.expert_gate_up;
    experts_node.experts = experts;
    experts_node.top_k = top_k;
    const RT::ValueId routed = em.value(hidden_size, RT::ValueDType::F32);
    em.emit(RT::PrimitiveNode{experts_node}, {mixed, top_idx, top_w}, {routed});
    name_last("moe_experts");

    const RT::ValueId sh_gate =
        em.linear(lw.moe.shared.gate, idx.shared_gate, mixed, RT::ValueDType::F32);
    name_last("shared_gate");
    const RT::ValueId sh_up =
        em.linear(lw.moe.shared.up, idx.shared_up, mixed, RT::ValueDType::F32);
    name_last("shared_up");
    const RT::ValueId sh_act = em.unary(
        RT::PrimitiveNode{RT::SiLUNode{RT::RowwiseShapeKey{shared_inter}}},
        sh_gate, shared_inter, RT::ValueDType::F32);
    name_last("shared_act");
    RT::ValueId sh_mul = em.value(shared_inter, RT::ValueDType::F32);
    em.emit(RT::PrimitiveNode{RT::MulNode{RT::RowwiseShapeKey{shared_inter}}},
            {sh_act, sh_up}, {sh_mul});
    name_last("shared_mul");
    const RT::ValueId sh_down =
        em.linear(lw.moe.shared.down, idx.shared_down, sh_mul, RT::ValueDType::F32);
    name_last("shared_down");

    MW gate_scale;
    gate_scale.rows = 1;
    gate_scale.cols = hidden_size;
    const RT::ValueId sh_scale =
        em.linear(gate_scale, idx.shared_gate_scale, mixed, RT::ValueDType::F32);
    name_last("shared_gate_scale");
    const RT::ValueId sh_scale_sig = em.unary(
        RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{1}}}, sh_scale, 1,
        RT::ValueDType::F32);
    name_last("shared_gate_sigmoid");
    RT::ValueId sh_out = em.value(hidden_size, RT::ValueDType::F32);
    {
        RT::GroupBroadcastMulNode gm;
        gm.shape = RT::RowwiseShapeKey{hidden_size};
        gm.group_size = 1;
        em.emit(RT::PrimitiveNode{gm}, {sh_down, sh_scale_sig}, {sh_out});
        name_last("shared_gate_mul");
    }

    RT::ValueId moe_out = em.value(hidden_size, RT::ValueDType::F32);
    {
        RT::ResidualAddNode ra{RT::RowwiseShapeKey{hidden_size}};
        em.emit(RT::PrimitiveNode{ra}, {routed, sh_out}, {moe_out});
        name_last("moe_add");
    }
    return moe_out;
}

struct Geometry {
    std::uint32_t hidden_size = 0;
    std::uint32_t streams = 0;
    std::uint32_t lowrank = 0;
    std::uint32_t q_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rotary_dim = 0;
    std::uint32_t key_heads = 0;
    std::uint32_t value_heads = 0;
    std::uint32_t key_head_dim = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel = 0;
    std::uint32_t hc_hidden = 0;
    float eps = 1.0e-6f;
    float rope_theta = 10000000.0f;
};

Status parse_geometry(const Qwen4ExpTextConfig& config, Geometry& g) {
    std::uint32_t hidden_size = 0;
    std::uint32_t streams = 0;
    std::uint32_t lowrank = 0;
    auto status = checked_u32(config.hidden_size, "invalid hidden size", hidden_size);
    if (!status.ok()) return status;
    status = checked_u32(config.hc_count, "invalid hyper connection count", streams);
    if (!status.ok()) return status;
    status = checked_u32(config.hc_lowrank, "invalid hyper connection lowrank", lowrank);
    if (!status.ok()) return status;

    g.hidden_size = hidden_size;
    g.streams = streams;
    g.lowrank = lowrank;
    g.q_heads = static_cast<std::uint32_t>(config.num_attention_heads);
    g.kv_heads = static_cast<std::uint32_t>(config.num_key_value_heads);
    g.head_dim = static_cast<std::uint32_t>(config.head_dim);
    g.key_heads = static_cast<std::uint32_t>(config.linear_num_key_heads);
    g.value_heads = static_cast<std::uint32_t>(config.linear_num_value_heads);
    g.key_head_dim = static_cast<std::uint32_t>(config.linear_key_head_dim);
    g.value_head_dim = static_cast<std::uint32_t>(config.linear_value_head_dim);
    g.conv_kernel = static_cast<std::uint32_t>(config.linear_conv_kernel_dim);
    g.eps = config.rms_norm_eps != 0.0f ? config.rms_norm_eps : 1.0e-6f;
    g.rope_theta = config.rope_theta != 0.0f ? config.rope_theta : 10000000.0f;
    float partial = config.partial_rotary_factor;
    if (partial == 0.0f) partial = 0.25f;
    const std::uint32_t head_dim = g.head_dim;
    g.rotary_dim =
        config.rotary_dim != 0
            ? static_cast<std::uint32_t>(config.rotary_dim)
            : static_cast<std::uint32_t>(static_cast<double>(head_dim) * partial);
    if (g.rotary_dim == 0 || (g.rotary_dim & 1u) != 0u || g.rotary_dim > head_dim) {
        return Status::invalid_argument("invalid rotary dimension", __FILE__, __LINE__);
    }
    std::uint32_t hc_hidden = 0;
    status = checked_product(streams, hidden_size, "invalid hyper hidden size",
                             hc_hidden);
    if (!status.ok()) return status;
    g.hc_hidden = hc_hidden;
    return Status::make_ok();
}

struct SlotBuilder {
    std::vector<RT::WeightSlot> weights;
    std::vector<RT::StaticParameterSlot> parameters;
    RT::PrimitiveGraph* graph = nullptr;

    std::uint32_t add_weight(const MW& w) {
        weights.push_back(make_weight_slot(w));
        graph->weight_table.push_back(RT::MatrixwiseShapeKey{w.rows, w.cols});
        return static_cast<std::uint32_t>(weights.size() - 1u);
    }

    std::uint32_t add_tensor_weight(const gpu::Tensor& t, std::uint32_t rows,
                                    std::uint32_t cols) {
        MW w;
        w.bf16 = t;
        w.rows = rows;
        w.cols = cols;
        w.k_padded = cols;
        return add_weight(w);
    }

    std::uint32_t add_parameter(const gpu::Tensor& t) {
        RT::StaticParameterSlot s{};
        s.device_ptr = t.data<void>();
        s.elements = static_cast<std::uint32_t>(tensor_element_count(t));
        s.dtype = t.element_size() == 2 ? RT::ValueDType::BF16 : RT::ValueDType::F32;
        parameters.push_back(s);
        return static_cast<std::uint32_t>(parameters.size() - 1u);
    }
};

void register_layer_slots(const Qwen4ExpLayerWeights& lw,
                          const Qwen4ExpLowerOptions& options, std::uint32_t hidden_size,
                          SlotBuilder& sb, LayerWeightIndices& idx) {
    idx.attn_mix_down = sb.add_weight(lw.attn_hyper.mix_down);
    idx.attn_mix_up = sb.add_weight(lw.attn_hyper.mix_up);
    idx.attn_block_inject = sb.add_weight(lw.attn_hyper.block_inject);
    if (options.connect_moe) {
        idx.mlp_mix_down = sb.add_weight(lw.mlp_hyper.mix_down);
        idx.mlp_mix_up = sb.add_weight(lw.mlp_hyper.mix_up);
        idx.mlp_block_inject = sb.add_weight(lw.mlp_hyper.block_inject);
    }
    if (lw.is_gdn) {
        idx.gdn_qkv = sb.add_weight(lw.gdn.qkv);
        idx.gdn_z = sb.add_weight(lw.gdn.z);
        idx.gdn_b = sb.add_weight(lw.gdn.b);
        idx.gdn_a = sb.add_weight(lw.gdn.a);
        idx.gdn_out = sb.add_weight(lw.gdn.out);
    } else {
        idx.qsa_q = sb.add_weight(lw.qsa.q);
        idx.qsa_k = sb.add_weight(lw.qsa.k);
        idx.qsa_v = sb.add_weight(lw.qsa.v);
        idx.qsa_o = sb.add_weight(lw.qsa.o);
    }
    if (options.connect_moe) {
        idx.moe_router = sb.add_weight(lw.moe.router);
        idx.shared_gate = sb.add_weight(lw.moe.shared.gate);
        idx.shared_up = sb.add_weight(lw.moe.shared.up);
        idx.shared_down = sb.add_weight(lw.moe.shared.down);
        idx.shared_gate_scale =
            sb.add_tensor_weight(lw.moe.shared.gate_scale, 1, hidden_size);
        idx.expert_gate_up = static_cast<std::uint32_t>(sb.weights.size());
        for (const MW& w : lw.moe.expert_gate_up) (void)sb.add_weight(w);
        idx.expert_down = static_cast<std::uint32_t>(sb.weights.size());
        for (const MW& w : lw.moe.expert_down) (void)sb.add_weight(w);
    }
}

void register_layer_parameters(const Qwen4ExpLayerWeights& lw,
                               const Qwen4ExpLowerOptions& options, SlotBuilder& sb,
                               LayerWeightIndices& idx) {
    idx.p_attn_hc = sb.add_parameter(lw.attn_hyper.hc_norm);
    if (options.connect_moe) {
        idx.p_mlp_hc = sb.add_parameter(lw.mlp_hyper.hc_norm);
    }
    if (lw.is_gdn) {
        idx.p_conv = sb.add_parameter(lw.gdn.conv1d);
        idx.p_gdn_norm = sb.add_parameter(lw.gdn.norm);
        idx.p_dt_bias = sb.add_parameter(lw.gdn.dt_bias);
        idx.p_a_log = sb.add_parameter(lw.gdn.a_log);
    } else {
        idx.p_q_norm = sb.add_parameter(lw.qsa.q_norm);
        idx.p_k_norm = sb.add_parameter(lw.qsa.k_norm);
    }
}

struct LayerEmitter {
    const Qwen4ExpTextConfig& config;
    const Qwen4ExpModelWeights& weights;
    const Qwen4ExpLowerOptions& options;
    const Geometry& g;
    const std::vector<LayerWeightIndices>& indices;

    void emit(QH::Emitter& em, std::size_t li, RT::ValueId& x4,
              std::uint32_t& next_gdn_state_index,
              std::uint32_t& next_attention_state_index,
              std::vector<std::pair<std::uint32_t, RT::ValueId>>* layer_inputs) const {
        const Qwen4ExpLayerWeights& lw = weights.layers[li];
        const LayerWeightIndices& idx = indices[li];
        const std::string prefix = "L" + std::to_string(li) + ".";
        const QH::HyperLoweringGeometry hyper_geo{
            .streams = g.streams,
            .hidden = g.hidden_size,
            .lowrank = g.lowrank,
            .eps = g.eps,
        };

        if (layer_inputs != nullptr) {
            for (std::size_t o = 0; o < options.override_layers.size(); ++o) {
                if (static_cast<std::size_t>(options.override_layers[o]) == li) {
                    layer_inputs->emplace_back(static_cast<std::uint32_t>(li), x4);
                }
            }
        }

        const QH::HyperLoweringWeights attn_hyper_weights{
            .mix_down = &lw.attn_hyper.mix_down,
            .mix_up = &lw.attn_hyper.mix_up,
            .block_inject = &lw.attn_hyper.block_inject,
        };
        const QH::HyperLoweringIndices attn_hyper_indices{
            .w_mix_down = idx.attn_mix_down,
            .w_mix_up = idx.attn_mix_up,
            .w_block_inject = idx.attn_block_inject,
            .p_hc_norm = idx.p_attn_hc,
        };
        const QH::HyperReadOutputs attn_read = QH::emit_hyper_read(
            em, hyper_geo, attn_hyper_weights, attn_hyper_indices, x4,
            prefix + "attn.");

        RT::ValueId sub_out{0};
        if (lw.is_gdn) {
            const QH::GdnLoweringGeometry gdn_geometry{
                .key_heads = g.key_heads,
                .value_heads = g.value_heads,
                .key_head_dim = g.key_head_dim,
                .value_head_dim = g.value_head_dim,
                .conv_kernel = g.conv_kernel,
                .eps = g.eps,
                .output_gate_silu = config.output_gate_silu,
                .out_dtype = RT::ValueDType::F32,
            };
            const QH::GdnLoweringWeights gdn_weights{
                .qkv = &lw.gdn.qkv,
                .z = &lw.gdn.z,
                .b = &lw.gdn.b,
                .a = &lw.gdn.a,
                .out = &lw.gdn.out,
            };
            const QH::GdnLoweringIndices gdn_indices{
                .w_qkv = idx.gdn_qkv,
                .w_z = idx.gdn_z,
                .w_b = idx.gdn_b,
                .w_a = idx.gdn_a,
                .w_out = idx.gdn_out,
                .p_conv = idx.p_conv,
                .p_dt_bias = idx.p_dt_bias,
                .p_a_log = idx.p_a_log,
                .p_gdn_norm = idx.p_gdn_norm,
                .state_index = next_gdn_state_index++,
            };
            sub_out = QH::emit_gdn(em, gdn_geometry, gdn_weights, gdn_indices,
                                   attn_read.mixed, prefix + "attn.");
        } else {
            const QH::GatedAttentionLoweringGeometry attn_geometry{
                .query_heads = g.q_heads,
                .kv_heads = g.kv_heads,
                .head_dim = g.head_dim,
                .rotary_dim = g.rotary_dim,
                .rope_theta = g.rope_theta,
                .rope_attention_factor =
                    static_cast<float>(options.rope_profile.attention_factor),
                .eps = g.eps,
                .out_dtype = RT::ValueDType::F32,
            };
            const QH::GatedAttentionLoweringWeights attn_weights{
                .q = &lw.qsa.q,
                .k = &lw.qsa.k,
                .v = &lw.qsa.v,
                .o = &lw.qsa.o,
            };
            const QH::GatedAttentionLoweringIndices attn_indices{
                .w_q = idx.qsa_q,
                .w_k = idx.qsa_k,
                .w_v = idx.qsa_v,
                .w_o = idx.qsa_o,
                .p_q_norm = idx.p_q_norm,
                .p_k_norm = idx.p_k_norm,
                .state_index = next_attention_state_index++,
            };
            sub_out = QH::emit_gated_attention(em, attn_geometry, attn_weights,
                                               attn_indices, attn_read.mixed,
                                               prefix + "attn.");
        }
        x4 = QH::emit_hyper_write(em, hyper_geo, x4, sub_out, attn_read.inject,
                                  prefix + "attn.");

        if (options.connect_moe) {
            const QH::HyperLoweringWeights mlp_hyper_weights{
                .mix_down = &lw.mlp_hyper.mix_down,
                .mix_up = &lw.mlp_hyper.mix_up,
                .block_inject = &lw.mlp_hyper.block_inject,
            };
            const QH::HyperLoweringIndices mlp_hyper_indices{
                .w_mix_down = idx.mlp_mix_down,
                .w_mix_up = idx.mlp_mix_up,
                .w_block_inject = idx.mlp_block_inject,
                .p_hc_norm = idx.p_mlp_hc,
            };
            const QH::HyperReadOutputs mlp_read = QH::emit_hyper_read(
                em, hyper_geo, mlp_hyper_weights, mlp_hyper_indices, x4,
                prefix + "mlp.");
            const RT::ValueId moe_out =
                emit_moe_sublayer(em, config, lw, idx, mlp_read.mixed, prefix + "mlp.",
                                  g.hidden_size);
            x4 = QH::emit_hyper_write(em, hyper_geo, x4, moe_out, mlp_read.inject,
                                      prefix + "mlp.");
        }
    }
};

RT::ValueId emit_token_embedding(QH::Emitter& em, RT::PrimitiveGraph& pg,
                                 const weights::MatrixWeight& embed,
                                 std::uint32_t embed_index, std::uint32_t hidden_size,
                                 std::uint32_t streams) {
    RT::ValueId tokens = em.value(0, RT::ValueDType::I32);
    pg.external_inputs.push_back(tokens);

    RT::ValueId emb = em.value(hidden_size);
    {
        RT::EmbeddingLookupNode n;
        n.vocab_size = embed.rows;
        n.hidden_size = hidden_size;
        n.weight_shape = RT::MatrixwiseShapeKey{embed.rows, embed.cols};
        n.weight_index = embed_index;
        em.emit(RT::PrimitiveNode{n}, {tokens}, {emb});
        pg.nodes.back().debug_name = "embedding";
    }

    RT::ValueId x4 = emb;
    std::uint32_t feats = hidden_size;
    for (std::uint32_t s = 1; s < streams; ++s) {
        const std::uint32_t expanded_feats = feats + hidden_size;
        RT::ValueId expanded = em.value(expanded_feats);
        RT::ConcatNode cn;
        cn.shape = RT::RowwiseShapeKey{expanded_feats};
        cn.input_shape_a = RT::RowwiseShapeKey{feats};
        cn.input_shape_b = RT::RowwiseShapeKey{hidden_size};
        em.emit(RT::PrimitiveNode{cn}, {x4, emb}, {expanded});
        pg.nodes.back().debug_name = "hc_expand_" + std::to_string(s - 1);
        x4 = expanded;
        feats = expanded_feats;
    }
    return x4;
}

void emit_head(QH::Emitter& em, RT::PrimitiveGraph& pg, const Geometry& g,
               RT::ValueId x4, const weights::MatrixWeight& lm_weight,
               std::uint32_t lm_index, const Qwen4ExpHyperWeights& final_hyper,
               std::uint32_t final_mix_down_index, std::uint32_t final_mix_up_index,
               std::uint32_t final_hc_param) {
    const QH::HyperLoweringGeometry hyper_geo{
        .streams = g.streams,
        .hidden = g.hidden_size,
        .lowrank = g.lowrank,
        .eps = g.eps,
    };
    const QH::HyperLoweringWeights final_weights{
        .mix_down = &final_hyper.mix_down,
        .mix_up = &final_hyper.mix_up,
        .block_inject = nullptr,
    };
    const QH::HyperLoweringIndices final_indices{
        .w_mix_down = final_mix_down_index,
        .w_mix_up = final_mix_up_index,
        .w_block_inject = 0,
        .p_hc_norm = final_hc_param,
    };
    const RT::ValueId final_hidden = QH::emit_hyper_collapse(
        em, hyper_geo, final_weights, final_indices, x4, "final.");

    RT::ValueId selected_hidden = em.value(g.hidden_size, RT::ValueDType::F32,
                                           RT::ValueRowDomain::OUTPUT_ROWS);
    {
        RT::OutputGatherNode og{RT::RowwiseShapeKey{g.hidden_size}};
        em.emit(RT::PrimitiveNode{og}, {final_hidden}, {selected_hidden});
        pg.nodes.back().debug_name = "selected_hidden";
    }

    RT::ValueId selected_logits =
        em.value(lm_weight.rows, RT::ValueDType::F32, RT::ValueRowDomain::OUTPUT_ROWS);
    {
        RT::LinearNode ln;
        ln.shape = RT::MatrixwiseShapeKey{lm_weight.rows, lm_weight.cols};
        ln.weight_index = lm_index;
        ln.out_dtype = RT::ValueDType::F32;
        em.emit(RT::PrimitiveNode{ln}, {selected_hidden}, {selected_logits});
        pg.nodes.back().debug_name = "logits";
    }

    RT::ValueId sampled =
        em.value(1, RT::ValueDType::I32, RT::ValueRowDomain::OUTPUT_ROWS);
    em.emit(RT::PrimitiveNode{RT::SamplingNode{lm_weight.rows}}, {selected_logits},
            {sampled});
    pg.nodes.back().debug_name = "sampled";

    pg.external_outputs.push_back(selected_hidden);
    pg.external_outputs.push_back(selected_logits);
    pg.external_outputs.push_back(sampled);
    pg.external_outputs.push_back(final_hidden);
}

}

Result<Qwen4ExpLoweredPrimitives> lower_qwen4exp_to_primitives(
    const Qwen4ExpTextConfig& config,
    const Qwen4ExpModelWeights& weights,
    const Qwen4ExpLowerOptions& options) {

    auto geometry_status = validate_geometry(config, weights, options);
    if (!geometry_status.ok()) return geometry_status;

    for (std::size_t t = 0; t < options.hidden_taps.size(); ++t) {
        const std::size_t layer = options.hidden_taps[t];
        if (layer >= weights.layers.size()) {
            return Status::invalid_argument("hidden tap layer index out of range",
                                            __FILE__, __LINE__);
        }
        for (std::size_t u = 0; u < t; ++u) {
            if (static_cast<std::size_t>(options.hidden_taps[u]) == layer) {
                return Status::invalid_argument("duplicate hidden tap layer index",
                                                __FILE__, __LINE__);
            }
        }
    }
    for (std::size_t t = 0; t < options.override_layers.size(); ++t) {
        const std::size_t layer = options.override_layers[t];
        if (layer >= weights.layers.size()) {
            return Status::invalid_argument("override layer index out of range",
                                            __FILE__, __LINE__);
        }
        for (std::size_t u = 0; u < t; ++u) {
            if (static_cast<std::size_t>(options.override_layers[u]) == layer) {
                return Status::invalid_argument("duplicate override layer index",
                                                __FILE__, __LINE__);
            }
        }
    }
    if (options.connect_ple && weights.lm_head_tied) {
        return Status::invalid_argument("connect_ple requires a separate lm_head weight",
                                        __FILE__, __LINE__);
    }

    const std::size_t layer_count_all = weights.layers.size();
    const std::size_t stage_begin =
        options.stage_begin > layer_count_all
            ? layer_count_all
            : static_cast<std::size_t>(options.stage_begin);
    const std::size_t stage_end = options.stage_end == 0
                                      ? layer_count_all
                                      : static_cast<std::size_t>(options.stage_end);
    if (stage_begin >= stage_end || stage_end > layer_count_all) {
        return Status::invalid_argument("empty or out-of-range pp stage",
                                        __FILE__, __LINE__);
    }
    if (options.connect_ple && stage_begin > 0) {
        return Status::invalid_argument(
            "connect_ple belongs to the pp stage that starts at layer 0",
            __FILE__, __LINE__);
    }
    if (options.connect_ple && (!options.hidden_taps.empty() || !options.override_layers.empty())) {
        return Status::invalid_argument(
            "connect_ple cannot be combined with hidden taps or overrides", __FILE__,
            __LINE__);
    }

    Geometry g;
    auto status = parse_geometry(config, g);
    if (!status.ok()) return status;

    const std::size_t layer_count = layer_count_all;
    const std::size_t ple_layer = config.ple_layer_index;
    const std::size_t prefix_layer_end =
        options.connect_ple && stage_begin < ple_layer ? ple_layer : 0;
    const std::size_t main_layer_begin =
        prefix_layer_end > stage_begin ? prefix_layer_end : stage_begin;
    const bool has_embedding = stage_begin == 0;
    const bool has_lm_head = stage_end == layer_count;

    Qwen4ExpLoweredPrimitives lowered;
    RT::PrimitiveGraph& pg = lowered.graph;
    std::vector<RT::ValueId> hidden_tap_values(options.hidden_taps.size());
    std::vector<std::uint8_t> hidden_tap_found(options.hidden_taps.size(), 0);

    SlotBuilder prefix_slots;
    prefix_slots.graph = &lowered.ple_prefix_graph;
    SlotBuilder main_slots;
    main_slots.graph = &pg;

    std::vector<LayerWeightIndices> layer_indices(layer_count);
    std::uint32_t embed_index = 0;
    if (has_embedding && options.connect_ple) {
        embed_index = prefix_slots.add_weight(weights.embed_tokens);
        for (std::size_t li = 0; li < prefix_layer_end; ++li) {
            register_layer_slots(weights.layers[li], options, g.hidden_size,
                                 prefix_slots, layer_indices[li]);
        }
    } else if (has_embedding) {
        embed_index = main_slots.add_weight(weights.embed_tokens);
    }
    for (std::size_t li = main_layer_begin; li < stage_end; ++li) {
        register_layer_slots(weights.layers[li], options, g.hidden_size, main_slots,
                                 layer_indices[li]);
    }

    const std::uint32_t lm_index =
        !has_lm_head ? 0u : (weights.lm_head_tied && !options.connect_ple
                                 ? embed_index
                                 : main_slots.add_weight(weights.lm_head));
    const std::uint32_t final_mix_down_index =
        has_lm_head ? main_slots.add_weight(weights.final_hyper.mix_down) : 0u;
    const std::uint32_t final_mix_up_index =
        has_lm_head ? main_slots.add_weight(weights.final_hyper.mix_up) : 0u;

    for (std::size_t li = 0; li < prefix_layer_end; ++li) {
        register_layer_parameters(weights.layers[li], options, prefix_slots,
                                  layer_indices[li]);
    }
    for (std::size_t li = main_layer_begin; li < stage_end; ++li) {
        register_layer_parameters(weights.layers[li], options, main_slots,
                                  layer_indices[li]);
    }
    const std::uint32_t final_hc_param =
        has_lm_head ? main_slots.add_parameter(weights.final_hyper.hc_norm) : 0u;

    const LayerEmitter layer_emitter{config, weights, options, g, layer_indices};
    std::uint32_t next_gdn_state_index = 0;
    std::uint32_t next_attention_state_index = 0;

    if (options.connect_ple && has_embedding) {
        QH::Emitter em_pre{lowered.ple_prefix_graph};
        RT::ValueId prefix_x4 =
            emit_token_embedding(em_pre, lowered.ple_prefix_graph, weights.embed_tokens,
                                 embed_index, g.hidden_size, g.streams);
        for (std::size_t li = 0; li < prefix_layer_end; ++li) {
            layer_emitter.emit(em_pre, li, prefix_x4, next_gdn_state_index,
                               next_attention_state_index, nullptr);
        }
        lowered.ple_prefix_graph.external_outputs.push_back(prefix_x4);
        lowered.has_ple_prefix = true;
        status = RT::validate_primitive_graph(lowered.ple_prefix_graph);
        if (!status.ok()) return status;
    }

    QH::Emitter em{pg};
    RT::ValueId x4{0};
    if (options.connect_ple && has_embedding) {
        const RT::ValueId x4_entry = em.value(g.hc_hidden, RT::ValueDType::F32);
        pg.external_inputs.push_back(x4_entry);
        const RT::ValueId ple_out = em.value(g.hc_hidden, RT::ValueDType::F32);
        pg.external_inputs.push_back(ple_out);
        x4 = em.value(g.hc_hidden, RT::ValueDType::F32);
        RT::ResidualAddNode ra{RT::RowwiseShapeKey{g.hc_hidden}};
        em.emit(RT::PrimitiveNode{ra}, {x4_entry, ple_out}, {x4});
        pg.nodes.back().debug_name =
            "L" + std::to_string(config.ple_layer_index) + ".ple.add";
    } else if (has_embedding) {
        x4 = emit_token_embedding(em, pg, weights.embed_tokens, embed_index,
                                  g.hidden_size, g.streams);
    } else {
        x4 = em.value(g.hc_hidden, RT::ValueDType::F32);
        pg.external_inputs.push_back(x4);
    }

    for (std::size_t li = main_layer_begin; li < stage_end; ++li) {
        layer_emitter.emit(em, li, x4, next_gdn_state_index, next_attention_state_index,
                       &lowered.layer_input_values);
        for (std::size_t t = 0; t < options.hidden_taps.size(); ++t) {
            if (static_cast<std::size_t>(options.hidden_taps[t]) == li) {
                hidden_tap_values[t] = x4;
                hidden_tap_found[t] = 1;
            }
        }
    }

    if (has_lm_head) {
        const MW* lmw = weights.lm_head_tied ? &weights.embed_tokens : &weights.lm_head;
        emit_head(em, pg, g, x4, *lmw, lm_index, weights.final_hyper,
                  final_mix_down_index, final_mix_up_index, final_hc_param);
    } else {
        pg.external_outputs.push_back(x4);
    }

    for (std::size_t t = 0; t < hidden_tap_values.size(); ++t) {
        if (hidden_tap_found[t] == 0) {
            return Status::invalid_state("hidden tap layer was not lowered", __FILE__,
                                         __LINE__);
        }
        pg.external_outputs.push_back(hidden_tap_values[t]);
    }

    lowered.weights = std::move(main_slots.weights);
    lowered.parameters = std::move(main_slots.parameters);
    lowered.ple_prefix_weights = std::move(prefix_slots.weights);
    lowered.ple_prefix_parameters = std::move(prefix_slots.parameters);

    status = RT::validate_primitive_graph(pg);
    if (!status.ok()) return status;

    return lowered;
}

}
