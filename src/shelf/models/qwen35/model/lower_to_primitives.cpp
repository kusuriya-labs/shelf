#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/runtime/graph/primitive_node.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <shelf/models/common/qwen_hybrid/gdn_geometry.h>
#include <shelf/models/common/qwen_hybrid/attention_geometry.h>
#include <shelf/models/common/qwen_hybrid/gdn_lowering.h>
#include <shelf/models/common/qwen_hybrid/attention_lowering.h>

namespace shelf::qwen35 {

namespace {

namespace RT = ::shelf::runtime;

namespace QH = ::shelf::qwen_hybrid;
using QH::checked_u32;
using QH::checked_product;
using QH::checked_double;
using QH::validate_matrix;
using QH::Emitter;
using MW = ::shelf::weights::MatrixWeight;

std::size_t tensor_element_count(const gpu::Tensor& t) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < t.ndim(); ++i) {
        n *= t.dim(i);
    }
    return n;
}

Status validate_mtp_matrix(
    const shelf::weights::MatrixWeight& weight,
    std::uint32_t expected_rows,
    std::uint32_t expected_cols,
    const char* tensor_name) {

    if (weight.rows == expected_rows && weight.cols == expected_cols) {
        return Status::make_ok();
    }
    char message[512];
    std::snprintf(
        message, sizeof(message),
        "invalid MTP matrix %s: expected [%u,%u], got [%u,%u]",
        tensor_name,
        expected_rows, expected_cols,
        weight.rows, weight.cols);
    return Status::invalid_argument(message, __FILE__, __LINE__);
}

Status validate_mtp_vector(
    const gpu::Tensor& tensor,
    std::uint32_t expected_elements,
    const char* tensor_name) {

    const std::size_t got = tensor.ndim() == 0 ? 0 : tensor_element_count(tensor);
    if (tensor.ndim() != 0 && got == expected_elements) {
        return Status::make_ok();
    }
    char message[512];
    std::snprintf(
        message, sizeof(message),
        "invalid MTP vector %s: expected [%u], got [%zu]",
        tensor_name, expected_elements, got);
    return Status::invalid_argument(message, __FILE__, __LINE__);
}

Status validate_mtp_geometry(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights) {

    if (!weights.mtp.present) {
        return Status::unsupported("model has no MTP weights", __FILE__, __LINE__);
    }

    std::uint32_t hidden_size = 0;
    std::uint32_t fc_input_features = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t query_features = 0;
    std::uint32_t kv_features = 0;
    std::uint32_t q_projection_features = 0;
    std::uint32_t head_dim = 0;
    auto status = checked_u32(config.hidden_size, "invalid MTP hidden size", hidden_size);
    if (!status.ok()) return status;
    status = checked_double(
        hidden_size,
        "invalid MTP fc geometry",
        fc_input_features);
    if (!status.ok()) return status;
    status = checked_u32(config.ffn.intermediate_size, "invalid MTP intermediate size", intermediate_size);
    if (!status.ok()) return status;
    status = checked_product(
        config.num_attention_heads,
        config.attention_head_dim,
        "invalid MTP attention query geometry",
        query_features);
    if (!status.ok()) return status;
    status = checked_product(
        config.num_key_value_heads,
        config.attention_head_dim,
        "invalid MTP attention KV geometry",
        kv_features);
    if (!status.ok()) return status;
    status = checked_double(
        query_features,
        "invalid MTP attention gated query geometry",
        q_projection_features);
    if (!status.ok()) return status;
    status = checked_u32(config.attention_head_dim, "invalid MTP attention head dimension", head_dim);
    if (!status.ok()) return status;

    const Qwen35MtpWeights& m = weights.mtp;

    status = validate_mtp_vector(
        m.pre_fc_norm_embedding_weight, hidden_size, "mtp.pre_fc_norm_embedding.weight");
    if (!status.ok()) return status;
    status = validate_mtp_vector(
        m.pre_fc_norm_hidden_weight, hidden_size, "mtp.pre_fc_norm_hidden.weight");
    if (!status.ok()) return status;
    status = validate_mtp_vector(
        m.final_norm_weight, hidden_size, "mtp.norm.weight");
    if (!status.ok()) return status;

    status = validate_mtp_matrix(
        m.fc, hidden_size, fc_input_features, "mtp.fc.weight");
    if (!status.ok()) return status;

    const Qwen35LayerWeights& layer = m.layer;
    status = validate_mtp_vector(
        layer.input_layernorm_weight, hidden_size, "mtp.layers.0.input_layernorm.weight");
    if (!status.ok()) return status;
    status = validate_mtp_vector(
        layer.post_attention_layernorm_weight, hidden_size, "mtp.layers.0.post_attention_layernorm.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.ffn.gate, intermediate_size, hidden_size, "mtp.layers.0.mlp.gate_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.ffn.up, intermediate_size, hidden_size, "mtp.layers.0.mlp.up_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.ffn.down, hidden_size, intermediate_size, "mtp.layers.0.mlp.down_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.attn_q_proj, q_projection_features, hidden_size, "mtp.layers.0.self_attn.q_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.attn_k_proj, kv_features, hidden_size, "mtp.layers.0.self_attn.k_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.attn_v_proj, kv_features, hidden_size, "mtp.layers.0.self_attn.v_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.attn_o_proj, hidden_size, query_features, "mtp.layers.0.self_attn.o_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_vector(
        layer.attn_q_norm_weight, head_dim, "mtp.layers.0.self_attn.q_norm.weight");
    if (!status.ok()) return status;
    return validate_mtp_vector(
        layer.attn_k_norm_weight, head_dim, "mtp.layers.0.self_attn.k_norm.weight");
}

Status validate_mlp_shapes(
    const Qwen35LayerWeights& weights,
    std::uint32_t hidden_size,
    std::uint32_t intermediate_size) {

    const RT::MatrixwiseShapeKey gate{intermediate_size, hidden_size};
    const RT::MatrixwiseShapeKey down{hidden_size, intermediate_size};

    auto status = validate_matrix(weights.ffn.gate, gate, "invalid MLP gate projection");
    if (!status.ok()) return status;
    status = validate_matrix(weights.ffn.up, gate, "invalid MLP up projection");
    if (!status.ok()) return status;
    return validate_matrix(weights.ffn.down, down, "invalid MLP down projection");
}

Status validate_full_attention_shapes(
    const Qwen35LayerWeights& weights,
    const Qwen35TextConfig& config,
    std::uint32_t hidden_size,
    std::uint32_t local_query_heads,
    std::uint32_t local_kv_heads) {

    std::uint32_t head_dim = 0;
    auto status = checked_u32(config.attention_head_dim,
                              "invalid full attention head dimension", head_dim);
    if (!status.ok()) return status;

    const QH::GatedAttentionGeometry geometry{
        .query_heads = local_query_heads,
        .kv_heads = local_kv_heads,
        .head_dim = head_dim,
        .partial_rotary_factor = config.partial_rotary_factor,
    };
    const QH::GatedAttentionWeightView view{
        .q_proj = &weights.attn_q_proj,
        .k_proj = &weights.attn_k_proj,
        .v_proj = &weights.attn_v_proj,
        .o_proj = &weights.attn_o_proj,
    };
    return QH::validate_gated_attention_geometry(geometry, hidden_size, view);
}

Status validate_gdn_shapes(
    const Qwen35LayerWeights& weights,
    const Qwen35TextConfig& config,
    std::uint32_t hidden_size,
    std::uint32_t local_key_heads,
    std::uint32_t local_value_heads) {

    std::uint32_t key_head_dim = 0;
    auto status = checked_u32(config.linear_key_head_dim,
                              "invalid GDN key head dimension", key_head_dim);
    if (!status.ok()) return status;
    std::uint32_t value_head_dim = 0;
    status = checked_u32(config.linear_value_head_dim,
                         "invalid GDN value head dimension", value_head_dim);
    if (!status.ok()) return status;
    std::uint32_t conv_kernel = 0;
    status = checked_u32(config.linear_conv_kernel_dim,
                         "invalid GDN convolution kernel", conv_kernel);
    if (!status.ok()) return status;

    const QH::GdnGeometry geometry{
        .key_heads = local_key_heads,
        .value_heads = local_value_heads,
        .key_head_dim = key_head_dim,
        .value_head_dim = value_head_dim,
        .conv_kernel = conv_kernel,
    };
    const QH::GdnWeightView view{
        .qkv = &weights.attn_in_proj_qkv,
        .z = &weights.attn_in_proj_z,
        .b = &weights.attn_in_proj_b,
        .a = &weights.attn_in_proj_a,
        .out = &weights.attn_out_proj,
    };
    return QH::validate_gdn_geometry(geometry, hidden_size, view);
}

Status validate_geometry(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const Qwen35TensorParallelContext* tp) {

    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    auto status = checked_u32(config.hidden_size, "invalid hidden size", hidden_size);
    if (!status.ok()) return status;
    status = checked_u32(config.ffn.intermediate_size, "invalid intermediate size", intermediate_size);
    if (!status.ok()) return status;
    const bool tp_active = tp != nullptr && tp->tp_size > 1;
    std::uint32_t local_query_heads = config.num_attention_heads;
    std::uint32_t local_kv_heads = config.num_key_value_heads;
    std::uint32_t local_key_heads = config.linear_num_key_heads;
    std::uint32_t local_value_heads = config.linear_num_value_heads;
    if (tp_active) {
        intermediate_size = tp->local_intermediate_size;
        local_query_heads = tp->local_attention_heads;
        local_kv_heads = tp->local_key_value_heads;
        local_key_heads = tp->local_gdn_key_heads;
        local_value_heads = tp->local_gdn_value_heads;
    }
    if (config.num_hidden_layers == 0 ||
        weights.layers.size() != config.num_hidden_layers) {
        return Status::invalid_argument(
            "Qwen3.5 layer metadata count mismatch", __FILE__, __LINE__);
    }
    if (!config.layer_types.empty() &&
        config.layer_types.size() != config.num_hidden_layers) {
        return Status::invalid_argument(
            "Qwen3.5 layer type count mismatch", __FILE__, __LINE__);
    }

    auto embed_status = validate_matrix(
        weights.embed_tokens, RT::MatrixwiseShapeKey{weights.embed_tokens.rows, hidden_size},
        "invalid embedding weight");
    if (!embed_status.ok()) return embed_status;
    if (!weights.lm_head_tied) {
        auto lm_status = validate_matrix(
            weights.lm_head, RT::MatrixwiseShapeKey{weights.lm_head.rows, hidden_size},
            "invalid lm head weight");
        if (!lm_status.ok()) return lm_status;
    }

    for (std::size_t layer_index = 0;
         layer_index < config.num_hidden_layers;
         ++layer_index) {

        const auto& layer_weights = weights.layers[layer_index];
        bool is_gdn = layer_weights.is_gdn;
        if (!config.layer_types.empty()) {
            const int layer_type = config.layer_types[layer_index];
            if (layer_type != 0 && layer_type != 1) {
                return Status::unsupported(
                    "unknown Qwen3.5 layer type", __FILE__, __LINE__);
            }
            const bool config_is_gdn = layer_type == 0;
            if (config_is_gdn != is_gdn) {
                return Status::invalid_argument(
                    "Qwen3.5 layer type conflicts with weight metadata",
                    __FILE__, __LINE__);
            }
            is_gdn = config_is_gdn;
        }

        status = validate_mlp_shapes(layer_weights, hidden_size, intermediate_size);
        if (!status.ok()) return status;

        if (is_gdn) {
            status = validate_gdn_shapes(layer_weights, config, hidden_size,
                                         local_key_heads, local_value_heads);
        } else {
            status = validate_full_attention_shapes(layer_weights, config, hidden_size,
                                                    local_query_heads, local_kv_heads);
        }
        if (!status.ok()) return status;
    }
    return Status::make_ok();
}

RT::WeightSlot make_weight_slot(const MW& w) {
    RT::WeightSlot s{};
    s.rows = w.rows;
    s.cols = w.cols;
    s.k_padded = w.k_padded;
    s.encoding = static_cast<uint8_t>(w.encoding);
    s.compute_spec = static_cast<uint8_t>(w.compute_spec);
    if (w.encoding == shelf::weights::MatrixEncoding::Bf16) {
        s.device_ptr = w.bf16.data<void>();
        s.bytes = static_cast<uint64_t>(w.rows) * w.cols * sizeof(uint16_t);
        s.codes = nullptr;
        s.scales = nullptr;
    } else {
        s.device_ptr = nullptr;
        s.bytes = 0;
        s.codes = w.codes.ndim() != 0 ? w.codes.data<void>() : nullptr;
        s.scales = w.scales.ndim() != 0 ? w.scales.data<void>() : nullptr;
        s.row_ref = w.row_ref.ndim() != 0 ? w.row_ref.data<void>() : nullptr;
        s.row_ref_exact = w.row_ref_exact;
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
    uint32_t mlp_gate = 0;
    uint32_t mlp_up = 0;
    uint32_t mlp_down = 0;
};

}

Result<Qwen35LoweredPrimitives> lower_qwen35_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const Qwen35LowerOptions& options) {
    auto geometry_status = validate_geometry(config, weights, options.tp);
    if (!geometry_status.ok()) return geometry_status;

    for (std::size_t t = 0; t < options.hidden_taps.size(); ++t) {
        const std::size_t layer = options.hidden_taps[t];
        if (layer >= weights.layers.size()) {
            return Status::invalid_argument(
                "hidden tap layer index out of range", __FILE__, __LINE__);
        }
        for (std::size_t u = 0; u < t; ++u) {
            if (static_cast<std::size_t>(options.hidden_taps[u]) == layer) {
                return Status::invalid_argument(
                    "duplicate hidden tap layer index", __FILE__, __LINE__);
            }
        }
    }

    Qwen35LoweredPrimitives lowered;
    RT::PrimitiveGraph& pg = lowered.graph;
    std::vector<RT::ValueId> hidden_tap_values(options.hidden_taps.size());
    std::vector<uint8_t> hidden_tap_found(options.hidden_taps.size(), 0);

    std::vector<const MW*> weight_pointers;
    auto add_weight = [&](const MW& w) -> uint32_t {
        weight_pointers.push_back(&w);
        lowered.weights.push_back(make_weight_slot(w));
        pg.weight_table.push_back(RT::MatrixwiseShapeKey{w.rows, w.cols});
        return static_cast<uint32_t>(weight_pointers.size() - 1);
    };

    const uint32_t embed_index = add_weight(weights.embed_tokens);
    std::vector<LayerWeightIndices> layer_weights(weights.layers.size());
    {
        size_t cursor = 1;
        for (size_t li = 0; li < weights.layers.size(); ++li) {
            const auto& lw = weights.layers[li];
            auto& idx = layer_weights[li];
            if (lw.is_gdn) {
                idx.gdn_qkv = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_in_proj_qkv);
                idx.gdn_z = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_in_proj_z);
                idx.gdn_b = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_in_proj_b);
                idx.gdn_a = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_in_proj_a);
                idx.gdn_out = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_out_proj);
            } else {
                idx.attn_q = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_q_proj);
                idx.attn_k = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_k_proj);
                idx.attn_v = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_v_proj);
                idx.attn_o = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_o_proj);
            }
            idx.mlp_gate = static_cast<uint32_t>(cursor++);
            (void)add_weight(lw.ffn.gate);
            idx.mlp_up = static_cast<uint32_t>(cursor++);
            (void)add_weight(lw.ffn.up);
            idx.mlp_down = static_cast<uint32_t>(cursor++);
            (void)add_weight(lw.ffn.down);
        }
    }
    const bool lm_tied = weights.lm_head_tied;
    const uint32_t lm_index =
        lm_tied ? embed_index : add_weight(weights.lm_head);

    uint32_t key_heads = static_cast<uint32_t>(config.linear_num_key_heads);
    uint32_t value_heads = static_cast<uint32_t>(config.linear_num_value_heads);
    uint32_t key_head_dim = static_cast<uint32_t>(config.linear_key_head_dim);
    uint32_t value_head_dim = static_cast<uint32_t>(config.linear_value_head_dim);
    uint32_t conv_kernel = static_cast<uint32_t>(config.linear_conv_kernel_dim);
    uint32_t q_heads = static_cast<uint32_t>(config.num_attention_heads);
    uint32_t kv_heads = static_cast<uint32_t>(config.num_key_value_heads);
    uint32_t head_dim = static_cast<uint32_t>(config.attention_head_dim);
    uint32_t mlp_intermediate = static_cast<uint32_t>(config.ffn.intermediate_size);
    if (options.tp != nullptr && options.tp->tp_size > 1) {
        key_heads = options.tp->local_gdn_key_heads;
        value_heads = options.tp->local_gdn_value_heads;
        q_heads = options.tp->local_attention_heads;
        kv_heads = options.tp->local_key_value_heads;
        mlp_intermediate = options.tp->local_intermediate_size;
    }
    float rope_theta = config.rope_theta;
    float eps = config.rms_norm_eps;
    float partial = config.partial_rotary_factor;
    if (rope_theta == 0) rope_theta = 10000000.0f;
    if (eps == 0) eps = 1e-6f;
    if (partial == 0) partial = 0.25f;
    uint32_t rotary_dim = static_cast<uint32_t>((double)head_dim * (double)partial);

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
    std::vector<LayerParamIndices> layer_params(weights.layers.size());
    {
        for (size_t li = 0; li < weights.layers.size(); ++li) {
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
    }
    const uint32_t final_norm_param =
        weights.final_norm_weight.ndim() > 0 ? reg_param(weights.final_norm_weight) : UINT32_MAX;

    uint32_t next_gdn_state_index = 0;
    uint32_t next_attention_state_index = 0;

    Emitter em{pg};

    const uint32_t hidden_size = static_cast<uint32_t>(config.hidden_size);
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

    for (size_t li = 0; li < weights.layers.size(); ++li) {
        const auto& lw = weights.layers[li];
        const auto& lw_idx = layer_weights[li];
        const std::string layer_prefix = "L" + std::to_string(li) + ".";
        const auto name_last = [&](const char* suffix) {
            pg.nodes.back().debug_name = layer_prefix + suffix;
        };
        for (std::size_t o = 0; o < options.override_layers.size(); ++o) {
            if (static_cast<size_t>(options.override_layers[o]) == li) {
                lowered.layer_input_values.emplace_back(static_cast<uint32_t>(li),
                                                        layer_input);
            }
        }

        const auto& lp = layer_params[li];
        RT::RmsNormNode input_rms;
        input_rms.shape = hidden_key;
        input_rms.eps = eps;
        input_rms.group_size = hidden_size;
        input_rms.parameter_index = lp.input_norm;
        input_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
        RT::ValueId normed = em.unary(RT::PrimitiveNode{input_rms}, layer_input, hidden_size);
        name_last("input_norm");

        RT::ValueId block_out{0};
        if (lw.is_gdn) {
            const MW& w_qkv = *weight_pointers[lw_idx.gdn_qkv];
            const MW& w_z = *weight_pointers[lw_idx.gdn_z];
            const MW& w_b = *weight_pointers[lw_idx.gdn_b];
            const MW& w_a = *weight_pointers[lw_idx.gdn_a];
            const MW& w_out = *weight_pointers[lw_idx.gdn_out];

            const QH::GdnLoweringGeometry geo{
                .key_heads = key_heads,
                .value_heads = value_heads,
                .key_head_dim = key_head_dim,
                .value_head_dim = value_head_dim,
                .conv_kernel = conv_kernel,
                .eps = eps,
                .out_dtype = options.tp_combine_f32 ? RT::ValueDType::F32
                                                    : RT::ValueDType::BF16,
            };
            const QH::GdnLoweringWeights gdn_weights{
                .qkv = &w_qkv,
                .z = &w_z,
                .b = &w_b,
                .a = &w_a,
                .out = &w_out,
            };
            const QH::GdnLoweringIndices gdn_indices{
                .w_qkv = lw_idx.gdn_qkv,
                .w_z = lw_idx.gdn_z,
                .w_b = lw_idx.gdn_b,
                .w_a = lw_idx.gdn_a,
                .w_out = lw_idx.gdn_out,
                .p_conv = lp.conv,
                .p_dt_bias = lp.dt_bias,
                .p_a_log = lp.a_log,
                .p_gdn_norm = lp.gdn_norm,
                .state_index = next_gdn_state_index++,
            };
            block_out = QH::emit_gdn(em, geo, gdn_weights, gdn_indices, normed,
                                     layer_prefix);
        } else {
            const MW& w_q = *weight_pointers[lw_idx.attn_q];
            const MW& w_k = *weight_pointers[lw_idx.attn_k];
            const MW& w_v = *weight_pointers[lw_idx.attn_v];
            const MW& w_o = *weight_pointers[lw_idx.attn_o];

            const QH::GatedAttentionLoweringGeometry geo{
                .query_heads = q_heads,
                .kv_heads = kv_heads,
                .head_dim = head_dim,
                .rotary_dim = rotary_dim,
                .rope_theta = rope_theta,
                .rope_attention_factor =
                    static_cast<float>(options.rope_profile.attention_factor),
                .eps = eps,
                .out_dtype = options.tp_combine_f32 ? RT::ValueDType::F32
                                                    : RT::ValueDType::BF16,
            };
            const QH::GatedAttentionLoweringWeights attn_weights{
                .q = &w_q,
                .k = &w_k,
                .v = &w_v,
                .o = &w_o,
            };
            const QH::GatedAttentionLoweringIndices attn_indices{
                .w_q = lw_idx.attn_q,
                .w_k = lw_idx.attn_k,
                .w_v = lw_idx.attn_v,
                .w_o = lw_idx.attn_o,
                .p_q_norm = lp.q_norm,
                .p_k_norm = lp.k_norm,
                .state_index = next_attention_state_index++,
            };
            block_out = QH::emit_gated_attention(
                em, geo, attn_weights, attn_indices, normed, layer_prefix);
        }

        RT::ValueId attn_residual{0};
        {
            RT::ResidualAddNode ra{hidden_key};
            attn_residual = em.value(hidden_size);
            em.emit(RT::PrimitiveNode{ra}, {block_out, layer_input}, {attn_residual});
            name_last("residual_attn");
        }

        RT::RmsNormNode post_rms;
        post_rms.shape = hidden_key;
        post_rms.eps = eps;
        post_rms.group_size = hidden_size;
        post_rms.parameter_index = lp.post_norm;
        post_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
        RT::ValueId post_norm = em.unary(RT::PrimitiveNode{post_rms}, attn_residual,
                                         hidden_size);
        name_last("post_norm");

        const MW& w_gate = *weight_pointers[lw_idx.mlp_gate];
        const MW& w_up = *weight_pointers[lw_idx.mlp_up];
        const MW& w_down = *weight_pointers[lw_idx.mlp_down];

        RT::ValueId v_mlp_gate = em.linear(w_gate, lw_idx.mlp_gate, post_norm);
        name_last("mlp_gate");
        RT::ValueId v_mlp_up = em.linear(w_up, lw_idx.mlp_up, post_norm);
        name_last("mlp_up");
        RT::ValueId v_swiglu = em.value(mlp_intermediate);
        {
            RT::SwiGluNode sn{RT::RowwiseShapeKey{mlp_intermediate}};
            em.emit(RT::PrimitiveNode{sn}, {v_mlp_gate, v_mlp_up}, {v_swiglu});
            name_last("mlp_swiglu");
        }
        RT::ValueId v_down = em.linear(w_down, lw_idx.mlp_down, v_swiglu,
                                       options.tp_combine_f32 ? RT::ValueDType::F32
                                                              : RT::ValueDType::BF16);
        name_last("mlp_down");
        pg.nodes.back().tp_combine = 1;

        RT::ValueId layer_output{0};
        {
            RT::ResidualAddNode ra{hidden_key};
            layer_output = em.value(hidden_size);
            em.emit(RT::PrimitiveNode{ra}, {v_down, attn_residual}, {layer_output});
            name_last("output");
        }
        layer_input = layer_output;
        for (std::size_t t = 0; t < options.hidden_taps.size(); ++t) {
            if (static_cast<std::size_t>(options.hidden_taps[t]) == li) {
                hidden_tap_values[t] = layer_output;
                hidden_tap_found[t] = 1;
            }
        }
    }

    {
        const MW* lmw = lm_tied ? &weights.embed_tokens : &weights.lm_head;
        const uint64_t global_vocab = config.vocab_size;
        bool vocab_parallel = false;
        uint32_t lm_head_col_offset = 0;
        if (!lm_tied && options.tp != nullptr &&
            options.tp->vocab_parallel_lm_head &&
            lmw->encoding == shelf::weights::MatrixEncoding::Bf16 &&
            lmw->partition.has_value()) {
            const auto& part = lmw->partition.value();
            vocab_parallel = part.axis == 0 && part.count == options.tp->tp_size &&
                part.index == options.tp->tp_rank &&
                lmw->rows == options.tp->local_vocab_size &&
                static_cast<uint64_t>(lmw->rows) * part.count == global_vocab;
            if (vocab_parallel) {
                lm_head_col_offset = static_cast<uint32_t>(part.index) * lmw->rows;
            }
        }
        if (!vocab_parallel && lmw->rows != global_vocab) {
            return Status::unsupported(
                "vocab sharded lm_head requires vocab parallel lowering", __FILE__,
                __LINE__);
        }
        const uint32_t out_vocab =
            vocab_parallel ? static_cast<uint32_t>(global_vocab) : lmw->rows;

        RT::ValueId selected_hidden = em.value(
            hidden_size, RT::ValueDType::BF16, RT::ValueRowDomain::OUTPUT_ROWS);
        RT::OutputGatherNode og{RT::RowwiseShapeKey{hidden_size}};
        em.emit(RT::PrimitiveNode{og}, {layer_input}, {selected_hidden});
        pg.nodes.back().debug_name = "selected_hidden";

        RT::RmsNormNode head_rms;
        head_rms.shape = hidden_key;
        head_rms.eps = eps;
        head_rms.group_size = hidden_size;
        head_rms.parameter_index = final_norm_param;
        head_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
        RT::ValueId normed_hidden = em.value(hidden_size);
        em.emit(RT::PrimitiveNode{head_rms}, {selected_hidden}, {normed_hidden});
        pg.nodes.back().debug_name = "final_norm";

        RT::ValueId selected_logits = em.value(
            out_vocab, RT::ValueDType::F32, RT::ValueRowDomain::OUTPUT_ROWS);
        RT::LinearNode ln;
        ln.shape = RT::MatrixwiseShapeKey{lmw->rows, lmw->cols};
        ln.weight_index = lm_index;
        ln.out_dtype = RT::ValueDType::F32;
        em.emit(RT::PrimitiveNode{ln}, {normed_hidden}, {selected_logits});
        pg.nodes.back().debug_name = "logits";
        if (vocab_parallel) {
            pg.nodes.back().tp_combine = 1;
            pg.nodes.back().tp_combine_local_cols = lmw->rows;
            lowered.weights[lm_index].output_col_offset = lm_head_col_offset;
            lowered.weights[lm_index].output_col_total = out_vocab;
        }

        RT::ValueId sampled = em.value(
            1, RT::ValueDType::I32, RT::ValueRowDomain::OUTPUT_ROWS);
        em.emit(RT::PrimitiveNode{RT::SamplingNode{out_vocab}}, {selected_logits}, {sampled});
        pg.nodes.back().debug_name = "sampled";

        pg.external_outputs.push_back(selected_hidden);
        pg.external_outputs.push_back(selected_logits);
        pg.external_outputs.push_back(sampled);
        pg.external_outputs.push_back(layer_input);
    }

    for (std::size_t t = 0; t < hidden_tap_values.size(); ++t) {
        if (hidden_tap_found[t] == 0) {
            return Status::invalid_state(
                "hidden tap layer was not lowered", __FILE__, __LINE__);
        }
        pg.external_outputs.push_back(hidden_tap_values[t]);
    }

    auto st = RT::validate_primitive_graph(pg);
    if (!st.ok()) return st;

    return lowered;
}

Result<Qwen35LoweredPrimitives> lower_qwen35_mtp_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const EffectiveRopeProfile& rope_profile) {

    auto mtp_geometry_status = validate_mtp_geometry(config, weights);
    if (!mtp_geometry_status.ok()) return mtp_geometry_status;

    const uint32_t hidden_size = static_cast<uint32_t>(config.hidden_size);
    const uint32_t q_heads = static_cast<uint32_t>(config.num_attention_heads);
    const uint32_t kv_heads = static_cast<uint32_t>(config.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(config.attention_head_dim);
    const uint32_t intermediate = static_cast<uint32_t>(config.ffn.intermediate_size);
    const uint32_t q_features = q_heads * head_dim;
    const uint32_t kv_features = kv_heads * head_dim;
    const uint32_t q_proj_features = 2u * q_features;

    float eps = config.rms_norm_eps;
    float rope_theta = config.rope_theta;
    float partial = config.partial_rotary_factor;
    if (eps == 0) eps = 1e-6f;
    if (rope_theta == 0) rope_theta = 10000000.0f;
    if (partial == 0) partial = 0.25f;
    const uint32_t rotary_dim = static_cast<uint32_t>((double)head_dim * (double)partial);

    Qwen35LoweredPrimitives lowered;
    RT::PrimitiveGraph& pg = lowered.graph;

    auto add_weight = [&](const MW& w) -> uint32_t {
        lowered.weights.push_back(make_weight_slot(w));
        pg.weight_table.push_back(RT::MatrixwiseShapeKey{w.rows, w.cols});
        return static_cast<uint32_t>(lowered.weights.size() - 1u);
    };
    auto reg_param = [&](const gpu::Tensor& t) -> uint32_t {
        RT::StaticParameterSlot s{};
        s.device_ptr = t.data<void>();
        std::size_t n = 1;
        for (std::size_t i = 0; i < t.ndim(); ++i) n *= t.dim(i);
        s.elements = static_cast<uint32_t>(n);
        s.dtype = t.element_size() == 2 ? RT::ValueDType::BF16 : RT::ValueDType::F32;
        lowered.parameters.push_back(s);
        return static_cast<uint32_t>(lowered.parameters.size() - 1u);
    };

    const uint32_t embed_index = add_weight(weights.embed_tokens);
    const uint32_t lm_index =
        weights.lm_head_tied ? embed_index : add_weight(weights.lm_head);
    const uint32_t fc_index = add_weight(weights.mtp.fc);

    const auto& lw = weights.mtp.layer;
    const uint32_t q_i = add_weight(lw.attn_q_proj);
    const uint32_t k_i = add_weight(lw.attn_k_proj);
    const uint32_t v_i = add_weight(lw.attn_v_proj);
    const uint32_t o_i = add_weight(lw.attn_o_proj);
    const uint32_t gate_i = add_weight(lw.ffn.gate);
    const uint32_t up_i = add_weight(lw.ffn.up);
    const uint32_t down_i = add_weight(lw.ffn.down);

    const uint32_t p_enorm = reg_param(weights.mtp.pre_fc_norm_embedding_weight);
    const uint32_t p_hnorm = reg_param(weights.mtp.pre_fc_norm_hidden_weight);
    const uint32_t p_inorm = reg_param(lw.input_layernorm_weight);
    const uint32_t p_postnorm = reg_param(lw.post_attention_layernorm_weight);
    const uint32_t p_qnorm = reg_param(lw.attn_q_norm_weight);
    const uint32_t p_knorm = reg_param(lw.attn_k_norm_weight);
    const uint32_t p_finalnorm = reg_param(weights.mtp.final_norm_weight);

    Emitter em{pg};
    const RT::RowwiseShapeKey hidden_key{hidden_size};
    const RT::RowwiseShapeKey concat_key{2u * hidden_size};

    RT::ValueId tokens = em.value(0, RT::ValueDType::I32);
    RT::ValueId hidden_in = em.value(hidden_size);
    pg.external_inputs.push_back(tokens);
    pg.external_inputs.push_back(hidden_in);

    RT::ValueId v_embed = em.value(hidden_size);
    {
        RT::EmbeddingLookupNode n;
        n.vocab_size = weights.embed_tokens.rows;
        n.hidden_size = hidden_size;
        n.weight_shape =
            RT::MatrixwiseShapeKey{weights.embed_tokens.rows, weights.embed_tokens.cols};
        n.weight_index = embed_index;
        em.emit(RT::PrimitiveNode{n}, {tokens}, {v_embed});
    }

    RT::RmsNormNode enorm;
    enorm.shape = hidden_key;
    enorm.eps = eps;
    enorm.group_size = hidden_size;
    enorm.parameter_index = p_enorm;
    enorm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId v_en = em.unary(RT::PrimitiveNode{enorm}, v_embed, hidden_size);

    RT::RmsNormNode hnorm;
    hnorm.shape = hidden_key;
    hnorm.eps = eps;
    hnorm.group_size = hidden_size;
    hnorm.parameter_index = p_hnorm;
    hnorm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId v_hn = em.unary(RT::PrimitiveNode{hnorm}, hidden_in, hidden_size);

    RT::ValueId v_cat = em.value(concat_key.features);
    {
        RT::ConcatNode cn;
        cn.shape = concat_key;
        cn.input_shape_a = hidden_key;
        cn.input_shape_b = hidden_key;
        em.emit(RT::PrimitiveNode{cn}, {v_en, v_hn}, {v_cat});
    }

    RT::ValueId x = em.linear(weights.mtp.fc, fc_index, v_cat);

    RT::RmsNormNode input_rms;
    input_rms.shape = hidden_key;
    input_rms.eps = eps;
    input_rms.group_size = hidden_size;
    input_rms.parameter_index = p_inorm;
    input_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId normed = em.unary(RT::PrimitiveNode{input_rms}, x, hidden_size);

    RT::AttentionShapeKey attn_key{q_heads, kv_heads, head_dim, rotary_dim};
    RT::ValueId v_qproj = em.linear(lw.attn_q_proj, q_i, normed);
    RT::ValueId v_k = em.linear(lw.attn_k_proj, k_i, normed);
    RT::ValueId v_v = em.linear(lw.attn_v_proj, v_i, normed);

    RT::ValueId v_q = em.value(q_features);
    RT::ValueId v_gate = em.value(q_features);
    {
        RT::SplitNode sn;
        sn.input_shape = RT::RowwiseShapeKey{q_proj_features};
        sn.output_shape_a = RT::RowwiseShapeKey{q_features};
        sn.output_shape_b = RT::RowwiseShapeKey{q_features};
        sn.split_dim = 1;
        sn.layout = RT::SplitLayout::INTERLEAVED_HEADS;
        sn.head_dim = head_dim;
        em.emit(RT::PrimitiveNode{sn}, {v_qproj}, {v_q, v_gate});
    }

    RT::RmsNormNode q_rms;
    q_rms.shape = RT::RowwiseShapeKey{q_features};
    q_rms.eps = eps;
    q_rms.group_size = head_dim;
    q_rms.parameter_index = p_qnorm;
    q_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId v_qn = em.unary(RT::PrimitiveNode{q_rms}, v_q, q_features, RT::ValueDType::F32);
    RT::ValueId v_qr = em.unary(
        RT::PrimitiveNode{RT::RoPENode{attn_key, rotary_dim, head_dim, rope_theta,
                                       static_cast<float>(rope_profile.attention_factor)}},
        v_qn, q_features);

    RT::RmsNormNode k_rms;
    k_rms.shape = RT::RowwiseShapeKey{kv_features};
    k_rms.eps = eps;
    k_rms.group_size = head_dim;
    k_rms.parameter_index = p_knorm;
    k_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId v_kn = em.unary(RT::PrimitiveNode{k_rms}, v_k, kv_features, RT::ValueDType::F32);
    RT::ValueId v_kr = em.unary(
        RT::PrimitiveNode{RT::RoPENode{attn_key, rotary_dim, head_dim, rope_theta,
                                       static_cast<float>(rope_profile.attention_factor)}},
        v_kn, kv_features);

    const uint32_t mtp_attn_state_index = 0;
    RT::StateId kv_state =
        pg.alloc_state(RT::PrimitiveStateKind::KV_CACHE, mtp_attn_state_index);
    RT::KvAppendNode kan{attn_key, mtp_attn_state_index};
    em.emit(RT::PrimitiveNode{kan}, {v_kr, v_v}, {}, {}, {kv_state});

    RT::ValueId v_ctx = em.value(q_features, RT::ValueDType::F32);
    {
        RT::PagedAttentionNode pan{
            attn_key, head_dim ? 1.0f / sqrtf(static_cast<float>(head_dim)) : 1.0f,
            mtp_attn_state_index};
        auto& pgn = pg.add_node(RT::PrimitiveNode{pan});
        pgn.inputs = {v_qr};
        pgn.outputs = {v_ctx};
        pgn.state_inputs = {kv_state};
    }

    RT::ValueId v_gs = em.unary(
        RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{q_features}}},
        v_gate, q_features, RT::ValueDType::F32);
    RT::ValueId v_att = em.value(q_features);
    {
        RT::MulNode mn{RT::RowwiseShapeKey{q_features}};
        em.emit(RT::PrimitiveNode{mn}, {v_ctx, v_gs}, {v_att});
    }
    RT::ValueId block_out = em.linear(lw.attn_o_proj, o_i, v_att);

    RT::ValueId attn_residual = em.value(hidden_size);
    {
        RT::ResidualAddNode ra{hidden_key};
        em.emit(RT::PrimitiveNode{ra}, {block_out, x}, {attn_residual});
    }

    RT::RmsNormNode post_rms;
    post_rms.shape = hidden_key;
    post_rms.eps = eps;
    post_rms.group_size = hidden_size;
    post_rms.parameter_index = p_postnorm;
    post_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId post_norm = em.unary(RT::PrimitiveNode{post_rms}, attn_residual, hidden_size);

    RT::ValueId v_mlp_gate = em.linear(lw.ffn.gate, gate_i, post_norm);
    RT::ValueId v_mlp_up = em.linear(lw.ffn.up, up_i, post_norm);
    RT::ValueId v_swiglu = em.value(intermediate);
    {
        RT::SwiGluNode sn{RT::RowwiseShapeKey{intermediate}};
        em.emit(RT::PrimitiveNode{sn}, {v_mlp_gate, v_mlp_up}, {v_swiglu});
    }
    RT::ValueId v_down = em.linear(lw.ffn.down, down_i, v_swiglu);

    RT::ValueId layer_output = em.value(hidden_size);
    {
        RT::ResidualAddNode ra{hidden_key};
        em.emit(RT::PrimitiveNode{ra}, {v_down, attn_residual}, {layer_output});
    }

    RT::RmsNormNode final_rms;
    final_rms.shape = hidden_key;
    final_rms.eps = eps;
    final_rms.group_size = hidden_size;
    final_rms.parameter_index = p_finalnorm;
    final_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;

    RT::ValueId selected_hidden = em.value(
        hidden_size, RT::ValueDType::BF16, RT::ValueRowDomain::OUTPUT_ROWS);
    RT::OutputGatherNode og{RT::RowwiseShapeKey{hidden_size}};
    em.emit(RT::PrimitiveNode{og}, {layer_output}, {selected_hidden});

    RT::ValueId normed_hidden =
        em.unary(RT::PrimitiveNode{final_rms}, selected_hidden, hidden_size);

    const MW* lmw = weights.lm_head_tied ? &weights.embed_tokens : &weights.lm_head;
    if (lmw->rows != config.vocab_size) {
        return Status::unsupported(
            "mtp lowering requires an unsharded lm_head", __FILE__, __LINE__);
    }
    RT::ValueId selected_logits = em.value(
        lmw->rows, RT::ValueDType::F32, RT::ValueRowDomain::OUTPUT_ROWS);
    {
        RT::LinearNode ln;
        ln.shape = RT::MatrixwiseShapeKey{lmw->rows, lmw->cols};
        ln.weight_index = lm_index;
        ln.out_dtype = RT::ValueDType::F32;
        em.emit(RT::PrimitiveNode{ln}, {normed_hidden}, {selected_logits});
    }

    RT::ValueId sampled = em.value(1, RT::ValueDType::I32, RT::ValueRowDomain::OUTPUT_ROWS);
    em.emit(RT::PrimitiveNode{RT::SamplingNode{lmw->rows}}, {selected_logits}, {sampled});

    pg.external_outputs.push_back(selected_hidden);
    pg.external_outputs.push_back(selected_logits);
    pg.external_outputs.push_back(sampled);
    pg.external_outputs.push_back(normed_hidden);

    auto st = RT::validate_primitive_graph(pg);
    if (!st.ok()) return st;

    return lowered;
}

}
