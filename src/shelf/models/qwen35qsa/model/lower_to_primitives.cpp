#include <shelf/models/qwen35qsa/model/lower_to_primitives.h>

#include <shelf/models/qwen35/model/qwen35_model.h>

#include <shelf/kernels/optimized/qsa.h>
#include <shelf/models/common/qwen_hybrid/attention_geometry.h>
#include <shelf/models/common/qwen_hybrid/block_lowering.h>
#include <shelf/models/common/qwen_hybrid/gdn_geometry.h>
#include <shelf/models/common/qwen_hybrid/head_lowering.h>
#include <shelf/models/common/qwen_hybrid/weight_slot.h>
#include <shelf/models/qwen35qsa/model/qsa_indexer_lowering.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace shelf {
namespace qwen35qsa {

namespace {

namespace RT = ::shelf::runtime;
namespace QH = ::shelf::qwen_hybrid;
using QH::Emitter;
using QH::make_weight_slot;
using QH::validate_matrix;
using MW = ::shelf::weights::MatrixWeight;

std::size_t tensor_elements(const gpu::Tensor& t) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < t.ndim(); ++i) n *= t.dim(i);
    return n;
}

Status validate_dense_ffn(const shelf::qwen35::DenseFfnWeights& ffn,
                          std::uint32_t hidden,
                          std::uint32_t intermediate) {
    const RT::MatrixwiseShapeKey gate{intermediate, hidden};
    const RT::MatrixwiseShapeKey down{hidden, intermediate};
    Status st = validate_matrix(ffn.gate, gate, "invalid MLP gate projection");
    if (!st.ok()) return st;
    st = validate_matrix(ffn.up, gate, "invalid MLP up projection");
    if (!st.ok()) return st;
    return validate_matrix(ffn.down, down, "invalid MLP down projection");
}

Status validate_attention_layer(const shelf::qwen35::Qwen35LayerWeights& lw,
                                const Qwen35QsaTextConfig& config,
                                std::uint32_t hidden,
                                std::uint32_t q_heads,
                                std::uint32_t kv_heads,
                                std::uint32_t head_dim) {
    const QH::GatedAttentionGeometry geometry{
        .query_heads = q_heads,
        .kv_heads = kv_heads,
        .head_dim = head_dim,
        .partial_rotary_factor = config.partial_rotary_factor,
    };
    const QH::GatedAttentionWeightView view{
        .q_proj = &lw.attn_q_proj,
        .k_proj = &lw.attn_k_proj,
        .v_proj = &lw.attn_v_proj,
        .o_proj = &lw.attn_o_proj,
    };
    return QH::validate_gated_attention_geometry(geometry, hidden, view);
}

Status validate_gdn_layer(const shelf::qwen35::Qwen35LayerWeights& lw,
                          const Qwen35QsaTextConfig& config,
                          std::uint32_t hidden,
                          const QH::GdnGeometry& geometry,
                          const QH::GdnFeatureSizes& features) {
    const QH::GdnWeightView view{
        .qkv = &lw.attn_in_proj_qkv,
        .z = &lw.attn_in_proj_z,
        .b = &lw.attn_in_proj_b,
        .a = &lw.attn_in_proj_a,
        .out = &lw.attn_out_proj,
    };
    Status st = QH::validate_gdn_geometry(geometry, hidden, view);
    if (!st.ok()) return st;
    if (tensor_elements(lw.attn_conv1d_weight) !=
        static_cast<std::size_t>(features.conv_features) *
            static_cast<std::size_t>(config.linear_conv_kernel_dim)) {
        return Status::invalid_argument(
            "qwen35qsa: GDN conv1d shape mismatch", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status validate_indexer_layer(const QsaIndexerLayerWeights& w,
                              const Qwen35QsaTextConfig& config,
                              std::uint32_t hidden) {
    const std::uint64_t q_rows = config.indexer.heads * config.indexer.head_dim;
    if (w.q_proj.rows != q_rows || w.q_proj.cols != hidden ||
        w.k_proj.rows != config.indexer.head_dim || w.k_proj.cols != hidden) {
        return Status::invalid_argument(
            "qwen35qsa: indexer projection shape mismatch", __FILE__, __LINE__);
    }
    const std::uint64_t dim = config.indexer.head_dim;
    if (tensor_elements(w.q_norm_weight) != dim ||
        tensor_elements(w.k_norm_weight) != dim ||
        tensor_elements(w.k_block_norm_weight) != dim) {
        return Status::invalid_argument(
            "qwen35qsa: indexer norm shape mismatch", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

struct QsaGeometry {
    std::uint32_t heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rotary_dim = 0;
    std::uint32_t compress_ratio = 0;
    std::uint32_t blocks = 0;
    std::uint32_t top_k = 0;
    std::uint32_t mask_bytes = 0;
};

struct LayerWeightIndices {
    std::uint32_t attn_q = 0;
    std::uint32_t attn_k = 0;
    std::uint32_t attn_v = 0;
    std::uint32_t attn_o = 0;
    std::uint32_t indexer_q = UINT32_MAX;
    std::uint32_t indexer_k = UINT32_MAX;
    std::uint32_t gdn_qkv = 0;
    std::uint32_t gdn_z = 0;
    std::uint32_t gdn_b = 0;
    std::uint32_t gdn_a = 0;
    std::uint32_t gdn_out = 0;
    std::uint32_t mlp_gate = 0;
    std::uint32_t mlp_up = 0;
    std::uint32_t mlp_down = 0;
};

struct LayerParamIndices {
    std::uint32_t input_norm = UINT32_MAX;
    std::uint32_t post_norm = UINT32_MAX;
    std::uint32_t q_norm = UINT32_MAX;
    std::uint32_t k_norm = UINT32_MAX;
    std::uint32_t indexer_q_norm = UINT32_MAX;
    std::uint32_t indexer_k_norm = UINT32_MAX;
    std::uint32_t indexer_k_block_norm = UINT32_MAX;
    std::uint32_t conv = UINT32_MAX;
    std::uint32_t gdn_norm = UINT32_MAX;
    std::uint32_t dt_bias = UINT32_MAX;
    std::uint32_t a_log = UINT32_MAX;
};

}  // namespace

Result<shelf::qwen35::Qwen35LoweredPrimitives> lower_qwen35qsa_to_primitives(
    const Qwen35QsaTextConfig& config,
    const Qwen35QsaModelWeights& weights,
    const Qwen35QsaLowerOptions& options) {
    const auto& backbone_options = options.backbone;
    const bool tp_active =
        backbone_options.tp != nullptr && backbone_options.tp->tp_size > 1u;
    if (config.indexer.declared == false) {
        return Status::invalid_argument(
            "qwen35qsa: the config does not declare an indexer", __FILE__, __LINE__);
    }
    if (options.max_context_tokens == 0u) {
        return Status::invalid_argument(
            "qwen35qsa: lowering requires max_context_tokens", __FILE__, __LINE__);
    }
    if (config.indexer.head_dim != ::shelf::kernel::kQsaHeadDim) {
        return Status::unsupported(
            "qwen35qsa: indexer head_dim is not supported by the select kernel",
            __FILE__, __LINE__);
    }
    if (config.indexer.heads > ::shelf::kernel::kQsaSelectMaxHeads) {
        return Status::unsupported(
            "qwen35qsa: indexer heads exceed the select kernel limit", __FILE__, __LINE__);
    }
    if (config.indexer.kept_blocks() > ::shelf::kernel::kQsaSelectMaxTopK) {
        return Status::unsupported(
            "qwen35qsa: kept block count exceeds the select kernel limit", __FILE__,
            __LINE__);
    }
    Status model_st =
        shelf::qwen35::validate_qwen35_weights(
            weights.backbone, config,
            backbone_options.stage.enabled ? backbone_options.stage
                                           : shelf::qwen_hybrid::PipelineStage{});
    if (!model_st.ok()) return model_st;

    const std::uint32_t num_layers = static_cast<std::uint32_t>(config.num_hidden_layers);
    const bool stage_load = backbone_options.stage.enabled;
    const std::uint32_t layer_lo = stage_load ? backbone_options.stage.layer_lo() : 0u;
    const std::uint32_t layer_hi = stage_load ? backbone_options.stage.layer_hi() : num_layers;
    const bool first_stage = !stage_load || backbone_options.stage.first();
    const bool last_stage = !stage_load || backbone_options.stage.last(num_layers);
    if (stage_load && (layer_lo >= layer_hi || layer_hi > num_layers)) {
        return Status::invalid_argument(
            "qwen35qsa: pipeline stage layer range mismatch", __FILE__, __LINE__);
    }
    const std::uint32_t hidden_size = static_cast<std::uint32_t>(config.hidden_size);
    const std::uint32_t q_heads = static_cast<std::uint32_t>(config.num_attention_heads);
    const std::uint32_t kv_heads = static_cast<std::uint32_t>(config.num_key_value_heads);
    const std::uint32_t head_dim = static_cast<std::uint32_t>(config.attention_head_dim);
    const std::uint32_t global_intermediate =
        static_cast<std::uint32_t>(config.ffn.intermediate_size);
    const std::uint32_t key_heads = static_cast<std::uint32_t>(config.linear_num_key_heads);
    const std::uint32_t value_heads =
        static_cast<std::uint32_t>(config.linear_num_value_heads);
    const std::uint32_t key_head_dim =
        static_cast<std::uint32_t>(config.linear_key_head_dim);
    const std::uint32_t value_head_dim =
        static_cast<std::uint32_t>(config.linear_value_head_dim);
    const std::uint32_t conv_kernel =
        static_cast<std::uint32_t>(config.linear_conv_kernel_dim);

    std::uint32_t intermediate = global_intermediate;
    std::uint32_t local_q_heads = q_heads;
    std::uint32_t local_kv_heads = kv_heads;
    std::uint32_t local_key_heads = key_heads;
    std::uint32_t local_value_heads = value_heads;
    if (tp_active) {
        intermediate = backbone_options.tp->local_intermediate_size;
        local_q_heads = backbone_options.tp->local_attention_heads;
        local_kv_heads = backbone_options.tp->local_key_value_heads;
        local_key_heads = backbone_options.tp->local_gdn_key_heads;
        local_value_heads = backbone_options.tp->local_gdn_value_heads;
    }

    float rope_theta = config.rope_theta;
    float eps = config.rms_norm_eps;
    float partial = config.partial_rotary_factor;
    if (rope_theta == 0) rope_theta = 10000000.0f;
    if (eps == 0) eps = 1e-6f;
    if (partial == 0) partial = 0.25f;
    const std::uint32_t rotary_dim =
        static_cast<std::uint32_t>((double)head_dim * (double)partial);
    if (config.indexer.rope_dim > rotary_dim) {
        return Status::unsupported(
            "qwen35qsa: indexer rope_dim exceeds the backbone rope width", __FILE__,
            __LINE__);
    }

    QH::GdnGeometry gdn_geometry;
    gdn_geometry.key_heads = local_key_heads;
    gdn_geometry.value_heads = local_value_heads;
    gdn_geometry.key_head_dim = key_head_dim;
    gdn_geometry.value_head_dim = value_head_dim;
    gdn_geometry.conv_kernel = conv_kernel;
    QH::GdnFeatureSizes gdn_features;
    Status gdn_st = QH::gdn_feature_sizes(gdn_geometry, gdn_features);
    if (!gdn_st.ok()) return gdn_st;

    QH::GdnLoweringGeometry gdn_lowering_geometry;
    gdn_lowering_geometry.key_heads = local_key_heads;
    gdn_lowering_geometry.value_heads = local_value_heads;
    gdn_lowering_geometry.key_head_dim = key_head_dim;
    gdn_lowering_geometry.value_head_dim = value_head_dim;
    gdn_lowering_geometry.conv_kernel = conv_kernel;
    gdn_lowering_geometry.eps = eps;
    gdn_lowering_geometry.out_dtype = backbone_options.tp_combine_f32
        ? RT::ValueDType::F32
        : RT::ValueDType::BF16;
    gdn_lowering_geometry.tp_combine = tp_active;

    if (weights.indexer.size() != weights.backbone.layers.size()) {
        return Status::invalid_argument(
            "qwen35qsa: indexer count does not match the layer count", __FILE__,
            __LINE__);
    }
    for (std::uint32_t li = layer_lo; li < layer_hi; ++li) {
        const auto& lw = weights.backbone.layers[li];
        Status st = validate_dense_ffn(lw.ffn, hidden_size, intermediate);
        if (!st.ok()) return st;
        if (lw.is_gdn) {
            st = validate_gdn_layer(lw, config, hidden_size, gdn_geometry,
                                    gdn_features);
        } else {
            st = validate_attention_layer(lw, config, hidden_size, local_q_heads,
                                          local_kv_heads, head_dim);
            if (st.ok()) st = validate_indexer_layer(weights.indexer[li], config, hidden_size);
        }
        if (!st.ok()) return st;
    }

    // A context that fits in the indexer budget keeps every block, so selection cannot
    // prune anything and the layer runs plain attention: the graph then emits no indexer
    // and the runtime allocates no QSA state.
    const bool qsa_active = options.max_context_tokens > config.indexer.budget;

    QsaGeometry geometry;
    geometry.heads = static_cast<std::uint32_t>(config.indexer.heads);
    geometry.head_dim = static_cast<std::uint32_t>(config.indexer.head_dim);
    geometry.rotary_dim = static_cast<std::uint32_t>(config.indexer.rope_dim);
    geometry.compress_ratio = static_cast<std::uint32_t>(config.indexer.block_size);
    geometry.blocks =
        (options.max_context_tokens + geometry.compress_ratio - 1u) / geometry.compress_ratio;
    geometry.top_k = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(config.indexer.kept_blocks(), geometry.blocks));
    geometry.mask_bytes = ((geometry.blocks + 31u) / 32u) * 4u;

    shelf::qwen35::Qwen35LoweredPrimitives lowered;
    RT::PrimitiveGraph& pg = lowered.graph;

    std::vector<const MW*> weight_pointers;
    auto add_weight = [&](const MW& w) -> std::uint32_t {
        weight_pointers.push_back(&w);
        lowered.weights.push_back(make_weight_slot(w));
        pg.weight_table.push_back(RT::MatrixwiseShapeKey{w.rows, w.cols});
        return static_cast<std::uint32_t>(weight_pointers.size() - 1);
    };
    auto reg_param = [&](const gpu::Tensor& t) -> std::uint32_t {
        RT::StaticParameterSlot s{};
        s.device_ptr = t.data<void>();
        s.elements = static_cast<std::uint32_t>(tensor_elements(t));
        s.dtype = t.element_size() == 2 ? RT::ValueDType::BF16 : RT::ValueDType::F32;
        lowered.parameters.push_back(s);
        return static_cast<std::uint32_t>(lowered.parameters.size() - 1);
    };

    const std::uint32_t embed_index =
        first_stage ? add_weight(weights.backbone.embed_tokens) : UINT32_MAX;
    const bool lm_tied = weights.backbone.lm_head_tied;
    const std::uint32_t lm_index =
        !last_stage ? UINT32_MAX
                    : (lm_tied ? embed_index : add_weight(weights.backbone.lm_head));
    std::vector<LayerWeightIndices> layer_weights(weights.backbone.layers.size());
    for (std::uint32_t li = layer_lo; li < layer_hi; ++li) {
        const auto& lw = weights.backbone.layers[li];
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
            if (qsa_active) {
                idx.indexer_q = add_weight(weights.indexer[li].q_proj);
                idx.indexer_k = add_weight(weights.indexer[li].k_proj);
            }
        }
        idx.mlp_gate = add_weight(lw.ffn.gate);
        idx.mlp_up = add_weight(lw.ffn.up);
        idx.mlp_down = add_weight(lw.ffn.down);
    }

    std::vector<LayerParamIndices> layer_params(weights.backbone.layers.size());
    for (std::uint32_t li = layer_lo; li < layer_hi; ++li) {
        const auto& lw = weights.backbone.layers[li];
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
            if (qsa_active) {
                lp.indexer_q_norm = reg_param(weights.indexer[li].q_norm_weight);
                lp.indexer_k_norm = reg_param(weights.indexer[li].k_norm_weight);
                lp.indexer_k_block_norm = reg_param(weights.indexer[li].k_block_norm_weight);
            }
        }
    }
    const std::uint32_t final_norm_param =
        weights.backbone.final_norm_weight.ndim() > 0
            ? reg_param(weights.backbone.final_norm_weight)
            : UINT32_MAX;

    std::uint32_t next_gdn_state_index = 0;
    std::uint32_t next_attention_state_index = 0;

    Emitter em{pg};
    const RT::RowwiseShapeKey hidden_key{hidden_size};

    RT::ValueId layer_input = em.value(hidden_size);
    if (first_stage) {
        RT::ValueId tokens = em.value(0, RT::ValueDType::I32);
        RT::EmbeddingLookupNode n;
        n.vocab_size = static_cast<std::uint32_t>(config.vocab_size);
        n.hidden_size = hidden_size;
        n.weight_shape = RT::MatrixwiseShapeKey{weights.backbone.embed_tokens.rows,
                                                weights.backbone.embed_tokens.cols};
        n.weight_index = embed_index;
        em.emit(RT::PrimitiveNode{n}, {tokens}, {layer_input});
        pg.nodes.back().debug_name = "embedding";
        if (weights.backbone.embed_tokens.partition.has_value())
            pg.nodes.back().tp_combine = 1;
        pg.external_inputs.push_back(tokens);
    } else {
        pg.external_inputs.push_back(layer_input);
    }

    std::vector<RT::ValueId> hidden_tap_values(backbone_options.hidden_taps.size());
    std::vector<std::uint8_t> hidden_tap_found(backbone_options.hidden_taps.size(), 0);

    for (std::uint32_t li = layer_lo; li < layer_hi; ++li) {
        const auto& lw = weights.backbone.layers[li];
        const auto& lw_idx = layer_weights[li];
        const auto& lp = layer_params[li];
        const std::string layer_prefix = "L" + std::to_string(li) + ".";
        const auto name_last = [&](const char* suffix) {
            pg.nodes.back().debug_name = layer_prefix + suffix;
        };
        for (std::size_t o = 0; o < backbone_options.override_layers.size(); ++o) {
            if (static_cast<std::size_t>(backbone_options.override_layers[o]) == li) {
                lowered.layer_input_values.emplace_back(li, layer_input);
            }
        }

        QH::HybridBlockLowering block;
        block.is_gdn = lw.is_gdn;
        block.eps = eps;
        block.weight_input_norm = lp.input_norm;
        block.weight_post_norm = lp.post_norm;
        if (lw.is_gdn) {
            block.gdn_geometry = gdn_lowering_geometry;
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
            block.attention_geometry.query_heads = local_q_heads;
            block.attention_geometry.kv_heads = local_kv_heads;
            block.attention_geometry.head_dim = head_dim;
            block.attention_geometry.rotary_dim = rotary_dim;
            block.attention_geometry.rope_theta = rope_theta;
            block.attention_geometry.rope_attention_factor =
                static_cast<float>(backbone_options.rope_profile.attention_factor);
            block.attention_geometry.eps = eps;
            block.attention_geometry.out_dtype = backbone_options.tp_combine_f32
                ? RT::ValueDType::F32
                : RT::ValueDType::BF16;
            block.attention_geometry.tp_combine = tp_active;
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

        const RT::ValueId normed = QH::emit_hybrid_block_input_norm(
            em, hidden_key, block, layer_input, layer_prefix);
        bool has_mask = false;
        RT::ValueId mask{};
        if (!lw.is_gdn && qsa_active) {
            QsaIndexerLowering indexer;
            indexer.hidden = hidden_size;
            indexer.heads = geometry.heads;
            indexer.head_dim = geometry.head_dim;
            indexer.rotary_dim = geometry.rotary_dim;
            indexer.compress_ratio = geometry.compress_ratio;
            indexer.blocks = geometry.blocks;
            indexer.top_k = geometry.top_k;
            indexer.mask_bytes = geometry.mask_bytes;
            indexer.theta = rope_theta;
            indexer.eps = eps;
            indexer.state_index = block.attention_indices.state_index;
            indexer.weights = &weights.indexer[li];
            indexer.w_q_proj = lw_idx.indexer_q;
            indexer.w_k_proj = lw_idx.indexer_k;
            indexer.p_q_norm = lp.indexer_q_norm;
            indexer.p_k_norm = lp.indexer_k_norm;
            indexer.p_k_block_norm = lp.indexer_k_block_norm;
            mask = emit_qsa_block_mask(em, indexer, normed, layer_prefix);
            has_mask = true;
        }
        const QH::HybridBlockOutput blk = QH::emit_hybrid_block_attention(
            em, hidden_key, block, layer_input, normed, layer_prefix, has_mask, mask,
            geometry.compress_ratio);

        const RT::ValueId post_norm = blk.post_norm;
        const MW& w_gate = *weight_pointers[lw_idx.mlp_gate];
        const MW& w_up = *weight_pointers[lw_idx.mlp_up];
        const MW& w_down = *weight_pointers[lw_idx.mlp_down];

        const RT::ValueId v_mlp_gate = em.linear(w_gate, lw_idx.mlp_gate, post_norm);
        name_last("mlp_gate");
        const RT::ValueId v_mlp_up = em.linear(w_up, lw_idx.mlp_up, post_norm);
        name_last("mlp_up");
        const RT::ValueId v_swiglu = em.value(intermediate);
        {
            RT::SwiGluNode sn{RT::RowwiseShapeKey{intermediate}};
            em.emit(RT::PrimitiveNode{sn}, {v_mlp_gate, v_mlp_up}, {v_swiglu});
            name_last("mlp_swiglu");
        }
        const RT::ValueId v_down =
            em.linear(w_down, lw_idx.mlp_down, v_swiglu,
                      backbone_options.tp_combine_f32 ? RT::ValueDType::F32
                                                      : RT::ValueDType::BF16);
        name_last("mlp_down");
        if (tp_active) pg.nodes.back().tp_combine = 1;

        const RT::ValueId layer_output = em.value(hidden_size);
        {
            RT::ResidualAddNode ra{hidden_key};
            em.emit(RT::PrimitiveNode{ra}, {v_down, blk.attention_residual}, {layer_output});
            name_last("output");
        }
        layer_input = layer_output;
        for (std::size_t t = 0; t < backbone_options.hidden_taps.size(); ++t) {
            if (static_cast<std::size_t>(backbone_options.hidden_taps[t]) == li) {
                hidden_tap_values[t] = layer_output;
                hidden_tap_found[t] = 1;
            }
        }
    }

    if (last_stage) {
        const MW* lmw = lm_tied ? &weights.backbone.embed_tokens : &weights.backbone.lm_head;
        const std::uint64_t global_vocab = config.vocab_size;
        bool vocab_parallel = false;
        std::uint32_t lm_head_col_offset = 0;
        const bool shardable_encoding =
            lmw->encoding == shelf::weights::MatrixEncoding::Bf16 ||
            lmw->encoding == shelf::weights::MatrixEncoding::Int8W32 ||
            lmw->encoding == shelf::weights::MatrixEncoding::Exl3;
        if (!lm_tied && backbone_options.tp != nullptr &&
            backbone_options.tp->vocab_parallel_lm_head && shardable_encoding &&
            lmw->partition.has_value()) {
            const auto& part = lmw->partition.value();
            vocab_parallel = part.axis == 0 && part.count == backbone_options.tp->tp_size &&
                part.index == backbone_options.tp->tp_rank &&
                lmw->rows == backbone_options.tp->local_vocab_size &&
                static_cast<std::uint64_t>(lmw->rows) * part.count == global_vocab;
            if (vocab_parallel) {
                lm_head_col_offset = static_cast<std::uint32_t>(part.index) * lmw->rows;
            }
        }
        if (!vocab_parallel && lmw->rows != global_vocab) {
            return Status::unsupported(
                "qwen35qsa: vocab sharded lm_head requires vocab parallel lowering",
                __FILE__, __LINE__);
        }
        const std::uint32_t out_vocab =
            vocab_parallel ? static_cast<std::uint32_t>(global_vocab)
                           : static_cast<std::uint32_t>(lmw->rows);
        QH::HybridHeadLowering head;
        head.hidden_features = hidden_size;
        head.eps = eps;
        head.final_norm_param = final_norm_param;
        head.lm_shape = RT::MatrixwiseShapeKey{lmw->rows, lmw->cols};
        head.lm_weight_index = lm_index;
        head.out_vocab = out_vocab;
        const QH::HybridHeadOutput head_out =
            QH::emit_hybrid_head(em, head, layer_input);
        if (vocab_parallel) {
            pg.nodes[head_out.logits_node_index].tp_combine = 1;
            pg.nodes[head_out.logits_node_index].tp_combine_local_cols = lmw->rows;
            lowered.weights[lm_index].output_col_offset = lm_head_col_offset;
            lowered.weights[lm_index].output_col_total = out_vocab;
        }
    } else {
        pg.external_outputs.push_back(layer_input);
    }

    for (std::size_t t = 0; t < hidden_tap_values.size(); ++t) {
        if (hidden_tap_found[t] == 0) {
            return Status::invalid_state(
                "hidden tap layer was not lowered", __FILE__, __LINE__);
        }
        pg.external_outputs.push_back(hidden_tap_values[t]);
    }
    if (backbone_options.emit_normed_token_hidden) {
        RT::RmsNormNode token_rms;
        token_rms.shape = hidden_key;
        token_rms.eps = eps;
        token_rms.group_size = hidden_size;
        token_rms.parameter_index = final_norm_param;
        token_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
        pg.external_outputs.push_back(
            em.unary(RT::PrimitiveNode{token_rms}, layer_input, hidden_size));
    }

    Status st = RT::validate_primitive_graph(pg);
    if (!st.ok()) return st;
    return lowered;
}

}
}
