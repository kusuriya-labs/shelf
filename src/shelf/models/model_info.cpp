#include <shelf/models/model_info.h>

#include <shelf/models/loaded_model.h>
#include <shelf/runtime/gdn_state_layout.h>

namespace shelf::models {
namespace {

void add_matrix(const shelf::weights::MatrixWeight& matrix, DenseWeightBytes& out) {
    using Encoding = shelf::weights::MatrixEncoding;
    if (matrix.rows == 0) return;
    const std::uint64_t rows = matrix.rows;
    if (matrix.encoding == Encoding::Bf16) {
        out.bf16_bytes += rows * matrix.cols * 2u;
    } else if (matrix.encoding == Encoding::Fp8Finegrained) {
        out.fp8_code_bytes += rows * matrix.k_padded;
        out.fp8_scale_bytes +=
            ((rows + 127u) / 128u) * (matrix.k_padded / 128u) * 4u;
    }
}

}  // namespace

ModelInfo make_model_info(const LoadedModel& model) {
    ModelInfo info;
    const auto& text_config = model.hybrid_config();
    const auto gdn_layout = shelf::runtime::make_gdn_state_layout(text_config);
    const auto topology = shelf::qwen_hybrid::count_layer_topology(text_config);
    const ModelFamily family = model.family();

    info.family_name = family == ModelFamily::HunyuanDense
        ? "hunyuan_v1_dense"
        : (family == ModelFamily::Plamo3
               ? "plamo3"
               : (family == ModelFamily::Qsa
                      ? "qsa"
                      : (model.runs_grouped_experts() ? "moe" : "dense")));
    info.hidden_size = text_config.hidden_size;
    info.num_hidden_layers = text_config.num_hidden_layers;
    info.num_attention_heads = text_config.num_attention_heads;
    info.num_key_value_heads = text_config.num_key_value_heads;
    info.attention_head_dim = text_config.attention_head_dim;
    info.linear_num_key_heads = text_config.linear_num_key_heads;
    info.vocab_size = text_config.vocab_size;
    info.gdn_layers = gdn_layout.num_gdn_states;
    info.attention_layers = topology.attention_layers;
    info.gdn_num_value_heads = gdn_layout.num_v_heads;
    info.gdn_key_head_dim = gdn_layout.head_k;
    info.gdn_value_head_dim = gdn_layout.head_v;
    info.gdn_conv_dim = gdn_layout.conv_dim;
    info.gdn_conv_history = gdn_layout.conv_history;

    if (const auto* dense = model.dense_model()) {
        info.has_intermediate_size = true;
        info.intermediate_size = dense->text_config().ffn.intermediate_size;
    }
    if (const auto* qsa = model.qsa_model()) {
        info.has_intermediate_size = true;
        info.intermediate_size = qsa->text_config().ffn.intermediate_size;
    }
    if (const auto* moe = model.moe_model()) {
        info.has_experts = true;
        info.num_experts = moe->text_config().ffn.num_experts;
        info.num_experts_per_tok = moe->text_config().ffn.num_experts_per_tok;
        info.moe_intermediate_size = moe->text_config().ffn.moe_intermediate_size;
        info.shared_expert_intermediate_size =
            moe->text_config().ffn.shared_expert_intermediate_size;
    }
    return info;
}

bool has_dense_weight_view(const LoadedModel& model) noexcept {
    return model.dense_model() != nullptr;
}

DenseWeightBytes dense_weight_bytes(const LoadedModel& model) {
    DenseWeightBytes out;
    const auto* dense = model.dense_model();
    if (dense == nullptr) return out;
    const auto& weights = dense->weights();
    add_matrix(weights.embed_tokens, out);
    if (!weights.lm_head_tied) add_matrix(weights.lm_head, out);
    for (const auto& layer : weights.layers) {
        add_matrix(layer.ffn.gate, out);
        add_matrix(layer.ffn.up, out);
        add_matrix(layer.ffn.down, out);
        if (layer.is_gdn) {
            add_matrix(layer.attn_in_proj_a, out);
            add_matrix(layer.attn_in_proj_b, out);
            add_matrix(layer.attn_in_proj_qkv, out);
            add_matrix(layer.attn_in_proj_z, out);
            add_matrix(layer.attn_out_proj, out);
        } else {
            add_matrix(layer.attn_q_proj, out);
            add_matrix(layer.attn_k_proj, out);
            add_matrix(layer.attn_v_proj, out);
            add_matrix(layer.attn_o_proj, out);
        }
    }
    return out;
}

}  // namespace shelf::models
