#include <shelf/models/common/qwen_hybrid/weight_validation.h>

#include <string>

namespace shelf {
namespace qwen_hybrid {
namespace {

using shelf::weights::MatrixWeight;

std::size_t full_rows(const MatrixWeight& w) {
    if (w.partition && w.partition->axis == 0 && w.partition->count > 1u)
        return static_cast<std::size_t>(w.rows) * w.partition->count;
    return w.rows;
}

std::size_t full_cols(const MatrixWeight& w) {
    if (w.partition && w.partition->axis == 1 && w.partition->count > 1u)
        return static_cast<std::size_t>(w.cols) * w.partition->count;
    return w.cols;
}

// The tensor that carries a matrix's GPU bytes for its encoding. A tied LM head aliases
// its embedding, so this is the pointer the tie flag has to agree with.
const void* payload(const MatrixWeight& w) {
    if (w.encoding == shelf::weights::MatrixEncoding::Bf16) return w.bf16.data();
    if (w.codes.ndim() != 0) return w.codes.data();
    return w.data.data();
}

bool vector_of_width(const gpu::Tensor& t, std::size_t width) {
    return t.ndim() == 1 && static_cast<std::size_t>(t.dim(0)) == width;
}

bool loaded_vector(const gpu::Tensor& t) {
    return t.ndim() >= 1;
}

Status fail(const char* tag, const std::string& what) {
    return Status::invalid_argument((std::string(tag) + ": " + what).c_str(), __FILE__,
                                    __LINE__);
}

Status check_matrix(const MatrixWeight& w, std::size_t rows, std::size_t cols, const char* tag) {
    if (w.rows == 0 || w.cols == 0) return fail(tag, "weight is not loaded");
    if (rows != 0 && (full_rows(w) != rows || full_cols(w) != cols))
        return fail(tag, "shape is not [rows, cols] for the model");
    return Status::make_ok();
}

bool loaded_matrix(const MatrixWeight& w) {
    return w.rows != 0 && w.cols != 0;
}

}  // namespace

Status check_matrix_shape(const MatrixWeight& w, std::size_t rows, std::size_t cols,
                          const char* tag) {
    return check_matrix(w, rows, cols, tag);
}

Status validate_hybrid_model_level(const HybridModelLevelWeights& model,
                                   const QwenHybridConfig& config,
                                   std::size_t layer_count) {
    if (model.embed_tokens.rows == 0 || model.embed_tokens.cols == 0)
        return Status::invalid_argument("Qwen hybrid: embed_tokens is not loaded", __FILE__,
                                        __LINE__);
    if (model.lm_head.rows == 0 || model.lm_head.cols == 0)
        return Status::invalid_argument("Qwen hybrid: lm_head is not loaded", __FILE__, __LINE__);
    if (full_rows(model.embed_tokens) != config.vocab_size)
        return Status::invalid_argument("Qwen hybrid: embed_tokens vocab size mismatch",
                                        __FILE__, __LINE__);
    if (full_rows(model.lm_head) != config.vocab_size)
        return Status::invalid_argument("Qwen hybrid: lm_head vocab size mismatch", __FILE__,
                                        __LINE__);
    if (model.embed_tokens.cols != config.hidden_size)
        return Status::invalid_argument("Qwen hybrid: embed_tokens hidden size mismatch",
                                        __FILE__, __LINE__);
    if (!vector_of_width(model.final_norm_weight, config.hidden_size))
        return Status::invalid_argument("Qwen hybrid: final_norm_weight is not [hidden]",
                                        __FILE__, __LINE__);
    if (model.lm_head_tied != (payload(model.lm_head) == payload(model.embed_tokens)))
        return Status::invalid_argument("Qwen hybrid: lm_head_tied disagrees with lm_head storage",
                                        __FILE__, __LINE__);
    if (layer_count != config.num_hidden_layers)
        return Status::invalid_argument("Qwen hybrid: layer count mismatch", __FILE__, __LINE__);
    return Status::make_ok();
}

Status validate_hybrid_layer_body(const HybridLayerWeights& layer,
                                  std::size_t hidden,
                                  bool is_gdn,
                                  const char* tag) {
    if (!vector_of_width(layer.input_layernorm_weight, hidden))
        return fail(tag, "input_layernorm is not [hidden]");
    if (!vector_of_width(layer.post_attention_layernorm_weight, hidden))
        return fail(tag, "post_attention_layernorm is not [hidden]");

    if (is_gdn) {
        if (!loaded_matrix(layer.attn_in_proj_a) || !loaded_matrix(layer.attn_in_proj_b) ||
            !loaded_matrix(layer.attn_in_proj_qkv) || !loaded_matrix(layer.attn_in_proj_z))
            return fail(tag, "linear_attn in_proj is not loaded");
        if (!loaded_matrix(layer.attn_out_proj)) return fail(tag, "linear_attn.out_proj");
        if (!loaded_vector(layer.attn_conv1d_weight)) return fail(tag, "linear_attn.conv1d");
        if (!loaded_vector(layer.attn_norm_weight)) return fail(tag, "linear_attn.norm");
        if (!loaded_vector(layer.attn_dt_bias)) return fail(tag, "linear_attn.dt_bias");
        if (!loaded_vector(layer.attn_A_log)) return fail(tag, "linear_attn.A_log");
        if (layer.attn_q_proj.rows != 0)
            return fail(tag, "a gdn layer carries self_attn weights");
    } else {
        if (!loaded_matrix(layer.attn_q_proj) || !loaded_matrix(layer.attn_k_proj) ||
            !loaded_matrix(layer.attn_v_proj) || !loaded_matrix(layer.attn_o_proj))
            return fail(tag, "self_attn projection is not loaded");
        if (!loaded_vector(layer.attn_q_norm_weight)) return fail(tag, "self_attn.q_norm");
        if (!loaded_vector(layer.attn_k_norm_weight)) return fail(tag, "self_attn.k_norm");
        if (layer.attn_in_proj_a.rows != 0)
            return fail(tag, "a full attention layer carries linear_attn weights");
    }
    return Status::make_ok();
}

Status validate_hybrid_layer_body(const HybridLayerWeights& layer,
                                  const QwenHybridConfig& config,
                                  std::size_t layer_index) {
    const bool is_gdn = layer_is_gdn(config, layer_index);
    if (layer.is_gdn != is_gdn)
        return Status::invalid_argument("Qwen hybrid: unexpected layer topology", __FILE__,
                                        __LINE__);
    return validate_hybrid_layer_body(layer, config.hidden_size, is_gdn,
                                      "Qwen hybrid layer body");
}

}  // namespace qwen_hybrid
}  // namespace shelf
