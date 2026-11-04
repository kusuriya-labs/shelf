#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/quantization/bundle/hf_manifest.h>

#include <cstdio>
#include <filesystem>

namespace shelf {
namespace qwen_hybrid {
namespace {

namespace fs = std::filesystem;
namespace qqb = shelf::quantization::bundle;
namespace qqm = shelf::quantization::manifest;

Status load_matrix(
    const qqb::QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::weights::WeightLoadOptions& weight_options,
    shelf::weights::MatrixWeight& dst)
{
    auto m = shelf::weights::load_quantized_matrix(reader, name, arena, stream, weight_options);
    if (!m.ok()) return m.status();
    dst = m.release();
    return Status::make_ok();
}

Status load_small(
    const qqb::QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    gpu::Tensor& dst,
    const shelf::weights::WeightLoadOptions& weight_options = {})
{
    auto t = shelf::weights::load_quantized_small(reader, name, arena, stream, weight_options);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

}  // namespace

std::string layer_tensor_prefix(std::size_t layer) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "model.language_model.layers.%zu.", layer);
    return std::string(buf);
}

std::vector<std::string> hf_quantized_body_names(const QwenHybridConfig& config) {
    std::vector<std::string> names = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.lm_head.weight",
        "model.language_model.norm.weight",
    };
    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        std::vector<std::string> layer_names =
            hf_quantized_layer_body_names(config, l);
        names.insert(names.end(), layer_names.begin(), layer_names.end());
    }
    return names;
}

std::vector<std::string> hf_quantized_layer_body_names(const QwenHybridConfig& config,
                                                       std::size_t layer) {
    std::vector<std::string> names;
    const std::string prefix = layer_tensor_prefix(layer);
    const bool gdn = layer_is_gdn(config, layer);
    names.push_back(prefix + "input_layernorm.weight");
    names.push_back(prefix + "post_attention_layernorm.weight");
    if (gdn) {
        names.push_back(prefix + "linear_attn.in_proj_a.weight");
        names.push_back(prefix + "linear_attn.in_proj_b.weight");
        names.push_back(prefix + "linear_attn.in_proj_qkv.weight");
        names.push_back(prefix + "linear_attn.in_proj_z.weight");
        names.push_back(prefix + "linear_attn.conv1d.weight");
        names.push_back(prefix + "linear_attn.out_proj.weight");
        names.push_back(prefix + "linear_attn.norm.weight");
        names.push_back(prefix + "linear_attn.dt_bias");
        names.push_back(prefix + "linear_attn.A_log");
    } else {
        names.push_back(prefix + "self_attn.q_proj.weight");
        names.push_back(prefix + "self_attn.k_proj.weight");
        names.push_back(prefix + "self_attn.v_proj.weight");
        names.push_back(prefix + "self_attn.o_proj.weight");
        names.push_back(prefix + "self_attn.q_norm.weight");
        names.push_back(prefix + "self_attn.k_norm.weight");
    }
    return names;
}

Result<qqb::QuantizedModelReader> open_quantized_model_reader(
    const std::string& model_dir,
    const std::vector<std::string>& logical_names,
    bool verify_payload_crc)
{
    const fs::path dir(model_dir);
    if (fs::exists(dir / qqm::kQuantizedMetadataFile)) {
        return qqb::QuantizedModelReader::open(model_dir, verify_payload_crc);
    }
    if (!qqb::is_hf_quantized_model_dir(model_dir)) {
        return Status::invalid_argument(
            "model dir is neither a Shelf quantized model dir nor an HF real-quantized dir",
            __FILE__, __LINE__);
    }
    auto source = qqb::synthesize_hf_quantized_source(model_dir, logical_names, verify_payload_crc);
    if (!source.ok()) return source.status();
    return qqb::QuantizedModelReader::open_hf_source(source.release(), verify_payload_crc);
}

Status load_hybrid_model_level_quantized(
    const qqb::QuantizedModelReader& reader,
    const QwenHybridConfig& config,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const HybridLoadOptions& options,
    HybridModelLevelWeights& weights)
{
    const bool need_embedding =
        options.load_embedding ||
        (options.load_lm_head && config.tie_word_embeddings);
    if (need_embedding) {
        auto embed = shelf::weights::load_quantized_matrix(
            reader, "model.language_model.embed_tokens.weight", arena, stream,
            options.weights);
        if (!embed.ok()) return embed.status();
        weights.embed_tokens = embed.release();
    }

    if (config.tie_word_embeddings) {
        weights.lm_head = weights.embed_tokens;
        weights.lm_head_tied = true;
    } else if (options.load_lm_head) {
        auto head = shelf::weights::load_quantized_matrix(
            reader, "model.language_model.lm_head.weight", arena, stream, options.weights);
        if (!head.ok()) {
            head = shelf::weights::load_quantized_matrix(
                reader, "lm_head.weight", arena, stream, options.weights);
            if (!head.ok()) return head.status();
        }
        weights.lm_head = head.release();
        weights.lm_head_tied = false;
    } else {
        weights.lm_head_tied = false;
    }

    if (options.load_final_norm) {
        auto norm = shelf::weights::load_quantized_small(
            reader, "model.language_model.norm.weight", arena, stream);
        if (!norm.ok()) return norm.status();
        weights.final_norm_weight = norm.release();
    }
    return Status::make_ok();
}

Status load_hybrid_layer_body_quantized(
    gpu::GpuArena& arena,
    hipStream_t stream,
    const qqb::QuantizedModelReader& reader,
    const QwenHybridConfig& config,
    std::size_t layer,
    const HybridLoadOptions& options,
    HybridLayerWeights& w)
{
    const std::string prefix = layer_tensor_prefix(layer);
    const bool is_gdn = layer_is_gdn(config, layer);
    w.is_gdn = is_gdn;

    Status st = load_small(reader, prefix + "input_layernorm.weight", arena, stream,
                           w.input_layernorm_weight);
    if (!st.ok()) return st;
    st = load_small(reader, prefix + "post_attention_layernorm.weight", arena, stream,
                    w.post_attention_layernorm_weight);
    if (!st.ok()) return st;

    if (is_gdn) {
        st = load_matrix(reader, prefix + "linear_attn.in_proj_a.weight", arena, stream,
                         options.weights, w.attn_in_proj_a);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "linear_attn.in_proj_b.weight", arena, stream,
                         options.weights, w.attn_in_proj_b);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "linear_attn.in_proj_qkv.weight", arena, stream,
                         options.weights, w.attn_in_proj_qkv);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "linear_attn.in_proj_z.weight", arena, stream,
                         options.weights, w.attn_in_proj_z);
        if (!st.ok()) return st;
        st = load_small(reader, prefix + "linear_attn.conv1d.weight", arena, stream,
                        w.attn_conv1d_weight, options.weights);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "linear_attn.out_proj.weight", arena, stream,
                         options.weights, w.attn_out_proj);
        if (!st.ok()) return st;
        st = load_small(reader, prefix + "linear_attn.norm.weight", arena, stream,
                        w.attn_norm_weight);
        if (!st.ok()) return st;
        st = load_small(reader, prefix + "linear_attn.dt_bias", arena, stream, w.attn_dt_bias,
                        options.weights);
        if (!st.ok()) return st;
        st = load_small(reader, prefix + "linear_attn.A_log", arena, stream, w.attn_A_log,
                        options.weights);
        if (!st.ok()) return st;
    } else {
        st = load_matrix(reader, prefix + "self_attn.q_proj.weight", arena, stream,
                         options.weights, w.attn_q_proj);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "self_attn.k_proj.weight", arena, stream,
                         options.weights, w.attn_k_proj);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "self_attn.v_proj.weight", arena, stream,
                         options.weights, w.attn_v_proj);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "self_attn.o_proj.weight", arena, stream,
                         options.weights, w.attn_o_proj);
        if (!st.ok()) return st;
        st = load_small(reader, prefix + "self_attn.q_norm.weight", arena, stream,
                        w.attn_q_norm_weight);
        if (!st.ok()) return st;
        st = load_small(reader, prefix + "self_attn.k_norm.weight", arena, stream,
                        w.attn_k_norm_weight);
        if (!st.ok()) return st;
    }

    return Status::make_ok();
}

}  // namespace qwen_hybrid
}  // namespace shelf
