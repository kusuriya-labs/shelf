#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/models/qwen35/model/tensor_parallel_context.h>
#include <shelf/models/qwen35/weights/tensor_parallel_plan.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/weights/weight_loader.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace shelf {
namespace qwen35 {
namespace fs = std::filesystem;
namespace {

using shelf::weights::MatrixWeight;
using shelf::weights::SafetensorsCollection;
using Tensor = gpu::Tensor;

std::string layer_prefix(std::size_t layer) {
    return shelf::qwen_hybrid::layer_tensor_prefix(layer);
}

const std::vector<std::string>& mtp_required_tensors() {
    static const std::vector<std::string> names = {
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
    };
    return names;
}

Result<std::size_t> count_mtp_layers(const std::vector<std::string>& names) {
    std::vector<std::size_t> indices;
    const std::string prefix = "mtp.layers.";
    for (const auto& name : names) {
        if (name.rfind(prefix, 0) != 0) continue;
        std::size_t i = prefix.size();
        std::size_t j = i;
        while (j < name.size() && std::isdigit(static_cast<unsigned char>(name[j]))) ++j;
        if (j == i || j >= name.size() || name[j] != '.') continue;
        indices.push_back(std::stoull(name.substr(i, j - i)));
    }
    if (indices.empty()) return std::size_t{0};
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] != i) {
            std::string msg = "unsupported MTP layer index layout:";
            for (auto v : indices) msg += " " + std::to_string(v);
            return Status::unsupported(msg.c_str(), __FILE__, __LINE__);
        }
    }
    return indices.size();
}

Status check_mtp_required_tensors(const std::vector<std::string>& names) {
    const std::set<std::string> have(names.begin(), names.end());
    std::string missing;
    for (const auto& name : mtp_required_tensors()) {
        if (have.find(name) == have.end()) {
            if (!missing.empty()) missing += ", ";
            missing += name;
        }
    }
    if (!missing.empty()) {
        return Status::invalid_argument(
            ("missing required MTP tensors: " + missing).c_str(), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status check_mtp_layer_count(const std::vector<std::string>& names) {
    auto layer_count = count_mtp_layers(names);
    if (!layer_count.ok()) return layer_count.status();
    if (layer_count.value() != 1) {
        return Status::unsupported(
            ("checkpoint contains " + std::to_string(layer_count.value()) +
             " MTP layers, runtime currently supports 1").c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status load_matrix(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::weights::WeightLoadOptions& weight_options,
    MatrixWeight& dst)
{
    auto m = shelf::weights::load_bf16_matrix(collection, name, arena, stream, weight_options);
    if (!m.ok()) return m.status();
    dst = m.release();
    return Status::make_ok();
}

Status load_vector(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    Tensor& dst)
{
    auto t = shelf::weights::load_bf16_tensor(collection, name, arena, stream);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

Status load_norm_vector(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    Tensor& dst)
{
    auto t = shelf::weights::load_bf16_or_f32_tensor(collection, name, arena, stream);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

Status load_q_matrix(
    const shelf::quantization::bundle::QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::weights::WeightLoadOptions& weight_options,
    MatrixWeight& dst)
{
    auto m = shelf::weights::load_quantized_matrix(reader, name, arena, stream, weight_options);
    if (!m.ok()) return m.status();
    dst = m.release();
    return Status::make_ok();
}

Status load_q_vector(
    const shelf::quantization::bundle::QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    Tensor& dst,
    const shelf::weights::WeightLoadOptions& weight_options = {})
{
    auto t = shelf::weights::load_quantized_small(reader, name, arena, stream, weight_options);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

Status load_layer_bf16(
    gpu::GpuArena& arena, hipStream_t stream,
    const SafetensorsCollection& collection, std::size_t layer,
    const Qwen35LoadOptions& options,
    Qwen35LayerWeights& w)
{
    const std::string prefix = layer_prefix(layer);

    Status st = load_norm_vector(collection, prefix + "input_layernorm.weight", arena, stream,
                                 w.input_layernorm_weight);
    if (!st.ok()) return st;
    st = load_norm_vector(collection, prefix + "post_attention_layernorm.weight", arena, stream,
                          w.post_attention_layernorm_weight);
    if (!st.ok()) return st;
    st = load_matrix(collection, prefix + "mlp.gate_proj.weight", arena, stream, options.weights, w.ffn.gate);
    if (!st.ok()) return st;
    st = load_matrix(collection, prefix + "mlp.up_proj.weight", arena, stream, options.weights, w.ffn.up);
    if (!st.ok()) return st;
    st = load_matrix(collection, prefix + "mlp.down_proj.weight", arena, stream, options.weights, w.ffn.down);
    if (!st.ok()) return st;

    const bool is_gdn = !collection.tensor_spec(prefix + "self_attn.q_proj.weight").ok();
    w.is_gdn = is_gdn;

    if (is_gdn) {
        st = load_matrix(collection, prefix + "linear_attn.in_proj_a.weight", arena, stream, options.weights, w.attn_in_proj_a);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "linear_attn.in_proj_b.weight", arena, stream, options.weights, w.attn_in_proj_b);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "linear_attn.in_proj_qkv.weight", arena, stream, options.weights, w.attn_in_proj_qkv);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "linear_attn.in_proj_z.weight", arena, stream, options.weights, w.attn_in_proj_z);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "linear_attn.conv1d.weight", arena, stream, w.attn_conv1d_weight);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "linear_attn.out_proj.weight", arena, stream, options.weights, w.attn_out_proj);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "linear_attn.norm.weight", arena, stream, w.attn_norm_weight);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "linear_attn.dt_bias", arena, stream, w.attn_dt_bias);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "linear_attn.A_log", arena, stream, w.attn_A_log);
        if (!st.ok()) return st;
    } else {
        st = load_matrix(collection, prefix + "self_attn.q_proj.weight", arena, stream, options.weights, w.attn_q_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "self_attn.k_proj.weight", arena, stream, options.weights, w.attn_k_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "self_attn.v_proj.weight", arena, stream, options.weights, w.attn_v_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "self_attn.o_proj.weight", arena, stream, options.weights, w.attn_o_proj);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "self_attn.q_norm.weight", arena, stream, w.attn_q_norm_weight);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "self_attn.k_norm.weight", arena, stream, w.attn_k_norm_weight);
        if (!st.ok()) return st;
    }

    return Status::make_ok();
}

Status load_mtp_bf16(
    gpu::GpuArena& arena, hipStream_t stream,
    const SafetensorsCollection& collection,
    const Qwen35LoadOptions& options,
    Qwen35ModelWeights& weights)
{
    if (!collection.tensor_spec("mtp.fc.weight").ok()) {
        return Status::make_ok();
    }

    auto names = collection.tensor_names();
    if (!names.ok()) return names.status();
    Status mtp_contract = check_mtp_layer_count(names.value());
    if (!mtp_contract.ok()) return mtp_contract;
    mtp_contract = check_mtp_required_tensors(names.value());
    if (!mtp_contract.ok()) return mtp_contract;

    Qwen35MtpWeights& m = weights.mtp;
    m.present = true;
    m.layer.is_gdn = false;

    Status st = load_vector(collection, "mtp.pre_fc_norm_embedding.weight", arena, stream,
                            m.pre_fc_norm_embedding_weight);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.pre_fc_norm_hidden.weight", arena, stream, m.pre_fc_norm_hidden_weight);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.norm.weight", arena, stream, m.final_norm_weight);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.fc.weight", arena, stream, options.weights, m.fc);
    if (!st.ok()) return st;

    st = load_vector(collection, "mtp.layers.0.input_layernorm.weight", arena, stream,
                     m.layer.input_layernorm_weight);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.layers.0.post_attention_layernorm.weight", arena, stream,
                     m.layer.post_attention_layernorm_weight);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.mlp.gate_proj.weight", arena, stream, options.weights,
                     m.layer.ffn.gate);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.mlp.up_proj.weight", arena, stream, options.weights,
                     m.layer.ffn.up);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.mlp.down_proj.weight", arena, stream, options.weights,
                     m.layer.ffn.down);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.self_attn.q_proj.weight", arena, stream, options.weights,
                     m.layer.attn_q_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.self_attn.k_proj.weight", arena, stream, options.weights,
                     m.layer.attn_k_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.self_attn.v_proj.weight", arena, stream, options.weights,
                     m.layer.attn_v_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.self_attn.o_proj.weight", arena, stream, options.weights,
                     m.layer.attn_o_proj);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.layers.0.self_attn.q_norm.weight", arena, stream,
                     m.layer.attn_q_norm_weight);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.layers.0.self_attn.k_norm.weight", arena, stream,
                     m.layer.attn_k_norm_weight);
    if (!st.ok()) return st;

    return Status::make_ok();
}

Result<Qwen35ModelWeights> load_qwen35_weights_bf16(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    auto collection_result = SafetensorsCollection::open(model_dir);
    if (!collection_result.ok()) return collection_result.status();
    SafetensorsCollection collection = collection_result.release();

    auto config_result = read_qwen35_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    const Qwen35TextConfig config = config_result.release();

    Qwen35ModelWeights weights;
    weights.layers.resize(config.num_hidden_layers);

    shelf::weights::WeightLoadOptions embed_load;
    embed_load.preshuffle = false;

    {
        auto m = shelf::weights::load_bf16_matrix(
            collection, "model.language_model.embed_tokens.weight", arena, stream, embed_load);
        if (!m.ok()) return m.status();
        weights.embed_tokens = m.release();
    }

    if (config.tie_word_embeddings) {
        weights.lm_head = weights.embed_tokens;
        weights.lm_head_tied = true;
    } else {
        auto m = shelf::weights::load_bf16_matrix(
            collection, "model.language_model.lm_head.weight", arena, stream, embed_load);
        if (!m.ok()) return m.status();
        weights.lm_head = m.release();
        weights.lm_head_tied = false;
    }

    {
        auto t = shelf::weights::load_bf16_tensor(
            collection, "model.language_model.norm.weight", arena, stream);
        if (!t.ok()) return t.status();
        weights.final_norm_weight = t.release();
    }

    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        Status st = load_layer_bf16(arena, stream, collection, l, options, weights.layers[l]);
        if (!st.ok()) return st;
    }

    if (options.load_mtp_layers) {
        Status mtp = load_mtp_bf16(arena, stream, collection, options, weights);
        if (!mtp.ok()) return mtp;
    }

    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error(
            "hipStreamSynchronize after weight load",
            hipGetErrorString(sync_err), __FILE__, __LINE__);
    }

    return weights;
}

std::vector<std::string> dense_ffn_names(std::size_t layer) {
    const std::string prefix = layer_prefix(layer);
    return {prefix + "mlp.gate_proj.weight", prefix + "mlp.up_proj.weight",
            prefix + "mlp.down_proj.weight"};
}

Status load_dense_ffn_quantized(
    gpu::GpuArena& arena, hipStream_t stream,
    const shelf::quantization::bundle::QuantizedModelReader& reader, std::size_t layer,
    const shelf::weights::WeightLoadOptions& weight_options,
    Qwen35LayerWeights& w)
{
    const std::string prefix = layer_prefix(layer);
    Status st = load_q_matrix(reader, prefix + "mlp.gate_proj.weight", arena, stream, weight_options, w.ffn.gate);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, prefix + "mlp.up_proj.weight", arena, stream, weight_options, w.ffn.up);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, prefix + "mlp.down_proj.weight", arena, stream, weight_options, w.ffn.down);
    if (!st.ok()) return st;
    return Status::make_ok();
}

Status load_mtp_quantized(
    gpu::GpuArena& arena, hipStream_t stream,
    const shelf::quantization::bundle::QuantizedModelReader& reader,
    const shelf::weights::WeightLoadOptions& weight_options,
    Qwen35ModelWeights& weights)
{
    if (reader.manifest().tensors.count("mtp.fc.weight") == 0u) {
        return Status::make_ok();
    }

    auto names = reader.list_logical_tensors();
    if (!names.ok()) return names.status();
    Status mtp_contract = check_mtp_layer_count(names.value());
    if (!mtp_contract.ok()) return mtp_contract;
    mtp_contract = check_mtp_required_tensors(names.value());
    if (!mtp_contract.ok()) return mtp_contract;

    Qwen35MtpWeights& m = weights.mtp;
    m.present = true;
    m.layer.is_gdn = false;

    Status st = load_q_vector(reader, "mtp.pre_fc_norm_embedding.weight", arena, stream,
                              m.pre_fc_norm_embedding_weight);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.pre_fc_norm_hidden.weight", arena, stream, m.pre_fc_norm_hidden_weight);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.norm.weight", arena, stream, m.final_norm_weight);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.fc.weight", arena, stream, weight_options, m.fc);
    if (!st.ok()) return st;

    st = load_q_vector(reader, "mtp.layers.0.input_layernorm.weight", arena, stream,
                       m.layer.input_layernorm_weight);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.layers.0.post_attention_layernorm.weight", arena, stream,
                       m.layer.post_attention_layernorm_weight);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.mlp.gate_proj.weight", arena, stream, weight_options, m.layer.ffn.gate);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.mlp.up_proj.weight", arena, stream, weight_options, m.layer.ffn.up);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.mlp.down_proj.weight", arena, stream, weight_options, m.layer.ffn.down);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.self_attn.q_proj.weight", arena, stream, weight_options, m.layer.attn_q_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.self_attn.k_proj.weight", arena, stream, weight_options, m.layer.attn_k_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.self_attn.v_proj.weight", arena, stream, weight_options, m.layer.attn_v_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.self_attn.o_proj.weight", arena, stream, weight_options, m.layer.attn_o_proj);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.layers.0.self_attn.q_norm.weight", arena, stream,
                       m.layer.attn_q_norm_weight);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.layers.0.self_attn.k_norm.weight", arena, stream,
                       m.layer.attn_k_norm_weight);
    if (!st.ok()) return st;

    return Status::make_ok();
}

}
Result<Qwen35ModelWeights> load_qwen35_weights_from_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options) {

    if (options.weights.partition_plan != nullptr) {
        return Status::unsupported(
            "qwen35 single-GPU weight load path does not accept a tensor partition plan",
            __FILE__, __LINE__);
    }
    if (shelf::weights::is_quantized_model_dir(model_dir)) {
        return load_qwen35_weights_from_quantized_safetensors(model_dir, arena, stream, options);
    }
    return load_qwen35_weights_bf16(model_dir, arena, stream, options);
}

namespace {

Result<Qwen35ModelWeights> load_qwen35_weights_quantized_impl(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    auto config_result = read_qwen35_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    Qwen35TextConfig config = config_result.release();

    std::vector<std::string> requested = shelf::qwen_hybrid::hf_quantized_body_names(config);
    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        const std::vector<std::string> ffn = dense_ffn_names(l);
        requested.insert(requested.end(), ffn.begin(), ffn.end());
    }
    if (options.load_mtp_layers) {
        const auto& mtp_names = mtp_required_tensors();
        requested.insert(requested.end(), mtp_names.begin(), mtp_names.end());
    }
    auto reader_result = shelf::qwen_hybrid::open_quantized_model_reader(
        model_dir, requested, options.verify_quantized_payload_crc);
    if (!reader_result.ok()) return reader_result.status();
    shelf::quantization::bundle::QuantizedModelReader reader = reader_result.release();

    Status contract = shelf::weights::validate_quantized_model(reader);
    if (!contract.ok()) {
        return contract;
    }

    const auto& manifest = reader.manifest();
    if (manifest.architecture != "qwen35_dense") {
        return Status::unsupported(
            ("unsupported quantized model architecture: " + manifest.architecture).c_str(),
            __FILE__, __LINE__);
    }

    Qwen35ModelWeights weights;
    weights.layers.resize(config.num_hidden_layers);

    Status model_level = shelf::qwen_hybrid::load_hybrid_model_level_quantized(
        reader, config, arena, stream, options, weights);
    if (!model_level.ok()) return model_level;

    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        Status st = shelf::qwen_hybrid::load_hybrid_layer_body_quantized(
            arena, stream, reader, config, l, options, weights.layers[l]);
        if (!st.ok()) return st;
        st = load_dense_ffn_quantized(arena, stream, reader, l, options.weights, weights.layers[l]);
        if (!st.ok()) return st;
    }

    if (options.load_mtp_layers) {
        Status mtp = load_mtp_quantized(arena, stream, reader, options.weights, weights);
        if (!mtp.ok()) return mtp;
    }

    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error(
            "hipStreamSynchronize after quantized weight load",
            hipGetErrorString(sync_err), __FILE__, __LINE__);
    }

    return weights;
}

}  // namespace

Result<Qwen35ModelWeights> load_qwen35_weights_from_quantized_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    if (options.weights.partition_plan != nullptr) {
        return Status::unsupported(
            "qwen35 single-GPU weight load path does not accept a tensor partition plan",
            __FILE__, __LINE__);
    }
    return load_qwen35_weights_quantized_impl(model_dir, arena, stream, options);
}

namespace {

bool quantized_lm_head_is_bf16(const std::string& model_dir) {
    namespace fqb = shelf::quantization::bundle;
    namespace fqm = shelf::quantization::manifest;
    if (!fs::exists(fs::path(model_dir) / fqm::kQuantizedMetadataFile)) {
        auto source_result = fqb::synthesize_hf_quantized_source(
            model_dir, {"model.language_model.lm_head.weight"}, true);
        if (!source_result.ok()) return false;
        const auto& tensors = source_result.value().manifest.tensors;
        auto it = tensors.find("lm_head.weight");
        if (it == tensors.end()) it = tensors.find("model.language_model.lm_head.weight");
        return it != tensors.end() &&
               it->second.encoding == fqm::QuantizedEncoding::Bf16;
    }
    auto reader_result = fqb::QuantizedModelReader::open(model_dir, false);
    if (!reader_result.ok()) return false;
    auto alias_result =
        reader_result.value().resolve_alias("model.language_model.lm_head.weight");
    const std::string canonical =
        alias_result.ok() ? alias_result.value() : std::string("lm_head.weight");
    const auto& tensors = reader_result.value().manifest().tensors;
    auto it = tensors.find(canonical);
    if (it == tensors.end()) it = tensors.find("lm_head.weight");
    if (it == tensors.end()) return false;
    return it->second.encoding == fqm::QuantizedEncoding::Bf16;
}

}

Result<Qwen35ModelWeights> load_qwen35_weights_tensor_parallel_rank(
    const std::string& model_dir,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    if (options.weights.partition_plan != nullptr) {
        return Status::invalid_argument(
            "tensor parallel rank load builds its own partition plan",
            __FILE__, __LINE__);
    }
    if (!shelf::weights::is_quantized_model_dir(model_dir)) {
        return Status::unsupported(
            "tensor parallel rank load requires a Shelf quantized model directory",
            __FILE__, __LINE__);
    }
    auto config_result = read_qwen35_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    const Qwen35TextConfig& config = config_result.value();

    auto context_result = make_qwen35_tensor_parallel_context(
        config, tp_size, tp_rank, options.vocab_parallel_lm_head);
    if (!context_result.ok()) return context_result.status();

    const bool shard_lm_head =
        context_result.value().vocab_parallel_lm_head &&
        quantized_lm_head_is_bf16(model_dir);

    auto plan_result =
        build_qwen35_tensor_partition_plan(config, tp_size, tp_rank, shard_lm_head);
    if (!plan_result.ok()) return plan_result.status();

    Qwen35LoadOptions tp_options = options;
    tp_options.weights.partition_plan = &plan_result.value();
    return load_qwen35_weights_quantized_impl(model_dir, arena, stream, tp_options);
}

}
}
