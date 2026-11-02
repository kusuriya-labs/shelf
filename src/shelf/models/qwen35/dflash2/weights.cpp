#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/io/safetensors_reader.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <set>
#include <string>
#include <utility>

namespace {
constexpr const char* kDFlash2QuantizedArchitecture = "dflash2_draft";
}

namespace ps {
namespace qwen35 {
namespace dflash2 {

namespace {

std::string layer_prefix(std::size_t layer) {
    return "layers." + std::to_string(layer) + ".";
}

void push_tensor(
    std::vector<DFlash2ExpectedTensor>& out,
    std::string name,
    std::vector<std::size_t> shape) {
    out.push_back(DFlash2ExpectedTensor{std::move(name), std::move(shape)});
}

std::string shape_to_string(const std::vector<std::size_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

Status load_matrix(
    const ps::weights::SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const ps::weights::WeightLoadOptions& options,
    ps::weights::MatrixWeight& dst) {
    auto m = ps::weights::load_bf16_matrix(collection, name, arena, stream, options);
    if (!m.ok()) return m.status();
    dst = m.release();
    return Status::make_ok();
}

Status load_vector(
    const ps::weights::SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    gpu::Tensor& dst) {
    auto t = ps::weights::load_bf16_tensor(collection, name, arena, stream);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

struct WeightSource {
    bool quantized = false;
    const ps::weights::SafetensorsCollection* collection = nullptr;
    const ps::quantization::bundle::QuantizedModelReader* reader = nullptr;
    gpu::GpuArena* arena = nullptr;
    hipStream_t stream = nullptr;
    ps::weights::WeightLoadOptions options;
};

Status source_load_matrix(
    const WeightSource& src, const std::string& name, ps::weights::MatrixWeight& dst) {
    if (src.quantized) {
        auto m = ps::weights::load_quantized_matrix(*src.reader, name, *src.arena, src.stream,
                                                    src.options);
        if (!m.ok()) return m.status();
        dst = m.release();
        return Status::make_ok();
    }
    return load_matrix(*src.collection, name, *src.arena, src.stream, src.options, dst);
}

Status source_load_vector(const WeightSource& src, const std::string& name, gpu::Tensor& dst) {
    if (src.quantized) {
        auto t = ps::weights::load_quantized_small(*src.reader, name, *src.arena, src.stream);
        if (!t.ok()) return t.status();
        dst = t.release();
        return Status::make_ok();
    }
    return load_vector(*src.collection, name, *src.arena, src.stream, dst);
}

Status validate_quantized_contract(
    const ps::quantization::bundle::QuantizedModelReader& reader,
    const DFlash2Config& config) {
    using ps::quantization::manifest::QuantizedEncoding;

    Status model_st = ps::weights::validate_quantized_model(reader);
    if (!model_st.ok()) return model_st;

    const auto& manifest = reader.manifest();
    if (manifest.architecture != kDFlash2QuantizedArchitecture) {
        return Status::invalid_argument(
            ("DFlash2 quantized manifest architecture is '" + manifest.architecture +
             "', expected 'dflash2_draft'").c_str(),
            __FILE__, __LINE__);
    }

    auto names_result = reader.list_logical_tensors();
    if (!names_result.ok()) return names_result.status();
    const std::vector<std::string> names = names_result.release();
    const std::vector<DFlash2ExpectedTensor> expected = dflash2_expected_tensors(config);

    std::set<std::string> expected_set;
    for (const DFlash2ExpectedTensor& e : expected) expected_set.insert(e.name);
    std::string unexpected;
    for (const std::string& n : names) {
        if (expected_set.find(n) == expected_set.end()) {
            if (!unexpected.empty()) unexpected += ", ";
            unexpected += n;
        }
    }
    if (!unexpected.empty()) {
        return Status::invalid_argument(
            ("DFlash2 quantized bundle has unexpected tensors: " + unexpected).c_str(),
            __FILE__, __LINE__);
    }

    for (const DFlash2ExpectedTensor& e : expected) {
        auto view_result = reader.resolve(e.name);
        if (!view_result.ok()) {
            return Status::invalid_argument(
                ("DFlash2 quantized bundle is missing tensor: " + e.name).c_str(),
                __FILE__, __LINE__);
        }
        const auto& v = view_result.value();
        if (v.encoding != QuantizedEncoding::Bf16) {
            return Status::invalid_argument(
                ("DFlash2 quantized tensor '" + e.name + "' must be BF16").c_str(),
                __FILE__, __LINE__);
        }
        if (v.logical_shape.size() != e.shape.size()) {
            return Status::invalid_argument(
                ("DFlash2 quantized tensor '" + e.name + "' shape rank mismatch").c_str(),
                __FILE__, __LINE__);
        }
        for (std::size_t d = 0; d < e.shape.size(); ++d) {
            if (static_cast<std::size_t>(v.logical_shape[d]) != e.shape[d]) {
                return Status::invalid_argument(
                    ("DFlash2 quantized tensor '" + e.name + "' shape mismatch").c_str(),
                    __FILE__, __LINE__);
            }
        }
    }
    return Status::make_ok();
}

Status load_dflash2_weights_from(
    const WeightSource& src, const DFlash2Config& config, DFlash2Weights& weights) {
    weights.layers.resize(config.num_hidden_layers);

    Status st = source_load_vector(src, "hidden_norm.weight", weights.hidden_norm_weight);
    if (!st.ok()) return st;
    st = source_load_vector(src, "norm.weight", weights.final_norm_weight);
    if (!st.ok()) return st;
    st = source_load_matrix(src, "fc.weight", weights.fc);
    if (!st.ok()) return st;

    st = source_load_matrix(src, "candidate_selector.hidden_projection.weight",
                            weights.selector.hidden_projection);
    if (!st.ok()) return st;
    st = source_load_vector(src, "candidate_selector.predecessor_codebook",
                            weights.selector.predecessor_codebook);
    if (!st.ok()) return st;
    st = source_load_vector(src, "candidate_selector.successor_codebook",
                            weights.selector.successor_codebook);
    if (!st.ok()) return st;

    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string pre = layer_prefix(layer);
        DFlash2LayerWeights& lw = weights.layers[layer];

        st = source_load_vector(src, pre + "input_layernorm.weight",
                                lw.input_layernorm_weight);
        if (!st.ok()) return st;
        st = source_load_vector(src, pre + "post_attention_layernorm.weight",
                                lw.post_attention_layernorm_weight);
        if (!st.ok()) return st;

        st = source_load_matrix(src, pre + "self_attn.q_proj.weight", lw.attn_q_proj);
        if (!st.ok()) return st;
        st = source_load_matrix(src, pre + "self_attn.k_proj.weight", lw.attn_k_proj);
        if (!st.ok()) return st;
        st = source_load_matrix(src, pre + "self_attn.v_proj.weight", lw.attn_v_proj);
        if (!st.ok()) return st;
        st = source_load_matrix(src, pre + "self_attn.o_proj.weight", lw.attn_o_proj);
        if (!st.ok()) return st;
        st = source_load_vector(src, pre + "self_attn.q_norm.weight", lw.attn_q_norm_weight);
        if (!st.ok()) return st;
        st = source_load_vector(src, pre + "self_attn.k_norm.weight", lw.attn_k_norm_weight);
        if (!st.ok()) return st;

        st = source_load_matrix(src, pre + "mlp.gate_proj.weight", lw.mlp_gate_proj);
        if (!st.ok()) return st;
        st = source_load_matrix(src, pre + "mlp.up_proj.weight", lw.mlp_up_proj);
        if (!st.ok()) return st;
        st = source_load_matrix(src, pre + "mlp.down_proj.weight", lw.mlp_down_proj);
        if (!st.ok()) return st;

        st = source_load_vector(src, pre + "attention_conv.base_kernel",
                                lw.attention_conv_base_kernel);
        if (!st.ok()) return st;
        st = source_load_matrix(src, pre + "attention_conv.kernel_projection.weight",
                                lw.attention_conv_kernel_projection);
        if (!st.ok()) return st;
        st = source_load_vector(src, pre + "mlp_conv.base_kernel", lw.mlp_conv_base_kernel);
        if (!st.ok()) return st;
        st = source_load_matrix(src, pre + "mlp_conv.kernel_projection.weight",
                                lw.mlp_conv_kernel_projection);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

Status validate_quantized_loaded(DFlash2Weights& weights) {
    auto check_bf16 = [](const ps::weights::MatrixWeight& w,
                         const std::string& name) -> Status {
        if (w.encoding != ps::weights::MatrixEncoding::Bf16) {
            return Status::invalid_argument(
                ("DFlash2 backbone matrix not BF16: " + name).c_str(), __FILE__, __LINE__);
        }
        if (w.bf16.ndim() == 0) {
            return Status::invalid_argument(
                ("DFlash2 backbone matrix payload missing: " + name).c_str(), __FILE__, __LINE__);
        }
        return Status::make_ok();
    };
    Status st = check_bf16(weights.fc, "fc.weight");
    if (!st.ok()) return st;
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        DFlash2LayerWeights& lw = weights.layers[layer];
        ps::weights::MatrixWeight* matrices[] = {
            &lw.attn_q_proj, &lw.attn_k_proj, &lw.attn_v_proj, &lw.attn_o_proj,
            &lw.mlp_gate_proj, &lw.mlp_up_proj, &lw.mlp_down_proj,
            &lw.attention_conv_kernel_projection, &lw.mlp_conv_kernel_projection,
        };
        const char* names[] = {
            "self_attn.q_proj",        "self_attn.k_proj",
            "self_attn.v_proj",        "self_attn.o_proj",
            "mlp.gate_proj",           "mlp.up_proj",
            "mlp.down_proj",           "attention_conv.kernel_projection",
            "mlp_conv.kernel_projection",
        };
        for (std::size_t i = 0; i < 9; ++i) {
            st = check_bf16(*matrices[i], std::to_string(layer) + "." + names[i]);
            if (!st.ok()) return st;
        }
    }
    if (weights.selector.hidden_projection.encoding != ps::weights::MatrixEncoding::Bf16) {
        return Status::invalid_argument(
            "DFlash2 selector hidden_projection must remain BF16", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

std::vector<DFlash2ExpectedTensor> dflash2_expected_tensors(const DFlash2Config& config) {
    std::vector<DFlash2ExpectedTensor> out;
    out.reserve(6 + 15 * config.num_hidden_layers);

    const std::size_t h = config.hidden_size;
    const std::size_t q = config.attention_q_rows();
    const std::size_t kv = config.attention_kv_rows();
    const std::size_t i = config.intermediate_size;
    const std::size_t k = config.conv_kernel_size;
    const std::size_t p = config.conv_projection_rows();

    push_tensor(out, "fc.weight", {h, config.tap_feature_size()});
    push_tensor(out, "hidden_norm.weight", {h});
    push_tensor(out, "norm.weight", {h});
    push_tensor(out, "candidate_selector.hidden_projection.weight",
                {config.selector_rank, h});
    push_tensor(out, "candidate_selector.predecessor_codebook",
                {config.vocab_size, config.selector_rank});
    push_tensor(out, "candidate_selector.successor_codebook",
                {config.vocab_size, config.selector_rank});

    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string pre = layer_prefix(layer);
        push_tensor(out, pre + "input_layernorm.weight", {h});
        push_tensor(out, pre + "post_attention_layernorm.weight", {h});
        push_tensor(out, pre + "self_attn.q_proj.weight", {q, h});
        push_tensor(out, pre + "self_attn.k_proj.weight", {kv, h});
        push_tensor(out, pre + "self_attn.v_proj.weight", {kv, h});
        push_tensor(out, pre + "self_attn.o_proj.weight", {h, q});
        push_tensor(out, pre + "self_attn.q_norm.weight", {config.head_dim});
        push_tensor(out, pre + "self_attn.k_norm.weight", {config.head_dim});
        push_tensor(out, pre + "mlp.gate_proj.weight", {i, h});
        push_tensor(out, pre + "mlp.up_proj.weight", {i, h});
        push_tensor(out, pre + "mlp.down_proj.weight", {h, i});
        push_tensor(out, pre + "attention_conv.base_kernel", {k, k, h});
        push_tensor(out, pre + "attention_conv.kernel_projection.weight", {p, h});
        push_tensor(out, pre + "mlp_conv.base_kernel", {k, k, h});
        push_tensor(out, pre + "mlp_conv.kernel_projection.weight", {p, h});
    }

    return out;
}

Status validate_dflash2_tensor_contract(
    const std::string& model_dir,
    const DFlash2Config& config) {
    if (ps::weights::is_quantized_model_dir(model_dir)) {
        auto reader_result = ps::quantization::bundle::QuantizedModelReader::open(model_dir, false);
        if (!reader_result.ok()) return reader_result.status();
        ps::quantization::bundle::QuantizedModelReader reader = reader_result.release();
        return validate_quantized_contract(reader, config);
    }

    auto collection_result = ps::weights::SafetensorsCollection::open(model_dir);
    if (!collection_result.ok()) return collection_result.status();
    ps::weights::SafetensorsCollection collection = collection_result.release();

    auto names_result = collection.tensor_names();
    if (!names_result.ok()) return names_result.status();
    const std::vector<std::string>& names = names_result.value();

    const std::vector<DFlash2ExpectedTensor> expected = dflash2_expected_tensors(config);
    std::set<std::string> expected_names;
    for (const DFlash2ExpectedTensor& e : expected) expected_names.insert(e.name);
    const std::set<std::string> actual_names(names.begin(), names.end());

    std::string missing;
    for (const DFlash2ExpectedTensor& e : expected) {
        if (actual_names.find(e.name) == actual_names.end()) {
            if (!missing.empty()) missing += ", ";
            missing += e.name;
        }
    }
    if (!missing.empty()) {
        return Status::invalid_argument(
            ("DFlash2 checkpoint is missing tensors: " + missing).c_str(),
            __FILE__, __LINE__);
    }

    std::string unexpected;
    for (const std::string& n : names) {
        if (expected_names.find(n) == expected_names.end()) {
            if (!unexpected.empty()) unexpected += ", ";
            unexpected += n;
        }
    }
    if (!unexpected.empty()) {
        return Status::invalid_argument(
            ("DFlash2 checkpoint has unexpected tensors: " + unexpected).c_str(),
            __FILE__, __LINE__);
    }

    for (const DFlash2ExpectedTensor& e : expected) {
        auto spec_result = collection.tensor_spec(e.name);
        if (!spec_result.ok()) return spec_result.status();
        const ps::io::StTensorSpec& spec = spec_result.value();
        if (spec.dtype != ps::io::SType::BF16) {
            return Status::invalid_argument(
                ("DFlash2 tensor '" + e.name + "' must be BF16").c_str(),
                __FILE__, __LINE__);
        }
        if (spec.shape != e.shape) {
            return Status::invalid_argument(
                ("DFlash2 tensor '" + e.name + "' has shape " + shape_to_string(spec.shape) +
                 ", expected " + shape_to_string(e.shape)).c_str(),
                __FILE__, __LINE__);
        }
    }

    return Status::make_ok();
}

Result<DFlash2Weights> load_dflash2_weights(
    const std::string& model_dir,
    const DFlash2Config& config,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const ps::weights::WeightLoadOptions& options) {
    DFlash2Weights weights;

    if (ps::weights::is_quantized_model_dir(model_dir)) {
        auto reader_result =
            ps::quantization::bundle::QuantizedModelReader::open(model_dir, true);
        if (!reader_result.ok()) return reader_result.status();
        ps::quantization::bundle::QuantizedModelReader reader = reader_result.release();

        Status contract = validate_quantized_contract(reader, config);
        if (!contract.ok()) return contract;

        WeightSource src;
        src.quantized = true;
        src.reader = &reader;
        src.arena = &arena;
        src.stream = stream;
        src.options = options;

        Status st = load_dflash2_weights_from(src, config, weights);
        if (!st.ok()) return st;
        st = validate_quantized_loaded(weights);
        if (!st.ok()) return st;
    } else {
        Status contract = validate_dflash2_tensor_contract(model_dir, config);
        if (!contract.ok()) return contract;

        auto collection_result = ps::weights::SafetensorsCollection::open(model_dir);
        if (!collection_result.ok()) return collection_result.status();
        ps::weights::SafetensorsCollection collection = collection_result.release();

        WeightSource src;
        src.quantized = false;
        src.collection = &collection;
        src.arena = &arena;
        src.stream = stream;
        src.options = options;

        Status st = load_dflash2_weights_from(src, config, weights);
        if (!st.ok()) return st;
    }

    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error(
            "hipStreamSynchronize after DFlash2 weight load",
            hipGetErrorString(sync_err), __FILE__, __LINE__);
    }

    weights.present = true;
    return weights;
}

}  // namespace dflash2
}  // namespace qwen35
}  // namespace ps
