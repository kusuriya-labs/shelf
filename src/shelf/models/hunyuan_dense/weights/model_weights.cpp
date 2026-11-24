#include <shelf/models/hunyuan_dense/weights/model_weights.h>

#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/exl3/exl3_int8.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace shelf::hunyuan_dense {
namespace {

using shelf::weights::SafetensorsCollection;

Status check_bf16_or_f16_vector(const SafetensorsCollection& collection,
                                const std::string& name,
                                std::initializer_list<std::size_t> shape) {
    auto spec = collection.tensor_spec(name);
    if (!spec.ok()) return spec.status();
    if ((spec.value().dtype != io::SType::BF16 && spec.value().dtype != io::SType::F16) ||
        spec.value().shape != std::vector<std::size_t>(shape)) {
        return Status::invalid_argument(
            ("hunyuan_v1_dense: invalid vector shape or dtype: " + name).c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status check_bf16_matrix(const SafetensorsCollection& collection,
                         const std::string& name,
                         std::initializer_list<std::size_t> shape) {
    auto spec = collection.tensor_spec(name);
    if (!spec.ok()) return spec.status();
    if (spec.value().dtype != io::SType::BF16 ||
        spec.value().shape != std::vector<std::size_t>(shape)) {
        return Status::invalid_argument(
            ("hunyuan_v1_dense: invalid BF16 matrix shape or dtype: " + name).c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

bool valid_lm_head_rows(std::size_t rows, std::size_t vocab_size) noexcept {
    if (rows == vocab_size) return true;
    return rows > vocab_size && (rows % 128u) == 0u && (rows - vocab_size) < 128u;
}

std::string layer_prefix(std::size_t layer) {
    return "model.layers." + std::to_string(layer) + ".";
}

Status load_vector(const SafetensorsCollection& collection, const std::string& name,
                   gpu::GpuArena& arena, hipStream_t stream, gpu::Tensor& dst) {
    auto result = shelf::weights::load_bf16_tensor(collection, name, arena, stream);
    if (!result.ok()) return result.status();
    dst = result.release();
    return Status::make_ok();
}

Status load_matrix(const SafetensorsCollection& collection, const std::string& name,
                   gpu::GpuArena& arena, hipStream_t stream,
                   const shelf::weights::WeightLoadOptions& opts,
                   shelf::weights::MatrixWeight& dst) {
    auto result = shelf::weights::load_bf16_matrix(collection, name, arena, stream, opts);
    if (!result.ok()) return result.status();
    dst = result.release();
    return Status::make_ok();
}

Result<bool> detect_standalone_quantized_lm_head(const std::string& dir) {
    if (!shelf::quantization::bundle::is_hf_quantized_model_dir(dir)) {
        return true;
    }
    auto coll = SafetensorsCollection::open(dir);
    if (!coll.ok()) return coll.status();
    const auto& c = coll.value();
    return c.tensor_spec("lm_head.trellis").ok() || c.tensor_spec("lm_head.weight").ok();
}

Result<ModelWeights> load_quantized_weights(const std::string& dir, const Config& cfg,
                                            gpu::GpuArena& arena, hipStream_t stream,
                                            const LoadOptions& options) {
    using namespace shelf::quantization::bundle;
    auto probe_head = detect_standalone_quantized_lm_head(dir);
    if (!probe_head.ok()) return probe_head.status();
    const bool has_standalone_head = probe_head.value();

    std::vector<std::string> names = {"model.embed_tokens.weight", "model.norm.weight"};
    if (has_standalone_head) {
        names.emplace_back("lm_head.weight");
    } else if (!cfg.tie_word_embeddings) {
        return Status::invalid_argument(
            "hunyuan_v1_dense: missing lm_head.weight in quantized checkpoint",
            __FILE__, __LINE__);
    }
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const auto p = layer_prefix(i);
        for (const char* n : {
                 "input_layernorm.weight",
                 "post_attention_layernorm.weight",
                 "self_attn.query_layernorm.weight",
                 "self_attn.key_layernorm.weight",
                 "self_attn.q_proj.weight",
                 "self_attn.k_proj.weight",
                 "self_attn.v_proj.weight",
                 "self_attn.o_proj.weight",
                 "mlp.gate_proj.weight",
                 "mlp.up_proj.weight",
                 "mlp.down_proj.weight",
             }) {
            names.push_back(p + n);
        }
    }

    auto rr = shelf::qwen_hybrid::open_quantized_model_reader(
        dir, names, options.verify_quantized_payload_crc);
    if (!rr.ok()) return rr.status();
    auto reader = rr.release();
    Status st = shelf::weights::validate_quantized_model(reader);
    if (!st.ok()) return st;

    const bool resolve_head = reader.resolve("lm_head.weight").ok();
    if (!resolve_head && !cfg.tie_word_embeddings) {
        return Status::invalid_argument(
            "hunyuan_v1_dense: missing lm_head.weight in quantized bundle",
            __FILE__, __LINE__);
    }

    auto check = [&](const std::string& name, std::initializer_list<int64_t> shape,
                     bool linear, bool allow_padded_vocab = false) -> Status {
        auto result = reader.resolve(name);
        if (!result.ok()) return result.status();
        const auto& v = result.value();
        const std::vector<int64_t> expected(shape);
        bool shape_ok = (v.logical_shape == expected);
        if (!shape_ok && allow_padded_vocab && v.logical_shape.size() == 2u &&
            expected.size() == 2u && v.logical_shape[1] == expected[1] &&
            v.logical_shape[0] >= expected[0]) {
            shape_ok = valid_lm_head_rows(
                static_cast<std::size_t>(v.logical_shape[0]),
                static_cast<std::size_t>(expected[0]));
        }
        if (!shape_ok ||
            (v.encoding != QuantizedEncoding::Bf16 &&
             (!linear || v.encoding != QuantizedEncoding::Exl3))) {
            return Status::invalid_argument(
                ("hunyuan_v1_dense: invalid quantized tensor shape or encoding: " + name).c_str(),
                __FILE__, __LINE__);
        }
        if (v.encoding == QuantizedEncoding::Exl3) {
            std::uint32_t multiplier = 0;
            if (v.marker.data == nullptr || v.marker.size != sizeof(multiplier)) {
                return Status::invalid_argument(
                    ("hunyuan_v1_dense: invalid EXL3 marker: " + name).c_str(),
                    __FILE__, __LINE__);
            }
            std::memcpy(&multiplier, v.marker.data, sizeof(multiplier));
            if (multiplier != shelf::quantization::exl3::kExl3Mul24Multiplier || v.exl3_half) {
                return Status::unsupported(
                    ("hunyuan_v1_dense: EXL3 requires integer-rate mul24: " + name).c_str(),
                    __FILE__, __LINE__);
            }
        }
        return Status::make_ok();
    };

    const auto h = static_cast<int64_t>(cfg.hidden_size);
    const auto q = static_cast<int64_t>(cfg.num_attention_heads * cfg.head_dim);
    const auto kv = static_cast<int64_t>(cfg.num_key_value_heads * cfg.head_dim);
    const auto f = static_cast<int64_t>(cfg.intermediate_size);
    const auto d = static_cast<int64_t>(cfg.head_dim);
    const auto vocab = static_cast<int64_t>(cfg.vocab_size);

    st = check("model.embed_tokens.weight", {vocab, h}, false);
    if (!st.ok()) return st;
    if (resolve_head) {
        st = check("lm_head.weight", {vocab, h}, true, true);
        if (!st.ok()) return st;
    }
    st = check("model.norm.weight", {h}, false);
    if (!st.ok()) return st;

    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const auto p = layer_prefix(i);
        for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight"}) {
            st = check(p + n, {h}, false);
            if (!st.ok()) return st;
        }
        for (const char* n : {"self_attn.query_layernorm.weight",
                              "self_attn.key_layernorm.weight"}) {
            st = check(p + n, {d}, false);
            if (!st.ok()) return st;
        }
        for (const auto& item : std::initializer_list<std::pair<const char*, int64_t>>{
                 {"self_attn.q_proj.weight", q},
                 {"self_attn.k_proj.weight", kv},
                 {"self_attn.v_proj.weight", kv},
                 {"mlp.gate_proj.weight", f},
                 {"mlp.up_proj.weight", f}}) {
            st = check(p + item.first, {item.second, h}, true);
            if (!st.ok()) return st;
        }
        st = check(p + "self_attn.o_proj.weight", {h, q}, true);
        if (!st.ok()) return st;
        st = check(p + "mlp.down_proj.weight", {h, f}, true);
        if (!st.ok()) return st;
    }

    ModelWeights w;
    auto matrix = [&](const std::string& name, const shelf::weights::WeightLoadOptions& opts,
                      shelf::weights::MatrixWeight& dst) -> Status {
        auto r = shelf::weights::load_quantized_matrix(reader, name, arena, stream, opts);
        if (!r.ok()) return r.status();
        dst = r.release();
        return Status::make_ok();
    };
    auto vector = [&](const std::string& name, gpu::Tensor& dst) -> Status {
        auto r = shelf::weights::load_quantized_small(reader, name, arena, stream);
        if (!r.ok()) return r.status();
        dst = r.release();
        return Status::make_ok();
    };

    st = matrix("model.embed_tokens.weight", {}, w.embed_tokens);
    if (!st.ok()) return st;
    if (resolve_head) {
        shelf::weights::WeightLoadOptions head_opts;
        head_opts.quantize_int8_w32 = options.quantize_lm_head_int8_w32;
        st = matrix("lm_head.weight", head_opts, w.lm_head);
        if (!st.ok()) return st;
        w.lm_head_tied = false;
    } else if (options.quantize_lm_head_int8_w32) {
        shelf::weights::WeightLoadOptions head_opts;
        head_opts.quantize_int8_w32 = true;
        st = matrix("model.embed_tokens.weight", head_opts, w.lm_head);
        if (!st.ok()) return st;
        w.lm_head_tied = false;
    } else {
        w.lm_head = w.embed_tokens;
        w.lm_head_tied = true;
    }
    st = vector("model.norm.weight", w.final_norm);
    if (!st.ok()) return st;

    w.layers.resize(cfg.num_hidden_layers);
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& l = w.layers[i];
        const auto p = layer_prefix(i);
        for (const auto& item : std::initializer_list<std::pair<const char*, gpu::Tensor*>>{
                 {"input_layernorm.weight", &l.input_layernorm},
                 {"post_attention_layernorm.weight", &l.post_attention_layernorm},
                 {"self_attn.query_layernorm.weight", &l.query_layernorm},
                 {"self_attn.key_layernorm.weight", &l.key_layernorm}}) {
            st = vector(p + item.first, *item.second);
            if (!st.ok()) return st;
        }
        for (const auto& item :
             std::initializer_list<std::pair<const char*, shelf::weights::MatrixWeight*>>{
                 {"self_attn.q_proj.weight", &l.q_proj},
                 {"self_attn.k_proj.weight", &l.k_proj},
                 {"self_attn.v_proj.weight", &l.v_proj},
                 {"self_attn.o_proj.weight", &l.o_proj},
                 {"mlp.gate_proj.weight", &l.gate_proj},
                 {"mlp.up_proj.weight", &l.up_proj},
                 {"mlp.down_proj.weight", &l.down_proj}}) {
            st = matrix(p + item.first, {}, *item.second);
            if (!st.ok()) return st;
        }
    }

    const hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error(
            "hipStreamSynchronize after quantized hunyuan_v1_dense load",
            hipGetErrorString(sync_err), __FILE__, __LINE__);
    }
    return w;
}

}

Status validate_checkpoint(const SafetensorsCollection& c, const Config& cfg) {
    const auto h = cfg.hidden_size;
    const auto q = cfg.num_attention_heads * cfg.head_dim;
    const auto kv = cfg.num_key_value_heads * cfg.head_dim;
    const auto f = cfg.intermediate_size;

    Status st = check_bf16_matrix(c, "model.embed_tokens.weight", {cfg.vocab_size, h});
    if (!st.ok()) return st;
    if (c.tensor_spec("lm_head.weight").ok()) {
        auto spec = c.tensor_spec("lm_head.weight");
        if (!spec.ok()) return spec.status();
        const auto& s = spec.value();
        if (s.dtype != io::SType::BF16 || s.shape.size() != 2u || s.shape[1] != h ||
            !valid_lm_head_rows(s.shape[0], cfg.vocab_size)) {
            return Status::invalid_argument(
                "hunyuan_v1_dense: invalid lm_head.weight shape or dtype",
                __FILE__, __LINE__);
        }
    } else if (!cfg.tie_word_embeddings) {
        return Status::invalid_argument(
            "hunyuan_v1_dense: missing lm_head.weight", __FILE__, __LINE__);
    }
    st = check_bf16_or_f16_vector(c, "model.norm.weight", {h});
    if (!st.ok()) return st;

    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const auto p = layer_prefix(i);
        for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight"}) {
            st = check_bf16_or_f16_vector(c, p + n, {h});
            if (!st.ok()) return st;
        }
        for (const char* n : {"self_attn.query_layernorm.weight",
                              "self_attn.key_layernorm.weight"}) {
            st = check_bf16_or_f16_vector(c, p + n, {cfg.head_dim});
            if (!st.ok()) return st;
        }
        st = check_bf16_matrix(c, p + "self_attn.q_proj.weight", {q, h});
        if (!st.ok()) return st;
        st = check_bf16_matrix(c, p + "self_attn.k_proj.weight", {kv, h});
        if (!st.ok()) return st;
        st = check_bf16_matrix(c, p + "self_attn.v_proj.weight", {kv, h});
        if (!st.ok()) return st;
        st = check_bf16_matrix(c, p + "self_attn.o_proj.weight", {h, q});
        if (!st.ok()) return st;
        st = check_bf16_matrix(c, p + "mlp.gate_proj.weight", {f, h});
        if (!st.ok()) return st;
        st = check_bf16_matrix(c, p + "mlp.up_proj.weight", {f, h});
        if (!st.ok()) return st;
        st = check_bf16_matrix(c, p + "mlp.down_proj.weight", {h, f});
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

Result<ModelWeights> load_weights(const std::string& dir, const Config& cfg,
                                  gpu::GpuArena& arena, hipStream_t stream,
                                  const LoadOptions& options) {
    if (shelf::weights::is_quantized_model_dir(dir)) {
        return load_quantized_weights(dir, cfg, arena, stream, options);
    }
    auto collection = SafetensorsCollection::open(dir);
    if (!collection.ok()) return collection.status();
    const auto& c = collection.value();
    Status st = validate_checkpoint(c, cfg);
    if (!st.ok()) return st;

    ModelWeights w;
    st = load_matrix(c, "model.embed_tokens.weight", arena, stream, {}, w.embed_tokens);
    if (!st.ok()) return st;

    const bool has_standalone_head = c.tensor_spec("lm_head.weight").ok();
    if (has_standalone_head) {
        shelf::weights::WeightLoadOptions head_opts;
        head_opts.quantize_int8_w32 = options.quantize_lm_head_int8_w32;
        st = load_matrix(c, "lm_head.weight", arena, stream, head_opts, w.lm_head);
        if (!st.ok()) return st;
        w.lm_head_tied = false;
    } else if (options.quantize_lm_head_int8_w32) {
        shelf::weights::WeightLoadOptions head_opts;
        head_opts.quantize_int8_w32 = true;
        st = load_matrix(c, "model.embed_tokens.weight", arena, stream, head_opts, w.lm_head);
        if (!st.ok()) return st;
        w.lm_head_tied = false;
    } else {
        w.lm_head = w.embed_tokens;
        w.lm_head_tied = true;
    }
    st = load_vector(c, "model.norm.weight", arena, stream, w.final_norm);
    if (!st.ok()) return st;

    w.layers.resize(cfg.num_hidden_layers);
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& l = w.layers[i];
        const auto p = layer_prefix(i);
        st = load_vector(c, p + "input_layernorm.weight", arena, stream, l.input_layernorm);
        if (!st.ok()) return st;
        st = load_vector(c, p + "post_attention_layernorm.weight", arena, stream,
                         l.post_attention_layernorm);
        if (!st.ok()) return st;
        st = load_vector(c, p + "self_attn.query_layernorm.weight", arena, stream,
                         l.query_layernorm);
        if (!st.ok()) return st;
        st = load_vector(c, p + "self_attn.key_layernorm.weight", arena, stream,
                         l.key_layernorm);
        if (!st.ok()) return st;
        st = load_matrix(c, p + "self_attn.q_proj.weight", arena, stream, {}, l.q_proj);
        if (!st.ok()) return st;
        st = load_matrix(c, p + "self_attn.k_proj.weight", arena, stream, {}, l.k_proj);
        if (!st.ok()) return st;
        st = load_matrix(c, p + "self_attn.v_proj.weight", arena, stream, {}, l.v_proj);
        if (!st.ok()) return st;
        st = load_matrix(c, p + "self_attn.o_proj.weight", arena, stream, {}, l.o_proj);
        if (!st.ok()) return st;
        st = load_matrix(c, p + "mlp.gate_proj.weight", arena, stream, {}, l.gate_proj);
        if (!st.ok()) return st;
        st = load_matrix(c, p + "mlp.up_proj.weight", arena, stream, {}, l.up_proj);
        if (!st.ok()) return st;
        st = load_matrix(c, p + "mlp.down_proj.weight", arena, stream, {}, l.down_proj);
        if (!st.ok()) return st;
    }

    const hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error(
            "hipStreamSynchronize after BF16 hunyuan_v1_dense load",
            hipGetErrorString(sync_err), __FILE__, __LINE__);
    }
    return w;
}

}
