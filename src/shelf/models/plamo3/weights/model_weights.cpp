#include <shelf/models/plamo3/weights/model_weights.h>

#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/quantization/exl3/exl3_int8.h>

#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

namespace shelf::plamo3 {
namespace {
using shelf::weights::SafetensorsCollection;

Status check_tensor(const SafetensorsCollection& collection, const std::string& name,
                    std::initializer_list<std::size_t> shape) {
    auto spec = collection.tensor_spec(name);
    if (!spec.ok()) return spec.status();
    if (spec.value().dtype != io::SType::BF16 ||
        spec.value().shape != std::vector<std::size_t>(shape)) {
        return Status::invalid_argument(("plamo3: invalid BF16 tensor shape or dtype: " + name).c_str(),
                                        __FILE__, __LINE__);
    }
    return Status::make_ok();
}

std::string prefix(std::size_t layer) {
    return "model.layers.layers." + std::to_string(layer) + ".";
}

Status load_vector(const SafetensorsCollection& c, const std::string& name,
                   gpu::GpuArena& arena, hipStream_t stream, gpu::Tensor& dst) {
    auto result = shelf::weights::load_bf16_tensor(c, name, arena, stream);
    if (!result.ok()) return result.status();
    dst = result.release();
    return Status::make_ok();
}
Status load_matrix(const SafetensorsCollection& c, const std::string& name,
                   gpu::GpuArena& arena, hipStream_t stream, shelf::weights::MatrixWeight& dst) {
    auto result = shelf::weights::load_bf16_matrix(c, name, arena, stream, {});
    if (!result.ok()) return result.status();
    dst = result.release();
    return Status::make_ok();
}
}  // namespace

Status validate_checkpoint(const SafetensorsCollection& c, const Config& cfg) {
    const auto h = cfg.hidden_size, q = cfg.num_attention_heads * cfg.head_dim;
    const auto kv = cfg.num_key_value_heads * cfg.head_dim, f = cfg.intermediate_size;
    auto check = [&](const std::string& name, std::initializer_list<std::size_t> shape) {
        return check_tensor(c, name, shape);
    };
    Status st = check("model.embed_tokens.weight", {cfg.vocab_size, h});
    if (!st.ok()) return st;
    st = check("lm_head.weight", {cfg.vocab_size, h}); if (!st.ok()) return st;
    st = check("model.norm.weight", {h}); if (!st.ok()) return st;
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const auto p = prefix(i);
        for (const char* n : {"pre_mixer_norm.weight", "post_mixer_norm.weight",
                              "pre_mlp_norm.weight", "post_mlp_norm.weight"}) {
            st = check(p + n, {h}); if (!st.ok()) return st;
        }
        for (const char* n : {"mixer.q_norm.weight", "mixer.k_norm.weight"}) {
            st = check(p + n, {cfg.head_dim}); if (!st.ok()) return st;
        }
        st = check(p + "mixer.qkv_proj.weight", {q + 2 * kv, h}); if (!st.ok()) return st;
        st = check(p + "mixer.o_proj.weight", {h, q}); if (!st.ok()) return st;
        st = check(p + "mlp.gate_up_proj.weight", {2 * f, h}); if (!st.ok()) return st;
        st = check(p + "mlp.down_proj.weight", {h, f}); if (!st.ok()) return st;
    }
    return Status::make_ok();
}

namespace {
Result<ModelWeights> load_quantized_weights(const std::string& dir, const Config& cfg,
                                            gpu::GpuArena& arena, hipStream_t stream) {
    using namespace shelf::quantization::bundle;
    std::vector<std::string> names = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"};
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const auto p = prefix(i);
        for (const char* n : {"pre_mixer_norm.weight", "post_mixer_norm.weight",
                              "pre_mlp_norm.weight", "post_mlp_norm.weight",
                              "mixer.q_norm.weight", "mixer.k_norm.weight",
                              "mixer.q_proj.weight", "mixer.k_proj.weight", "mixer.v_proj.weight",
                              "mixer.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                              "mlp.down_proj.weight"}) names.push_back(p + n);
    }
    auto rr = shelf::qwen_hybrid::open_quantized_model_reader(dir, names, false);
    if (!rr.ok()) return rr.status();
    auto reader = rr.release();
    auto st = shelf::weights::validate_quantized_model(reader);
    if (!st.ok()) return st;
    auto check = [&](const std::string& name, std::initializer_list<int64_t> shape, bool linear) -> Status {
        auto result = reader.resolve(name);
        if (!result.ok()) return result.status();
        const auto& v = result.value();
        if (v.logical_shape != std::vector<int64_t>(shape) ||
            (v.encoding != QuantizedEncoding::Bf16 && (!linear || v.encoding != QuantizedEncoding::Exl3)))
            return Status::invalid_argument(("plamo3: invalid quantized tensor shape or encoding: " + name).c_str(), __FILE__, __LINE__);
        if (v.encoding == QuantizedEncoding::Exl3) {
            uint32_t multiplier = 0;
            if (v.marker.data == nullptr || v.marker.size != sizeof(multiplier))
                return Status::invalid_argument(("plamo3: invalid EXL3 marker: " + name).c_str(), __FILE__, __LINE__);
            std::memcpy(&multiplier, v.marker.data, sizeof(multiplier));
            if (multiplier != shelf::quantization::exl3::kExl3Mul24Multiplier || v.exl3_half)
                return Status::unsupported(("plamo3: EXL3 requires integer-rate mul24: " + name).c_str(), __FILE__, __LINE__);
        }
        return Status::make_ok();
    };
    const auto h = static_cast<int64_t>(cfg.hidden_size), q = static_cast<int64_t>(cfg.num_attention_heads * cfg.head_dim);
    const auto kv = static_cast<int64_t>(cfg.num_key_value_heads * cfg.head_dim), f = static_cast<int64_t>(cfg.intermediate_size);
    const auto d = static_cast<int64_t>(cfg.head_dim), vocab = static_cast<int64_t>(cfg.vocab_size);
    st = check("model.embed_tokens.weight", {vocab, h}, false); if (!st.ok()) return st;
    st = check("lm_head.weight", {vocab, h}, true); if (!st.ok()) return st;
    st = check("model.norm.weight", {h}, false); if (!st.ok()) return st;
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const auto p = prefix(i);
        for (const char* n : {"pre_mixer_norm.weight", "post_mixer_norm.weight", "pre_mlp_norm.weight", "post_mlp_norm.weight"}) {
            st = check(p + n, {h}, false); if (!st.ok()) return st;
        }
        for (const char* n : {"mixer.q_norm.weight", "mixer.k_norm.weight"}) {
            st = check(p + n, {d}, false); if (!st.ok()) return st;
        }
        for (const auto& item : std::initializer_list<std::pair<const char*, int64_t>>{
                 {"mixer.q_proj.weight", q}, {"mixer.k_proj.weight", kv}, {"mixer.v_proj.weight", kv},
                 {"mlp.gate_proj.weight", f}, {"mlp.up_proj.weight", f}}) {
            st = check(p + item.first, {item.second, h}, true); if (!st.ok()) return st;
        }
        st = check(p + "mixer.o_proj.weight", {h, q}, true); if (!st.ok()) return st;
        st = check(p + "mlp.down_proj.weight", {h, f}, true); if (!st.ok()) return st;
    }
    ModelWeights w;
    auto matrix = [&](const std::string& name, shelf::weights::MatrixWeight& dst) -> Status {
        auto r = shelf::weights::load_quantized_matrix(reader, name, arena, stream, {});
        if (!r.ok()) return r.status(); dst = r.release(); return Status::make_ok();
    };
    auto vector = [&](const std::string& name, gpu::Tensor& dst) -> Status {
        auto r = shelf::weights::load_quantized_small(reader, name, arena, stream);
        if (!r.ok()) return r.status(); dst = r.release(); return Status::make_ok();
    };
    st = matrix("model.embed_tokens.weight", w.embed_tokens); if (!st.ok()) return st;
    st = matrix("lm_head.weight", w.lm_head); if (!st.ok()) return st;
    st = vector("model.norm.weight", w.final_norm); if (!st.ok()) return st;
    w.layers.resize(cfg.num_hidden_layers);
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& l = w.layers[i]; const auto p = prefix(i); l.split_projections = true;
        for (const auto& item : std::initializer_list<std::pair<const char*, gpu::Tensor*>>{
                 {"pre_mixer_norm.weight", &l.pre_mixer_norm}, {"post_mixer_norm.weight", &l.post_mixer_norm},
                 {"pre_mlp_norm.weight", &l.pre_mlp_norm}, {"post_mlp_norm.weight", &l.post_mlp_norm},
                 {"mixer.q_norm.weight", &l.q_norm}, {"mixer.k_norm.weight", &l.k_norm}}) {
            st = vector(p + item.first, *item.second); if (!st.ok()) return st;
        }
        for (const auto& item : std::initializer_list<std::pair<const char*, shelf::weights::MatrixWeight*>>{
                 {"mixer.q_proj.weight", &l.q}, {"mixer.k_proj.weight", &l.k}, {"mixer.v_proj.weight", &l.v},
                 {"mixer.o_proj.weight", &l.o_proj}, {"mlp.gate_proj.weight", &l.gate},
                 {"mlp.up_proj.weight", &l.up}, {"mlp.down_proj.weight", &l.down}}) {
            st = matrix(p + item.first, *item.second); if (!st.ok()) return st;
        }
    }
    return w;
}
}

Result<ModelWeights> load_weights(const std::string& dir, const Config& cfg,
                                  gpu::GpuArena& arena, hipStream_t stream) {
    if (shelf::weights::is_quantized_model_dir(dir)) return load_quantized_weights(dir, cfg, arena, stream);
    auto collection = SafetensorsCollection::open(dir);
    if (!collection.ok()) return collection.status();
    auto& c = collection.value();
    Status st = validate_checkpoint(c, cfg);
    if (!st.ok()) return st;
    ModelWeights w;
    st = load_matrix(c, "model.embed_tokens.weight", arena, stream, w.embed_tokens); if (!st.ok()) return st;
    st = load_matrix(c, "lm_head.weight", arena, stream, w.lm_head); if (!st.ok()) return st;
    st = load_vector(c, "model.norm.weight", arena, stream, w.final_norm); if (!st.ok()) return st;
    w.layers.resize(cfg.num_hidden_layers);
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& l = w.layers[i]; const auto p = prefix(i);
        st = load_vector(c, p + "pre_mixer_norm.weight", arena, stream, l.pre_mixer_norm); if (!st.ok()) return st;
        st = load_vector(c, p + "post_mixer_norm.weight", arena, stream, l.post_mixer_norm); if (!st.ok()) return st;
        st = load_vector(c, p + "pre_mlp_norm.weight", arena, stream, l.pre_mlp_norm); if (!st.ok()) return st;
        st = load_vector(c, p + "post_mlp_norm.weight", arena, stream, l.post_mlp_norm); if (!st.ok()) return st;
        st = load_vector(c, p + "mixer.q_norm.weight", arena, stream, l.q_norm); if (!st.ok()) return st;
        st = load_vector(c, p + "mixer.k_norm.weight", arena, stream, l.k_norm); if (!st.ok()) return st;
        st = load_matrix(c, p + "mixer.qkv_proj.weight", arena, stream, l.qkv); if (!st.ok()) return st;
        st = load_matrix(c, p + "mixer.o_proj.weight", arena, stream, l.o_proj); if (!st.ok()) return st;
        st = load_matrix(c, p + "mlp.gate_up_proj.weight", arena, stream, l.gate_up); if (!st.ok()) return st;
        st = load_matrix(c, p + "mlp.down_proj.weight", arena, stream, l.down); if (!st.ok()) return st;
    }
    return w;
}
}  // namespace shelf::plamo3
