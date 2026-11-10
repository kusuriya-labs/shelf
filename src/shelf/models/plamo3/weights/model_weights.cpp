#include <shelf/models/plamo3/weights/model_weights.h>

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

Result<ModelWeights> load_weights(const std::string& dir, const Config& cfg,
                                  gpu::GpuArena& arena, hipStream_t stream) {
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
