// shelf model/qwen35 — config.json (text_config) parsing.
#include <nlohmann/json.hpp>

#include <shelf/model/qwen35/text_config.hpp>

namespace shelf::model::qwen35 {
namespace {

Status bad(const std::string& msg) {
    return Status::invalid_argument("qwen35/config: " + msg, __FILE__, __LINE__);
}

std::size_t sz(const nlohmann::json& j, const char* key) {
    return static_cast<std::size_t>(j.at(key).get<std::int64_t>());
}

} // namespace

Result<TextConfig> load_text_config_json(const std::string& json_text) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(std::string("qwen35/config: invalid json: ") + e.what(),
                                        __FILE__, __LINE__);
    }
    if (!root.contains("text_config")) {
        return bad("missing text_config (text-only runtime)");
    }
    const auto& tc = root.at("text_config");

    TextConfig c;
    try {
        c.hidden_size = sz(tc, "hidden_size");
        c.num_layers = sz(tc, "num_hidden_layers");
        c.vocab_size = sz(tc, "vocab_size");
        c.num_heads = sz(tc, "num_attention_heads");
        c.num_kv_heads = sz(tc, "num_key_value_heads");
        c.head_dim = sz(tc, "head_dim");
        c.intermediate_size = sz(tc, "intermediate_size");
        c.rms_eps = tc.at("rms_norm_eps").get<double>();
        c.eos_token_id = tc.at("eos_token_id").get<std::int64_t>();

        const auto& rp = tc.at("rope_parameters");
        c.rope_theta = rp.at("rope_theta").get<double>();
        const double partial = rp.contains("partial_rotary_factor")
                                   ? rp.at("partial_rotary_factor").get<double>() : 1.0;
        c.rotary_dim = static_cast<std::size_t>(c.head_dim * partial);
        if (rp.value("rope_type", std::string("default")) != "default") {
            return bad("only rope_type=default supported");
        }
        if (rp.contains("mrope_interleaved") && rp.at("mrope_interleaved").get<bool>() == false) {
            return bad("expected interleaved mrope (text path assumes equal channels)");
        }

        c.lin_key_heads = sz(tc, "linear_num_key_heads");
        c.lin_value_heads = sz(tc, "linear_num_value_heads");
        c.lin_key_head_dim = sz(tc, "linear_key_head_dim");
        c.lin_value_head_dim = sz(tc, "linear_value_head_dim");
        c.conv_kernel = sz(tc, "linear_conv_kernel_dim");
        if (c.conv_kernel != 4) return bad("causal_conv_step contract requires conv kernel 4");

        c.tie_word_embeddings = tc.value("tie_word_embeddings", false) ||
                                root.value("tie_word_embeddings", false);

        const auto& types = tc.at("layer_types");
        for (const auto& t : types) {
            const std::string s = t.get<std::string>();
            if (s == "linear_attention") c.layer_kinds.push_back(LayerKind::linear_attention);
            else if (s == "full_attention") c.layer_kinds.push_back(LayerKind::full_attention);
            else return bad("unsupported layer type: " + s);
        }
    } catch (const nlohmann::json::exception& e) {
        return bad(std::string("missing/invalid field: ") + e.what());
    }

    if (c.num_layers != c.layer_kinds.size()) return bad("layer_types length mismatch");
    if (c.rotary_dim == 0 || c.rotary_dim % 2 != 0) return bad("rotary_dim invalid");
    if (c.conv_channels() % 2 != 0) return bad("conv channels invalid");
    return c;
}

} // namespace shelf::model::qwen35
