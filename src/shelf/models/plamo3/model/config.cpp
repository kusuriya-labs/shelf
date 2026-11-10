#include <shelf/models/plamo3/model/config.h>

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

namespace shelf::plamo3 {
namespace {

Result<nlohmann::json> read_json(const std::string& model_dir) {
    std::ifstream file(model_dir + "/config.json");
    if (!file) return Status::invalid_argument("plamo3: cannot open config.json", __FILE__, __LINE__);
    try {
        auto root = nlohmann::json::parse(file);
        if (!root.is_object()) return Status::invalid_argument("plamo3: config.json must be an object", __FILE__, __LINE__);
        return root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument((std::string("plamo3: invalid config.json: ") + e.what()).c_str(), __FILE__, __LINE__);
    }
}

}

Result<bool> is_plamo3_config(const std::string& model_dir) {
    auto root = read_json(model_dir);
    if (!root.ok()) return root.status();
    return root.value().contains("model_type") && root.value()["model_type"].is_string() &&
           root.value()["model_type"] == "plamo3";
}

Result<Config> read_config(const std::string& model_dir) {
    auto result = read_json(model_dir);
    if (!result.ok()) return result.status();
    const auto& root = result.value();
    try {
        if (root.at("model_type") != "plamo3")
            return Status::unsupported("plamo3: wrong model_type", __FILE__, __LINE__);
        for (const char* key : {"num_hidden_layers", "hidden_size", "num_attention_heads",
                                "num_key_value_heads", "head_dim", "intermediate_size",
                                "vocab_size", "window_size", "sliding_window_pattern",
                                "max_position_embeddings"}) {
            const auto& value = root.at(key);
            if (!value.is_number_integer() || value.get<std::int64_t>() <= 0)
                return Status::invalid_argument("plamo3: invalid positive integer parameter", __FILE__, __LINE__);
        }
        Config c;
        c.num_hidden_layers = root.at("num_hidden_layers").get<std::size_t>();
        c.hidden_size = root.at("hidden_size").get<std::size_t>();
        c.num_attention_heads = root.at("num_attention_heads").get<std::size_t>();
        c.num_key_value_heads = root.at("num_key_value_heads").get<std::size_t>();
        c.head_dim = root.at("head_dim").get<std::size_t>();
        c.intermediate_size = root.at("intermediate_size").get<std::size_t>();
        c.vocab_size = root.at("vocab_size").get<std::size_t>();
        c.window_size = root.at("window_size").get<std::size_t>();
        c.sliding_window_pattern = root.at("sliding_window_pattern").get<std::size_t>();
        c.max_position_embeddings = root.at("max_position_embeddings").get<std::size_t>();
        c.rms_norm_eps = root.at("rms_norm_eps").get<double>();
        c.local_rope.theta = root.at("rope_local_theta").get<double>();
        c.full_rope.theta = root.at("rope_theta").get<double>();
        c.full_rope.factor = root.at("rope_scaling_factor").get<double>();
        if (c.full_rope.factor != 1)
            c.full_rope.original_max_position_embeddings = root.at("initial_context_length").get<std::size_t>();
        if (!c.num_hidden_layers || !c.hidden_size || !c.num_attention_heads ||
            !c.num_key_value_heads || !c.head_dim || !c.intermediate_size || !c.vocab_size ||
            !c.window_size || !c.sliding_window_pattern || !c.max_position_embeddings ||
            c.num_attention_heads % c.num_key_value_heads || c.head_dim % 2 ||
            !std::isfinite(c.rms_norm_eps) || c.rms_norm_eps <= 0 ||
            !std::isfinite(c.local_rope.theta) || c.local_rope.theta <= 0 ||
            !std::isfinite(c.full_rope.theta) || c.full_rope.theta <= 0 ||
            !std::isfinite(c.full_rope.factor) || c.full_rope.factor < 1 ||
            (c.full_rope.factor != 1 && !c.full_rope.original_max_position_embeddings))
            return Status::invalid_argument("plamo3: invalid geometry or RoPE parameters", __FILE__, __LINE__);
        if (root.value("scale_embedding", false) || root.value("tie_word_embeddings", false) ||
            root.value("linear_type", std::string("normal")) != "normal")
            return Status::unsupported("plamo3: unsupported embedding or linear configuration", __FILE__, __LINE__);
        return c;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument((std::string("plamo3: invalid config: ") + e.what()).c_str(), __FILE__, __LINE__);
    }
}

}
