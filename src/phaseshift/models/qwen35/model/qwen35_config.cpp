#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <fstream>
#include <nlohmann/json.hpp>

namespace ps {
namespace qwen35 {

namespace {

bool tie_word_embeddings_from_json(const nlohmann::json& root) {
    if (root.contains("tie_word_embeddings") && root["tie_word_embeddings"].is_boolean()) {
        return root["tie_word_embeddings"].get<bool>();
    }
    if (root.contains("text_config") && root["text_config"].is_object()) {
        const nlohmann::json& t = root["text_config"];
        if (t.contains("tie_word_embeddings") && t["tie_word_embeddings"].is_boolean()) {
            return t["tie_word_embeddings"].get<bool>();
        }
    }
    return false;
}

}  // namespace

Result<bool> read_qwen35_tie_word_embeddings(const std::string& model_dir) {
    const std::string path = model_dir + "/config.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        return Status::invalid_argument(
            "read_qwen35_tie_word_embeddings: cannot open config.json", __FILE__, __LINE__);
    }

    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("read_qwen35_tie_word_embeddings: bad config.json: " + std::string(e.what())).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object()) {
        return Status::invalid_argument(
            "read_qwen35_tie_word_embeddings: config.json is not an object",
            __FILE__, __LINE__);
    }
    return tie_word_embeddings_from_json(root);
}

Result<Qwen35TextConfig> read_qwen35_text_config(const std::string& model_dir) {
    std::string path = model_dir + "/config.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        return Status::invalid_argument(
            "read_qwen35_text_config: cannot open config.json", __FILE__, __LINE__);
    }

    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("read_qwen35_text_config: bad config.json: " + std::string(e.what())).c_str(),
            __FILE__, __LINE__);
    }

    if (!root.is_object() || !root.contains("text_config")) {
        return Status::invalid_argument(
            "read_qwen35_text_config: config.json missing text_config", __FILE__, __LINE__);
    }

    const nlohmann::json& t = root["text_config"];
    Qwen35TextConfig cfg;

    auto read_size = [&](const char* key, std::size_t& out) -> bool {
        if (!t.contains(key) || !t[key].is_number()) return false;
        out = t[key].get<std::size_t>();
        return true;
    };
    auto read_float = [&](const char* key, float& out) -> bool {
        if (!t.contains(key) || !t[key].is_number()) return false;
        out = t[key].get<float>();
        return true;
    };

    read_size("vocab_size", cfg.vocab_size);
    read_size("hidden_size", cfg.hidden_size);
    read_size("intermediate_size", cfg.intermediate_size);
    read_size("num_hidden_layers", cfg.num_hidden_layers);
    read_size("full_attention_interval", cfg.full_attention_interval);

    read_size("linear_num_key_heads", cfg.linear_num_key_heads);
    read_size("linear_num_value_heads", cfg.linear_num_value_heads);
    read_size("linear_key_head_dim", cfg.linear_key_head_dim);
    read_size("linear_value_head_dim", cfg.linear_value_head_dim);
    read_size("linear_conv_kernel_dim", cfg.linear_conv_kernel_dim);

    read_size("num_attention_heads", cfg.num_attention_heads);
    read_size("num_key_value_heads", cfg.num_key_value_heads);
    read_size("head_dim", cfg.attention_head_dim);

    // Generation stop tokens come from generation_config.json, which may list
    // several (Qwen3.5/3.8 use <|im_end|> and <|endoftext|>). Fall back to the
    // single text_config.eos_token_id when no generation config is present.
    {
        std::ifstream gen_file(model_dir + "/generation_config.json");
        if (gen_file.is_open()) {
            nlohmann::json gen;
            try {
                gen_file >> gen;
            } catch (const nlohmann::json::exception&) {
                gen = nlohmann::json();
            }
            if (gen.is_object() && gen.contains("eos_token_id")) {
                const nlohmann::json& eos = gen["eos_token_id"];
                if (eos.is_number_integer()) {
                    cfg.stop_tokens.push_back(eos.get<int32_t>());
                } else if (eos.is_array()) {
                    for (const auto& value : eos) {
                        if (value.is_number_integer()) {
                            cfg.stop_tokens.push_back(value.get<int32_t>());
                        }
                    }
                }
            }
        }
        if (cfg.stop_tokens.empty()) {
            std::size_t eos_token_id = 0;
            if (read_size("eos_token_id", eos_token_id)) {
                cfg.stop_tokens.push_back(static_cast<int32_t>(eos_token_id));
            }
        }
    }

    read_float("rms_norm_eps", cfg.rms_norm_eps);

    if (t.contains("rope_parameters") && t["rope_parameters"].is_object()) {
        const nlohmann::json& rp = t["rope_parameters"];
        if (rp.contains("rope_theta") && rp["rope_theta"].is_number()) {
            cfg.rope_theta = rp["rope_theta"].get<float>();
        }
        if (rp.contains("partial_rotary_factor") && rp["partial_rotary_factor"].is_number()) {
            cfg.partial_rotary_factor = rp["partial_rotary_factor"].get<float>();
        }
    }

    cfg.tie_word_embeddings = tie_word_embeddings_from_json(root);

    if (t.contains("output_gate_type") && t["output_gate_type"].is_string()) {
        const std::string gate_type = t["output_gate_type"].get<std::string>();
        if (gate_type == "swish" || gate_type == "silu") {
            cfg.output_gate_silu = true;
        } else if (gate_type != "sigmoid") {
            return Status::invalid_argument(
                ("read_qwen35_text_config: unsupported output_gate_type '" + gate_type + "'")
                    .c_str(),
                __FILE__, __LINE__);
        }
    }

    if (t.contains("layer_types") && t["layer_types"].is_array()) {
        cfg.layer_types.reserve(t["layer_types"].size());
        for (const auto& lt : t["layer_types"]) {
            if (!lt.is_string()) continue;
            const std::string s = lt.get<std::string>();
            if (s == "linear_attention") {
                cfg.layer_types.push_back(0);
            } else if (s == "full_attention") {
                cfg.layer_types.push_back(1);
            } else {
                cfg.layer_types.push_back(-1);
            }
        }
    }

    if (cfg.vocab_size == 0 || cfg.hidden_size == 0 || cfg.num_hidden_layers == 0) {
        return Status::invalid_argument(
            "read_qwen35_text_config: required dimensions missing", __FILE__, __LINE__);
    }

    return cfg;
}

}
}
