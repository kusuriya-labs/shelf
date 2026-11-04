#include <shelf/models/common/qwen_hybrid/config.h>

#include <fstream>
#include <nlohmann/json.hpp>

namespace shelf {
namespace qwen_hybrid {
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

struct ModelIdentity {
    const char* model_type;
    // What this identity declares about the FFN of the model. It is only used to reject a
    // reader, never to choose one: the FFN kind of a loaded model stays decided by its
    // structure, which is the part that can be checked against the weight set.
    bool declares_routed_experts;
};

constexpr ModelIdentity kModelIdentities[] = {
    {"qwen3_5", false},
    {"qwen3_5_text", false},
    {"qwen3_5_moe", true},
    {"qwen3_5_moe_text", true},
};

const ModelIdentity* find_model_identity(const std::string& model_type) {
    for (const auto& identity : kModelIdentities) {
        if (identity.model_type == model_type)
            return &identity;
    }
    return nullptr;
}

struct DeclaredIdentity {
    std::string model_type;
    std::string text_model_type;
    std::vector<std::string> architectures;
    bool declares_routed_experts = false;
};

DeclaredIdentity identity_from_root(const nlohmann::json& root) {
    DeclaredIdentity id;
    if (root.contains("model_type") && root["model_type"].is_string())
        id.model_type = root["model_type"].get<std::string>();
    if (root.contains("text_config") && root["text_config"].is_object()) {
        const nlohmann::json& t = root["text_config"];
        if (t.contains("model_type") && t["model_type"].is_string())
            id.text_model_type = t["model_type"].get<std::string>();
    }
    if (root.contains("architectures") && root["architectures"].is_array()) {
        for (const auto& a : root["architectures"]) {
            if (a.is_string()) id.architectures.push_back(a.get<std::string>());
        }
    }
    for (const std::string& candidate : {id.model_type, id.text_model_type}) {
        const ModelIdentity* found = find_model_identity(candidate);
        if (found != nullptr && found->declares_routed_experts)
            id.declares_routed_experts = true;
    }
    return id;
}

// The gate. An identity is a declaration, so it can only reject: what a model is read as
// stays decided by the loaded structure (num_experts, tensor names, shapes).
Status gate_declared_identity(const DeclaredIdentity& id, const char* who) {
    const bool declared = !id.model_type.empty() || !id.text_model_type.empty();
    if (!declared) {
        return Status::unsupported(
            (std::string(who) + ": config.json declares no model identity (model_type)").c_str(),
            __FILE__, __LINE__);
    }
    for (const std::string& candidate : {id.model_type, id.text_model_type}) {
        if (find_model_identity(candidate) != nullptr)
            return Status::make_ok();
    }
    std::string msg = std::string(who) + ": unsupported model identity model_type='" +
                      id.model_type + "'";
    if (!id.text_model_type.empty())
        msg += " text_config.model_type='" + id.text_model_type + "'";
    if (!id.architectures.empty()) {
        msg += " architectures=" + id.architectures[0];
        if (id.architectures.size() > 1) msg += "...";
    }
    msg += " (this build implements";
    for (const auto& identity : kModelIdentities)
        msg += std::string(" ") + identity.model_type;
    msg += ")";
    return Status::unsupported(msg.c_str(), __FILE__, __LINE__);
}

}  // namespace

const std::vector<std::string>& supported_model_identities() {
    static const std::vector<std::string> supported = [] {
        std::vector<std::string> names;
        for (const auto& identity : kModelIdentities)
            names.emplace_back(identity.model_type);
        return names;
    }();
    return supported;
}

Status check_config_identity(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f.is_open()) {
        return Status::invalid_argument(
            "check_config_identity: cannot open config.json", __FILE__, __LINE__);
    }
    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            (std::string("check_config_identity: bad config.json: ") + e.what()).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object()) {
        return Status::invalid_argument(
            "check_config_identity: config.json is not an object", __FILE__, __LINE__);
    }
    return gate_declared_identity(identity_from_root(root), "check_config_identity");
}

Result<bool> read_tie_word_embeddings(const std::string& model_dir) {
    const std::string path = model_dir + "/config.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        return Status::invalid_argument(
            "read_tie_word_embeddings: cannot open config.json", __FILE__, __LINE__);
    }

    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("read_tie_word_embeddings: bad config.json: " + std::string(e.what())).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object()) {
        return Status::invalid_argument(
            "read_tie_word_embeddings: config.json is not an object",
            __FILE__, __LINE__);
    }
    return tie_word_embeddings_from_json(root);
}

namespace {

Result<nlohmann::json> open_text_config(const std::string& model_dir, const char* who) {
    std::ifstream f(model_dir + "/config.json");
    if (!f.is_open()) {
        return Status::invalid_argument(
            (std::string(who) + ": cannot open config.json").c_str(), __FILE__, __LINE__);
    }
    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            (std::string(who) + ": bad config.json: " + std::string(e.what())).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object() || !root.contains("text_config") || !root["text_config"].is_object()) {
        return Status::invalid_argument(
            (std::string(who) + ": config.json missing text_config").c_str(), __FILE__,
            __LINE__);
    }
    return root["text_config"];
}

}  // namespace

Result<std::vector<std::size_t>> read_text_config_sizes(
    const std::string& model_dir,
    const std::vector<std::string>& keys) {
    auto text = open_text_config(model_dir, "read_text_config_sizes");
    if (!text.ok()) return text.status();
    const nlohmann::json& t = text.value();
    std::vector<std::size_t> out;
    out.reserve(keys.size());
    for (const auto& key : keys) {
        std::size_t value = 0;
        if (t.contains(key) && t[key].is_number()) value = t[key].get<std::size_t>();
        out.push_back(value);
    }
    return out;
}

Result<bool> has_text_config_key(const std::string& model_dir, const char* key) {
    auto text = open_text_config(model_dir, "has_text_config_key");
    if (!text.ok()) return text.status();
    return text.value().contains(key);
}

Result<QwenHybridConfig> read_qwen_hybrid_config(const std::string& model_dir) {
    std::string path = model_dir + "/config.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        return Status::invalid_argument(
            "read_qwen_hybrid_config: cannot open config.json", __FILE__, __LINE__);
    }

    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("read_qwen_hybrid_config: bad config.json: " + std::string(e.what())).c_str(),
            __FILE__, __LINE__);
    }

    if (!root.is_object()) {
        return Status::invalid_argument(
            "read_qwen_hybrid_config: config.json is not an object", __FILE__, __LINE__);
    }

    const DeclaredIdentity identity = identity_from_root(root);
    Status identity_status = gate_declared_identity(identity, "read_qwen_hybrid_config");
    if (!identity_status.ok()) return identity_status;

    if (!root.contains("text_config")) {
        return Status::invalid_argument(
            "read_qwen_hybrid_config: config.json missing text_config", __FILE__, __LINE__);
    }

    const nlohmann::json& t = root["text_config"];
    QwenHybridConfig cfg;
    cfg.declared_model_type = identity.model_type;
    cfg.declared_text_model_type = identity.text_model_type;
    cfg.declared_architectures = identity.architectures;
    cfg.identity_declares_routed_experts = identity.declares_routed_experts;

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
            "read_qwen_hybrid_config: required dimensions missing", __FILE__, __LINE__);
    }

    return cfg;
}

HybridLayerCounts count_layer_topology(const QwenHybridConfig& config) noexcept {
    HybridLayerCounts counts;
    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        if (layer_is_gdn(config, l)) ++counts.gdn_layers;
        else ++counts.attention_layers;
    }
    return counts;
}

bool layer_is_gdn(const QwenHybridConfig& config, std::size_t layer) {
    if (config.layer_types.size() == config.num_hidden_layers) {
        return config.layer_types[layer] == 0;
    }
    if (config.full_attention_interval == 0) {
        return layer % 4 < 3;
    }
    return layer % config.full_attention_interval < config.full_attention_interval - 1;
}

}
}
