#include <shelf/models/qwen35/eagle4/config.h>
#include <fstream>
#include <nlohmann/json.hpp>

namespace ps::qwen35::eagle4 {

namespace {

Status config_error(const std::string& message) {
    return Status::invalid_argument(message.c_str(), __FILE__, __LINE__);
}

Result<std::size_t> read_size(
    const nlohmann::json& obj, const char* key, const char* where) {
    if (!obj.contains(key)) {
        return config_error(std::string(where) + ": missing key '" + key + "'");
    }
    const nlohmann::json& v = obj.at(key);
    if (!v.is_number_integer() && !v.is_number_unsigned()) {
        return config_error(std::string(where) + ": key '" + key + "' is not an integer");
    }
    const long long raw = v.get<long long>();
    if (raw < 0) {
        return config_error(std::string(where) + ": key '" + key + "' is negative");
    }
    return static_cast<std::size_t>(raw);
}

Result<bool> read_bool(const nlohmann::json& obj, const char* key, const char* where) {
    if (!obj.contains(key)) {
        return config_error(std::string(where) + ": missing key '" + key + "'");
    }
    const nlohmann::json& v = obj.at(key);
    if (!v.is_boolean()) {
        return config_error(std::string(where) + ": key '" + key + "' is not a boolean");
    }
    return v.get<bool>();
}

Result<float> read_float(const nlohmann::json& obj, const char* key, const char* where) {
    if (!obj.contains(key)) {
        return config_error(std::string(where) + ": missing key '" + key + "'");
    }
    const nlohmann::json& v = obj.at(key);
    if (!v.is_number()) {
        return config_error(std::string(where) + ": key '" + key + "' is not a number");
    }
    return v.get<float>();
}

Status read_aux_layer_ids(
    const nlohmann::json& root, const char* where, Eagle4Config& c) {
    if (!root.contains("eagle_config") || !root.at("eagle_config").is_object()) {
        return config_error(std::string(where) + ": missing object 'eagle_config'");
    }
    const nlohmann::json& ec = root.at("eagle_config");
    if (!ec.contains("eagle_aux_hidden_state_layer_ids") ||
        !ec.at("eagle_aux_hidden_state_layer_ids").is_array()) {
        return config_error(
            std::string(where) +
            ": 'eagle_config.eagle_aux_hidden_state_layer_ids' must be an array");
    }
    const nlohmann::json& ids = ec.at("eagle_aux_hidden_state_layer_ids");
    if (ids.empty() || ids.size() > kEagle4MaxAuxLayers) {
        return config_error(
            std::string(where) + ": 'eagle_aux_hidden_state_layer_ids' count out of range");
    }
    for (const auto& id : ids) {
        if (!id.is_number_integer() && !id.is_number_unsigned()) {
            return config_error(
                std::string(where) + ": aux layer id entries must be integers");
        }
        const long long raw = id.get<long long>();
        if (raw < 0) {
            return config_error(std::string(where) + ": aux layer id is negative");
        }
        c.aux_layer_ids[c.num_aux_layers] = static_cast<std::size_t>(raw);
        ++c.num_aux_layers;
    }
    if (root.contains("num_aux_hidden_states")) {
        auto stated = read_size(root, "num_aux_hidden_states", where);
        if (!stated.ok()) return stated.status();
        if (stated.value() != c.num_aux_layers) {
            return config_error(
                std::string(where) +
                ": num_aux_hidden_states does not match the aux layer id count");
        }
    }
    return Status::make_ok();
}

Status read_candidate_head(
    const nlohmann::json& root, const char* where, Eagle4Config& c) {
    if (!root.contains("efficient_lm_head_config") ||
        !root.at("efficient_lm_head_config").is_object()) {
        return config_error(
            std::string(where) + ": missing object 'efficient_lm_head_config'");
    }
    const nlohmann::json& head = root.at("efficient_lm_head_config");
    if (!head.contains("type") || !head.at("type").is_string()) {
        return config_error(std::string(where) + ": efficient lm head type is missing");
    }
    const std::string type = head.at("type").get<std::string>();
    if (type != "low_rank") {
        return config_error(
            std::string(where) + ": unsupported efficient lm head type '" + type + "'");
    }
    auto rank = read_size(head, "rank", where);
    if (!rank.ok()) return rank.status();
    c.candidate_rank = rank.value();
    auto candidates = read_size(head, "num_candidates", where);
    if (!candidates.ok()) return candidates.status();
    c.num_candidates = candidates.value();
    return Status::make_ok();
}

Status validate_eagle4_config(const Eagle4Config& c) {
    if (c.architecture != "LlamaForCausalLMEagle4") {
        return config_error("eagle4: unsupported architecture '" + c.architecture + "'");
    }
    if (c.num_hidden_layers != 1) {
        return config_error("eagle4: only num_hidden_layers=1 is supported");
    }
    if (c.num_attention_heads != c.num_key_value_heads) {
        return config_error("eagle4: draft attention must be MHA");
    }
    if (c.num_attention_heads == 0 || c.head_dim == 0) {
        return config_error("eagle4: attention dimensions are zero");
    }
    if (c.num_attention_heads * c.head_dim != c.hidden_size) {
        return config_error("eagle4: hidden_size is not num_attention_heads * head_dim");
    }
    if (c.sliding_window == 0) {
        return config_error("eagle4: sliding_window is zero");
    }
    if (c.vocab_size == 0 || c.hidden_size == 0 || c.intermediate_size == 0) {
        return config_error("eagle4: required dimensions are zero");
    }
    if (c.target_hidden_size == 0) {
        return config_error("eagle4: target_hidden_size is zero");
    }
    if (c.candidate_rank <= 0 || c.num_candidates <= 0 || c.num_candidates > c.vocab_size) {
        return config_error("eagle4: invalid efficient_lm_head geometry");
    }
    if (c.num_aux_layers == 0) {
        return config_error("eagle4: no auxiliary hidden state layers");
    }
    for (std::size_t i = 1; i < c.num_aux_layers; ++i) {
        if (c.aux_layer_ids[i] <= c.aux_layer_ids[i - 1]) {
            return config_error("eagle4: aux_layer_ids must be strictly increasing");
        }
    }
    return Status::make_ok();
}

}  // namespace

Result<Eagle4Config> read_eagle4_config(const std::string& model_dir) {
    const std::string path = model_dir + "/config.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        return config_error("read_eagle4_config: cannot open " + path);
    }

    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return config_error(
            "read_eagle4_config: bad config.json: " + std::string(e.what()));
    }
    if (!root.is_object()) {
        return config_error("read_eagle4_config: config.json is not an object");
    }

    constexpr const char* kWhere = "read_eagle4_config";

    if (!root.contains("architectures")) {
        return config_error("read_eagle4_config: missing key 'architectures'");
    }
    const nlohmann::json& archs = root.at("architectures");
    if (!archs.is_array() || archs.empty() || !archs.at(0).is_string()) {
        return config_error(
            "read_eagle4_config: 'architectures' must be a non-empty string array");
    }

    Eagle4Config c;
    c.architecture = archs.at(0).get<std::string>();

    struct SizeKey {
        const char* key;
        std::size_t* out;
    };
    const SizeKey size_keys[] = {
        {"vocab_size", &c.vocab_size},
        {"hidden_size", &c.hidden_size},
        {"intermediate_size", &c.intermediate_size},
        {"num_hidden_layers", &c.num_hidden_layers},
        {"num_attention_heads", &c.num_attention_heads},
        {"num_key_value_heads", &c.num_key_value_heads},
        {"head_dim", &c.head_dim},
        {"sliding_window", &c.sliding_window},
        {"target_hidden_size", &c.target_hidden_size},
    };
    for (const SizeKey& k : size_keys) {
        auto v = read_size(root, k.key, kWhere);
        if (!v.ok()) return v.status();
        *k.out = v.value();
    }

    auto eps = read_float(root, "rms_norm_eps", kWhere);
    if (!eps.ok()) return eps.status();
    c.rms_norm_eps = eps.value();

    auto fc_norm = read_bool(root, "fc_norm", kWhere);
    if (!fc_norm.ok()) return fc_norm.status();
    c.fc_norm = fc_norm.value();

    auto norm_output = read_bool(root, "norm_output", kWhere);
    if (!norm_output.ok()) return norm_output.status();
    c.norm_output = norm_output.value();

    auto tied = read_bool(root, "tie_word_embeddings", kWhere);
    if (!tied.ok()) return tied.status();
    c.tie_word_embeddings = tied.value();

    Status aux_st = read_aux_layer_ids(root, kWhere, c);
    if (!aux_st.ok()) return aux_st;
    Status head_st = read_candidate_head(root, kWhere, c);
    if (!head_st.ok()) return head_st;

    if (root.contains("rope_parameters") && root.at("rope_parameters").is_object()) {
        const nlohmann::json& rp = root.at("rope_parameters");
        if (rp.contains("rope_theta") && rp.at("rope_theta").is_number()) {
            c.rope_theta = rp.at("rope_theta").get<float>();
        }
        if (rp.contains("rope_type") && rp.at("rope_type").is_string()) {
            const std::string rope_type = rp.at("rope_type").get<std::string>();
            if (rope_type != "default") {
                return config_error(
                    "read_eagle4_config: unsupported rope_type '" + rope_type + "'");
            }
        }
    }

    if (root.contains("layer_types")) {
        if (!root.at("layer_types").is_array()) {
            return config_error("read_eagle4_config: 'layer_types' must be an array");
        }
        for (const auto& lt : root.at("layer_types")) {
            if (!lt.is_string()) {
                return config_error(
                    "read_eagle4_config: 'layer_types' entries must be strings");
            }
            const std::string s = lt.get<std::string>();
            if (s != "sliding_attention") {
                return config_error(
                    "read_eagle4_config: unsupported layer type '" + s + "'");
            }
        }
    }

    if (c.tie_word_embeddings) {
        return config_error("read_eagle4_config: tie_word_embeddings must be false");
    }

    Status st = validate_eagle4_config(c);
    if (!st.ok()) return st;
    return c;
}

}  // namespace ps::qwen35::eagle4
