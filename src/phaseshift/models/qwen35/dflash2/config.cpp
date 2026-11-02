#include <phaseshift/models/qwen35/dflash2/config.h>
#include <fstream>
#include <nlohmann/json.hpp>

namespace ps {
namespace qwen35 {
namespace dflash2 {

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

}  // namespace

Result<DFlash2Config> read_dflash2_config(const std::string& model_dir) {
    const std::string path = model_dir + "/config.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        return config_error("read_dflash2_config: cannot open " + path);
    }

    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return config_error(
            "read_dflash2_config: bad config.json: " + std::string(e.what()));
    }
    if (!root.is_object()) {
        return config_error("read_dflash2_config: config.json is not an object");
    }

    constexpr const char* kWhere = "read_dflash2_config";

    if (!root.contains("architectures")) {
        return config_error("read_dflash2_config: missing key 'architectures'");
    }
    const nlohmann::json& archs = root.at("architectures");
    if (!archs.is_array() || archs.empty() || !archs.at(0).is_string()) {
        return config_error(
            "read_dflash2_config: 'architectures' must be a non-empty string array");
    }

    DFlash2Config c;
    c.architecture = archs.at(0).get<std::string>();
    if (c.architecture != "DFlash2DraftModel") {
        return config_error(
            "read_dflash2_config: unsupported architecture '" + c.architecture + "'");
    }

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
        {"num_target_layers", &c.num_target_layers},
    };
    for (const SizeKey& k : size_keys) {
        auto v = read_size(root, k.key, kWhere);
        if (!v.ok()) return v.status();
        *k.out = v.value();
    }

    auto is_causal = read_bool(root, "is_causal", kWhere);
    if (!is_causal.ok()) return is_causal.status();
    c.is_causal = is_causal.value();

    auto eps = read_float(root, "rms_norm_eps", kWhere);
    if (!eps.ok()) return eps.status();
    c.rms_norm_eps = eps.value();

    auto tied = read_bool(root, "tie_word_embeddings", kWhere);
    if (!tied.ok()) return tied.status();
    c.tie_word_embeddings = tied.value();

    if (!root.contains("dflash_config") || !root.at("dflash_config").is_object()) {
        return config_error("read_dflash2_config: missing object 'dflash_config'");
    }
    const nlohmann::json& dc = root.at("dflash_config");

    const SizeKey draft_keys[] = {
        {"block_size", &c.block_size},
        {"conv_group_size", &c.conv_group_size},
        {"conv_kernel_size", &c.conv_kernel_size},
        {"mask_token_id", &c.mask_token_id},
        {"selector_rank", &c.selector_rank},
        {"selector_top_k", &c.selector_top_k},
    };
    for (const SizeKey& k : draft_keys) {
        auto v = read_size(dc, k.key, kWhere);
        if (!v.ok()) return v.status();
        *k.out = v.value();
    }

    if (!dc.contains("target_layer_ids") || !dc.at("target_layer_ids").is_array() ||
        dc.at("target_layer_ids").empty()) {
        return config_error(
            "read_dflash2_config: 'dflash_config.target_layer_ids' must be a non-empty array");
    }
    const nlohmann::json& tids = dc.at("target_layer_ids");
    if (tids.size() > kMaxTargetLayerIds) {
        return config_error(
            "read_dflash2_config: 'target_layer_ids' exceeds the supported count");
    }
    for (const auto& t : tids) {
        if (!t.is_number_integer() && !t.is_number_unsigned()) {
            return config_error(
                "read_dflash2_config: 'target_layer_ids' entries must be integers");
        }
        const long long raw = t.get<long long>();
        if (raw < 0) {
            return config_error("read_dflash2_config: 'target_layer_ids' entry is negative");
        }
        c.target_layer_ids[c.num_target_layer_ids] = static_cast<std::size_t>(raw);
        ++c.num_target_layer_ids;
    }

    if (root.contains("rope_parameters") && root.at("rope_parameters").is_object()) {
        const nlohmann::json& rp = root.at("rope_parameters");
        if (rp.contains("rope_theta") && rp.at("rope_theta").is_number()) {
            c.rope_theta = rp.at("rope_theta").get<float>();
        }
        if (rp.contains("rope_type") && rp.at("rope_type").is_string()) {
            const std::string rope_type = rp.at("rope_type").get<std::string>();
            if (rope_type != "default") {
                return config_error(
                    "read_dflash2_config: unsupported rope_type '" + rope_type + "'");
            }
        }
    }

    if (root.contains("layer_types")) {
        if (!root.at("layer_types").is_array()) {
            return config_error("read_dflash2_config: 'layer_types' must be an array");
        }
        for (const auto& lt : root.at("layer_types")) {
            if (!lt.is_string()) {
                return config_error("read_dflash2_config: 'layer_types' entries must be strings");
            }
            const std::string s = lt.get<std::string>();
            if (s != "sliding_attention") {
                return config_error(
                    "read_dflash2_config: unsupported layer type '" + s + "'");
            }
        }
    }

    if (c.vocab_size == 0 || c.hidden_size == 0 || c.intermediate_size == 0 ||
        c.num_hidden_layers == 0) {
        return config_error("read_dflash2_config: required dimensions are zero");
    }
    if (c.num_attention_heads == 0 || c.num_key_value_heads == 0 || c.head_dim == 0) {
        return config_error("read_dflash2_config: attention dimensions are zero");
    }
    if (c.num_attention_heads % c.num_key_value_heads != 0) {
        return config_error(
            "read_dflash2_config: num_attention_heads is not a multiple of num_key_value_heads");
    }
    if (c.sliding_window == 0) {
        return config_error("read_dflash2_config: sliding_window is zero");
    }
    if (c.is_causal) {
        return config_error(
            "read_dflash2_config: draft attention must be non-causal (is_causal=false)");
    }
    if (c.block_size < 2 || c.block_size > kMaxBlockSize) {
        return config_error("read_dflash2_config: block_size out of range");
    }
    if (c.conv_group_size == 0 || c.conv_kernel_size == 0) {
        return config_error("read_dflash2_config: convolution geometry is zero");
    }
    if (c.hidden_size % c.conv_group_size != 0) {
        return config_error(
            "read_dflash2_config: conv_group_size does not divide hidden_size");
    }
    if (c.mask_token_id >= c.vocab_size) {
        return config_error("read_dflash2_config: mask_token_id exceeds vocab_size");
    }
    if (c.selector_rank == 0 || c.selector_top_k == 0 ||
        c.selector_top_k > c.vocab_size) {
        return config_error("read_dflash2_config: invalid selector geometry");
    }
    if (c.num_target_layers == 0) {
        return config_error("read_dflash2_config: num_target_layers is zero");
    }
    std::size_t previous = 0;
    for (std::size_t i = 0; i < c.num_target_layer_ids; ++i) {
        const std::size_t id = c.target_layer_ids[i];
        if (id >= c.num_target_layers) {
            return config_error(
                "read_dflash2_config: target_layer_ids entry exceeds num_target_layers");
        }
        if (i != 0 && id <= previous) {
            return config_error(
                "read_dflash2_config: target_layer_ids must be strictly increasing");
        }
        previous = id;
    }
    if (c.tie_word_embeddings) {
        return config_error(
            "read_dflash2_config: tie_word_embeddings must be false for the drafter");
    }

    return c;
}

}  // namespace dflash2
}  // namespace qwen35
}  // namespace ps
