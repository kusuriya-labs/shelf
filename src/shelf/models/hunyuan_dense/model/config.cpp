#include <shelf/models/hunyuan_dense/model/config.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

namespace shelf::hunyuan_dense {
namespace {

Result<nlohmann::json> parse_json_object(const std::string& path, const char* label) {
    std::ifstream file(path);
    if (!file) {
        return Status::invalid_argument(
            (std::string("hunyuan_v1_dense: cannot open ") + label).c_str(),
            __FILE__, __LINE__);
    }
    try {
        nlohmann::json root = nlohmann::json::parse(file);
        if (!root.is_object()) {
            return Status::invalid_argument(
                (std::string("hunyuan_v1_dense: ") + label + " must be an object").c_str(),
                __FILE__, __LINE__);
        }
        return root;
    } catch (const std::exception& e) {
        return Status::invalid_argument(
            (std::string("hunyuan_v1_dense: invalid ") + label + ": " + e.what()).c_str(),
            __FILE__, __LINE__);
    }
}

void collect_eos_ids(const nlohmann::json& obj, std::vector<std::int32_t>& out) {
    if (!obj.is_object() || !obj.contains("eos_token_id")) return;
    const auto& eos = obj.at("eos_token_id");
    auto push_unique = [&](std::int32_t id) {
        if (id >= 0 && std::find(out.begin(), out.end(), id) == out.end()) {
            out.push_back(id);
        }
    };
    if (eos.is_number_integer()) {
        push_unique(eos.get<std::int32_t>());
    } else if (eos.is_array()) {
        for (const auto& item : eos) {
            if (item.is_number_integer()) {
                push_unique(item.get<std::int32_t>());
            }
        }
    }
}

}

Result<bool> is_hunyuan_dense_config(const std::string& model_dir) {
    const auto root = parse_json_object(model_dir + "/config.json", "config.json");
    if (!root.ok()) return root.status();
    const auto it = root.value().find("model_type");
    return it != root.value().end() && it->is_string() && *it == "hunyuan_v1_dense";
}

Result<Config> read_config(const std::string& model_dir) {
    const auto parsed = parse_json_object(model_dir + "/config.json", "config.json");
    if (!parsed.ok()) return parsed.status();
    const auto& root = parsed.value();
    try {
        if (!root.contains("model_type") || !root.at("model_type").is_string() ||
            root.at("model_type").get<std::string>() != "hunyuan_v1_dense") {
            return Status::unsupported(
                "hunyuan_v1_dense: wrong model_type", __FILE__, __LINE__);
        }
        if (root.contains("architectures") && !root.at("architectures").is_null()) {
            const auto& archs = root.at("architectures");
            if (!archs.is_array() || archs.empty()) {
                return Status::unsupported(
                    "hunyuan_v1_dense: invalid architectures declaration", __FILE__, __LINE__);
            }
            bool matched = false;
            for (const auto& item : archs) {
                if (item.is_string() &&
                    item.get<std::string>() == "HunYuanDenseV1ForCausalLM") {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                return Status::unsupported(
                    "hunyuan_v1_dense: unsupported architecture class", __FILE__, __LINE__);
            }
        }
        if (root.value("use_cla", false)) {
            return Status::unsupported(
                "hunyuan_v1_dense: cross-layer attention (use_cla) is not supported",
                __FILE__, __LINE__);
        }
        if (root.value("attention_bias", false) || root.value("bias", false) ||
            root.value("mlp_bias", false)) {
            return Status::unsupported(
                "hunyuan_v1_dense: attention or MLP bias is not supported",
                __FILE__, __LINE__);
        }
        if (root.value("add_classification_head", false)) {
            return Status::unsupported(
                "hunyuan_v1_dense: classification head is not supported",
                __FILE__, __LINE__);
        }
        if (root.contains("use_qk_norm") && root.at("use_qk_norm").is_boolean() &&
            !root.at("use_qk_norm").get<bool>()) {
            return Status::unsupported(
                "hunyuan_v1_dense: disabling QK RMSNorm is not supported",
                __FILE__, __LINE__);
        }
        if (root.contains("use_rotary_pos_emb") &&
            root.at("use_rotary_pos_emb").is_boolean() &&
            !root.at("use_rotary_pos_emb").get<bool>()) {
            return Status::unsupported(
                "hunyuan_v1_dense: disabling RoPE is not supported",
                __FILE__, __LINE__);
        }
        if (root.contains("hidden_act") && root.at("hidden_act").is_string() &&
            root.at("hidden_act").get<std::string>() != "silu") {
            return Status::unsupported(
                "hunyuan_v1_dense: only silu activation is supported",
                __FILE__, __LINE__);
        }
        if (root.contains("norm_type") && root.at("norm_type").is_string() &&
            root.at("norm_type").get<std::string>() != "rms") {
            return Status::unsupported(
                "hunyuan_v1_dense: only RMSNorm is supported",
                __FILE__, __LINE__);
        }
        if (root.contains("quantization_config") && !root.at("quantization_config").is_null()) {
            const auto& qcfg = root.at("quantization_config");
            if (!qcfg.is_object() ||
                qcfg.value("quant_method", std::string{}) != "exl3") {
                return Status::unsupported(
                    "hunyuan_v1_dense: unsupported quantization_config",
                    __FILE__, __LINE__);
            }
        }

        auto pos = [&](const char* key) -> Result<std::size_t> {
            const auto value = root.at(key).get<std::int64_t>();
            if (value <= 0) {
                return Status::invalid_argument(
                    "hunyuan_v1_dense: invalid positive integer parameter",
                    __FILE__, __LINE__);
            }
            return static_cast<std::size_t>(value);
        };

        Config c;
        auto layers = pos("num_hidden_layers");
        if (!layers.ok()) return layers.status();
        c.num_hidden_layers = layers.value();

        auto hidden = pos("hidden_size");
        if (!hidden.ok()) return hidden.status();
        c.hidden_size = hidden.value();

        auto q_heads = pos("num_attention_heads");
        if (!q_heads.ok()) return q_heads.status();
        c.num_attention_heads = q_heads.value();

        if (root.contains("num_key_value_heads")) {
            auto kv_heads = pos("num_key_value_heads");
            if (!kv_heads.ok()) return kv_heads.status();
            c.num_key_value_heads = kv_heads.value();
        } else {
            c.num_key_value_heads = c.num_attention_heads;
        }

        if (root.contains("head_dim") && !root.at("head_dim").is_null()) {
            auto hd = pos("head_dim");
            if (!hd.ok()) return hd.status();
            c.head_dim = hd.value();
        } else {
            if (c.hidden_size % c.num_attention_heads != 0) {
                return Status::invalid_argument(
                    "hunyuan_v1_dense: hidden_size not divisible by num_attention_heads",
                    __FILE__, __LINE__);
            }
            c.head_dim = c.hidden_size / c.num_attention_heads;
        }

        auto inter = pos("intermediate_size");
        if (!inter.ok()) return inter.status();
        c.intermediate_size = inter.value();

        auto vocab = pos("vocab_size");
        if (!vocab.ok()) return vocab.status();
        c.vocab_size = vocab.value();

        if (root.contains("max_position_embeddings")) {
            auto max_pos = pos("max_position_embeddings");
            if (!max_pos.ok()) return max_pos.status();
            c.max_position_embeddings = max_pos.value();
        } else {
            c.max_position_embeddings = 262144;
        }

        c.rms_norm_eps = root.at("rms_norm_eps").get<double>();
        c.rope_theta = root.value("rope_theta", 10000.0);
        c.tie_word_embeddings = root.value("tie_word_embeddings", false);

        if (c.num_attention_heads % c.num_key_value_heads != 0 ||
            (c.head_dim % 2u) != 0u || c.head_dim < 4u || (c.head_dim / 2u) > 128u ||
            !(c.rms_norm_eps > 0.0) || !std::isfinite(c.rms_norm_eps) ||
            !(c.rope_theta > 1.0) || !std::isfinite(c.rope_theta)) {
            return Status::invalid_argument(
                "hunyuan_v1_dense: invalid geometry or RoPE parameters",
                __FILE__, __LINE__);
        }

        if (root.contains("rope_scaling") && !root.at("rope_scaling").is_null()) {
            const auto& rs = root.at("rope_scaling");
            if (!rs.is_object()) {
                return Status::invalid_argument(
                    "hunyuan_v1_dense: rope_scaling must be an object",
                    __FILE__, __LINE__);
            }
            std::string rs_type = rs.value("type", std::string{});
            if (rs_type.empty()) rs_type = rs.value("rope_type", std::string{});
            c.rope_scaling.type = rs_type;
            if (rs.contains("factor") && rs.at("factor").is_number()) {
                c.rope_scaling.factor = rs.at("factor").get<double>();
            }
            if (rs_type == "dynamic") {
                if (!rs.contains("alpha") || !rs.at("alpha").is_number()) {
                    return Status::unsupported(
                        "hunyuan_v1_dense: dynamic rope_scaling without alpha is not supported",
                        __FILE__, __LINE__);
                }
                c.rope_scaling.alpha = rs.at("alpha").get<double>();
                if (!(c.rope_scaling.alpha > 0.0) || !std::isfinite(c.rope_scaling.alpha)) {
                    return Status::invalid_argument(
                        "hunyuan_v1_dense: invalid rope_scaling.alpha",
                        __FILE__, __LINE__);
                }
            } else if (!rs_type.empty() && rs_type != "default") {
                return Status::unsupported(
                    ("hunyuan_v1_dense: unsupported rope_scaling.type: " + rs_type).c_str(),
                    __FILE__, __LINE__);
            }
        }

        const std::string gen_path = model_dir + "/generation_config.json";
        if (std::filesystem::exists(gen_path)) {
            const auto gen = parse_json_object(gen_path, "generation_config.json");
            if (!gen.ok()) return gen.status();
            collect_eos_ids(gen.value(), c.stop_tokens);
        }
        if (c.stop_tokens.empty()) {
            collect_eos_ids(root, c.stop_tokens);
        }

        return c;
    } catch (const std::exception& e) {
        return Status::invalid_argument(
            (std::string("hunyuan_v1_dense: invalid config: ") + e.what()).c_str(),
            __FILE__, __LINE__);
    }
}

Result<shelf::runtime::EffectiveRopeProfile> resolve_rope_profile(
    const Config& config, const std::string& rope_factor) {
    if (rope_factor != "auto") {
        return Status::invalid_argument(
            "hunyuan_v1_dense: --rope-factor override is not supported",
            __FILE__, __LINE__);
    }
    shelf::runtime::EffectiveRopeProfile profile;
    profile.theta = config.rope_theta;
    profile.rotary_dim = static_cast<std::uint32_t>(config.head_dim);
    profile.float32_frequencies = true;
    if (config.rope_scaling.type == "dynamic" && config.rope_scaling.alpha > 0.0) {
        profile.type = shelf::runtime::EffectiveRopeProfile::Type::NtkAlpha;
        profile.alpha = config.rope_scaling.alpha;
    } else {
        profile.type = shelf::runtime::EffectiveRopeProfile::Type::Default;
    }
    return profile;
}

}
