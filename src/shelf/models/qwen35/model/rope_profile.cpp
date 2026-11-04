#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

namespace shelf {
namespace qwen35 {

namespace {

constexpr double kFallbackTheta = 10000000.0;
constexpr double kFallbackPartial = 0.25;
constexpr std::uint64_t kOfficialMaxContextTokens = 1000000ull;

bool finite_positive(double v) {
    return std::isfinite(v) && v > 0.0;
}

}

Result<Qwen35RopeParameters> read_qwen35_rope_parameters(const std::string& model_dir) {
    Status identity = shelf::qwen_hybrid::check_config_identity(model_dir);
    if (!identity.ok()) return identity;
    std::ifstream f(model_dir + "/config.json");
    if (!f.is_open()) {
        return Status::invalid_argument(
            "read_qwen35_rope_parameters: cannot open config.json", __FILE__, __LINE__);
    }
    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("read_qwen35_rope_parameters: bad config.json: " + std::string(e.what())).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object() || !root.contains("text_config") || !root["text_config"].is_object()) {
        return Status::invalid_argument(
            "read_qwen35_rope_parameters: config.json missing text_config", __FILE__, __LINE__);
    }
    const nlohmann::json& t = root["text_config"];

    Qwen35RopeParameters params;
    if (t.contains("head_dim") && t["head_dim"].is_number()) {
        const double v = t["head_dim"].get<double>();
        if (!std::isfinite(v) || v <= 0.0 || v > 65535.0) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: invalid head_dim", __FILE__, __LINE__);
        }
        params.head_dim = static_cast<std::uint32_t>(v);
    }
    if (t.contains("max_position_embeddings") && t["max_position_embeddings"].is_number()) {
        const double v = t["max_position_embeddings"].get<double>();
        if (!std::isfinite(v) || v <= 0.0 || v > 4294967295.0) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: invalid max_position_embeddings", __FILE__, __LINE__);
        }
        params.max_position_embeddings = static_cast<std::uint32_t>(v);
    }

    if (!t.contains("rope_parameters") || !t["rope_parameters"].is_object()) {
        return params;
    }
    const nlohmann::json& rp = t["rope_parameters"];

    if (rp.contains("rope_type")) {
        if (!rp["rope_type"].is_string()) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: rope_type must be a string", __FILE__, __LINE__);
        }
        params.rope_type = rp["rope_type"].get<std::string>();
        if (params.rope_type != "default" && params.rope_type != "yarn") {
            return Status::unsupported(
                ("read_qwen35_rope_parameters: unsupported rope_type '" + params.rope_type + "'")
                    .c_str(),
                __FILE__, __LINE__);
        }
    }
    if (rp.contains("rope_theta")) {
        if (!rp["rope_theta"].is_number()) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: rope_theta must be a number", __FILE__, __LINE__);
        }
        params.theta = rp["rope_theta"].get<double>();
        if (!finite_positive(params.theta)) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: rope_theta must be finite and positive",
                __FILE__, __LINE__);
        }
    }
    if (rp.contains("partial_rotary_factor")) {
        if (!rp["partial_rotary_factor"].is_number()) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: partial_rotary_factor must be a number",
                __FILE__, __LINE__);
        }
        params.partial_rotary_factor = rp["partial_rotary_factor"].get<double>();
    } else if (t.contains("partial_rotary_factor") && t["partial_rotary_factor"].is_number()) {
        params.partial_rotary_factor = t["partial_rotary_factor"].get<double>();
    }
    if (!std::isfinite(params.partial_rotary_factor) || params.partial_rotary_factor <= 0.0 ||
        params.partial_rotary_factor > 1.0) {
        return Status::invalid_argument(
            "read_qwen35_rope_parameters: partial_rotary_factor must be in (0, 1]",
            __FILE__, __LINE__);
    }

    if (rp.contains("factor")) {
        if (!rp["factor"].is_number()) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: factor must be a number", __FILE__, __LINE__);
        }
        params.factor = rp["factor"].get<double>();
        params.has_factor = true;
        if (!finite_positive(params.factor)) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: factor must be finite and positive", __FILE__,
                __LINE__);
        }
    }
    if (rp.contains("original_max_position_embeddings")) {
        if (!rp["original_max_position_embeddings"].is_number()) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: original_max_position_embeddings must be a number",
                __FILE__, __LINE__);
        }
        const double v = rp["original_max_position_embeddings"].get<double>();
        if (!std::isfinite(v) || v <= 0.0 || v > 4294967295.0) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: invalid original_max_position_embeddings",
                __FILE__, __LINE__);
        }
        params.original_max_position_embeddings = static_cast<std::uint32_t>(v);
    }
    if (rp.contains("beta_fast")) {
        if (!rp["beta_fast"].is_number()) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: beta_fast must be a number", __FILE__, __LINE__);
        }
        params.beta_fast = rp["beta_fast"].get<double>();
        if (!finite_positive(params.beta_fast)) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: beta_fast must be finite and positive", __FILE__,
                __LINE__);
        }
    }
    if (rp.contains("beta_slow")) {
        if (!rp["beta_slow"].is_number()) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: beta_slow must be a number", __FILE__, __LINE__);
        }
        params.beta_slow = rp["beta_slow"].get<double>();
        if (!finite_positive(params.beta_slow)) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: beta_slow must be finite and positive", __FILE__,
                __LINE__);
        }
    }
    if (rp.contains("truncate")) {
        if (!rp["truncate"].is_boolean()) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: truncate must be a boolean", __FILE__, __LINE__);
        }
        params.truncate = rp["truncate"].get<bool>();
    }
    if (rp.contains("attention_factor")) {
        if (!rp["attention_factor"].is_number()) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: attention_factor must be a number", __FILE__,
                __LINE__);
        }
        params.attention_factor = rp["attention_factor"].get<double>();
        params.has_attention_factor = true;
        if (!finite_positive(params.attention_factor)) {
            return Status::invalid_argument(
                "read_qwen35_rope_parameters: attention_factor must be finite and positive",
                __FILE__, __LINE__);
        }
    }
    if (rp.contains("mscale") || rp.contains("mscale_all_dim")) {
        return Status::unsupported(
            "read_qwen35_rope_parameters: mscale / mscale_all_dim are not supported",
            __FILE__, __LINE__);
    }
    return params;
}

Result<RopeFactorSelection> parse_rope_factor_selection(const std::string& text) {
    if (text == "auto") return RopeFactorSelection::Auto;
    if (text == "1") return RopeFactorSelection::One;
    if (text == "2") return RopeFactorSelection::Two;
    if (text == "4") return RopeFactorSelection::Four;
    return Status::invalid_argument(
        "parse_rope_factor_selection: expected auto | 1 | 2 | 4", __FILE__, __LINE__);
}

Result<EffectiveRopeProfile> resolve_effective_rope_profile(
    const Qwen35RopeParameters& params, RopeFactorSelection selection) {
    if (!finite_positive(params.theta)) {
        return Status::invalid_argument("resolve_effective_rope_profile: invalid theta", __FILE__,
                                        __LINE__);
    }
    if (params.head_dim == 0) {
        return Status::invalid_argument("resolve_effective_rope_profile: head_dim missing",
                                        __FILE__, __LINE__);
    }
    const double rotary = static_cast<double>(params.head_dim) * params.partial_rotary_factor;
    if (!std::isfinite(rotary) || rotary < 2.0 || rotary > static_cast<double>(params.head_dim)) {
        return Status::invalid_argument(
            "resolve_effective_rope_profile: rotary dim out of range", __FILE__, __LINE__);
    }
    const auto rotary_dim = static_cast<std::uint32_t>(rotary);
    if (rotary != static_cast<double>(rotary_dim)) {
        return Status::invalid_argument(
            "resolve_effective_rope_profile: rotary dim must be an integer", __FILE__,
            __LINE__);
    }
    if (rotary_dim % 2u != 0u) {
        return Status::invalid_argument(
            "resolve_effective_rope_profile: rotary dim must be even", __FILE__, __LINE__);
    }

    std::uint32_t original_len = params.original_max_position_embeddings != 0
        ? params.original_max_position_embeddings
        : params.max_position_embeddings;

    bool use_yarn = false;
    double factor = 1.0;
    if (selection == RopeFactorSelection::Auto) {
        if (params.rope_type == "yarn") {
            use_yarn = true;
            factor = params.has_factor ? params.factor : 0.0;
        }
    } else if (selection == RopeFactorSelection::One) {
        use_yarn = false;
        factor = 1.0;
    } else {
        use_yarn = true;
        factor = selection == RopeFactorSelection::Two ? 2.0 : 4.0;
    }

    EffectiveRopeProfile profile;
    profile.theta = params.theta;
    profile.rotary_dim = rotary_dim;
    profile.from_cli = selection != RopeFactorSelection::Auto;
    if (!use_yarn) {
        profile.type = EffectiveRopeProfile::Type::Default;
        profile.factor = 1.0;
        profile.attention_factor = 1.0;
        profile.original_max_position_embeddings = original_len;
        return profile;
    }

    if (!finite_positive(factor) || factor <= 1.0) {
        return Status::invalid_argument(
            "resolve_effective_rope_profile: yarn requires factor > 1", __FILE__, __LINE__);
    }
    if (original_len == 0) {
        return Status::invalid_argument(
            "resolve_effective_rope_profile: yarn requires original_max_position_embeddings or "
            "max_position_embeddings",
            __FILE__, __LINE__);
    }
    if (!std::isfinite(params.beta_fast) || params.beta_fast <= 0.0 ||
        !std::isfinite(params.beta_slow) || params.beta_slow <= 0.0) {
        return Status::invalid_argument(
            "resolve_effective_rope_profile: beta_fast / beta_slow must be finite and positive",
            __FILE__, __LINE__);
    }

    profile.type = EffectiveRopeProfile::Type::Yarn;
    profile.factor = factor;
    profile.original_max_position_embeddings = original_len;
    profile.beta_fast = params.beta_fast;
    profile.beta_slow = params.beta_slow;
    profile.truncate = params.truncate;
    profile.attention_factor =
        params.has_attention_factor ? params.attention_factor : 1.0 + 0.1 * std::log(factor);
    if (!std::isfinite(profile.attention_factor) || profile.attention_factor <= 0.0) {
        return Status::invalid_argument(
            "resolve_effective_rope_profile: attention_factor must be finite and positive",
            __FILE__, __LINE__);
    }
    return profile;
}

std::uint32_t rope_max_seq_len_cap(const EffectiveRopeProfile& profile) {
    if (profile.original_max_position_embeddings == 0) return 0;
    if (profile.type != EffectiveRopeProfile::Type::Yarn ||
        !std::isfinite(profile.factor) || profile.factor <= 1.0) {
        return profile.original_max_position_embeddings;
    }
    const std::uint64_t extended =
        static_cast<std::uint64_t>(static_cast<double>(profile.original_max_position_embeddings) *
                                   profile.factor);
    const std::uint64_t capped = extended > kOfficialMaxContextTokens
        ? kOfficialMaxContextTokens
        : extended;
    return static_cast<std::uint32_t>(capped);
}

}
}
