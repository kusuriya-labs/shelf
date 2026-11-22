#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/qwen35/model/rope_profile.h>

#include <cstdio>
#include <cstdint>
#include <optional>
#include <string>

namespace shelf::bench {

inline shelf::Result<std::optional<shelf::qwen35::EffectiveRopeProfile>>
resolve_bench_rope_profile(const std::string& model_dir, const std::string& factor_text) {
    const shelf::Status identity = shelf::qwen_hybrid::check_config_identity(model_dir);
    if (!identity.ok()) {
        if (factor_text == "auto") {
            return std::optional<shelf::qwen35::EffectiveRopeProfile>{};
        }
        return shelf::Status::unsupported(
            "--rope-factor requires a supported qwen checkpoint", __FILE__, __LINE__);
    }
    auto selection = shelf::qwen35::parse_rope_factor_selection(factor_text);
    if (!selection.ok()) return selection.status();
    auto params = shelf::qwen35::read_qwen35_rope_parameters(model_dir);
    if (!params.ok()) return params.status();
    auto profile = shelf::qwen35::resolve_effective_rope_profile(
        params.value(), selection.value());
    if (!profile.ok()) return profile.status();
    return std::optional<shelf::qwen35::EffectiveRopeProfile>(profile.release());
}

inline shelf::Status check_bench_rope_len(
    const shelf::qwen35::EffectiveRopeProfile& profile,
    const std::string& factor_text,
    uint32_t required_seq_len) {
    const uint32_t cap = shelf::qwen35::rope_max_seq_len_cap(profile);
    if (cap == 0u || required_seq_len <= cap) return shelf::Status::make_ok();
    return shelf::Status::invalid_argument(
        ("required seq len " + std::to_string(required_seq_len) +
         " exceeds the rope factor limit " + std::to_string(cap) +
         " (--rope-factor " + factor_text + ", model original length " +
         std::to_string(profile.original_max_position_embeddings) + ")")
            .c_str(),
        __FILE__, __LINE__);
}

}
