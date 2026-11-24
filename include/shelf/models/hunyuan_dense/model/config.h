#pragma once

#include <shelf/core/status.h>
#include <shelf/runtime/rope_profile.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace shelf::hunyuan_dense {

struct RopeScalingConfig {
    std::string type;
    double alpha = 0.0;
    double factor = 1.0;
};

struct Config {
    std::size_t num_hidden_layers = 0;
    std::size_t hidden_size = 0;
    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t head_dim = 0;
    std::size_t intermediate_size = 0;
    std::size_t vocab_size = 0;
    std::size_t max_position_embeddings = 0;
    double rms_norm_eps = 0.0;
    double rope_theta = 10000.0;
    bool tie_word_embeddings = false;
    RopeScalingConfig rope_scaling;
    std::vector<std::int32_t> stop_tokens;
};

Result<bool> is_hunyuan_dense_config(const std::string& model_dir);
Result<Config> read_config(const std::string& model_dir);
Result<shelf::runtime::EffectiveRopeProfile> resolve_rope_profile(
    const Config& config, const std::string& rope_factor = "auto");

}
