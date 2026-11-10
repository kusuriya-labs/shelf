#pragma once

#include <shelf/core/status.h>

#include <cstddef>
#include <string>

namespace shelf::plamo3 {

struct RopeProfile {
    double theta = 0;
    double factor = 1;
    std::size_t original_max_position_embeddings = 0;
    double beta_fast = 32;
    double beta_slow = 1;
    bool truncate = false;
};

struct Config {
    std::size_t num_hidden_layers = 0;
    std::size_t hidden_size = 0;
    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t head_dim = 0;
    std::size_t intermediate_size = 0;
    std::size_t vocab_size = 0;
    std::size_t window_size = 0;
    std::size_t sliding_window_pattern = 0;
    std::size_t max_position_embeddings = 0;
    double rms_norm_eps = 0;
    RopeProfile local_rope;
    RopeProfile full_rope;

    bool is_full_attention(std::size_t layer) const {
        return (layer + 1) % sliding_window_pattern == 0;
    }
};

Result<bool> is_plamo3_config(const std::string& model_dir);
Result<Config> read_config(const std::string& model_dir);

}
