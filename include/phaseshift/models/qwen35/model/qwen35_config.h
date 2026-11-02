#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/stop_tokens.h>
#include <cstddef>
#include <string>
#include <vector>

namespace ps {
namespace qwen35 {

struct Qwen35TextConfig {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t num_hidden_layers = 0;

    std::size_t full_attention_interval = 0;

    std::size_t linear_num_key_heads = 0;
    std::size_t linear_num_value_heads = 0;
    std::size_t linear_key_head_dim = 0;
    std::size_t linear_value_head_dim = 0;
    std::size_t linear_conv_kernel_dim = 0;

    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t attention_head_dim = 0;

    StopTokens stop_tokens;

    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 0.25f;

    bool tie_word_embeddings = false;
    bool output_gate_silu = false;

    std::vector<int> layer_types;
};

Result<Qwen35TextConfig> read_qwen35_text_config(const std::string& model_dir);

Result<bool> read_qwen35_tie_word_embeddings(const std::string& model_dir);

}
}
