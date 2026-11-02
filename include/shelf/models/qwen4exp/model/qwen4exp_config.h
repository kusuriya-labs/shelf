#pragma once

#include <shelf/core/status.h>
#include <shelf/models/qwen35/stop_tokens.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ps::qwen4exp {

struct Qwen4ExpTextConfig {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t num_hidden_layers = 0;

    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t head_dim = 0;
    std::size_t rotary_dim = 0;

    std::size_t full_attention_interval = 0;

    std::size_t linear_num_key_heads = 0;
    std::size_t linear_num_value_heads = 0;
    std::size_t linear_key_head_dim = 0;
    std::size_t linear_value_head_dim = 0;
    std::size_t linear_conv_kernel_dim = 0;

    std::size_t num_experts = 0;
    std::size_t num_experts_per_tok = 0;
    std::size_t moe_intermediate_size = 0;
    std::size_t shared_expert_intermediate_size = 0;

    std::size_t hc_count = 0;
    std::size_t hc_lowrank = 0;

    std::size_t ngram_size = 0;
    std::size_t heads_per_ngram = 0;
    std::size_t ngram_vocab_size_base = 0;
    std::size_t make_ngram_vocab_size_divisible_by = 0;
    std::size_t split_ngram_parts = 0;
    std::uint64_t ngram_seed = 1234;
    std::size_t ple_layer_index = 0;
    std::size_t ple_slot = 0;
    std::size_t ple_embed_dim = 0;
    std::size_t ple_conv_kernel_size = 0;

    std::size_t indexer_n_heads = 0;
    std::size_t indexer_kv_heads = 0;
    std::size_t indexer_head_dim = 0;
    std::size_t indexer_budget = 0;
    std::size_t indexer_compress_ratio = 0;
    std::size_t block_topk = 0;

    qwen35::StopTokens stop_tokens;

    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 0.25f;

    bool tie_word_embeddings = false;
    bool output_gate_silu = false;
    bool norm_topk_prob = true;

    std::vector<int> layer_types;

    bool layer_is_qsa(std::size_t layer_index) const noexcept;
};

Result<Qwen4ExpTextConfig> read_qwen4exp_text_config(const std::string& model_dir);

}  // namespace ps::qwen4exp
