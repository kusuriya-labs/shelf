#pragma once
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>

#include <array>
#include <cstddef>
#include <string>

namespace ps::qwen35::eagle4 {

constexpr std::size_t kEagle4MaxAuxLayers = 8;
constexpr std::size_t kEagle4AuxLayers = 3;

struct Eagle4Config {
    std::string architecture;

    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t num_hidden_layers = 0;

    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t head_dim = 0;

    std::size_t sliding_window = 0;
    std::size_t target_hidden_size = 0;

    std::array<std::size_t, kEagle4MaxAuxLayers> aux_layer_ids{};
    std::size_t num_aux_layers = 0;

    std::size_t candidate_rank = 0;
    std::size_t num_candidates = 0;

    bool fc_norm = false;
    bool norm_output = false;

    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 10000000.0f;
    bool tie_word_embeddings = false;

    std::size_t attention_q_rows() const noexcept {
        return num_attention_heads * head_dim;
    }
    std::size_t attention_kv_rows() const noexcept {
        return num_key_value_heads * head_dim;
    }
    std::size_t tap_feature_size() const noexcept {
        return num_aux_layers * target_hidden_size;
    }
    std::size_t context_ring_bytes() const noexcept {
        return 2u * num_hidden_layers * sliding_window * attention_kv_rows() *
               sizeof(bf16_t);
    }
};

Result<Eagle4Config> read_eagle4_config(const std::string& model_dir);

}  // namespace ps::qwen35::eagle4
