#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/core/memory/types.h>
#include <array>
#include <cstddef>
#include <string>

namespace ps {
namespace qwen35 {
namespace dflash2 {

constexpr std::size_t kMaxTargetLayerIds = 8;
constexpr std::size_t kMaxBlockSize = 16;

struct DFlash2Config {
    std::string architecture;

    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t num_hidden_layers = 0;

    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t head_dim = 0;

    std::size_t sliding_window = 0;
    bool is_causal = true;

    std::size_t block_size = 0;
    std::size_t conv_group_size = 0;
    std::size_t conv_kernel_size = 0;
    std::size_t mask_token_id = 0;
    std::size_t selector_rank = 0;
    std::size_t selector_top_k = 0;

    std::size_t num_target_layers = 0;
    std::array<std::size_t, kMaxTargetLayerIds> target_layer_ids{};
    std::size_t num_target_layer_ids = 0;

    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 10000000.0f;
    bool tie_word_embeddings = false;

    std::size_t max_draft_tokens() const noexcept {
        return block_size == 0 ? 0 : block_size - 1;
    }
    std::size_t conv_groups() const noexcept {
        return conv_group_size == 0 ? 0 : hidden_size / conv_group_size;
    }
    std::size_t conv_projection_rows() const noexcept {
        return 2 * conv_kernel_size * conv_groups();
    }
    std::size_t attention_q_rows() const noexcept {
        return num_attention_heads * head_dim;
    }
    std::size_t attention_kv_rows() const noexcept {
        return num_key_value_heads * head_dim;
    }
    std::size_t context_ring_bytes() const noexcept {
        return 2u * num_hidden_layers * sliding_window * attention_kv_rows() *
               sizeof(bf16_t);
    }
    std::size_t tap_feature_size() const noexcept {
        return num_target_layer_ids * hidden_size;
    }
};

Result<DFlash2Config> read_dflash2_config(const std::string& model_dir);

}  // namespace dflash2
}  // namespace qwen35
}  // namespace ps
