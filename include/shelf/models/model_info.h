#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace shelf::models {

class LoadedModel;

// The values the compute application prints at startup. The family branch that used to
// live in the application is resolved here, next to the model that owns it.
struct ModelInfo {
    std::string family_name;
    std::uint64_t hidden_size = 0;
    std::uint64_t num_hidden_layers = 0;
    std::uint64_t num_attention_heads = 0;
    std::uint64_t num_key_value_heads = 0;
    std::uint64_t attention_head_dim = 0;
    std::uint64_t linear_num_key_heads = 0;
    std::uint64_t vocab_size = 0;

    bool has_intermediate_size = false;
    std::uint64_t intermediate_size = 0;

    bool has_experts = false;
    std::uint64_t num_experts = 0;
    std::uint64_t num_experts_per_tok = 0;
    std::uint64_t moe_intermediate_size = 0;
    std::uint64_t shared_expert_intermediate_size = 0;

    std::uint64_t gdn_layers = 0;
    std::uint64_t attention_layers = 0;
    std::uint64_t gdn_num_value_heads = 0;
    std::uint64_t gdn_key_head_dim = 0;
    std::uint64_t gdn_value_head_dim = 0;
    std::uint64_t gdn_conv_dim = 0;
    std::uint64_t gdn_conv_history = 0;
};

ModelInfo make_model_info(const LoadedModel& model);

// The weight-encoding breakdown reported by SHELF_MEM_DEBUG. A model whose FFN runs on
// grouped expert rows reports only the shared dense matrices; its resident bytes are read
// from the arena instead.
struct DenseWeightBytes {
    std::uint64_t bf16_bytes = 0;
    std::uint64_t fp8_code_bytes = 0;
    std::uint64_t fp8_scale_bytes = 0;
    std::uint64_t w4_code_bytes = 0;
    std::uint64_t w4_storage_scale_bytes = 0;
    std::uint64_t w4_compute_scale_bytes = 0;
    std::uint64_t w8_code_bytes = 0;
    std::uint64_t w8_storage_scale_bytes = 0;
    std::uint64_t w8_compute_scale_bytes = 0;
};

bool has_dense_weight_view(const LoadedModel& model) noexcept;

DenseWeightBytes dense_weight_bytes(const LoadedModel& model);

}  // namespace shelf::models
