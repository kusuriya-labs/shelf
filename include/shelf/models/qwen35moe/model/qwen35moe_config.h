#pragma once
#include <shelf/models/qwen35/model/qwen35_config.h>

#include <cstddef>
#include <string>

namespace shelf::qwen35moe {

// Qwen3.5 MoE (`model_type = "qwen3_5_moe"`) text config. The base geometry is read by
// the shared Qwen35 reader; this adds the routed / shared expert geometry and is the
// type the MoE weight plan and loader take. num_experts is the authority for MoE: the
// model_type string is never used to branch at runtime.
struct Qwen35MoeTextConfig : shelf::qwen35::Qwen35TextConfig {
    std::size_t num_experts = 0;
    std::size_t num_experts_per_tok = 0;
    std::size_t moe_intermediate_size = 0;
    std::size_t shared_expert_intermediate_size = 0;
};

Result<Qwen35MoeTextConfig> read_qwen35_moe_text_config(const std::string& model_dir);

}  // namespace shelf::qwen35moe
