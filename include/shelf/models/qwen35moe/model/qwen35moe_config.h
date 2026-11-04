#pragma once
#include <shelf/models/common/qwen_hybrid/config.h>

#include <cstddef>
#include <string>

namespace shelf::qwen35moe {

// The FFN of one MoE layer: routed experts of one intermediate width, a top_k selection
// width, and a shared expert of its own width. num_experts is the authority for MoE:
// the model_type string is never used to branch at runtime.
struct MoeFfnConfig {
    std::size_t num_experts = 0;
    std::size_t num_experts_per_tok = 0;
    std::size_t moe_intermediate_size = 0;
    std::size_t shared_expert_intermediate_size = 0;
};

// Qwen3.5 MoE (`model_type = "qwen3_5_moe"`) text config: the shared Qwen hybrid
// geometry plus the FFN section above. It carries no dense intermediate size, and it is
// not related to shelf::qwen35::Qwen35TextConfig. This is the type the MoE weight plan
// and loaders take.
struct Qwen35MoeTextConfig : shelf::qwen_hybrid::QwenHybridConfig {
    MoeFfnConfig ffn;
};

Result<Qwen35MoeTextConfig> read_qwen35_moe_text_config(const std::string& model_dir);

// num_experts as config.json declares it, without validating any other MoE field. Zero
// means the model is dense. This is the probe a loader above both families uses to pick
// a path; it fails only when config.json cannot be read at all.
Result<std::size_t> read_declared_num_experts(const std::string& model_dir);

}  // namespace shelf::qwen35moe
