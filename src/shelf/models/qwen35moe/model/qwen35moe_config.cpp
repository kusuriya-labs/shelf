#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

#include <string>
#include <vector>

namespace shelf::qwen35moe {

Result<std::size_t> read_declared_num_experts(const std::string& model_dir) {
    Status identity = shelf::qwen_hybrid::check_config_identity(model_dir);
    if (!identity.ok()) return identity;
    auto sizes = shelf::qwen_hybrid::read_text_config_sizes(model_dir, {"num_experts"});
    if (!sizes.ok()) return sizes.status();
    return sizes.value()[0];
}

Result<Qwen35MoeTextConfig> read_qwen35_moe_text_config(const std::string& model_dir) {
    auto hybrid = shelf::qwen_hybrid::read_qwen_hybrid_config(model_dir);
    if (!hybrid.ok()) return hybrid.status();

    auto sizes = shelf::qwen_hybrid::read_text_config_sizes(
        model_dir, {"num_experts", "num_experts_per_tok", "moe_intermediate_size",
                    "shared_expert_intermediate_size"});
    if (!sizes.ok()) return sizes.status();

    auto has_dense_ffn = shelf::qwen_hybrid::has_text_config_key(model_dir, "intermediate_size");
    if (!has_dense_ffn.ok()) return has_dense_ffn.status();


    Qwen35MoeTextConfig cfg;
    static_cast<shelf::qwen_hybrid::QwenHybridConfig&>(cfg) = hybrid.release();
    cfg.ffn.num_experts = sizes.value()[0];
    cfg.ffn.num_experts_per_tok = sizes.value()[1];
    cfg.ffn.moe_intermediate_size = sizes.value()[2];
    cfg.ffn.shared_expert_intermediate_size = sizes.value()[3];

    if (cfg.ffn.num_experts == 0) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: num_experts is required (config is not MoE)",
            __FILE__, __LINE__);
    }
    if (cfg.ffn.num_experts_per_tok == 0 ||
        cfg.ffn.num_experts_per_tok > cfg.ffn.num_experts) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: num_experts_per_tok must be in [1, num_experts]",
            __FILE__, __LINE__);
    }
    if (cfg.ffn.moe_intermediate_size == 0 || cfg.ffn.shared_expert_intermediate_size == 0) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: moe config requires moe_intermediate_size and "
            "shared_expert_intermediate_size",
            __FILE__, __LINE__);
    }
    // A MoE model has no dense FFN: an intermediate_size here means the config mixes
    // the two topologies and the lowering contract does not define it.
    if (has_dense_ffn.value()) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: moe config must not declare intermediate_size",
            __FILE__, __LINE__);
    }
    return cfg;
}

}  // namespace shelf::qwen35moe
