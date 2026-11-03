#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

#include <fstream>
#include <nlohmann/json.hpp>

namespace shelf::qwen35moe {

Result<Qwen35MoeTextConfig> read_qwen35_moe_text_config(const std::string& model_dir) {
    auto base = shelf::qwen35::read_qwen35_text_config(model_dir);
    if (!base.ok())
        return base.status();

    std::ifstream f(model_dir + "/config.json");
    if (!f.is_open()) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: cannot open config.json", __FILE__, __LINE__);
    }
    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("read_qwen35_moe_text_config: bad config.json: " + std::string(e.what())).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object() || !root.contains("text_config")) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: config.json missing text_config", __FILE__, __LINE__);
    }
    const nlohmann::json& t = root["text_config"];

    Qwen35MoeTextConfig cfg;
    static_cast<shelf::qwen35::Qwen35TextConfig&>(cfg) = base.value();

    auto read_size = [&](const char* key, std::size_t& out) {
        if (t.contains(key) && t[key].is_number())
            out = t[key].get<std::size_t>();
    };
    read_size("num_experts", cfg.num_experts);
    read_size("num_experts_per_tok", cfg.num_experts_per_tok);
    read_size("moe_intermediate_size", cfg.moe_intermediate_size);
    read_size("shared_expert_intermediate_size", cfg.shared_expert_intermediate_size);

    if (cfg.num_experts == 0) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: num_experts is required (config is not MoE)",
            __FILE__, __LINE__);
    }
    if (cfg.num_experts_per_tok == 0 || cfg.num_experts_per_tok > cfg.num_experts) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: num_experts_per_tok must be in [1, num_experts]",
            __FILE__, __LINE__);
    }
    if (cfg.moe_intermediate_size == 0 || cfg.shared_expert_intermediate_size == 0) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: moe config requires moe_intermediate_size and "
            "shared_expert_intermediate_size",
            __FILE__, __LINE__);
    }
    // A MoE model has no dense FFN: an intermediate_size here means the config mixes
    // the two topologies and the lowering contract does not define it.
    if (cfg.intermediate_size != 0) {
        return Status::invalid_argument(
            "read_qwen35_moe_text_config: moe config must not declare intermediate_size",
            __FILE__, __LINE__);
    }
    // The MoE checkpoint carries no output_gate_type key, and the reference applies
    // silu to the GDN output gate unconditionally. Only an explicit "sigmoid" selects
    // the other form.
    if (!(t.contains("output_gate_type") && t["output_gate_type"].is_string()))
        cfg.output_gate_silu = true;

    return cfg;
}

}  // namespace shelf::qwen35moe
