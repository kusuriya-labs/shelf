#include <shelf/models/qwen35/model/qwen35_config.h>

#include <string>
#include <vector>

namespace shelf {
namespace qwen35 {

Result<Qwen35TextConfig> read_qwen35_text_config(const std::string& model_dir) {
    auto hybrid = shelf::qwen_hybrid::read_qwen_hybrid_config(model_dir);
    if (!hybrid.ok()) return hybrid.status();

    auto sizes = shelf::qwen_hybrid::read_text_config_sizes(
        model_dir, {"intermediate_size", "num_experts"});
    if (!sizes.ok()) return sizes.status();

    Qwen35TextConfig cfg;
    static_cast<shelf::qwen_hybrid::QwenHybridConfig&>(cfg) = hybrid.release();
    cfg.ffn.intermediate_size = sizes.value()[0];

    if (cfg.ffn.intermediate_size == 0) {
        return Status::invalid_argument(
            "read_qwen35_text_config: dense config requires intermediate_size",
            __FILE__, __LINE__);
    }
    // num_experts is the authority for a routed expert FFN: a checkpoint that declares
    // experts is not a dense model, and reading it as one would load an FFN that does not
    // exist there.
    if (sizes.value()[1] != 0) {
        return Status::invalid_argument(
            "read_qwen35_text_config: dense config must not declare num_experts",
            __FILE__, __LINE__);
    }
    // Backstop for the case the structure cannot show: a routed-expert identity whose
    // expert count is spelled in a way this build does not know would be accepted here
    // and lose its expert tensors. The identity vocabulary lives in the shared config, so
    // the dense family names no expert vocabulary of its own.
    if (cfg.identity_declares_routed_experts) {
        const std::string who = !cfg.declared_text_model_type.empty() ? cfg.declared_text_model_type
                                                                      : cfg.declared_model_type;
        return Status::unsupported(
            ("read_qwen35_text_config: identity declares a routed-expert model (model_type='" +
             who + "')")
                .c_str(),
            __FILE__, __LINE__);
    }
    return cfg;
}

}
}
