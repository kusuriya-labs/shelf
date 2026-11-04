#include <shelf/models/qwen35/model/qwen35_config.h>

#include <string>
#include <vector>

namespace shelf {
namespace qwen35 {

Result<Qwen35TextConfig> read_qwen35_text_config(const std::string& model_dir) {
    auto hybrid = shelf::qwen_hybrid::read_qwen_hybrid_config(model_dir);
    if (!hybrid.ok()) return hybrid.status();

    auto sizes = shelf::qwen_hybrid::read_text_config_sizes(model_dir, {"intermediate_size"});
    if (!sizes.ok()) return sizes.status();

    Qwen35TextConfig cfg;
    static_cast<shelf::qwen_hybrid::QwenHybridConfig&>(cfg) = hybrid.release();
    cfg.ffn.intermediate_size = sizes.value()[0];
    return cfg;
}

}
}
