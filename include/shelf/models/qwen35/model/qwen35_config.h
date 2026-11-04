#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/config.h>

#include <cstddef>
#include <string>

namespace shelf {
namespace qwen35 {

// The dense FFN of one layer: one intermediate width shared by gate_proj, up_proj and
// down_proj.
struct DenseFfnConfig {
    std::size_t intermediate_size = 0;
};

struct Qwen35TextConfig : shelf::qwen_hybrid::QwenHybridConfig {
    DenseFfnConfig ffn;
};

Result<Qwen35TextConfig> read_qwen35_text_config(const std::string& model_dir);

}
}
