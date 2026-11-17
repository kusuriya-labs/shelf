#pragma once

#include <shelf/models/common/qwen_hybrid/stop_tokens.h>

#include <cstdint>

namespace shelf {
namespace runtime {

struct DFlash2SpecDecoderConfig {
    uint32_t num_drafts = 7u;
    bool tree = false;
    uint32_t tree_extra = 0u;
    uint32_t tree_budget = 0u;
    ::shelf::qwen_hybrid::StopTokens eos_tokens;
};

}  // namespace runtime
}  // namespace shelf
