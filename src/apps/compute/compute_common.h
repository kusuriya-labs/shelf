#pragma once

#include <shelf/models/qwen35/runtime/runtime_request.h>
#include <shelf/models/qwen35/model/qwen35_config.h>

#include <cstdint>

namespace shelf::app {

inline uint32_t count_attention_layers(const shelf::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt == 1) ++n;
    }
    return n;
}

inline const char* finish_reason_name(shelf::qwen35::runtime::FinishReason reason) {
    using shelf::qwen35::runtime::FinishReason;
    switch (reason) {
        case FinishReason::Eos: return "stop";
        case FinishReason::MaxNewTokens: return "length";
        case FinishReason::Cancelled: return "cancelled";
        case FinishReason::Error: return "error";
        case FinishReason::None: return "none";
    }
    return "none";
}

}  // namespace shelf::app
