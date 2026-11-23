#pragma once

#include <shelf/runtime/runtime_request.h>

namespace shelf::app {

inline const char* finish_reason_name(shelf::runtime::FinishReason reason) {
    using shelf::runtime::FinishReason;
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
