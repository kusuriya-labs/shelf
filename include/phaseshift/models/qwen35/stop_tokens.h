#pragma once

#include <cstdint>
#include <vector>

namespace ps {
namespace qwen35 {

// Generation stop tokens for one model.
//
// A Qwen3.5/3.8 assistant turn ends with <|im_end|> and a document may end with
// <|endoftext|>; `generation_config.json` lists both. Generation stops when any
// listed token is produced, matching the transformers generation contract.
using StopTokens = std::vector<int32_t>;

inline bool is_stop_token(const StopTokens& stop_tokens, int32_t token) {
    for (const int32_t id : stop_tokens) {
        if (id >= 0 && id == token) {
            return true;
        }
    }
    return false;
}

// Primary stop token, used when a single token must be emitted (for example
// when a constraint terminates a request). Returns -1 when none is configured.
inline int32_t primary_stop_token(const StopTokens& stop_tokens) {
    return stop_tokens.empty() ? -1 : stop_tokens.front();
}

}  // namespace qwen35
}  // namespace ps
