#pragma once

#include <cstdint>

namespace shelf::runtime {

constexpr uint32_t kv_pages_for_context(uint32_t max_seq_len, uint32_t page_tokens) noexcept {
    if (page_tokens == 0u) return 0u;
    const uint64_t context = static_cast<uint64_t>(max_seq_len) + 1u;
    const uint64_t pages = (context + page_tokens - 1u) / page_tokens + 1u;
    return pages > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(pages);
}

}
