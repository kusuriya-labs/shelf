#pragma once

#include <shelf/core/status.h>
#include <cstdint>
#include <unordered_map>

namespace shelf {
namespace scheduling {

enum class KVAdmissionPolicy : uint8_t {
    ConservativeFullClaim,
    BankerSafe,
};

struct KVCapacitySnapshot {
    uint32_t total_pages = 0;
    uint32_t physical_used_pages = 0;
    uint32_t physical_free_pages = 0;

    uint64_t max_claim_pages = 0;
    uint64_t remaining_need_pages = 0;
    uint64_t conservative_headroom_pages = 0;
    uint64_t overcommit_pages = 0;

    uint32_t active_requests = 0;
    bool banker_safe = false;
};

Result<uint32_t> compute_max_kv_tokens(
    uint32_t prompt_tokens,
    uint32_t max_new_tokens);

uint32_t kv_pages_for_tokens(uint32_t tokens, uint32_t page_tokens);

uint32_t additional_kv_pages(
    uint32_t position,
    uint32_t current_blocks,
    uint32_t append_tokens,
    uint32_t page_tokens);

class KVCapacityManager {
 public:
    explicit KVCapacityManager(uint32_t total_pages) : total_pages_(total_pages) {}

    Status register_claim(uint64_t request_id, uint32_t pages);

    Status release_claim(uint64_t request_id);

    bool can_admit(uint32_t pages) const noexcept {
        return static_cast<uint64_t>(pages) <= conservative_headroom_pages();
    }

    uint32_t total_pages() const noexcept { return total_pages_; }
    uint64_t max_claim_pages() const noexcept { return claimed_pages_; }
    uint64_t conservative_headroom_pages() const noexcept {
        const uint64_t total = total_pages_;
        return claimed_pages_ > total ? 0 : total - claimed_pages_;
    }

 private:
    uint32_t total_pages_;
    uint64_t claimed_pages_ = 0;
    std::unordered_map<uint64_t, uint32_t> claims_;
};

}
}
