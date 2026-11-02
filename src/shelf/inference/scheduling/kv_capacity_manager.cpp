#include <shelf/inference/scheduling/kv_capacity_manager.h>
#include <climits>

namespace shelf {
namespace scheduling {

Result<uint32_t> compute_max_kv_tokens(
    uint32_t prompt_tokens,
    uint32_t max_new_tokens) {
    const uint64_t generated =
        max_new_tokens > 0 ? static_cast<uint64_t>(max_new_tokens) - 1 : 0;
    const uint64_t total = static_cast<uint64_t>(prompt_tokens) + generated;
    if (total > UINT32_MAX) {
        return Status::overflow("max_kv_tokens exceeds uint32 range", __FILE__, __LINE__);
    }
    return static_cast<uint32_t>(total);
}

uint32_t kv_pages_for_tokens(uint32_t tokens, uint32_t page_tokens) {
    if (page_tokens == 0) {
        return 0;
    }
    return (tokens + page_tokens - 1) / page_tokens;
}

uint32_t additional_kv_pages(
    uint32_t position,
    uint32_t current_blocks,
    uint32_t append_tokens,
    uint32_t page_tokens) {
    if (append_tokens == 0 || page_tokens == 0) {
        return 0;
    }
    const uint32_t end_position = position + append_tokens;
    const uint32_t required_blocks = (end_position + page_tokens - 1) / page_tokens;
    return required_blocks > current_blocks ? required_blocks - current_blocks : 0;
}

Status KVCapacityManager::register_claim(uint64_t request_id, uint32_t pages) {
    auto it = claims_.find(request_id);
    if (it != claims_.end()) {
        return Status::invalid_state("claim already registered", __FILE__, __LINE__);
    }
    claims_[request_id] = pages;
    claimed_pages_ += pages;
    return Status::make_ok();
}

Status KVCapacityManager::release_claim(uint64_t request_id) {
    auto it = claims_.find(request_id);
    if (it == claims_.end()) {
        return Status::invalid_state("no claim registered", __FILE__, __LINE__);
    }
    claimed_pages_ -= it->second;
    claims_.erase(it);
    return Status::make_ok();
}

}
}
