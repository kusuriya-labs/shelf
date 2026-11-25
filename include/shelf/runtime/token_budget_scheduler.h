#pragma once

#include <shelf/runtime/runtime_request.h>
#include <shelf/runtime/scheduling_types.h>
#include <shelf/core/status.h>
#include <cstddef>
#include <cstdint>
#include <span>

namespace shelf {
namespace runtime {

Result<SchedulePlan> schedule_requests(
    std::span<const RuntimeRequest*> active,
    uint32_t max_num_batched_tokens,
    uint32_t max_scheduled_requests,
    uint32_t page_tokens,
    KVBankerState* kv, bool decode_only = false);

inline bool request_decode_ready(const RuntimeRequest& request) noexcept {
    return request.state == RequestState::Active &&
           !request.waiting_for_token_bitmask &&
           request.sequence.position >= request.input_tokens.size() &&
           request.pending_decode_token >= 0;
}

inline bool request_prefill_ready(const RuntimeRequest& request) noexcept {
    return request.state == RequestState::Active &&
           !request.waiting_for_token_bitmask &&
           request.sequence.position < request.input_tokens.size();
}

bool prefill_decode_should_interrupt(std::span<const RuntimeRequest*> active,
                                     uint32_t interval,
                                     uint32_t chunks_since_decode) noexcept;

}
}
