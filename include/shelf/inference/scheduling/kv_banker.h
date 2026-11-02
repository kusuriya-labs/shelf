#pragma once

#include <shelf/core/status.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace shelf {
namespace scheduling {

struct KVBankerProcess {
    uint32_t max_pages = 0;
    uint32_t allocated_pages = 0;
};

struct KVBankerState {
    uint32_t available_pages = 0;
    std::vector<KVBankerProcess> processes;
};

Status validate_banker_state(const KVBankerState& state);

bool is_safe(const KVBankerState& state);

bool can_add_process(const KVBankerState& state, uint32_t max_pages);

bool can_grant_pages(
    const KVBankerState& state,
    std::size_t process_index,
    uint32_t pages);

uint32_t max_safe_grant_pages(
    const KVBankerState& state,
    std::size_t process_index,
    uint32_t requested_pages);

}
}
