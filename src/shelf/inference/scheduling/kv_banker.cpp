#include <shelf/inference/scheduling/kv_banker.h>

#include <algorithm>

namespace shelf {
namespace scheduling {

Status validate_banker_state(const KVBankerState& state) {
    for (const KVBankerProcess& process : state.processes) {
        if (process.allocated_pages > process.max_pages) {
            return Status::invalid_state(
                "banker state: allocation exceeds max", __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

bool is_safe(const KVBankerState& state) {
    if (!validate_banker_state(state).ok()) {
        return false;
    }
    const std::size_t count = state.processes.size();
    std::vector<bool> finished(count, false);
    uint64_t work = state.available_pages;
    bool progress = true;
    while (progress) {
        progress = false;
        for (std::size_t i = 0; i < count; ++i) {
            if (finished[i]) {
                continue;
            }
            const KVBankerProcess& process = state.processes[i];
            const uint64_t need =
                process.max_pages - process.allocated_pages;
            if (need <= work) {
                work += process.allocated_pages;
                finished[i] = true;
                progress = true;
            }
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (!finished[i]) {
            return false;
        }
    }
    return true;
}

bool can_add_process(const KVBankerState& state, uint32_t max_pages) {
    KVBankerState simulated = state;
    simulated.processes.push_back({max_pages, 0});
    return is_safe(simulated);
}

bool can_grant_pages(
    const KVBankerState& state,
    std::size_t process_index,
    uint32_t pages) {
    if (pages == 0) {
        return is_safe(state);
    }
    if (process_index >= state.processes.size()) {
        return false;
    }
    if (pages > state.available_pages) {
        return false;
    }
    const KVBankerProcess& process = state.processes[process_index];
    if (process.allocated_pages > process.max_pages) {
        return false;
    }
    if (pages > process.max_pages - process.allocated_pages) {
        return false;
    }
    KVBankerState simulated = state;
    simulated.available_pages -= pages;
    simulated.processes[process_index].allocated_pages += pages;
    return is_safe(simulated);
}

uint32_t max_safe_grant_pages(
    const KVBankerState& state,
    std::size_t process_index,
    uint32_t requested_pages) {
    if (requested_pages == 0 || process_index >= state.processes.size()) {
        return 0;
    }
    const KVBankerProcess& process = state.processes[process_index];
    if (process.allocated_pages > process.max_pages) {
        return 0;
    }
    uint32_t high = requested_pages;
    high = std::min(high, process.max_pages - process.allocated_pages);
    high = std::min(high, state.available_pages);
    if (high == 0) {
        return 0;
    }
    uint32_t low = 0;
    while (low < high) {
        const uint32_t mid = low + (high - low + 1) / 2;
        if (can_grant_pages(state, process_index, mid)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

}
}
