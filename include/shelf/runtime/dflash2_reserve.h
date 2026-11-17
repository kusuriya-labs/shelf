#pragma once

#include <cstddef>

namespace shelf {
namespace runtime {

struct DFlash2Footprint {
    std::size_t draft_weights = 0u;
    std::size_t draft_scratch = 0u;
    std::size_t session_slot = 0u;
    std::size_t spec_batch = 0u;
    std::size_t sessions = 0u;

    std::size_t total() const noexcept {
        return draft_weights + draft_scratch + spec_batch + session_slot * sessions;
    }
};

}  // namespace runtime
}  // namespace shelf
