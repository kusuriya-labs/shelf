#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace ps::runtime {

inline constexpr uint32_t kInvalidRequestSlot =
    std::numeric_limits<uint32_t>::max();

inline constexpr uint32_t kInvalidRequestGeneration = 0u;

struct alignas(8) RequestHandle {
    uint32_t slot = kInvalidRequestSlot;
    uint32_t generation = kInvalidRequestGeneration;
};

constexpr bool request_handle_valid(RequestHandle handle) noexcept {
    return handle.slot != kInvalidRequestSlot &&
           handle.generation != kInvalidRequestGeneration;
}

constexpr bool request_handle_equal(
    RequestHandle lhs,
    RequestHandle rhs) noexcept {

    return lhs.slot == rhs.slot &&
           lhs.generation == rhs.generation;
}

constexpr uint64_t request_handle_key(RequestHandle handle) noexcept {
    return
        (static_cast<uint64_t>(handle.generation) << 32) |
        static_cast<uint64_t>(handle.slot);
}

static_assert(sizeof(RequestHandle) == 8);
static_assert(alignof(RequestHandle) == 8);
static_assert(std::is_standard_layout_v<RequestHandle>);
static_assert(std::is_trivially_copyable_v<RequestHandle>);

}  // namespace ps::runtime
