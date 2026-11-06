#pragma once
#include <cstddef>
#include <cstdint>

namespace shelf::gpu {

constexpr uint64_t align_up_16(uint64_t bytes) noexcept {
    return (bytes + 15ull) & ~15ull;
}

constexpr std::size_t align_up(std::size_t bytes, std::size_t alignment) noexcept {
    return (bytes + alignment - 1u) & ~(alignment - 1u);
}

constexpr uint32_t aligned_row_stride(
    uint32_t logical_elements,
    uint32_t element_bytes) noexcept {
    return static_cast<uint32_t>(
        align_up_16(
            static_cast<uint64_t>(logical_elements) *
            static_cast<uint64_t>(element_bytes)) /
        element_bytes);
}

inline bool is_aligned_16(const void* p) noexcept {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}

}
