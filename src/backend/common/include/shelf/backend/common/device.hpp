// shelf backend/common — device & memory value contracts (plan §4.5 Resource).
//
// Backend-neutral vocabulary ONLY: ids, descriptors, non-owning views.
// No HIP types here, and intentionally no polymorphic "universal
// Backend" interface — execution plans hold concrete backend handles chosen
// at compile time (plan §1.2).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <shelf/status.hpp>

namespace shelf::backend {

using DeviceId = std::int32_t;
inline constexpr DeviceId kInvalidDevice = -1;

enum class DeviceKind : std::uint8_t {
    host = 0,
    gpu,
};

struct DeviceInfo {
    DeviceId id = kInvalidDevice;
    DeviceKind kind = DeviceKind::host;
    std::string name;
    std::uint64_t total_memory_bytes = 0;
    std::uint64_t max_block_dim = 0;
    bool supports_unified_memory = false;
};

enum class MemoryKind : std::uint8_t {
    host = 0,        // pageable
    pinned_host,     // page-locked, transfer-friendly
    device,          // VRAM resident
    unified,         // device-visible managed
};

// Non-owning, transfer-time window into a buffer. The owning backend object
// decides lifetime; views must never outlive their owner (enforced by tests
// and, for graphs, by plan-time residency checks).
struct BufferView {
    void* base = nullptr;
    std::size_t bytes = 0;
    MemoryKind kind = MemoryKind::host;
    DeviceId device = kInvalidDevice;

    bool empty() const noexcept { return base == nullptr || bytes == 0; }

    Status contains(std::size_t offset, std::size_t count) const {
        if (base == nullptr) {
            return Status::invalid_state("BufferView has no storage", __FILE__, __LINE__);
        }
        if (offset > bytes) {
            return Status::out_of_range("BufferView offset exceeds capacity", __FILE__, __LINE__);
        }
        SHELF_RETURN_IF_ERROR(checked_add(offset, count, __FILE__, __LINE__));
        if (offset + count > bytes) {
            return Status::out_of_range("BufferView subrange exceeds capacity", __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
};

} // namespace shelf::backend
