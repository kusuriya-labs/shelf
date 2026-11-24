#pragma once

#include <shelf/core/status.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace shelf::gpu {

struct alignas(64) DebugRingRecord {
    uint64_t sequence;
    uint32_t kind;
    uint32_t rank;
    uint32_t stage;
    uint32_t value0;
    uint32_t value1;
    uint32_t value2;
    uint64_t data[3];
};

static_assert(sizeof(DebugRingRecord) == 64, "debug ring record must be 64 bytes");

struct alignas(64) DebugRingHeader {
    uint64_t head;
    uint64_t tail;
    uint64_t lost;
    uint64_t capacity;
    uint32_t producer_device;
    uint32_t flags;
    uint64_t reserved[2];
};

static_assert(sizeof(DebugRingHeader) == 64, "debug ring header must be 64 bytes");

struct DebugRingDevice {
    DebugRingRecord* slots;
    uint64_t* head;
    uint64_t* tail;
    uint64_t* lost;
    uint64_t capacity;
    uint32_t producer_device;
};

__device__ __forceinline__ bool shelf_debug_ring_push(const DebugRingDevice& ring,
                                                      DebugRingRecord record) {
    if (ring.slots == nullptr || ring.capacity == 0u) return false;
    const uint64_t tail = __hip_atomic_load(ring.tail, __ATOMIC_RELAXED,
                                            __HIP_MEMORY_SCOPE_SYSTEM);
    const uint64_t head = __hip_atomic_load(ring.head, __ATOMIC_ACQUIRE,
                                            __HIP_MEMORY_SCOPE_SYSTEM);
    if (tail - head >= ring.capacity) {
        (void)__hip_atomic_fetch_add(ring.lost, 1ull, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_SYSTEM);
        return false;
    }
    record.sequence = tail;
    ring.slots[tail & (ring.capacity - 1u)] = record;
    __hip_atomic_store(ring.tail, tail + 1ull, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    return true;
}

class DebugRing {
public:
    static Result<std::unique_ptr<DebugRing>> create(int device, std::size_t capacity = 1024);
    ~DebugRing();

    DebugRing(const DebugRing&) = delete;
    DebugRing& operator=(const DebugRing&) = delete;

    void shutdown() noexcept;
    DebugRingDevice device_view() const noexcept;

    bool try_pop(DebugRingRecord& out) noexcept;
    std::size_t drain(DebugRingRecord* out, std::size_t max) noexcept;

    std::size_t pending() const noexcept;
    uint64_t lost() const noexcept;
    int device() const noexcept { return device_; }
    std::size_t capacity() const noexcept { return capacity_; }

private:
    DebugRing() = default;

    DebugRingHeader* header_ = nullptr;
    DebugRingRecord* slots_ = nullptr;
    void* host_mapping_ = nullptr;
    void* device_mapping_ = nullptr;
    int device_ = -1;
    std::size_t capacity_ = 0;
};

}  // namespace shelf::gpu
