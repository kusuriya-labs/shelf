#include <shelf/core/gpu/debug_ring.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>

#include <atomic>
#include <cstring>

namespace shelf::gpu {

namespace {

constexpr uint32_t kDebugRingStopped = 1u;

bool capacity_is_power_of_two(std::size_t capacity) {
    return capacity >= 2u && (capacity & (capacity - 1u)) == 0u;
}

}  // namespace

Result<std::unique_ptr<DebugRing>> DebugRing::create(int device, std::size_t capacity) {
    if (device < 0) {
        return Status::invalid_argument("debug ring device is invalid", __FILE__, __LINE__);
    }
    if (!capacity_is_power_of_two(capacity)) {
        return Status::invalid_argument("debug ring capacity must be a power of two", __FILE__,
                                        __LINE__);
    }
    auto scope = ScopedDevice::create(device);
    if (!scope.ok()) return scope.status();

    int host_atomic = 0;
    const hipError_t attr = hipDeviceGetAttribute(
        &host_atomic, hipDeviceAttributeHostNativeAtomicSupported, device);
    if (attr != hipSuccess) {
        return Status::hip_error("debug ring attribute query", hipGetErrorString(attr),
                                 __FILE__, __LINE__);
    }
    if (host_atomic == 0) {
        return Status::unsupported("debug ring requires host native atomics", __FILE__,
                                   __LINE__);
    }

    const std::size_t bytes =
        sizeof(DebugRingHeader) + capacity * sizeof(DebugRingRecord);
    void* mapping = nullptr;
    const hipError_t alloc = hipHostMalloc(&mapping, bytes,
                                           hipHostMallocMapped | hipHostMallocCoherent);
    if (alloc != hipSuccess || mapping == nullptr) {
        return Status::hip_error("debug ring allocation", hipGetErrorString(alloc), __FILE__,
                                 __LINE__);
    }
    std::memset(mapping, 0, bytes);

    void* device_mapping = nullptr;
    if (hipHostGetDevicePointer(&device_mapping, mapping, 0) != hipSuccess ||
        device_mapping == nullptr) {
        device_mapping = mapping;
    }

    auto ring = std::unique_ptr<DebugRing>(new DebugRing());
    ring->host_mapping_ = mapping;
    ring->device_mapping_ = device_mapping;
    ring->header_ = static_cast<DebugRingHeader*>(mapping);
    ring->slots_ = reinterpret_cast<DebugRingRecord*>(
        static_cast<std::uint8_t*>(mapping) + sizeof(DebugRingHeader));
    ring->header_->capacity = capacity;
    ring->header_->producer_device = static_cast<std::uint32_t>(device);
    ring->device_ = device;
    ring->capacity_ = capacity;
    return ring;
}

DebugRing::~DebugRing() {
    shutdown();
}

void DebugRing::shutdown() noexcept {
    if (host_mapping_ == nullptr) return;
    header_->flags |= kDebugRingStopped;
    void* mapping = host_mapping_;
    host_mapping_ = nullptr;
    header_ = nullptr;
    slots_ = nullptr;
    device_mapping_ = nullptr;
    auto scope = ScopedDevice::create(device_);
    discard_cleanup_result(hipHostFree(mapping));
}

DebugRingDevice DebugRing::device_view() const noexcept {
    DebugRingDevice view{};
    if (host_mapping_ == nullptr || device_mapping_ == nullptr) return view;
    auto* base = static_cast<std::uint8_t*>(device_mapping_);
    auto* header = reinterpret_cast<DebugRingHeader*>(base);
    view.slots = reinterpret_cast<DebugRingRecord*>(base + sizeof(DebugRingHeader));
    view.head = &header->head;
    view.tail = &header->tail;
    view.lost = &header->lost;
    view.capacity = header->capacity;
    view.producer_device = header->producer_device;
    return view;
}

bool DebugRing::try_pop(DebugRingRecord& out) noexcept {
    if (header_ == nullptr) return false;
    std::atomic_ref<uint64_t> tail(header_->tail);
    std::atomic_ref<uint64_t> head(header_->head);
    const uint64_t tail_value = tail.load(std::memory_order_acquire);
    const uint64_t head_value = head.load(std::memory_order_relaxed);
    if (head_value == tail_value) return false;
    out = slots_[head_value & (capacity_ - 1u)];
    head.store(head_value + 1u, std::memory_order_release);
    return true;
}

std::size_t DebugRing::drain(DebugRingRecord* out, std::size_t max) noexcept {
    if (out == nullptr) return 0u;
    std::size_t count = 0u;
    while (count < max && try_pop(out[count])) ++count;
    return count;
}

std::size_t DebugRing::pending() const noexcept {
    if (header_ == nullptr) return 0u;
    std::atomic_ref<uint64_t> tail(header_->tail);
    std::atomic_ref<uint64_t> head(header_->head);
    const uint64_t tail_value = tail.load(std::memory_order_acquire);
    const uint64_t head_value = head.load(std::memory_order_acquire);
    if (tail_value <= head_value) return 0u;
    return static_cast<std::size_t>(tail_value - head_value);
}

uint64_t DebugRing::lost() const noexcept {
    if (header_ == nullptr) return 0u;
    std::atomic_ref<uint64_t> lost(header_->lost);
    return lost.load(std::memory_order_relaxed);
}

}  // namespace shelf::gpu
