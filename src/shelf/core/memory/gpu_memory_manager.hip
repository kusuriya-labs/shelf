#include <shelf/core/memory/gpu_memory_manager.h>

#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/size_format.h>

#include <hip/hip_runtime.h>

#include <cstdio>

namespace shelf {
namespace gpu {

namespace {

constexpr std::size_t kAutoMarginBytes = 820ull * 1024ull * 1024ull;

Result<std::size_t> resolve_capacity(int physical_device, std::size_t capacity_bytes) {
    if (capacity_bytes != 0u) {
        return capacity_bytes;
    }
    auto scope = ScopedDevice::create(physical_device);
    if (!scope.ok()) return scope.status();
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const hipError_t err = hipMemGetInfo(&free_bytes, &total_bytes);
    if (err != hipSuccess) {
        return Status::hip_error("hipMemGetInfo", hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (free_bytes == 0u) {
        return Status::insufficient_memory(
            "no VRAM available for the arena", __FILE__, __LINE__);
    }
    const std::size_t capacity = free_bytes > kAutoMarginBytes
        ? free_bytes - kAutoMarginBytes
        : free_bytes / 2u;
    if (capacity == 0u) {
        return Status::insufficient_memory(
            "no VRAM available for the arena", __FILE__, __LINE__);
    }
    std::fprintf(stderr, "ARENA auto: device=%d free=%s capacity=%s\n", physical_device,
                 format_gib(free_bytes).c_str(), format_gib(capacity).c_str());
    return capacity;
}

}  // namespace

Result<GpuMemoryManager> GpuMemoryManager::create(const std::vector<int>& devices,
                                                  std::size_t capacity_bytes) {
    if (devices.empty()) {
        return Status::invalid_argument("devices must not be empty", __FILE__, __LINE__);
    }
    for (std::size_t i = 0; i < devices.size(); ++i) {
        for (std::size_t j = i + 1; j < devices.size(); ++j) {
            if (devices[i] == devices[j]) {
                return Status::invalid_argument(
                    "duplicate device in GpuMemoryManager", __FILE__, __LINE__);
            }
        }
    }

    GpuMemoryManager manager;
    manager.devices_.reserve(devices.size());
    for (int device : devices) {
        auto capacity = resolve_capacity(device, capacity_bytes);
        if (!capacity.ok()) return capacity.status();
        auto arena_result = GpuArena::create(device, capacity.value());
        if (!arena_result.ok()) return arena_result.status();

        auto scope = ScopedDevice::create(device);
        if (!scope.ok()) return scope.status();
        hipStream_t stream = nullptr;
        const hipError_t stream_err = hipStreamCreate(&stream);
        if (stream_err != hipSuccess) {
            return Status::hip_error(
                "hipStreamCreate", hipGetErrorString(stream_err), __FILE__, __LINE__);
        }
        manager.devices_.push_back(DeviceResources{
            .physical_device = device,
            .arena = arena_result.release(),
            .stream = stream,
        });
    }
    return manager;
}

GpuMemoryManager::GpuMemoryManager(GpuMemoryManager&& other) noexcept
    : devices_(std::move(other.devices_)), shutdown_(other.shutdown_) {
    other.devices_.clear();
    other.shutdown_ = true;
}

GpuMemoryManager& GpuMemoryManager::operator=(GpuMemoryManager&& other) noexcept {
    if (this == &other) return *this;
    (void)shutdown();
    devices_ = std::move(other.devices_);
    shutdown_ = other.shutdown_;
    other.devices_.clear();
    other.shutdown_ = true;
    return *this;
}

GpuMemoryManager::~GpuMemoryManager() noexcept {
    (void)shutdown();
}

Result<GpuArena*> GpuMemoryManager::arena_for(int physical_device) noexcept {
    for (auto& device_resources : devices_) {
        if (device_resources.physical_device == physical_device) {
            return &device_resources.arena;
        }
    }
    return Status::out_of_range("device not in GpuMemoryManager", __FILE__, __LINE__);
}

Result<hipStream_t> GpuMemoryManager::stream_for(int physical_device) noexcept {
    for (auto& device_resources : devices_) {
        if (device_resources.physical_device == physical_device) {
            return device_resources.stream;
        }
    }
    return Status::out_of_range("device not in GpuMemoryManager", __FILE__, __LINE__);
}

Status GpuMemoryManager::shutdown() {
    if (shutdown_) {
        return Status::make_ok();
    }
    Status first_error = Status::make_ok();
    for (auto& device_resources : devices_) {
        auto scope = ScopedDevice::create(device_resources.physical_device);
        if (!scope.ok()) {
            if (first_error.ok()) first_error = scope.status();
        }
        if (device_resources.stream != nullptr) {
            const hipError_t err = hipStreamDestroy(device_resources.stream);
            device_resources.stream = nullptr;
            if (first_error.ok() && err != hipSuccess) {
                first_error = Status::hip_error(
                    "hipStreamDestroy", hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
    }
    for (auto& device_resources : devices_) {
        Status st = device_resources.arena.shutdown();
        if (first_error.ok() && !st.ok()) first_error = st;
    }
    shutdown_ = true;
    return first_error;
}

}  // namespace gpu
}  // namespace shelf
