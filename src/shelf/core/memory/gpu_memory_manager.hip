#include <shelf/core/memory/gpu_memory_manager.h>

#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/size_format.h>

#include <hip/hip_runtime.h>

#include <cstdio>

namespace shelf {
namespace gpu {

namespace {

constexpr std::size_t kAutoMarginBytes = 512ull * 1024ull * 1024ull;

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
    manager.arenas_.reserve(devices.size());
    for (int device : devices) {
        auto capacity = resolve_capacity(device, capacity_bytes);
        if (!capacity.ok()) return capacity.status();
        auto arena_result = GpuArena::create(device, capacity.value());
        if (!arena_result.ok()) return arena_result.status();
        manager.arenas_.push_back(DeviceArena{
            .physical_device = device,
            .arena = arena_result.release(),
        });
    }
    return manager;
}

GpuMemoryManager::~GpuMemoryManager() noexcept {
    (void)shutdown();
}

Result<GpuArena*> GpuMemoryManager::arena_for(int physical_device) noexcept {
    for (auto& device_arena : arenas_) {
        if (device_arena.physical_device == physical_device) {
            return &device_arena.arena;
        }
    }
    return Status::out_of_range("device not in GpuMemoryManager", __FILE__, __LINE__);
}

Status GpuMemoryManager::shutdown() {
    if (shutdown_) {
        return Status::make_ok();
    }
    Status first_error = Status::make_ok();
    for (auto& device_arena : arenas_) {
        Status st = device_arena.arena.shutdown();
        if (!first_error.ok()) continue;
        if (!st.ok()) first_error = st;
    }
    shutdown_ = true;
    return first_error;
}

}  // namespace gpu
}  // namespace shelf
