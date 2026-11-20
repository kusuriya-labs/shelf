#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>

#include <cstddef>
#include <vector>

namespace shelf {
namespace gpu {

// Owns one arena per physical device and lends them to the runtimes that execute a
// parallel plan. Runtimes borrow, they do not allocate or release device memory.
class GpuMemoryManager {
public:
    // capacity_bytes == 0 sizes each device's arena from its free VRAM minus the runtime
    // margin. A non-zero capacity is applied to every device.
    static Result<GpuMemoryManager> create(const std::vector<int>& devices,
                                           std::size_t capacity_bytes = 0);

    ~GpuMemoryManager() noexcept;

    GpuMemoryManager(GpuMemoryManager&& other) noexcept = default;
    GpuMemoryManager& operator=(GpuMemoryManager&& other) noexcept = default;

    GpuMemoryManager(const GpuMemoryManager&) = delete;
    GpuMemoryManager& operator=(const GpuMemoryManager&) = delete;

    Result<GpuArena*> arena_for(int physical_device) noexcept;

    std::size_t device_count() const noexcept { return arenas_.size(); }
    bool is_shutdown() const noexcept { return shutdown_; }

    Status shutdown();

private:
    struct DeviceArena {
        int physical_device = -1;
        GpuArena arena;
    };

    GpuMemoryManager() noexcept = default;

    std::vector<DeviceArena> arenas_;
    bool shutdown_ = false;
};

}  // namespace gpu
}  // namespace shelf
