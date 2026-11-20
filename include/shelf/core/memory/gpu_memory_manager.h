#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <vector>

namespace shelf {
namespace gpu {

// Owns one arena and one stream per physical device and lends them to the runtimes that
// execute a parallel plan. Runtimes borrow these handles; they never create or destroy them.
class GpuMemoryManager {
public:
    // capacity_bytes == 0 sizes each device's arena from its free VRAM minus the runtime
    // margin. A non-zero capacity is applied to every device.
    static Result<GpuMemoryManager> create(const std::vector<int>& devices,
                                           std::size_t capacity_bytes = 0);

    ~GpuMemoryManager() noexcept;

    GpuMemoryManager(GpuMemoryManager&& other) noexcept;
    GpuMemoryManager& operator=(GpuMemoryManager&& other) noexcept;

    GpuMemoryManager(const GpuMemoryManager&) = delete;
    GpuMemoryManager& operator=(const GpuMemoryManager&) = delete;

    Result<GpuArena*> arena_for(int physical_device) noexcept;
    Result<hipStream_t> stream_for(int physical_device) noexcept;

    std::size_t device_count() const noexcept { return devices_.size(); }
    bool is_shutdown() const noexcept { return shutdown_; }

    Status shutdown();

private:
    struct DeviceResources {
        int physical_device = -1;
        GpuArena arena;
        hipStream_t stream = nullptr;
    };

    GpuMemoryManager() noexcept = default;

    std::vector<DeviceResources> devices_;
    bool shutdown_ = false;
};

}  // namespace gpu
}  // namespace shelf
