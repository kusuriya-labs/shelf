#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/core/memory/device_allocation_view.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace ps {
namespace gpu {

class GpuArena {
public:
    struct VmmMapping {
        std::size_t offset = 0;
        std::size_t bytes = 0;
        hipMemGenericAllocationHandle_t handle{};
    };

    static Result<GpuArena> create(int physical_device, std::size_t capacity_bytes);

    ~GpuArena() noexcept;
    GpuArena(GpuArena&& other) noexcept;
    GpuArena& operator=(GpuArena&& other) noexcept;

    GpuArena(const GpuArena&) = delete;
    GpuArena& operator=(const GpuArena&) = delete;

    Result<DeviceAllocationView> allocate_aligned(std::size_t bytes, std::size_t alignment);

    Status shutdown();
    bool is_shutdown() const noexcept { return shutdown_; }

    int device() const noexcept { return device_; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t used() const noexcept { return offset_; }
    std::size_t committed() const noexcept { return committed_bytes_; }
    bool is_vmm() const noexcept { return vmm_; }

    static DeviceAllocationView make_view(void* data, std::size_t bytes, int physical_device) noexcept;

    Result<DeviceAllocationView> used_prefix_view() const;

 private:
    GpuArena() = default;

    Status ensure_committed(std::size_t required_end);
    void release_noexcept();

    int device_ = -1;
    void* base_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t offset_ = 0;
    bool shutdown_ = false;
    bool vmm_ = false;
    std::size_t reserved_bytes_ = 0;
    std::size_t committed_bytes_ = 0;
    std::size_t minimum_granularity_ = 0;
    std::size_t recommended_granularity_ = 0;
    std::size_t commit_quantum_ = 0;
    std::vector<VmmMapping> mappings_;
};

class ArenaSet {
public:
    static Result<ArenaSet> create(
        const std::vector<int>& devices,
        std::size_t capacity_bytes);

    ~ArenaSet() = default;

    ArenaSet(ArenaSet&& other) noexcept = default;
    ArenaSet& operator=(ArenaSet&& other) noexcept = default;

    ArenaSet(const ArenaSet&) = delete;
    ArenaSet& operator=(const ArenaSet&) = delete;

    ArenaSet() noexcept = default;

    Result<GpuArena*> for_device(int physical_device) noexcept;

    Status shutdown();
    bool is_shutdown() const noexcept { return shutdown_; }

private:
    struct DeviceArena {
        int physical_device;
        GpuArena arena;
    };
    std::vector<DeviceArena> arenas_;
    bool shutdown_ = false;
};

} // namespace gpu
} // namespace ps
