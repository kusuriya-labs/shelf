#pragma once

#include <shelf/core/status.h>
#include <hip/hip_runtime.h>

namespace shelf::gpu {

enum class DeviceArch { gfx1201, gfx1030 };

class ScopedDevice {
public:
    static Result<ScopedDevice> create(int target_device, DeviceArch required = DeviceArch::gfx1201);

    ~ScopedDevice();

    ScopedDevice(ScopedDevice&& other) noexcept
        : previous_device_(other.previous_device_),
          restore_required_(other.restore_required_)
    {
        other.previous_device_ = -1;
        other.restore_required_ = false;
    }
    ScopedDevice& operator=(ScopedDevice&&) = delete;

    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    ScopedDevice() = default;

    int previous_device_ = -1;
    bool restore_required_ = false;
};

} // namespace shelf::gpu
