#pragma once

#include <cstddef>

namespace ps {
namespace gpu {

class Tensor;
class GpuArena;

class DeviceAllocationView {

 public:
    void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }
    int physical_device() const noexcept { return physical_device_; }

    explicit operator bool() const noexcept { return data_ != nullptr; }

    DeviceAllocationView() = delete;

 public:
    DeviceAllocationView(const DeviceAllocationView&) noexcept = default;
    DeviceAllocationView& operator=(const DeviceAllocationView&) noexcept = default;

    DeviceAllocationView(DeviceAllocationView&&) noexcept = default;
    DeviceAllocationView& operator=(DeviceAllocationView&&) noexcept = default;

 private:
    friend class Tensor;
    friend class GpuArena;

    DeviceAllocationView(void* data, std::size_t bytes, int physical_device) noexcept
        : data_(data), bytes_(bytes), physical_device_(physical_device) {}

    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    int physical_device_ = -1;
};

} // namespace gpu
} // namespace ps
