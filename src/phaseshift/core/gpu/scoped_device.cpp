#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/core/gpu/gfx_arch.h>

#include <string>

namespace ps::gpu {

Result<ScopedDevice> ScopedDevice::create(int target_device) {
    int device_count = 0;

    const hipError_t count_result = hipGetDeviceCount(&device_count);

    if (count_result != hipSuccess) {
        return Status::hip_error(
            "hipGetDeviceCount",
            hipGetErrorString(count_result),
            __FILE__,
            __LINE__);
    }

    if (target_device < 0 || target_device >= device_count) {
        return Status::out_of_range(
            "target device is out of range",
            __FILE__,
            __LINE__);
    }

    if (!is_gfx1201(static_cast<uint32_t>(target_device))) {
        hipDeviceProp_t props{};
        const char* name = (hipGetDeviceProperties(&props, target_device) == hipSuccess)
            ? props.gcnArchName
            : "unknown";
        return Status::invalid_argument(
            ("PhaseShift requires gfx1201 (AMD Radeon AI PRO R9700); device "
             + std::to_string(target_device) + " is " + name).c_str(),
            __FILE__,
            __LINE__);
    }

    ScopedDevice guard;

    const hipError_t get_result = hipGetDevice(&guard.previous_device_);

    if (get_result != hipSuccess) {
        return Status::hip_error(
            "hipGetDevice",
            hipGetErrorString(get_result),
            __FILE__,
            __LINE__);
    }

    if (guard.previous_device_ == target_device) {
        return guard;
    }

    const hipError_t set_result = hipSetDevice(target_device);

    if (set_result != hipSuccess) {
        return Status::hip_error(
            "hipSetDevice",
            hipGetErrorString(set_result),
            __FILE__,
            __LINE__);
    }

    guard.restore_required_ = true;
    return guard;
}

ScopedDevice::~ScopedDevice() {
    if (restore_required_ && previous_device_ >= 0) {
        const hipError_t e = hipSetDevice(previous_device_);
        if (e != hipSuccess) {
            std::fprintf(
                stderr,
                "ScopedDevice: restore hipSetDevice(%d) failed: %s\n",
                previous_device_,
                hipGetErrorString(e));
        }
    }
}

} // namespace ps::gpu
