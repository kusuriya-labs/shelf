#include <shelf/core/gpu/scoped_device.h>
#include <string_view>

#include <string>

namespace shelf::gpu {

Result<ScopedDevice> ScopedDevice::create(int target_device, DeviceArch required) {
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

    hipDeviceProp_t props{};
    const auto props_result = hipGetDeviceProperties(&props, target_device);
    if (props_result != hipSuccess)
        return Status::hip_error("hipGetDeviceProperties", hipGetErrorString(props_result), __FILE__, __LINE__);
    const char* expected = required == DeviceArch::gfx1201 ? "gfx1201" : "gfx1030";
    const std::string_view name(props.gcnArchName);
    if (name.substr(0, name.find(':')) != expected)
        return Status::invalid_argument(
            ("必要な GPU は " + std::string(expected) + "、実際の device " + std::to_string(target_device) +
             " は " + props.gcnArchName).c_str(), __FILE__, __LINE__);

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

} // namespace shelf::gpu
