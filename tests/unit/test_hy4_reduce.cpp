#include <shelf/models/hy4_flash/residency.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr size_t count = 65539;
    std::vector<uint16_t> a(count, 0x3f80), b(count, 0x4000), output(count);
    uint16_t *device_a = nullptr, *device_b = nullptr;
    auto check = [](hipError_t status) {
        if (status != hipSuccess) {
            std::fprintf(stderr, "%s\n", hipGetErrorString(status));
            return false;
        }
        return true;
    };
    bool ok = check(hipMalloc(&device_a, count * sizeof(uint16_t))) &&
              check(hipMalloc(&device_b, count * sizeof(uint16_t)));
    if (ok)
        ok = check(hipMemcpy(device_a, a.data(), count * sizeof(uint16_t), hipMemcpyHostToDevice)) &&
             check(hipMemcpy(device_b, b.data(), count * sizeof(uint16_t), hipMemcpyHostToDevice));
    if (ok) {
        shelf::hy4::tp_add_bf16(device_a, device_b, count * sizeof(uint16_t));
        ok = check(hipGetLastError()) &&
             check(hipMemcpy(output.data(), device_a, count * sizeof(uint16_t), hipMemcpyDeviceToHost));
    }
    for (auto value : output)
        if (value != 0x4040)
            ok = false;
    if (device_b)
        ok = check(hipFree(device_b)) && ok;
    if (device_a)
        ok = check(hipFree(device_a)) && ok;
    return ok ? 0 : 1;
}
