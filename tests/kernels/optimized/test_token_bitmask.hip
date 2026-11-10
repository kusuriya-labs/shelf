#include <shelf/kernels/optimized/sampling.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr uint32_t vocab = 67, rows = 2, stride = 70;
    std::vector<float> input(rows * stride);
    for (uint32_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i);
    const uint32_t mask[] = {0x80000001u, 0x00000002u, 0x00000004u};
    uint32_t* device_mask = nullptr;
    float* device_logits = nullptr;
    shelf::runtime::DeviceSamplingParams* device_params = nullptr;
    shelf::runtime::DeviceSamplingParams params[rows]{};
    bool ok = hipMalloc(&device_mask, sizeof(mask)) == hipSuccess &&
        hipMalloc(&device_logits, input.size() * sizeof(float)) == hipSuccess &&
        hipMalloc(&device_params, sizeof(params)) == hipSuccess;
    params[0].token_bitmask = device_mask;
    if (ok) ok = hipMemcpy(device_mask, mask, sizeof(mask), hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(device_logits, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(device_params, params, sizeof(params), hipMemcpyHostToDevice) == hipSuccess;
    if (ok) ok = shelf::kernel::launch_apply_token_bitmask_f32(device_logits, device_params, rows, vocab, stride, nullptr) == hipSuccess &&
        hipDeviceSynchronize() == hipSuccess;
    std::vector<float> actual(input.size());
    if (ok) ok = hipMemcpy(actual.data(), device_logits, actual.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess;
    if (ok) {
        for (uint32_t row = 0; row < rows; ++row) for (uint32_t token = 0; token < stride; ++token) {
            const bool disallowed = row == 0 && token < vocab && !(mask[token / 32] & (1u << (token % 32)));
            const size_t i = row * stride + token;
            if (disallowed ? !(std::isinf(actual[i]) && actual[i] < 0) : actual[i] != input[i]) ok = false;
        }
    }
    if (device_params && hipFree(device_params) != hipSuccess) ok = false;
    if (device_logits && hipFree(device_logits) != hipSuccess) ok = false;
    if (device_mask && hipFree(device_mask) != hipSuccess) ok = false;
    std::printf("token bitmask CPU reference: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
