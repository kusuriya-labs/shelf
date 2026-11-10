#include <shelf/kernels/optimized/sampling.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// Exercise the same request-to-output expansion used by speculative verification.
static bool test_row_masks() {
    using namespace shelf::runtime;
    constexpr uint32_t rows = 3, vocab = 33, words = 2;
    const uint32_t masks[rows * words] = {1, 0, 2, 0, 0, 1};
    uint32_t *device_masks = nullptr, *output_rows = nullptr;
    uint32_t *slots = nullptr, *positions = nullptr;
    DeviceRequestDescriptor* request = nullptr;
    DeviceSamplingParams* sampling = nullptr;
    DeviceBatchContext* context = nullptr;
    float* logits = nullptr;
    bool ok = hipMalloc(&device_masks, sizeof(masks)) == hipSuccess &&
        hipMalloc(&output_rows, rows * sizeof(uint32_t)) == hipSuccess &&
        hipMalloc(&slots, rows * sizeof(int32_t)) == hipSuccess &&
        hipMalloc(&positions, rows * sizeof(int32_t)) == hipSuccess &&
        hipMalloc(&request, sizeof(*request)) == hipSuccess &&
        hipMalloc(&sampling, rows * sizeof(*sampling)) == hipSuccess &&
        hipMalloc(&context, sizeof(*context)) == hipSuccess &&
        hipMalloc(&logits, rows * vocab * sizeof(float)) == hipSuccess;
    DeviceRequestDescriptor host_request{};
    host_request.row_count = rows;
    host_request.compute_logits = 1;
    host_request.output_index = 0;
    host_request.output_count = rows;
    host_request.sampling.token_bitmask = device_masks;
    host_request.sampling.token_bitmask_row_stride = words;
    DeviceBatchContext host_context{};
    host_context.num_requests = 1;
    host_context.actual_rows = rows;
    host_context.requests = request;
    host_context.row_sequence_slots = slots;
    host_context.row_positions = positions;
    host_context.output_rows = output_rows;
    host_context.output_sampling_params = sampling;
    std::vector<float> result(rows * vocab, 1.0f);
    if (ok) ok = hipMemcpy(device_masks, masks, sizeof(masks), hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(request, &host_request, sizeof(host_request), hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(context, &host_context, sizeof(host_context), hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(logits, result.data(), result.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess &&
        launch_prepare_batch_descriptor(context, 1, nullptr).ok() &&
        shelf::kernel::launch_apply_token_bitmask_f32(logits, sampling, rows, vocab, vocab, nullptr) == hipSuccess &&
        hipDeviceSynchronize() == hipSuccess &&
        hipMemcpy(result.data(), logits, result.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess;
    for (uint32_t row = 0; ok && row < rows; ++row) for (uint32_t token = 0; token < vocab; ++token) {
        const bool allowed = (masks[row * words + token / 32] & (1u << (token % 32))) != 0;
        const float value = result[row * vocab + token];
        if (allowed ? value != 1.0f : !(std::isinf(value) && value < 0)) ok = false;
    }
    if (logits) hipFree(logits);
    if (context) hipFree(context);
    if (sampling) hipFree(sampling);
    if (request) hipFree(request);
    if (positions) hipFree(positions);
    if (slots) hipFree(slots);
    if (output_rows) hipFree(output_rows);
    if (device_masks) hipFree(device_masks);
    return ok;
}

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
    ok = test_row_masks() && ok;
    std::printf("token bitmask CPU reference: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
