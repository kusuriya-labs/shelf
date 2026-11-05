#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/ple_conv.h>
#include <shelf/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::fprintf(stderr, "ok: %s\n", what);
    }
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

std::vector<shelf::bf16_t> random_bf16(std::size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> out(count);
    for (auto& value : out) value.data = shelf::f32_to_bf16_rne(dist(rng));
    return out;
}

struct Case {
    std::uint32_t channels;
    std::uint32_t rows;
    std::vector<std::pair<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>>> requests;
};

bool run_case(const Case& c) {
    constexpr std::uint32_t kHistory = 9;
    std::mt19937 rng(1234u + c.channels);
    const auto input = random_bf16(c.rows * c.channels, rng);
    const auto weight = random_bf16(c.channels * 4u, rng);
    const std::uint64_t slot_stride = static_cast<std::uint64_t>(kHistory) * c.channels;
    const std::uint32_t max_sequences = 2u;
    std::vector<shelf::bf16_t> state(max_sequences * slot_stride, shelf::bf16_t{0});
    std::vector<shelf::bf16_t> want_state = state;
    std::vector<shelf::bf16_t> want_output(c.rows * c.channels, shelf::bf16_t{0});

    for (const auto& request : c.requests) {
        const std::uint32_t slot = request.first;
        const std::uint32_t begin = request.second.first;
        const std::uint32_t count = request.second.second;
        std::vector<std::array<float, kHistory>> history(c.channels);
        for (auto& row : history) row.fill(0.0f);
        for (std::uint32_t t = 0; t < count; ++t) {
            const std::uint32_t row = begin + t;
            for (std::uint32_t ch = 0; ch < c.channels; ++ch) {
                auto& h = history[ch];
                const float x = bf16_to_float(input[row * c.channels + ch].data);
                const float w0 = bf16_to_float(weight[ch * 4u + 0].data);
                const float w1 = bf16_to_float(weight[ch * 4u + 1].data);
                const float w2 = bf16_to_float(weight[ch * 4u + 2].data);
                const float w3 = bf16_to_float(weight[ch * 4u + 3].data);
                const float y = w0 * h[0] + w1 * h[3] + w2 * h[6] + w3 * x;
                const float silu = y / (1.0f + std::exp(-y));
                want_output[row * c.channels + ch].data = shelf::f32_to_bf16_rne(silu);
                for (std::uint32_t i = 0; i + 1u < kHistory; ++i) h[i] = h[i + 1u];
                h[kHistory - 1u] = x;
            }
        }
        for (std::uint32_t ch = 0; ch < c.channels; ++ch) {
            for (std::uint32_t i = 0; i < kHistory; ++i) {
                want_state[slot * slot_stride + i * c.channels + ch].data =
                    shelf::f32_to_bf16_rne(history[ch][i]);
            }
        }
    }

    void* d_input = nullptr;
    void* d_weight = nullptr;
    void* d_output = nullptr;
    void* d_state = nullptr;
    void* d_requests = nullptr;
    check(hipMalloc(&d_input, input.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc input");
    check(hipMalloc(&d_weight, weight.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc weight");
    check(hipMalloc(&d_output, want_output.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc output");
    check(hipMalloc(&d_state, state.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc state");
    check(hipMalloc(&d_requests, c.requests.size() *
                                     sizeof(shelf::runtime::DeviceRequestDescriptor)) ==
              hipSuccess,
          "hipMalloc requests");
    hipMemcpy(d_input, input.data(), input.size() * sizeof(shelf::bf16_t),
              hipMemcpyHostToDevice);
    hipMemcpy(d_weight, weight.data(), weight.size() * sizeof(shelf::bf16_t),
              hipMemcpyHostToDevice);
    hipMemcpy(d_output, want_output.data(), want_output.size() * sizeof(shelf::bf16_t),
              hipMemcpyHostToDevice);
    hipMemcpy(d_state, state.data(), state.size() * sizeof(shelf::bf16_t),
              hipMemcpyHostToDevice);

    std::vector<shelf::runtime::DeviceRequestDescriptor> requests(c.requests.size());
    for (std::size_t i = 0; i < c.requests.size(); ++i) {
        requests[i].request_handle.slot = c.requests[i].first;
        requests[i].row_begin = c.requests[i].second.first;
        requests[i].row_count = c.requests[i].second.second;
    }
    hipMemcpy(d_requests, requests.data(),
              requests.size() * sizeof(shelf::runtime::DeviceRequestDescriptor),
              hipMemcpyHostToDevice);

    shelf::kernel::PleConvArgs args;
    args.input = static_cast<const shelf::bf16_t*>(d_input);
    args.output = static_cast<shelf::bf16_t*>(d_output);
    args.weight = static_cast<const shelf::bf16_t*>(d_weight);
    args.state = static_cast<shelf::bf16_t*>(d_state);
    args.requests =
        static_cast<const shelf::runtime::DeviceRequestDescriptor*>(d_requests);
    args.num_requests = static_cast<std::uint32_t>(c.requests.size());
    args.actual_rows = c.rows;
    args.channels = c.channels;
    args.max_sequences = max_sequences;
    args.state_index = 0;
    args.input_row_stride = c.channels;
    args.output_row_stride = c.channels;
    args.state_slot_stride = slot_stride;
    args.state_layer_stride = slot_stride;
    args.state_history_stride = c.channels;
    check(shelf::kernel::launch_ple_conv(args, nullptr) == hipSuccess, "ple conv launch");
    check(hipDeviceSynchronize() == hipSuccess, "ple conv sync");

    std::vector<shelf::bf16_t> got_output(want_output.size());
    std::vector<shelf::bf16_t> got_state(want_state.size());
    hipMemcpy(got_output.data(), d_output, got_output.size() * sizeof(shelf::bf16_t),
              hipMemcpyDeviceToHost);
    hipMemcpy(got_state.data(), d_state, got_state.size() * sizeof(shelf::bf16_t),
              hipMemcpyDeviceToHost);

    double max_output = 0.0;
    double max_state = 0.0;
    for (std::size_t i = 0; i < got_output.size(); ++i) {
        max_output = std::max<double>(
            max_output, std::fabs(static_cast<double>(bf16_to_float(got_output[i].data)) -
                                   bf16_to_float(want_output[i].data)));
    }
    for (std::size_t i = 0; i < got_state.size(); ++i) {
        max_state = std::max<double>(
            max_state, std::fabs(static_cast<double>(bf16_to_float(got_state[i].data)) -
                                  bf16_to_float(want_state[i].data)));
    }
    std::fprintf(stderr, "channels=%u requests=%zu output max_abs=%g state max_abs=%g\n",
                 c.channels, c.requests.size(), max_output, max_state);
    check(max_output <= 2.0e-2, "ple conv output matches the CPU reference");
    check(max_state <= 2.0e-2, "ple conv state matches the CPU reference");

    (void)hipFree(d_input);
    (void)hipFree(d_weight);
    (void)hipFree(d_output);
    (void)hipFree(d_state);
    (void)hipFree(d_requests);
    return true;
}

}  // namespace

int main() {
    check(run_case(Case{16u, 5u, {{0u, {0u, 3u}}, {1u, {3u, 2u}}}}),
          "ple conv with two sequences");
    check(run_case(Case{32u, 1u, {{0u, {0u, 1u}}}}), "ple conv decode row");
    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
