#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/ple.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<shelf::bf16_t> out(count);
    for (auto& value : out) value.data = shelf::f32_to_bf16_rne(dist(rng));
    return out;
}

}  // namespace

int main() {
    constexpr std::uint32_t kRows = 3;
    constexpr std::uint32_t kHc = 4;
    constexpr std::uint32_t kHidden = 128;

    std::mt19937 rng(4321);
    const auto input = random_bf16(kRows * kHidden, rng);
    const auto value = random_bf16(kRows * kHidden, rng);

    std::vector<float> want_gate(kRows * kHidden);
    for (std::size_t i = 0; i < want_gate.size(); ++i) {
        const float x = bf16_to_float(input[i].data);
        const float root = std::sqrt(std::max(std::fabs(x), 1.0e-6f));
        want_gate[i] = std::copysign(root, x);
    }
    std::vector<float> want_broadcast(kRows * kHc * kHidden);
    for (std::uint32_t r = 0; r < kRows; ++r) {
        for (std::uint32_t i = 0; i < kHidden; ++i) {
            const float g = want_gate[r * kHidden + i];
            const float sigmoid = 1.0f / (1.0f + std::exp(-g));
            const float v = bf16_to_float(value[r * kHidden + i].data) * sigmoid;
            for (std::uint32_t h = 0; h < kHc; ++h) {
                want_broadcast[(r * kHc + h) * kHidden + i] = v;
            }
        }
    }

    void* d_input = nullptr;
    void* d_gate = nullptr;
    void* d_value = nullptr;
    void* d_broadcast = nullptr;
    check(hipMalloc(&d_input, input.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc input");
    check(hipMalloc(&d_gate, want_gate.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc gate");
    check(hipMalloc(&d_value, value.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc value");
    check(hipMalloc(&d_broadcast,
                    want_broadcast.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc broadcast");
    hipMemcpy(d_input, input.data(), input.size() * sizeof(shelf::bf16_t),
              hipMemcpyHostToDevice);
    hipMemcpy(d_value, value.data(), value.size() * sizeof(shelf::bf16_t),
              hipMemcpyHostToDevice);

    shelf::kernel::PleGateArgs gate_args;
    gate_args.input = static_cast<const shelf::bf16_t*>(d_input);
    gate_args.output = static_cast<shelf::bf16_t*>(d_gate);
    gate_args.rows = kRows;
    gate_args.hidden = kHidden;
    gate_args.input_stride = kHidden;
    gate_args.output_stride = kHidden;
    check(shelf::kernel::launch_ple_gate(gate_args, nullptr) == hipSuccess,
          "ple gate launch");
    check(hipDeviceSynchronize() == hipSuccess, "ple gate sync");

    shelf::kernel::PleBroadcastMulArgs mul_args;
    mul_args.gate = static_cast<const shelf::bf16_t*>(d_gate);
    mul_args.value = static_cast<const shelf::bf16_t*>(d_value);
    mul_args.output = static_cast<shelf::bf16_t*>(d_broadcast);
    mul_args.rows = kRows;
    mul_args.hc = kHc;
    mul_args.hidden = kHidden;
    mul_args.gate_stride = kHidden;
    mul_args.value_stride = kHidden;
    mul_args.output_stride = kHc * kHidden;
    check(shelf::kernel::launch_ple_broadcast_mul(mul_args, nullptr) == hipSuccess,
          "ple broadcast launch");
    check(hipDeviceSynchronize() == hipSuccess, "ple broadcast sync");

    std::vector<shelf::bf16_t> got_gate(kRows * kHidden);
    std::vector<shelf::bf16_t> got_broadcast(kRows * kHc * kHidden);
    hipMemcpy(got_gate.data(), d_gate, got_gate.size() * sizeof(shelf::bf16_t),
              hipMemcpyDeviceToHost);
    hipMemcpy(got_broadcast.data(), d_broadcast,
              got_broadcast.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost);

    double max_gate = 0.0;
    double max_broadcast = 0.0;
    for (std::size_t i = 0; i < got_gate.size(); ++i) {
        max_gate = std::max<double>(
            max_gate, std::fabs(static_cast<double>(bf16_to_float(got_gate[i].data)) -
                                want_gate[i]));
    }
    for (std::size_t i = 0; i < got_broadcast.size(); ++i) {
        max_broadcast = std::max<double>(
            max_broadcast,
            std::fabs(static_cast<double>(bf16_to_float(got_broadcast[i].data)) -
                      want_broadcast[i]));
    }
    std::fprintf(stderr, "ple gate max_abs=%g broadcast max_abs=%g\n", max_gate,
                 max_broadcast);
    check(max_gate <= 2.0e-2, "ple gate matches the CPU reference");
    check(max_broadcast <= 2.0e-2, "ple broadcast matches the CPU reference");

    (void)hipFree(d_input);
    (void)hipFree(d_gate);
    (void)hipFree(d_value);
    (void)hipFree(d_broadcast);

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
