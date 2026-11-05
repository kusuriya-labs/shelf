#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/elementwise.h>
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
    constexpr std::uint32_t kWidth = kHc * kHidden;

    std::mt19937 rng(4321);
    const auto key = random_bf16(kRows * kWidth, rng);
    const auto query = random_bf16(kRows * kWidth, rng);
    const auto value = random_bf16(kRows * kHidden, rng);

    std::vector<float> want_dot(kRows * kHc);
    std::vector<float> want_gate(kRows * kHc);
    for (std::uint32_t r = 0; r < kRows; ++r) {
        for (std::uint32_t h = 0; h < kHc; ++h) {
            float sum = 0.0f;
            for (std::uint32_t i = 0; i < kHidden; ++i) {
                sum += bf16_to_float(key[(r * kHc + h) * kHidden + i].data) *
                       bf16_to_float(query[(r * kHc + h) * kHidden + i].data);
            }
            want_dot[r * kHc + h] = sum;
            const float scaled = sum / std::sqrt(static_cast<float>(kHidden));
            const float root =
                std::sqrt(std::max(std::fabs(scaled), 1.0e-6f));
            want_gate[r * kHc + h] = std::copysign(root, scaled);
        }
    }
    std::vector<float> want_broadcast(kRows * kWidth);
    for (std::uint32_t r = 0; r < kRows; ++r) {
        for (std::uint32_t h = 0; h < kHc; ++h) {
            const float sigmoid =
                1.0f / (1.0f + std::exp(-want_gate[r * kHc + h]));
            for (std::uint32_t i = 0; i < kHidden; ++i) {
                want_broadcast[(r * kHc + h) * kHidden + i] =
                    bf16_to_float(value[r * kHidden + i].data) * sigmoid;
            }
        }
    }

    void* d_key = nullptr;
    void* d_query = nullptr;
    void* d_dot = nullptr;
    void* d_scaled = nullptr;
    void* d_gate = nullptr;
    void* d_value = nullptr;
    void* d_broadcast = nullptr;
    bool ok = true;
    ok = ok && hipMalloc(&d_key, key.size() * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_query, query.size() * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_dot, kRows * kHc * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_scaled, kRows * kHc * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_gate, kRows * kHc * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_value, value.size() * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_broadcast,
                         want_broadcast.size() * sizeof(shelf::bf16_t)) == hipSuccess;
    check(ok, "hipMalloc");
    if (!ok) return 1;
    (void)hipMemcpy(d_key, key.data(), key.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(d_query, query.data(), query.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(d_value, value.data(), value.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice);

    shelf::kernel::PleStreamDotArgs dot_args;
    dot_args.key = static_cast<const shelf::bf16_t*>(d_key);
    dot_args.query = static_cast<const shelf::bf16_t*>(d_query);
    dot_args.output = static_cast<shelf::bf16_t*>(d_dot);
    dot_args.rows = kRows;
    dot_args.hc = kHc;
    dot_args.hidden = kHidden;
    dot_args.key_stride = kWidth;
    dot_args.query_stride = kWidth;
    dot_args.output_stride = kHc;
    check(shelf::kernel::launch_ple_stream_dot(dot_args, nullptr) == hipSuccess,
          "ple stream dot launch");
    check(hipDeviceSynchronize() == hipSuccess, "ple stream dot sync");

    std::vector<shelf::bf16_t> got_dot(kRows * kHc);
    (void)hipMemcpy(got_dot.data(), d_dot, got_dot.size() * sizeof(shelf::bf16_t),
                    hipMemcpyDeviceToHost);
    double max_dot = 0.0;
    for (std::size_t i = 0; i < got_dot.size(); ++i) {
        max_dot = std::max<double>(
            max_dot, std::fabs(static_cast<double>(bf16_to_float(got_dot[i].data)) -
                               want_dot[i]));
    }
    std::fprintf(stderr, "ple stream dot max_abs=%g\n", max_dot);
    check(max_dot <= 2.0e-2 * kHidden, "ple stream dot matches the CPU reference");

    const auto scaled = [&]() {
        std::vector<shelf::bf16_t> out(kRows * kHc);
        const float scale = 1.0f / std::sqrt(static_cast<float>(kHidden));
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i].data = shelf::f32_to_bf16_rne(bf16_to_float(got_dot[i].data) * scale);
        return out;
    }();
    (void)hipMemcpy(d_scaled, scaled.data(), scaled.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice);

    shelf::kernel::PleGateArgs gate_args;
    gate_args.input = static_cast<const shelf::bf16_t*>(d_scaled);
    gate_args.output = static_cast<shelf::bf16_t*>(d_gate);
    gate_args.rows = kRows;
    gate_args.hidden = kHc;
    gate_args.input_stride = kHc;
    gate_args.output_stride = kHc;
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
    mul_args.gate_stride = kHc;
    mul_args.value_stride = kHidden;
    mul_args.output_stride = kWidth;
    check(shelf::kernel::launch_ple_broadcast_mul(mul_args, nullptr) == hipSuccess,
          "ple broadcast launch");
    check(hipDeviceSynchronize() == hipSuccess, "ple broadcast sync");

    std::vector<shelf::bf16_t> got_gate(kRows * kHc);
    std::vector<shelf::bf16_t> got_broadcast(kRows * kWidth);
    (void)hipMemcpy(got_gate.data(), d_gate, got_gate.size() * sizeof(shelf::bf16_t),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(got_broadcast.data(), d_broadcast,
                    got_broadcast.size() * sizeof(shelf::bf16_t),
                    hipMemcpyDeviceToHost);

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
    check(max_broadcast <= 2.0e-2, "ple per-stream broadcast matches the CPU reference");

    (void)hipFree(d_key);
    (void)hipFree(d_query);
    (void)hipFree(d_dot);
    (void)hipFree(d_scaled);
    (void)hipFree(d_gate);
    (void)hipFree(d_value);
    (void)hipFree(d_broadcast);

    {
        void* d_scaled_bf16 = nullptr;
        check(hipMalloc(&d_scaled_bf16, got_gate.size() * sizeof(shelf::bf16_t)) ==
                  hipSuccess,
              "hipMalloc scale bf16");
        const float factor = 0.25f;
        check(shelf::kernel::launch_scale_bf16(
                  static_cast<const shelf::bf16_t*>(d_gate), kHc,
                  static_cast<shelf::bf16_t*>(d_scaled_bf16), kHc, kRows, kHc, factor,
                  nullptr) == hipSuccess,
              "scale bf16 launch");
        check(hipDeviceSynchronize() == hipSuccess, "scale bf16 sync");
        std::vector<shelf::bf16_t> got_scaled(kRows * kHc);
        (void)hipMemcpy(got_scaled.data(), d_scaled_bf16,
                        got_scaled.size() * sizeof(shelf::bf16_t),
                        hipMemcpyDeviceToHost);
        double max_scaled = 0.0;
        for (std::size_t i = 0; i < got_scaled.size(); ++i) {
            max_scaled = std::max<double>(
                max_scaled,
                std::fabs(static_cast<double>(bf16_to_float(got_scaled[i].data)) -
                          bf16_to_float(got_gate[i].data) * factor));
        }
        std::fprintf(stderr, "scale bf16 max_abs=%g\n", max_scaled);
        check(max_scaled <= 2.0e-2, "scale bf16 matches the CPU reference");
        (void)hipFree(d_scaled_bf16);
    }

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
