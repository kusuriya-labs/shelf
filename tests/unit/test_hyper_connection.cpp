#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/hyper_connection.h>

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
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
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

    std::mt19937 rng(1234);
    const auto gate = random_bf16(kRows * kWidth, rng);
    const auto normed = random_bf16(kRows * kWidth, rng);
    const auto raw = random_bf16(kRows * kWidth, rng);
    const auto y = random_bf16(kRows * kHidden, rng);
    const auto injection = random_bf16(kRows * kHc, rng);

    std::vector<float> want_mix(kRows * kHidden);
    for (std::uint32_t row = 0; row < kRows; ++row) {
        for (std::uint32_t i = 0; i < kHidden; ++i) {
            float acc = 0.0f;
            for (std::uint32_t h = 0; h < kHc; ++h) {
                const std::uint32_t offset = row * kWidth + h * kHidden + i;
                acc += bf16_to_float(gate[offset].data) *
                       bf16_to_float(normed[offset].data);
            }
            want_mix[row * kHidden + i] = acc / static_cast<float>(kHc);
        }
    }
    std::vector<float> want_apply(kRows * kWidth);
    for (std::uint32_t row = 0; row < kRows; ++row) {
        for (std::uint32_t h = 0; h < kHc; ++h) {
            const float scale = bf16_to_float(injection[row * kHc + h].data);
            for (std::uint32_t i = 0; i < kHidden; ++i) {
                const std::uint32_t offset = row * kWidth + h * kHidden + i;
                want_apply[offset] = bf16_to_float(raw[offset].data) +
                                     bf16_to_float(y[row * kHidden + i].data) * scale;
            }
        }
    }

    void* d_gate = nullptr;
    void* d_normed = nullptr;
    void* d_mix = nullptr;
    void* d_raw = nullptr;
    void* d_y = nullptr;
    void* d_injection = nullptr;
    void* d_apply = nullptr;
    check(hipMalloc(&d_gate, gate.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc gate");
    check(hipMalloc(&d_normed, normed.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc normed");
    check(hipMalloc(&d_mix, kRows * kHidden * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc mix");
    check(hipMalloc(&d_raw, raw.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc raw");
    check(hipMalloc(&d_y, y.size() * sizeof(shelf::bf16_t)) == hipSuccess, "hipMalloc y");
    check(hipMalloc(&d_injection, injection.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc injection");
    check(hipMalloc(&d_apply, kRows * kWidth * sizeof(shelf::bf16_t)) == hipSuccess,
          "hipMalloc apply");

    hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_normed, normed.data(), normed.size() * sizeof(shelf::bf16_t),
              hipMemcpyHostToDevice);
    hipMemcpy(d_raw, raw.data(), raw.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y.data(), y.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_injection, injection.data(), injection.size() * sizeof(shelf::bf16_t),
              hipMemcpyHostToDevice);

    shelf::kernel::HyperMixArgs mix_args;
    mix_args.gate = d_gate;
    mix_args.normed = d_normed;
    mix_args.output = d_mix;
    mix_args.rows = kRows;
    mix_args.hc = kHc;
    mix_args.hidden = kHidden;
    mix_args.gate_stride = kWidth;
    mix_args.normed_stride = kWidth;
    mix_args.output_stride = kHidden;
    check(shelf::kernel::launch_hyper_mix(mix_args, nullptr) == hipSuccess,
          "hyper mix launch");
    check(hipDeviceSynchronize() == hipSuccess, "hyper mix sync");

    shelf::kernel::HyperApplyArgs apply_args;
    apply_args.raw = d_raw;
    apply_args.y = d_y;
    apply_args.injection = d_injection;
    apply_args.output = d_apply;
    apply_args.rows = kRows;
    apply_args.hc = kHc;
    apply_args.hidden = kHidden;
    apply_args.raw_stride = kWidth;
    apply_args.y_stride = kHidden;
    apply_args.injection_stride = kHc;
    apply_args.output_stride = kWidth;
    check(shelf::kernel::launch_hyper_apply(apply_args, nullptr) == hipSuccess,
          "hyper apply launch");
    check(hipDeviceSynchronize() == hipSuccess, "hyper apply sync");

    std::vector<shelf::bf16_t> got_mix(kRows * kHidden);
    std::vector<shelf::bf16_t> got_apply(kRows * kWidth);
    hipMemcpy(got_mix.data(), d_mix, got_mix.size() * sizeof(shelf::bf16_t),
              hipMemcpyDeviceToHost);
    hipMemcpy(got_apply.data(), d_apply, got_apply.size() * sizeof(shelf::bf16_t),
              hipMemcpyDeviceToHost);

    double max_mix = 0.0;
    double max_apply = 0.0;
    for (std::size_t i = 0; i < got_mix.size(); ++i) {
        max_mix = std::max(max_mix,
                           std::fabs(static_cast<double>(bf16_to_float(got_mix[i].data)) -
                                     want_mix[i]));
    }
    for (std::size_t i = 0; i < got_apply.size(); ++i) {
        max_apply = std::max(max_apply,
                             std::fabs(static_cast<double>(bf16_to_float(got_apply[i].data)) -
                                       want_apply[i]));
    }
    std::fprintf(stderr, "hyper mix max_abs=%g apply max_abs=%g\n", max_mix, max_apply);
    check(max_mix <= 2.0e-2, "hyper mix matches the CPU reference");
    check(max_apply <= 2.0e-2, "hyper apply matches the CPU reference");

    (void)hipFree(d_gate);
    (void)hipFree(d_normed);
    (void)hipFree(d_mix);
    (void)hipFree(d_raw);
    (void)hipFree(d_y);
    (void)hipFree(d_injection);
    (void)hipFree(d_apply);

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
