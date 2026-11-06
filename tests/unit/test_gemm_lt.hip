#include <shelf/core/memory/types.h>
#include <shelf/inference/execution/gemm_lt.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++failed;
    }
}

bool hip_ok(hipError_t error, const char* what) {
    if (error == hipSuccess) return true;
    std::printf("FAIL %s: %s\n", what, hipGetErrorString(error));
    ++failed;
    return false;
}

float bf16_to_f32(shelf::bf16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void run_case(std::uint32_t rows, std::uint32_t out_features, std::uint32_t k,
              bool output_bf16) {
    std::mt19937 rng(1234u + rows);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<shelf::bf16_t> weight(static_cast<std::size_t>(out_features) * k);
    std::vector<shelf::bf16_t> input(static_cast<std::size_t>(rows) * k);
    for (auto& v : weight) v = shelf::bf16_t{shelf::f32_to_bf16_rne(dist(rng))};
    for (auto& v : input) v = shelf::bf16_t{shelf::f32_to_bf16_rne(dist(rng))};

    shelf::bf16_t* d_weight = nullptr;
    shelf::bf16_t* d_input = nullptr;
    void* d_output = nullptr;
    const std::size_t out_bytes =
        static_cast<std::size_t>(rows) * out_features * (output_bf16 ? 2u : 4u);
    bool ok = true;
    ok = ok && hip_ok(hipMalloc(&d_weight, weight.size() * sizeof(shelf::bf16_t)),
                      "malloc weight");
    ok = ok && hip_ok(hipMalloc(&d_input, input.size() * sizeof(shelf::bf16_t)),
                      "malloc input");
    ok = ok && hip_ok(hipMalloc(&d_output, out_bytes), "malloc output");
    if (!ok) return;
    ok = ok && hip_ok(hipMemcpy(d_weight, weight.data(),
                                weight.size() * sizeof(shelf::bf16_t),
                                hipMemcpyHostToDevice),
                      "copy weight");
    ok = ok && hip_ok(hipMemcpy(d_input, input.data(),
                                input.size() * sizeof(shelf::bf16_t),
                                hipMemcpyHostToDevice),
                      "copy input");
    if (!ok) return;

    ok = ok && hip_ok(shelf::execution::gemm_bf16_lt(
                          d_weight, d_input, d_output, rows, out_features, k, k,
                          out_features, output_bf16, nullptr),
                      "gemm_bf16_lt");
    ok = ok && hip_ok(hipStreamSynchronize(nullptr), "sync");
    if (!ok) return;

    std::vector<float> got(static_cast<std::size_t>(rows) * out_features);
    if (output_bf16) {
        std::vector<shelf::bf16_t> tmp(got.size());
        ok = ok && hip_ok(hipMemcpy(tmp.data(), d_output, out_bytes,
                                    hipMemcpyDeviceToHost),
                          "read bf16 output");
        if (ok)
            for (std::size_t i = 0; i < tmp.size(); ++i) got[i] = bf16_to_f32(tmp[i]);
    } else {
        ok = ok && hip_ok(hipMemcpy(got.data(), d_output, out_bytes,
                                    hipMemcpyDeviceToHost),
                          "read f32 output");
    }
    (void)hipFree(d_weight);
    (void)hipFree(d_input);
    (void)hipFree(d_output);
    if (!ok) return;

    double worst = 0.0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t o = 0; o < out_features; ++o) {
            double acc = 0.0;
            for (std::uint32_t i = 0; i < k; ++i) {
                acc += static_cast<double>(
                           bf16_to_f32(input[static_cast<std::size_t>(r) * k + i])) *
                       static_cast<double>(
                           bf16_to_f32(weight[static_cast<std::size_t>(o) * k + i]));
            }
            const double value = got[static_cast<std::size_t>(r) * out_features + o];
            const double rel = std::fabs(value - acc) / (std::fabs(acc) + 1.0e-3);
            worst = std::max(worst, rel);
        }
    }
    const double tolerance = output_bf16 ? 1.0e-2 : 1.0e-3;
    std::printf("rows=%u out=%u k=%u dtype=%s max_rel=%.6g\n", rows, out_features, k,
                output_bf16 ? "bf16" : "f32", worst);
    check(worst <= tolerance, "hipBLASLt bf16 gemm matches the CPU reference");
}

}  // namespace

int main() {
    run_case(1u, 256u, 256u, true);
    run_case(5u, 256u, 256u, true);
    run_case(127u, 256u, 256u, true);
    run_case(128u, 256u, 256u, true);
    run_case(300u, 512u, 1024u, true);
    run_case(64u, 256u, 256u, false);
    std::printf(failed == 0 ? "PASS\n" : "FAIL (%d)\n", failed);
    return failed == 0 ? 0 : 1;
}
