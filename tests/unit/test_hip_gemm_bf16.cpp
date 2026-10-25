// shelf unit test — vendored PhaseShift bf16 GEMM on gfx1201 (PR-008a).
// Parity vs independent f64 reference + launch timing sanity.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <shelf/backend/hip/kernels/linear_bf16.h>
#include <shelf/foundation/numeric.hpp>

namespace {

using shelf::hipk::Bf16GemmConfig;
using shelf::hipk::Bf16GemmConfigId;
using shelf::hipk::Bf16GemmOutputDType;
using shelf::hipk::launch_gemm_bf16;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

#define HIP_OK(call)                                                        \
    do {                                                                    \
        const hipError_t e_ = (call);                                       \
        if (e_ != hipSuccess) {                                             \
            std::printf("FAIL hip: %s @%d: %s\n", #call, __LINE__,          \
                        hipGetErrorString(e_));                             \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int run_case(uint32_t rows, uint32_t n, uint32_t k) {
    std::mt19937 rng(1234 + rows * 7919u + n * 31u + k);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> w(static_cast<size_t>(n) * k);
    std::vector<shelf::bf16_t> x(static_cast<size_t>(rows) * k);
    for (auto& v : w) v = shelf::f32_to_bf16(dist(rng));
    for (auto& v : x) v = shelf::f32_to_bf16(dist(rng));

    // f64 reference over the EXACT bf16 operands (kernel accumulates fp32)
    std::vector<double> ref(static_cast<size_t>(rows) * n, 0.0);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t o = 0; o < n; ++o) {
            double acc = 0.0;
            for (uint32_t kk = 0; kk < k; ++kk)
                acc += static_cast<double>(shelf::bf16_to_f32(w[static_cast<size_t>(o) * k + kk])) *
                       static_cast<double>(shelf::bf16_to_f32(x[static_cast<size_t>(r) * k + kk]));
            ref[static_cast<size_t>(r) * n + o] = acc;
        }

    void *dw, *dx, *dy;
    HIP_OK(hipMalloc(&dw, w.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&dx, x.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&dy, ref.size() * sizeof(float)));
    HIP_OK(hipMemcpy(dw, w.data(), w.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dx, x.data(), x.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));

    std::vector<float> out(ref.size(), 0.0f);
    auto test_config = [&](const Bf16GemmConfig& cfg, const char* name) {
        const hipError_t me = hipMemset(dy, 0, ref.size() * sizeof(float));
        check(me == hipSuccess, "memset");
        if (me != hipSuccess) return;
        const hipError_t err = launch_gemm_bf16(cfg, static_cast<const shelf::bf16_t*>(dw),
                                                static_cast<const shelf::bf16_t*>(dx), dy,
                                                Bf16GemmOutputDType::F32, rows, n, k, k, n, 0);
        check(err == hipSuccess, name);
        if (err != hipSuccess) return;
        const hipError_t ce =
            hipMemcpy(out.data(), dy, out.size() * sizeof(float), hipMemcpyDeviceToHost);
        check(ce == hipSuccess, "copyback");
        if (ce != hipSuccess) return;
        double max_abs = 0.0;
        for (size_t i = 0; i < out.size(); ++i) {
            const double d = std::fabs(out[i] - ref[i]);
            max_abs = std::max(max_abs, d);
        }
        // fp32 accumulation over bf16 products: error scales ~sqrt(K)*eps32*|x||w|
        const double tol = 1e-4 * std::sqrt(static_cast<double>(k));
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s rows=%u n=%u k=%u max|diff|=%.3e (tol %.1e)", name,
                      rows, n, k, max_abs, tol);
        check(max_abs <= tol, msg);
    };

    test_config(Bf16GemmConfig{Bf16GemmConfigId::ExactRows, static_cast<uint8_t>(rows)},
                "ExactRows");
    test_config(Bf16GemmConfig{Bf16GemmConfigId::Wmma, 0u}, "Wmma");

    // timing: rows=1 GEMV at real 4B shapes
    {
        Bf16GemmConfig cfg{Bf16GemmConfigId::ExactRows, static_cast<uint8_t>(rows)};
        for (int i = 0; i < 5; ++i)
            launch_gemm_bf16(cfg, static_cast<const shelf::bf16_t*>(dw),
                             static_cast<const shelf::bf16_t*>(dx), dy,
                             Bf16GemmOutputDType::F32, rows, n, k, k, n, 0);
        HIP_OK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i)
            launch_gemm_bf16(cfg, static_cast<const shelf::bf16_t*>(dw),
                             static_cast<const shelf::bf16_t*>(dx), dy,
                             Bf16GemmOutputDType::F32, rows, n, k, k, n, 0);
        HIP_OK(hipDeviceSynchronize());
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t0).count() / 100;
        const double gb = static_cast<double>(w.size()) * 2.0 / 1e9;
        std::printf("GEMM rows=%u n=%u k=%u: %lld us  (%.0f GB/s weight stream)\n", rows, n, k,
                    static_cast<long long>(us), gb / (us * 1e-6));
    }

    HIP_OK(hipFree(dw));
    HIP_OK(hipFree(dx));
    HIP_OK(hipFree(dy));
    return 0;
}

} // namespace

int main() {
    HIP_OK(hipSetDevice(0));
    if (run_case(1, 2560, 2560) != 0) return 1;   // q_proj-ish
    if (run_case(1, 9216, 2560) != 0) return 1;   // gate/up_proj
    if (run_case(4, 512, 256) != 0) return 1;     // small multi-row (chunked path)
    std::printf("test_hip_gemm_bf16: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
