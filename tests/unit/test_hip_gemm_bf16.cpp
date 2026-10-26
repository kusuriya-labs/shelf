// shelf unit test — vendored PhaseShift bf16 GEMM on gfx1201 (PR-008a).
// Parity vs independent f64 reference + launch timing sanity.
// tile-refactor PR-0b: covers all five Bf16GemmConfigId variants
// (Wmma / WmmaWide / WmmaKPartition / ExactRows / ExactRowsDot2), boundary
// row counts (1..17, 63..65), 16-nonmultiple N/K, non-trivial strides, and
// BF16 output dtype. Numeric gate vs f64 stays the same shape as before.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <shelf/backend/hip/kernels/gemm/linear_bf16.h>
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

// runs one (rows,n,k,istride,ostride) case against all five configs
int run_case(uint32_t rows, uint32_t n, uint32_t k,
             uint32_t istride = 0u, uint32_t ostride = 0u) {
    if (istride == 0u) istride = k;
    if (ostride == 0u) ostride = n;

    std::mt19937 rng(1234 + rows * 7919u + n * 31u + k);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> w(static_cast<size_t>(n) * k);
    std::vector<shelf::bf16_t> x(static_cast<size_t>(rows) * istride);
    for (auto& v : w) v = shelf::f32_to_bf16(dist(rng));
    for (auto& v : x) v = shelf::f32_to_bf16(dist(rng));
    // poison the padding rows of x beyond logical k (stride > k)
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t kk = k; kk < istride; ++kk)
            x[static_cast<size_t>(r) * istride + kk] = shelf::f32_to_bf16(123.5f);

    // f64 reference over the EXACT bf16 operands (kernel accumulates fp32)
    std::vector<double> ref(static_cast<size_t>(rows) * n, 0.0);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t o = 0; o < n; ++o) {
            double acc = 0.0;
            for (uint32_t kk = 0; kk < k; ++kk)
                acc += static_cast<double>(shelf::bf16_to_f32(w[static_cast<size_t>(o) * k + kk])) *
                       static_cast<double>(shelf::bf16_to_f32(x[static_cast<size_t>(r) * istride + kk]));
            ref[static_cast<size_t>(r) * n + o] = acc;
        }

    const size_t out_float_elems = static_cast<size_t>(rows) * ostride;
    void *dw, *dx, *dy;
    HIP_OK(hipMalloc(&dw, w.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&dx, x.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&dy, out_float_elems * sizeof(float)));  // max(F32,BF16) footprint
    HIP_OK(hipMemcpy(dw, w.data(), w.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dx, x.data(), x.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));

    std::vector<float> out(out_float_elems, 0.0f);
    auto test_config = [&](const Bf16GemmConfig& cfg, const char* name,
                           Bf16GemmOutputDType odt) {
        const bool expect_unsupported =
            (cfg.id == Bf16GemmConfigId::ExactRows ||
             cfg.id == Bf16GemmConfigId::ExactRowsDot2) &&
            (rows < (cfg.id == Bf16GemmConfigId::ExactRowsDot2 ? 2u : 1u) ||
             rows > shelf::hipk::kBf16GemmExactRowsMax);
        // sentinel fill so an unsupported variant must leave output untouched
        const hipError_t me = hipMemset(dy, 0x5A, out_float_elems * sizeof(float));
        check(me == hipSuccess, "memset");
        if (me != hipSuccess) return;
        const hipError_t err = launch_gemm_bf16(cfg, static_cast<const shelf::bf16_t*>(dw),
                                                static_cast<const shelf::bf16_t*>(dx), dy, odt,
                                                rows, n, k, istride, ostride, 0);
        if (expect_unsupported) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "%s rows=%u expected-not-supported", name, rows);
            check(err == hipErrorNotSupported, msg);
            return;
        }
        check(err == hipSuccess, name);
        if (err != hipSuccess) return;
        const hipError_t ce =
            hipMemcpy(out.data(), dy, out.size() * sizeof(float), hipMemcpyDeviceToHost);
        check(ce == hipSuccess, "copyback");
        if (ce != hipSuccess) return;
        // element-wise slack: fp32 accumulation error + (for BF16 out) one
        // bf16 rounding of the result (~|ref|/128 rel, so use 1/64 headroom)
        const double acc_tol = 1e-4 * std::sqrt(static_cast<double>(k));
        const bool bf_out = (odt == Bf16GemmOutputDType::BF16);
        double worst_excess = 0.0;  // max(diff - tol_i)
        double worst_diff = 0.0;
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t o = 0; o < n; ++o) {
                const double want = ref[static_cast<size_t>(r) * n + o];
                const double got = bf_out
                    ? shelf::bf16_to_f32(reinterpret_cast<const shelf::bf16_t*>(out.data())
                                         [static_cast<size_t>(r) * ostride + o])
                    : out[static_cast<size_t>(r) * ostride + o];
                const double tol_i = acc_tol + (bf_out ? 2.0 * std::fabs(want) / 128.0 : 0.0);
                const double d = std::fabs(got - want);
                worst_diff = std::max(worst_diff, d);
                worst_excess = std::max(worst_excess, d - tol_i);
            }
        }
        char msg[160];
        std::snprintf(msg, sizeof msg, "%s rows=%u n=%u k=%u is=%u os=%u out=%s max|diff|=%.3e (acc tol %.1e, worst excess %.3e)",
                      name, rows, n, k, istride, ostride, bf_out ? "bf16" : "f32",
                      worst_diff, acc_tol, worst_excess);
        check(worst_excess <= 0.0, msg);
    };

    const Bf16GemmConfigId ids[] = {
        Bf16GemmConfigId::ExactRows, Bf16GemmConfigId::ExactRowsDot2,
        Bf16GemmConfigId::Wmma, Bf16GemmConfigId::WmmaWide,
        Bf16GemmConfigId::WmmaKPartition,
    };
    const char* names[] = {"ExactRows", "ExactRowsDot2", "Wmma", "WmmaWide",
                           "WmmaKPartition"};
    for (int i = 0; i < 5; ++i) {
        Bf16GemmConfig cfg{ids[i], static_cast<uint8_t>(rows <= 255u ? rows : 0u)};
        test_config(cfg, names[i], Bf16GemmOutputDType::F32);
        // BF16 store path (skip huge outputs to keep the test light; Wmma-family
        // and exact rows both take the dtype branch inside the same bodies)
        if (static_cast<size_t>(rows) * n <= 32768u)
            test_config(cfg, names[i], Bf16GemmOutputDType::BF16);
    }

    // timing: rows=1 GEMV at real 4B shapes
    if (rows <= 64u) {
        Bf16GemmConfig cfg{Bf16GemmConfigId::ExactRows, static_cast<uint8_t>(rows)};
        for (int i = 0; i < 5; ++i)
            (void)launch_gemm_bf16(cfg, static_cast<const shelf::bf16_t*>(dw),
                                   static_cast<const shelf::bf16_t*>(dx), dy,
                                   Bf16GemmOutputDType::F32, rows, n, k, istride, ostride, 0);
        HIP_OK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i)
            (void)launch_gemm_bf16(cfg, static_cast<const shelf::bf16_t*>(dw),
                                   static_cast<const shelf::bf16_t*>(dx), dy,
                                   Bf16GemmOutputDType::F32, rows, n, k, istride, ostride, 0);
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
    // PR-0b boundary matrix (guide 2.3): rows 1..17 / 63..65, N/K 16-remainders
    if (run_case(1, 271, 40) != 0) return 1;
    if (run_case(2, 64, 16) != 0) return 1;
    if (run_case(4, 17, 1) != 0) return 1;
    if (run_case(15, 33, 63) != 0) return 1;
    if (run_case(16, 256, 256) != 0) return 1;
    if (run_case(17, 33, 63) != 0) return 1;      // crosses exact-rows kernel max 16
    if (run_case(63, 16, 16) != 0) return 1;      // exact-rows chunked
    if (run_case(64, 32, 15) != 0) return 1;
    if (run_case(65, 64, 272) != 0) return 1;     // crosses kBf16GemmWideMinRows=65
    // non-trivial strides (istride multiple of 8 elems = 16B for vector paths)
    if (run_case(4, 512, 256, 264, 520) != 0) return 1;
    if (run_case(16, 33, 17, 24, 40) != 0) return 1;
    std::printf("test_hip_gemm_bf16: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
