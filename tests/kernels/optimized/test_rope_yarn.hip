#include <shelf/kernels/optimized/attention/rope.h>
#include <shelf/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr uint32_t kHalf = 32u;
constexpr float kTheta = 10000000.0f;
constexpr uint32_t kOriginalLen = 262144u;

constexpr double kGoldenFactor2[kHalf] = {
    1, 0.60429638624191284, 0.36517414450645447, 0.22067339718341827,
    0.13335214555263519, 0.080584213137626648, 0.048696752637624741,
    0.029427273198962212, 0.017782794311642647, 0.010746078565716743,
    0.0064938166178762913, 0.0039241896010935307, 0.0023713738191872835,
    0.0014330126577988267, 0.00086596433538943529, 0.00049059290904551744,
    0.00027669931296259165, 0.00015526493370998651, 8.660864841658622e-05,
    4.7975852794479579e-05, 2.6356034140917473e-05, 1.4334168554341886e-05,
    7.6996320785838179e-06, 4.6528602979378775e-06, 2.8117067358834902e-06,
    1.6991041320579825e-06, 1.0267624475090997e-06, 6.2046888160693925e-07,
    3.7494712046282075e-07, 2.2657918918866926e-07, 1.3692098832507327e-07,
    8.274085416815069e-08,
};

constexpr double kGoldenFactor4[kHalf] = {
    1, 0.60429638624191284, 0.36517414450645447, 0.22067339718341827,
    0.13335214555263519, 0.080584213137626648, 0.048696752637624741,
    0.029427273198962212, 0.017782794311642647, 0.010746078565716743,
    0.0064938166178762913, 0.0039241896010935307, 0.0023713738191872835,
    0.0014330126577988267, 0.00086596433538943529, 0.00047423981595784426,
    0.00025693507632240653, 0.00013734975073020905, 7.2173868829850107e-05,
    3.707224823301658e-05, 1.844922371674329e-05, 8.7597700257902034e-06,
    3.849816039291909e-06, 2.3264301489689387e-06, 1.4058533679417451e-06,
    8.4955206602899125e-07, 5.1338122375454986e-07, 3.1023444080346962e-07,
    1.8747356023141037e-07, 1.1328959459433463e-07, 6.8460494162536634e-08,
    4.1370427084075345e-08,
};

int g_fail = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg);
    }
}

bool compare_table(const std::vector<double>& got, const double* golden, double tol,
                   const char* label) {
    double max_rel = 0.0;
    for (uint32_t i = 0; i < kHalf; ++i) {
        const double rel = std::fabs(got[i] - golden[i]) / std::fabs(golden[i]);
        if (rel > max_rel) max_rel = rel;
    }
    std::printf("%s max_rel=%.3e tol=%.1e\n", label, max_rel, tol);
    return max_rel <= tol;
}

}

int main() {
    int dev = 0;
    if (hipGetDeviceCount(&dev) != hipSuccess || dev == 0) {
        std::printf("SKIP no GPU\n");
        return 77;
    }
    double* d_table = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&d_table), kHalf * sizeof(double)) !=
        hipSuccess) {
        std::printf("FAIL hipMalloc\n");
        return 1;
    }

    for (int pass = 0; pass < 3; ++pass) {
        const float factor = pass == 0 ? 1.0f : (pass == 1 ? 2.0f : 4.0f);
        hipError_t err;
        if (pass == 0) {
            err = ::shelf::kernel::launch_rope_inv_freq_init(d_table, kHalf, kTheta, nullptr);
        } else {
            err = ::shelf::kernel::launch_rope_inv_freq_yarn_init(
                d_table, kHalf, kTheta, factor, kOriginalLen, 32.0f, 1.0f, true,
                nullptr);
        }
        if (err != hipSuccess) {
            std::printf("FAIL launch factor %g: %s\n", factor, hipGetErrorString(err));
            ++g_fail;
            continue;
        }
        std::vector<double> table(kHalf);
        if (hipMemcpy(table.data(), d_table, kHalf * sizeof(double),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL readback\n");
            ++g_fail;
            continue;
        }
        if (pass == 0) {
            std::vector<double> ref(kHalf);
            for (uint32_t i = 0; i < kHalf; ++i) {
                ref[i] = 1.0 / std::pow(static_cast<double>(kTheta),
                                        static_cast<double>(i) /
                                            static_cast<double>(kHalf));
            }
            check(compare_table(table, ref.data(), 1e-15, "native double identity"),
                  "native table equals 1/theta^(i/32) in double");
        } else if (pass == 1) {
            check(compare_table(table, kGoldenFactor2, 1e-6,
                                "yarn 2x vs transformers 5.14.1"),
                  "yarn 2x table matches transformers golden");
        } else {
            check(compare_table(table, kGoldenFactor4, 1e-6,
                                "yarn 4x vs transformers 5.14.1"),
                  "yarn 4x table matches transformers golden");
        }
        bool finite = true;
        for (uint32_t i = 0; i < kHalf; ++i) {
            if (!std::isfinite(table[i]) || table[i] <= 0.0) finite = false;
        }
        check(finite, "table entries are finite and positive");
    }

    shelf::gpu::discard_cleanup_result(hipFree(d_table));
    constexpr uint32_t plamo_half = 64u;
    const uint32_t indices[] = {0u, 8u, 16u, 24u, 32u, 40u, 48u, 56u, 63u};
    const double golden_local[] = {1.0, 0.3162277638912201, 0.10000000149011612,
        0.03162277862429619, 0.009999999776482582, 0.003162277862429619,
        0.0010000000474974513, 0.0003162277862429619, 0.00011547819303814322};
    const double golden_full[] = {1.0, 0.17782793939113617, 0.027626225724816322,
        0.0021215747110545635, 3.906250185536919e-06, 6.946403914298571e-07,
        1.2352646194813133e-07, 2.1966458874089767e-08, 4.847413137554213e-09};
    d_table = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&d_table), plamo_half * sizeof(double)) != hipSuccess) {
        std::printf("FAIL plamo hipMalloc\n");
        return 1;
    }
    for (int pass = 0; pass < 2; ++pass) {
        const hipError_t err = pass == 0
            ? ::shelf::kernel::launch_rope_inv_freq_init(d_table, plamo_half, 10000.0f, nullptr)
            : ::shelf::kernel::launch_rope_inv_freq_yarn_init(
                d_table, plamo_half, 1000000.0f, 256.0f, 4096u, 32.0f, 1.0f, false,
                nullptr);
        check(err == hipSuccess, "PLaMo RoPE table launch");
        std::vector<double> table(plamo_half);
        check(hipMemcpy(table.data(), d_table, plamo_half * sizeof(double),
                        hipMemcpyDeviceToHost) == hipSuccess, "PLaMo RoPE table readback");
        const double* golden = pass == 0 ? golden_local : golden_full;
        for (size_t i = 0; i < sizeof(indices) / sizeof(indices[0]); ++i) {
            check(std::fabs(table[indices[i]] - golden[i]) / golden[i] < 1e-6,
                  "PLaMo RoPE table matches reference");
        }
    }
    shelf::gpu::discard_cleanup_result(hipFree(d_table));
    if (g_fail != 0) {
        std::printf("%d FAIL\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
