// shelf unit test — N08: 複数行ぶんのルーター logits が 1 行版とビット一致すること。
//
// `moe_router_logit_rows` は行ごとの内訳（スレッド分担と加算順）を 1 行版と
// 同じにしてある。ここでは同じ入力から
//   A) `launch_moe_router_logit` を rows 回
//   B) `launch_moe_router_logit_rows` を 1 回
// を作り、f32 のビット列で比較する。
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/elementwise/aux_ops.h>
#include <shelf/foundation/numeric.hpp>

namespace {

using namespace shelf;
using namespace shelf::hipk;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}
void check_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        ++failures;
        std::printf("FAIL: %s (%s)\n", what, hipGetErrorString(e));
    }
}

void run_shape(std::uint32_t rows, std::uint32_t experts, std::uint32_t hidden,
               std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<bf16_t> x(std::size_t(rows) * hidden);
    std::vector<bf16_t> w(std::size_t(experts) * hidden);
    for (auto& v : x) v = f32_to_bf16(dist(rng));
    for (auto& v : w) v = f32_to_bf16(dist(rng));

    bf16_t* dx = nullptr;
    bf16_t* dw = nullptr;
    float* d_single = nullptr;
    float* d_rows = nullptr;
    check_hip(hipMalloc(&dx, x.size() * sizeof(bf16_t)), "malloc x");
    check_hip(hipMalloc(&dw, w.size() * sizeof(bf16_t)), "malloc w");
    check_hip(hipMalloc(&d_single, sizeof(float) * rows * experts), "malloc single");
    check_hip(hipMalloc(&d_rows, sizeof(float) * rows * experts), "malloc rows");
    check_hip(hipMemcpy(dx, x.data(), x.size() * sizeof(bf16_t), hipMemcpyHostToDevice),
              "copy x");
    check_hip(hipMemcpy(dw, w.data(), w.size() * sizeof(bf16_t), hipMemcpyHostToDevice),
              "copy w");

    for (std::uint32_t r = 0; r < rows; ++r)
        check_hip(launch_moe_router_logit(dx + std::size_t(r) * hidden, dw,
                                          d_single + std::size_t(r) * experts, experts,
                                          hidden, nullptr),
                  "router logits (1 row)");
    check_hip(launch_moe_router_logit_rows(dx, dw, d_rows, rows, experts, hidden, nullptr),
              "router logits (rows)");

    std::vector<float> single(std::size_t(rows) * experts);
    std::vector<float> multi(std::size_t(rows) * experts);
    check_hip(hipMemcpy(single.data(), d_single, sizeof(float) * single.size(),
                        hipMemcpyDeviceToHost),
              "copy single");
    check_hip(hipMemcpy(multi.data(), d_rows, sizeof(float) * multi.size(),
                        hipMemcpyDeviceToHost),
              "copy rows");
    const bool same = std::memcmp(single.data(), multi.data(),
                                  sizeof(float) * single.size()) == 0;
    if (!same) {
        double worst = 0.0;
        for (std::size_t i = 0; i < single.size(); ++i)
            worst = std::max(worst, std::fabs(double(single[i]) - double(multi[i])));
        std::printf("  不一致 rows=%u experts=%u hidden=%u 最大差 %.3e\n", rows, experts,
                    hidden, worst);
    }
    check(same, "router logits（複数行）は 1 行版の繰り返しとビット一致");

    hipFree(dx);
    hipFree(dw);
    hipFree(d_single);
    hipFree(d_rows);
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }
    std::printf("test_hip_moe_router: 複数行のルーター logits が 1 行版と一致（N08）\n");
    run_shape(1u, 128u, 2048u, 1u);
    run_shape(2u, 128u, 2048u, 2u);
    run_shape(3u, 128u, 2048u, 3u);
    run_shape(4u, 128u, 2048u, 4u);
    run_shape(4u, 64u, 512u, 5u);
    run_shape(4u, 256u, 1024u, 6u);
    run_shape(2u, 24u, 5120u, 7u);
    if (failures == 0) std::printf("test_hip_moe_router: OK\n");
    return failures == 0 ? 0 : 1;
}
