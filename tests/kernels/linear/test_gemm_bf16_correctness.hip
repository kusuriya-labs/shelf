#include <phaseshift/kernels/correctness/standalone/gemm_bf16_correctness.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <cstdint>

#define HIP_ASSERT(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        return 1; \
    } \
} while(0)

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        printf("PASS: %s\n", name);
    } else {
        failed++;
        printf("FAIL: %s\n", name);
    }
}

__host__ __device__ ps::bf16_t f32_to_bf16(float f) {
    uint32_t ui;
    __builtin_memcpy(&ui, &f, sizeof(float));
    uint16_t bf = static_cast<uint16_t>((ui + 0x8000U) >> 16);
    return ps::bf16_t{bf};
}

float bf16_to_f32(ps::bf16_t bf) {
    uint32_t ui = static_cast<uint32_t>(bf.data) << 16;
    float f;
    std::memcpy(&f, &ui, sizeof(float));
    return f;
}

__global__ void fill_weight_by_row(
    ps::bf16_t* weight,
    uint32_t rows,
    uint32_t k)
{
    const uint64_t count = static_cast<uint64_t>(rows) * k;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t row = i / k;
    float value = (row & 1u) ? 0.5f : 1.0f;
    weight[i] = f32_to_bf16(value);
}

__global__ void fill_input_pattern(ps::bf16_t* input, uint32_t k) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    input[i] = f32_to_bf16((i & 1u) ? 0.5f : 1.0f);
}

int test_small_random(ps::gpu::GpuArena& arena) {
    printf("\n--- BF16 GEMM small random (rows=3 N=33 K=65) ---\n");

    const uint32_t rows = 3;
    const uint32_t n = 33;
    const uint32_t k = 65;

    auto w_alloc = arena.allocate_aligned((std::size_t)n * k * sizeof(ps::bf16_t), 16);
    if (!w_alloc.ok()) { printf("FAIL: small weight alloc\n"); return -1; }
    auto x_alloc = arena.allocate_aligned((std::size_t)rows * k * sizeof(ps::bf16_t), 16);
    if (!x_alloc.ok()) { printf("FAIL: small input alloc\n"); return -1; }
    auto y_alloc = arena.allocate_aligned((std::size_t)rows * n * sizeof(ps::bf16_t), 16);
    if (!y_alloc.ok()) { printf("FAIL: small output alloc\n"); return -1; }

    auto* w = static_cast<ps::bf16_t*>(w_alloc.release().data());
    auto* x = static_cast<ps::bf16_t*>(x_alloc.release().data());
    auto* y = static_cast<ps::bf16_t*>(y_alloc.release().data());

    std::mt19937 rng(20260906u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<ps::bf16_t> h_w((std::size_t)n * k);
    std::vector<ps::bf16_t> h_x((std::size_t)rows * k);
    for (auto& v : h_w) v = f32_to_bf16(dist(rng));
    for (auto& v : h_x) v = f32_to_bf16(dist(rng));

    HIP_ASSERT(hipMemcpy(w, h_w.data(), h_w.size() * sizeof(ps::bf16_t), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(x, h_x.data(), h_x.size() * sizeof(ps::bf16_t), hipMemcpyHostToDevice));

    hipError_t launch = ps::kernel::detail::launch_gemm_bf16_baseline(w, x, y, n, rows, k, 0);
    check(launch == hipSuccess, "small launch");
    HIP_ASSERT(hipStreamSynchronize(0));

    std::vector<ps::bf16_t> h_y((std::size_t)rows * n);
    HIP_ASSERT(hipMemcpy(h_y.data(), y, h_y.size() * sizeof(ps::bf16_t), hipMemcpyDeviceToHost));

    float max_abs = 0.0f;
    bool finite = true;
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t m = 0; m < n; ++m) {
            float acc = 0.0f;
            for (uint32_t ki = 0; ki < k; ++ki) {
                acc += bf16_to_f32(h_w[(std::size_t)m * k + ki]) * bf16_to_f32(h_x[(std::size_t)t * k + ki]);
            }
            float got = bf16_to_f32(h_y[(std::size_t)t * n + m]);
            if (!std::isfinite(got)) finite = false;
            float d = std::abs(got - acc);
            if (d > max_abs) max_abs = d;
        }
    }
    printf("small max abs diff: %.6f\n", max_abs);
    check(finite, "small all finite");
    check(max_abs <= 5e-2f, "small atol 5e-2");

    return 0;
}

int test_27b_case(
    ps::gpu::GpuArena& arena,
    ps::bf16_t* weight,
    uint32_t out_features,
    uint32_t k,
    float expected_even,
    float expected_odd,
    const char* name)
{
    char title[128];
    std::snprintf(title, sizeof(title), "\n--- BF16 GEMM 27B %s (rows=1 N=%u K=%u) ---\n", name, out_features, k);
    printf("%s", title);

    auto x_alloc = arena.allocate_aligned((std::size_t)k * sizeof(ps::bf16_t), 16);
    if (!x_alloc.ok()) { printf("FAIL: 27B input alloc\n"); return -1; }
    auto y_alloc = arena.allocate_aligned((std::size_t)out_features * sizeof(ps::bf16_t), 16);
    if (!y_alloc.ok()) { printf("FAIL: 27B output alloc\n"); return -1; }

    auto* x = static_cast<ps::bf16_t*>(x_alloc.release().data());
    auto* y = static_cast<ps::bf16_t*>(y_alloc.release().data());

    const uint32_t block = 256;
    const uint32_t weight_count = out_features * k;
    fill_weight_by_row<<<(weight_count + block - 1) / block, block, 0, 0>>>(weight, out_features, k);
    fill_input_pattern<<<(k + block - 1) / block, block, 0, 0>>>(x, k);

    hipError_t launch = ps::kernel::detail::launch_gemm_bf16_baseline(weight, x, y, out_features, 1, k, 0);
    check(launch == hipSuccess, "27B launch");
    HIP_ASSERT(hipStreamSynchronize(0));

    std::vector<ps::bf16_t> h_y(out_features);
    HIP_ASSERT(hipMemcpy(h_y.data(), y, h_y.size() * sizeof(ps::bf16_t), hipMemcpyDeviceToHost));

    float max_abs = 0.0f;
    bool finite = true;
    for (uint32_t m = 0; m < out_features; ++m) {
        float expected = (m & 1u) ? expected_odd : expected_even;
        float got = bf16_to_f32(h_y[m]);
        if (!std::isfinite(got)) {
            finite = false;
            continue;
        }
        float d = std::abs(got - expected);
        if (d > max_abs) max_abs = d;
    }
    printf("27B max abs diff: %.6f\n", max_abs);
    check(finite, "27B all finite");
    check(max_abs <= 0.5f, "27B max abs within 0.5");

    return 0;
}

}

int main() {
    printf("=== BF16 GEMM Correctness Tests ===\n");

    int device_count = 0;
    HIP_ASSERT(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("SKIP: no GPU available\n");
        return 77;
    }
    HIP_ASSERT(hipSetDevice(0));

    auto arena_result = ps::gpu::GpuArena::create(0, 256 * 1024 * 1024);
    if (!arena_result.ok()) {
        printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    if (test_small_random(arena) != 0) {
        printf("\nResults: %d passed, %d failed\n", passed, failed);
        return 1;
    }

    const uint32_t elems = 17408u * 5120u;
    auto weight_alloc = arena.allocate_aligned((std::size_t)elems * sizeof(ps::bf16_t), 16);
    if (!weight_alloc.ok()) { printf("FAIL: 27B weight alloc\n"); return 1; }
    auto* weight = static_cast<ps::bf16_t*>(weight_alloc.release().data());

    if (test_27b_case(arena, weight, 17408, 5120, 3840.0f, 1920.0f, "gate/up") != 0) {
        printf("\nResults: %d passed, %d failed\n", passed, failed);
        return 1;
    }
    if (test_27b_case(arena, weight, 5120, 17408, 13056.0f, 6528.0f, "down") != 0) {
        printf("\nResults: %d passed, %d failed\n", passed, failed);
        return 1;
    }

    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
