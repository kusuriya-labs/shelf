#include <shelf/kernels/optimized/linear/bf16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define HIP_ASSERT(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(err)); \
        return 1; \
    } \
} while (0)

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

__host__ __device__ shelf::bf16_t f32_to_bf16(float f) {
    uint32_t ui;
    __builtin_memcpy(&ui, &f, sizeof(float));
    return shelf::bf16_t{static_cast<uint16_t>((ui + 0x8000u) >> 16)};
}

float bf16_to_f32(shelf::bf16_t b) {
    uint32_t ui = static_cast<uint32_t>(b.data) << 16;
    float f;
    std::memcpy(&f, &ui, sizeof(float));
    return f;
}

__global__ void fill_weight_row_parity(shelf::bf16_t* weight, uint32_t rows, uint32_t k) {
    const uint64_t count = static_cast<uint64_t>(rows) * k;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t row = i / k;
    const float v = ((row >> 3) & 1u) ? -0.75f : 1.0f;
    weight[i] = f32_to_bf16(v);
}

float weight_row_value(uint32_t row) {
    return ((row >> 3) & 1u) ? -0.75f : 1.0f;
}

struct CaseSpec {
    uint32_t n;
    uint32_t k;
    uint32_t rows;
    shelf::kernel::Bf16GemmOutputDType out_dtype;
    float rtol;
    float atol;
    uint32_t subset;
    const char* name;
};

int run_case(const CaseSpec& s) {
    char title[192];
    std::snprintf(title, sizeof(title),
                  "\n--- exact rows GEMM %s (n=%u k=%u rows=%u subset=%u) ---\n",
                  s.name, s.n, s.k, s.rows, s.subset);
    printf("%s", title);

    const size_t w_bytes = static_cast<size_t>(s.n) * s.k * sizeof(shelf::bf16_t);
    const size_t x_bytes = static_cast<size_t>(s.rows) * s.k * sizeof(shelf::bf16_t);
    const uint32_t out_elem =
        s.out_dtype == shelf::kernel::Bf16GemmOutputDType::BF16 ? 2u : 4u;
    const size_t y_bytes = static_cast<size_t>(s.rows) * s.n * out_elem;

    shelf::bf16_t* w = nullptr;
    shelf::bf16_t* x = nullptr;
    void* y_exact = nullptr;
    void* y_wmma = nullptr;
    void* y_dot2 = nullptr;
    void* y_dot2_repeat = nullptr;
    HIP_ASSERT(hipMalloc(&w, w_bytes));
    HIP_ASSERT(hipMalloc(&x, x_bytes));
    HIP_ASSERT(hipMalloc(&y_exact, y_bytes));
    HIP_ASSERT(hipMalloc(&y_wmma, y_bytes));
    HIP_ASSERT(hipMalloc(&y_dot2, y_bytes));
    HIP_ASSERT(hipMalloc(&y_dot2_repeat, y_bytes));

    const uint32_t block = 256;
    fill_weight_row_parity<<<(static_cast<uint32_t>(s.n * s.k) + block - 1) / block,
                             block, 0, 0>>>(w, s.n, s.k);

    std::mt19937 rng(20261017u + s.n + s.k);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> h_x(static_cast<size_t>(s.rows) * s.k);
    for (auto& v : h_x) v = f32_to_bf16(dist(rng));
    HIP_ASSERT(hipMemcpy(x, h_x.data(), h_x.size() * sizeof(shelf::bf16_t),
                         hipMemcpyHostToDevice));

    std::vector<float> rowsum(s.rows, 0.0f);
    for (uint32_t t = 0; t < s.rows; ++t) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < s.k; ++j) {
            acc += bf16_to_f32(h_x[static_cast<size_t>(t) * s.k + j]);
        }
        rowsum[t] = acc;
    }

    shelf::kernel::Bf16GemmConfig exact_cfg{shelf::kernel::Bf16GemmConfigId::ExactRows,
                                         static_cast<uint8_t>(s.rows)};
    shelf::kernel::Bf16GemmConfig wmma_cfg{shelf::kernel::Bf16GemmConfigId::Wmma, 0u};
    shelf::kernel::Bf16GemmConfig dot2_cfg{shelf::kernel::Bf16GemmConfigId::ExactRowsDot2,
                                        static_cast<uint8_t>(s.rows)};

    hipError_t err = shelf::kernel::launch_gemm_bf16(
        exact_cfg, w, x, y_exact, s.out_dtype, s.rows, s.n, s.k, s.k, s.n, 0);
    check(err == hipSuccess, "exact launch");
    err = shelf::kernel::launch_gemm_bf16(
        wmma_cfg, w, x, y_wmma, s.out_dtype, s.rows, s.n, s.k, s.k, s.n, 0);
    check(err == hipSuccess, "wmma launch");
    HIP_ASSERT(hipStreamSynchronize(0));
    err = shelf::kernel::launch_gemm_bf16(
        dot2_cfg, w, x, y_dot2, s.out_dtype, s.rows, s.n, s.k, s.k, s.n, 0);
    check(err == hipSuccess, "dot2 launch");
    err = shelf::kernel::launch_gemm_bf16(
        dot2_cfg, w, x, y_dot2_repeat, s.out_dtype, s.rows, s.n, s.k, s.k, s.n, 0);
    check(err == hipSuccess, "dot2 repeat launch");
    HIP_ASSERT(hipStreamSynchronize(0));

    std::vector<uint8_t> h_exact(y_bytes);
    std::vector<uint8_t> h_wmma(y_bytes);
    HIP_ASSERT(hipMemcpy(h_exact.data(), y_exact, y_bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(h_wmma.data(), y_wmma, y_bytes, hipMemcpyDeviceToHost));
    std::vector<uint8_t> h_dot2(y_bytes);
    std::vector<uint8_t> h_dot2_repeat(y_bytes);
    HIP_ASSERT(hipMemcpy(h_dot2.data(), y_dot2, y_bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(h_dot2_repeat.data(), y_dot2_repeat, y_bytes,
                         hipMemcpyDeviceToHost));

    std::vector<uint32_t> targets;
    if (s.subset == 0u || s.subset >= s.n) {
        targets.resize(s.n);
        for (uint32_t m = 0; m < s.n; ++m) targets[m] = m;
    } else {
        targets.reserve(s.subset);
        std::mt19937 idx_rng(777u + s.n);
        std::uniform_int_distribution<uint32_t> idx_dist(0, s.n - 1u);
        targets.push_back(0u);
        targets.push_back(s.n - 1u);
        while (targets.size() < s.subset) {
            const uint32_t v = idx_dist(idx_rng);
            bool dup = false;
            for (const uint32_t t : targets) {
                if (t == v) { dup = true; break; }
            }
            if (!dup) targets.push_back(v);
        }
    }

    uint64_t violations = 0;
    uint64_t nonfinite = 0;
    float worst_ratio = 0.0f;
    for (const uint32_t m : targets) {
        const float wr = weight_row_value(m);
        for (uint32_t t = 0; t < s.rows; ++t) {
            const float ref = wr * rowsum[t];
            float got = 0.0f;
            if (s.out_dtype == shelf::kernel::Bf16GemmOutputDType::BF16) {
                const auto* p = static_cast<const shelf::bf16_t*>(
                    static_cast<const void*>(h_exact.data()));
                got = bf16_to_f32(p[static_cast<size_t>(t) * s.n + m]);
            } else {
                const auto* p = static_cast<const float*>(
                    static_cast<const void*>(h_exact.data()));
                got = p[static_cast<size_t>(t) * s.n + m];
            }
            if (!std::isfinite(got)) {
                nonfinite++;
                continue;
            }
            const float tol = s.atol + s.rtol * std::fabs(ref);
            const float d = std::fabs(got - ref);
            if (d > tol) violations++;
            const float ratio = tol > 0.0f ? d / tol : 0.0f;
            if (ratio > worst_ratio) worst_ratio = ratio;
        }
    }
    printf("exact vs host ref: checks=%llu violations=%llu nonfinite=%llu worst_tol_ratio=%.3f\n",
           static_cast<unsigned long long>(targets.size()) * s.rows,
           static_cast<unsigned long long>(violations),
           static_cast<unsigned long long>(nonfinite), worst_ratio);
    check(nonfinite == 0u, "exact all finite");
    check(violations == 0u, "exact within tolerance");

    uint64_t byte_diff = 0;
    uint64_t byte_total = 0;
    const size_t elem_bytes = out_elem;
    const size_t elems = static_cast<size_t>(s.rows) * s.n;
    for (const uint32_t m : targets) {
        for (uint32_t t = 0; t < s.rows; ++t) {
            const size_t off = (static_cast<size_t>(t) * s.n + m) * elem_bytes;
            byte_total++;
            if (std::memcmp(h_exact.data() + off, h_wmma.data() + off, elem_bytes) != 0) {
                byte_diff++;
            }
        }
    }
    printf("exact-vs-wmma byte-diff (informational): %llu/%llu\n",
           static_cast<unsigned long long>(byte_diff),
           static_cast<unsigned long long>(byte_total));

    uint64_t dot2_violations = 0;
    uint64_t dot2_nonfinite = 0;
    float dot2_worst_ratio = 0.0f;
    for (const uint32_t m : targets) {
        const float wr = weight_row_value(m);
        for (uint32_t t = 0; t < s.rows; ++t) {
            const float ref = wr * rowsum[t];
            float got = 0.0f;
            if (s.out_dtype == shelf::kernel::Bf16GemmOutputDType::BF16) {
                const auto* p = static_cast<const shelf::bf16_t*>(
                    static_cast<const void*>(h_dot2.data()));
                got = bf16_to_f32(p[static_cast<size_t>(t) * s.n + m]);
            } else {
                const auto* p = static_cast<const float*>(
                    static_cast<const void*>(h_dot2.data()));
                got = p[static_cast<size_t>(t) * s.n + m];
            }
            if (!std::isfinite(got)) {
                dot2_nonfinite++;
                continue;
            }
            const float tol = s.atol + s.rtol * std::fabs(ref);
            const float d = std::fabs(got - ref);
            if (d > tol) dot2_violations++;
            const float ratio = tol > 0.0f ? d / tol : 0.0f;
            if (ratio > dot2_worst_ratio) dot2_worst_ratio = ratio;
        }
    }
    printf("dot2 vs host ref: violations=%llu nonfinite=%llu worst_tol_ratio=%.3f\n",
           static_cast<unsigned long long>(dot2_violations),
           static_cast<unsigned long long>(dot2_nonfinite), dot2_worst_ratio);
    check(dot2_nonfinite == 0u, "dot2 all finite");
    check(dot2_violations == 0u, "dot2 within tolerance");

    uint64_t dot2_repeat_diff = 0;
    for (const uint32_t m : targets) {
        for (uint32_t t = 0; t < s.rows; ++t) {
            const size_t off = (static_cast<size_t>(t) * s.n + m) * elem_bytes;
            if (std::memcmp(h_dot2.data() + off, h_dot2_repeat.data() + off, elem_bytes) !=
                0) {
                dot2_repeat_diff++;
            }
        }
    }
    check(dot2_repeat_diff == 0u, "dot2 deterministic");

    uint64_t dot2_byte_diff = 0;
    for (const uint32_t m : targets) {
        for (uint32_t t = 0; t < s.rows; ++t) {
            const size_t off = (static_cast<size_t>(t) * s.n + m) * elem_bytes;
            if (std::memcmp(h_exact.data() + off, h_dot2.data() + off, elem_bytes) != 0) {
                dot2_byte_diff++;
            }
        }
    }
    printf("exact-vs-dot2 byte-diff (informational): %llu/%llu\n",
           static_cast<unsigned long long>(dot2_byte_diff),
           static_cast<unsigned long long>(byte_total));

    HIP_ASSERT(hipFree(y_dot2_repeat));
    HIP_ASSERT(hipFree(y_dot2));

    HIP_ASSERT(hipFree(w));
    HIP_ASSERT(hipFree(x));
    HIP_ASSERT(hipFree(y_exact));
    HIP_ASSERT(hipFree(y_wmma));
    return 0;
}

}

int main() {
    printf("=== BF16 GEMM ExactRows(draft 経路) 正しさテスト ===\n");

    int device_count = 0;
    HIP_ASSERT(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("SKIP: no GPU available\n");
        return 77;
    }
    HIP_ASSERT(hipSetDevice(0));

    const CaseSpec cases[] = {
        {64u, 65u, 7u, shelf::kernel::Bf16GemmOutputDType::F32, 1.0e-3f, 1.0e-2f, 0u,
         "tail k=65"},
        {1024u, 5120u, 7u, shelf::kernel::Bf16GemmOutputDType::BF16, 8.0e-3f, 5.0e-2f, 0u,
         "k/v proj"},
        {4096u, 5120u, 7u, shelf::kernel::Bf16GemmOutputDType::BF16, 8.0e-3f, 5.0e-2f, 0u,
         "q/o proj"},
        {17408u, 5120u, 7u, shelf::kernel::Bf16GemmOutputDType::BF16, 8.0e-3f, 5.0e-2f, 0u,
         "gate/up"},
        {5120u, 17408u, 7u, shelf::kernel::Bf16GemmOutputDType::BF16, 8.0e-3f, 5.0e-2f, 0u,
         "down"},
        {248320u, 5120u, 7u, shelf::kernel::Bf16GemmOutputDType::F32, 1.0e-3f, 1.0e-2f,
         2048u, "proposal lm_head"},
    };

    for (const CaseSpec& c : cases) {
        if (run_case(c) != 0) {
            printf("\nResults: %d passed, %d failed\n", passed, failed);
            return 1;
        }
    }

    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
