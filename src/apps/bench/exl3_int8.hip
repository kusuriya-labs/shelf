#include <shelf/kernels/optimized/linear/exl3.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); return 1; } } while (0)
}

int run_exl3_int8(int argc, char** argv) {
    uint32_t n = 5120, k = 5120, rows = 1, bits = 4, split = 0, scratch_rows = 1024, working_mib = 256, iterations = 16;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--help")) {
            std::printf("shelf-bench exl3-int8 --n N --k K --rows R --bits B [--split S] [--scratch-rows R] [--working-mib M] [--iterations N]\n");
            return 0;
        }
        if (i + 1 >= argc) return 2;
        const char* key = argv[i++]; char* end = nullptr;
        unsigned long value = std::strtoul(argv[i], &end, 10);
        if (*end || value > UINT32_MAX) return 2;
        if (!std::strcmp(key, "--n")) n = value;
        else if (!std::strcmp(key, "--k")) k = value;
        else if (!std::strcmp(key, "--rows")) rows = value;
        else if (!std::strcmp(key, "--bits")) bits = value;
        else if (!std::strcmp(key, "--split")) split = value;
        else if (!std::strcmp(key, "--scratch-rows")) scratch_rows = value;
        else if (!std::strcmp(key, "--working-mib")) working_mib = value;
        else if (!std::strcmp(key, "--iterations")) iterations = value;
        else return 2;
    }
    if (!n || n % 128 || !k || k % 128 || !rows || rows > 16 || bits < 1 || bits > 8 || !iterations || !working_mib || scratch_rows < rows) return 2;
    if (!split) split = shelf::kernel::exl3_mul24_gemm_splits(rows, n, k, scratch_rows, bits);
    if (!split || split > k / 128 || split * rows > scratch_rows) return 2;
    const size_t weight_bytes = size_t(n) * k * bits / 8;
    const size_t copies = std::max<size_t>(1, (size_t(working_mib) * 1024 * 1024 + weight_bytes - 1) / weight_bytes);
    const uint32_t scale_stride = ((k / 128 * 4) + 15) & ~15u;
    void *weights, *x, *suh, *svh, *codes, *scales, *accum, *out;
    HIP_CHECK(hipMalloc(&weights, weight_bytes * copies)); HIP_CHECK(hipMalloc(&x, size_t(rows) * k * 2));
    HIP_CHECK(hipMalloc(&suh, k * 2)); HIP_CHECK(hipMalloc(&svh, n * 2));
    HIP_CHECK(hipMalloc(&codes, size_t(rows) * k)); HIP_CHECK(hipMalloc(&scales, size_t(rows) * scale_stride));
    HIP_CHECK(hipMalloc(&accum, size_t(split) * rows * n * 4)); HIP_CHECK(hipMalloc(&out, size_t(rows) * n * 2));
    HIP_CHECK(hipMemset(weights, 0x35, weight_bytes * copies)); HIP_CHECK(hipMemset(x, 0x3f, size_t(rows) * k * 2));
    HIP_CHECK(hipMemset(suh, 0x3c, k * 2)); HIP_CHECK(hipMemset(svh, 0x3c, n * 2));
    auto run = [&](int stage, size_t i) -> hipError_t {
        hipError_t e = hipSuccess;
        if (stage == 0 || stage == 3) e = shelf::kernel::launch_exl3_mul24_rotate_quantize(static_cast<shelf::bf16_t*>(x), k, static_cast<uint16_t*>(suh), static_cast<int8_t*>(codes), static_cast<float*>(scales), rows, k, k, scale_stride, nullptr);
        if (e == hipSuccess && (stage == 1 || stage == 3)) e = shelf::kernel::launch_exl3_mul24_gemm(static_cast<int8_t*>(codes), static_cast<float*>(scales), k, scale_stride, reinterpret_cast<uint16_t*>(static_cast<char*>(weights) + (i % copies) * weight_bytes), bits, static_cast<float*>(accum), rows, n, k, split, nullptr);
        if (e == hipSuccess && (stage == 2 || stage == 3)) e = shelf::kernel::launch_exl3_mul24_rotate_out(static_cast<float*>(accum), static_cast<uint16_t*>(svh), out, shelf::kernel::Exl3OutputDType::BF16, rows, n, n, split, nullptr);
        return e;
    };
    for (uint32_t i = 0; i < std::max<uint32_t>(iterations, copies); ++i) HIP_CHECK(run(3, i));
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t begin, end; HIP_CHECK(hipEventCreate(&begin)); HIP_CHECK(hipEventCreate(&end));
    std::printf("n=%u k=%u rows=%u bits=%u split=%u weight_bytes=%zu copies=%zu", n, k, rows, bits, split, weight_bytes, copies);
    for (int stage = 0; stage < 4; ++stage) {
        std::vector<float> samples;
        for (int rep = -2; rep < 5; ++rep) {
            HIP_CHECK(hipEventRecord(begin));
            for (uint32_t i = 0; i < iterations; ++i) HIP_CHECK(run(stage, size_t(rep + 2) * iterations + i));
            HIP_CHECK(hipEventRecord(end)); HIP_CHECK(hipEventSynchronize(end)); float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
            if (rep >= 0) samples.push_back(ms * 1000 / iterations);
        }
        std::sort(samples.begin(), samples.end());
        const char* names[] = {"rotate_in_us", "gemm_us", "rotate_out_us", "total_us"};
        std::printf(" %s=%.3f min=%.3f max=%.3f", names[stage], samples[2], samples[0], samples[4]);
        if (stage == 1 || stage == 3) std::printf(" weight_GBps=%.2f", weight_bytes / samples[2] / 1000.0);
    }
    std::printf("\n");
    HIP_CHECK(hipEventDestroy(begin)); HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipFree(weights)); HIP_CHECK(hipFree(x)); HIP_CHECK(hipFree(suh)); HIP_CHECK(hipFree(svh));
    HIP_CHECK(hipFree(codes)); HIP_CHECK(hipFree(scales)); HIP_CHECK(hipFree(accum)); HIP_CHECK(hipFree(out));
    return 0;
}
