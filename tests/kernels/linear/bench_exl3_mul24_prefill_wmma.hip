#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Shape {
    uint32_t n;
    uint32_t k;
};

void die(const char* what, hipError_t e) {
    if (e == hipSuccess) return;
    std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
}

template <typename T>
struct Buf {
    T* p = nullptr;
    explicit Buf(size_t count) { die("malloc", hipMalloc(&p, count * sizeof(T))); }
    ~Buf() { if (p != nullptr) (void)hipFree(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
};

}  // namespace

int main(int argc, char** argv) {
    uint32_t rows = 2048u;
    int warmup = 5;
    int iters = 50;
    std::vector<Shape> shapes = {
        {5120u, 5120u}, {10240u, 5120u}, {6144u, 5120u},
        {5120u, 6144u}, {17408u, 5120u}, {5120u, 17408u},
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) rows = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--shape" && i + 2 < argc) {
            if (shapes.size() == 6u) shapes.clear();
            shapes.push_back(Shape{static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10)),
                                   static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10))});
        } else {
            std::fprintf(stderr, "usage: %s [--rows N] [--warmup N] [--iters N] [--shape N K]...\n", argv[0]);
            return 2;
        }
    }
    if (shapes.empty()) return 2;

    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    die("event", hipEventCreate(&ev0));
    die("event", hipEventCreate(&ev1));

    std::printf("rows=%u warmup=%d iters=%d\n", rows, warmup, iters);
    std::printf("shape_n,shape_k,gpu_ms,weight_gb_s,tops\n");
    double total_ms = 0.0;
    for (const Shape& s : shapes) {
        const uint32_t groups = s.k / 128u;
        const uint32_t scale_stride = (groups * 4u + 15u) & ~15u;
        Buf<int8_t> codes(static_cast<size_t>(rows) * s.k);
        Buf<float> scales(static_cast<size_t>(rows) * scale_stride / 4u);
        Buf<int8_t> weights(static_cast<size_t>(s.n) * s.k);
        Buf<float> yprime(static_cast<size_t>(rows) * s.n);
        die("memset", hipMemset(codes.p, 0x11, static_cast<size_t>(rows) * s.k));
        die("memset", hipMemset(scales.p, 0x3c, static_cast<size_t>(rows) * scale_stride / 4u));
        die("memset", hipMemset(weights.p, 0x22, static_cast<size_t>(s.n) * s.k));
        die("memset", hipMemset(yprime.p, 0, static_cast<size_t>(rows) * s.n * sizeof(float)));

        for (int i = 0; i < warmup; ++i)
            die("launch", shelf::kernel::launch_exl3_mul24_prefill_wmma(
                              codes.p, scales.p, s.k, scale_stride, weights.p, yprime.p, rows, s.n, s.k, nullptr));
        die("sync", hipDeviceSynchronize());

        die("record", hipEventRecord(ev0));
        for (int i = 0; i < iters; ++i)
            die("launch", shelf::kernel::launch_exl3_mul24_prefill_wmma(
                              codes.p, scales.p, s.k, scale_stride, weights.p, yprime.p, rows, s.n, s.k, nullptr));
        die("record", hipEventRecord(ev1));
        die("sync", hipEventSynchronize(ev1));
        float ms = 0.0f;
        die("elapsed", hipEventElapsedTime(&ms, ev0, ev1));
        ms /= static_cast<float>(iters);
        const double weights_bytes = static_cast<double>(s.n) * s.k;
        const double macs = static_cast<double>(rows) * s.n * s.k;
        std::printf("%u,%u,%.4f,%.1f,%.1f\n", s.n, s.k, ms, weights_bytes / ms / 1e6,
                    2.0 * macs / ms / 1e9);
        total_ms += ms;
    }
    std::printf("sum_gpu_ms=%.4f\n", total_ms);
    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);
    return 0;
}
