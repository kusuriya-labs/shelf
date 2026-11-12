#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
void check(hipError_t e) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(e));
        std::exit(1);
    }
}
template <class T> struct Buffer {
    T *p = nullptr;
    explicit Buffer(size_t size) { check(hipMalloc(&p, size * sizeof(T))); }
    ~Buffer() {
        if (p)
            check(hipFree(p));
    }
};

bool run(uint32_t rows, uint32_t bits, uint32_t count) {
    constexpr uint32_t k = 1024, n = 384, input_stride = k + 4, out_stride = n + 4;
    constexpr uint32_t ss = k / 128 * 4;
    const size_t weights = size_t(n) * k * bits / 16;
    const size_t plane = size_t(rows) * out_stride;
    Buffer<uint16_t> x(size_t(rows) * input_stride), suh(k * count), svh(n * count), t(weights * count);
    Buffer<int8_t> codes(size_t(rows) * k * count);
    Buffer<float> scales(size_t(rows) * ss / 4 * count), y(size_t(rows) * n * 3 * count);
    Buffer<uint16_t> out(plane * count);
    std::mt19937 rng(17 + rows + bits * count);
    std::vector<uint16_t> hx(size_t(rows) * input_stride), ht(weights * count), hu(k * count), hv(n * count);
    for (auto &v : hx)
        v = static_cast<uint16_t>(0x3f00 + rng() % 256);
    for (auto &v : ht)
        v = static_cast<uint16_t>(rng());
    for (auto &v : hu)
        v = rng() & 1 ? 0x3c00 : 0xbc00;
    for (auto &v : hv)
        v = static_cast<uint16_t>(0x3800 + rng() % 2048);
    check(hipMemcpy(x.p, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    check(hipMemcpy(t.p, ht.data(), ht.size() * 2, hipMemcpyHostToDevice));
    check(hipMemcpy(suh.p, hu.data(), hu.size() * 2, hipMemcpyHostToDevice));
    check(hipMemcpy(svh.p, hv.data(), hv.size() * 2, hipMemcpyHostToDevice));
    check(hipMemset(out.p, 0x55, plane * count * 2));
    shelf::kernel::PlamoExl3ProjectionGroup args{};
    args.input = reinterpret_cast<const shelf::bf16_t *>(x.p);
    args.rows = rows;
    args.k = k;
    args.input_stride = input_stride;
    args.scale_stride_bytes = ss;
    args.count = count;
    for (uint32_t i = 0; i < count; ++i) {
        auto &p = args.p[i];
        p.n = 128 * (i + 1);
        p.splits = i + 1;
        p.out_stride = out_stride;
        p.trellis = t.p + weights * i;
        p.suh = suh.p + k * i;
        p.svh = svh.p + n * i;
        p.codes = codes.p + size_t(rows) * k * i;
        p.scales = scales.p + size_t(rows) * ss / 4 * i;
        p.yprime = y.p + size_t(rows) * n * 3 * i;
        p.out = reinterpret_cast<shelf::bf16_t *>(out.p + plane * i);
        check(shelf::kernel::launch_exl3_mul24_rotate_quantize(args.input, input_stride, p.suh, p.codes,
                                                               p.scales, rows, k, k, ss, nullptr));
        check(shelf::kernel::launch_exl3_mul24_gemm(p.codes, p.scales, k, ss, p.trellis, bits, p.yprime, rows,
                                                    p.n, k, p.splits, nullptr));
        check(shelf::kernel::launch_exl3_mul24_rotate_out(p.yprime, p.svh, p.out,
                                                          shelf::kernel::Exl3OutputDType::BF16, rows, p.n,
                                                          out_stride, p.splits, nullptr));
    }
    std::vector<uint16_t> baseline(plane * count), candidate(baseline.size());
    check(hipMemcpy(baseline.data(), out.p, baseline.size() * 2, hipMemcpyDeviceToHost));
    check(hipMemset(out.p, 0x55, plane * count * 2));
    check(shelf::kernel::launch_plamo_exl3_projection(args, bits, nullptr));
    check(hipMemcpy(candidate.data(), out.p, candidate.size() * 2, hipMemcpyDeviceToHost));
    if (baseline != candidate) {
        size_t diffs = 0;
        for (size_t i = 0; i < baseline.size(); ++i)
            diffs += baseline[i] != candidate[i];
        std::fprintf(stderr, "rows=%u bits=%u count=%u diffs=%zu\n", rows, bits, count, diffs);
        return false;
    }
    args.count = 4;
    if (shelf::kernel::launch_plamo_exl3_projection(args, bits, nullptr) != hipErrorInvalidValue)
        return false;
    return true;
}
}

int main() {
    for (uint32_t rows : {1u, 4u, 16u, 17u, 32u, 33u, 48u, 49u, 64u, 70u, 128u, 192u, 255u})
        for (uint32_t bits : {4u, 5u, 8u})
            for (uint32_t count : {1u, 2u, 3u})
                if (!run(rows, bits, count))
                    return 1;
    std::puts("PLaMo projection: BF16 outputs and padding match separate projections");
}
