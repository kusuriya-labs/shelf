#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <hip/hip_runtime.h>
#include <algorithm>
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

template <bool BF16>
__global__ void scalar_rotate_reference(const float* y, const uint16_t* svh, void* out,
                                        uint32_t rows, uint32_t n, uint32_t stride, uint32_t splits) {
    using namespace shelf::kernel::exl3_detail;
    using namespace shelf::kernel::exl3_detail::mul24;
    __shared__ float smem[128];
    const uint32_t m = blockIdx.x, col = blockIdx.y * 128u + threadIdx.x;
    float v = 0.0f;
    for (uint32_t z = 0; z < splits; ++z)
        v += y[(static_cast<size_t>(z) * rows + m) * n + col];
    v = fwht128_smem(v, smem, threadIdx.x) * kInvSqrt128 * f16_bits_to_f32(svh[col]);
    const size_t pos = static_cast<size_t>(m) * stride + col;
    if constexpr (BF16) reinterpret_cast<uint16_t*>(out)[pos] = f32_to_bf16_rne_device(v);
    else reinterpret_cast<float*>(out)[pos] = v;
}

bool run_out_exact(uint32_t rows, uint32_t n, uint32_t splits, bool bf16, uint32_t padding,
                   bool unaligned = false, bool special = false) {
    const uint32_t stride = n + padding;
    const size_t values = static_cast<size_t>(rows) * stride;
    Buffer<float> y(static_cast<size_t>(splits) * rows * n + 1u);
    Buffer<uint16_t> svh(n);
    Buffer<uint32_t> reference(values + 1u), candidate(values + 1u);
    std::mt19937 rng(917u + rows + n + splits);
    std::vector<float> hy(static_cast<size_t>(splits) * rows * n);
    std::vector<uint16_t> hs(n);
    for (auto& v : hy) v = static_cast<float>(static_cast<int32_t>(rng() % 65536u) - 32768) / 8192.0f;
    if (special) {
        constexpr uint32_t patterns[] = {0u, 0x80000000u, 0x7f800000u, 0xff800000u,
                                         0x7fc00001u, 0xffc01002u, 1u, 0x80000001u};
        for (size_t i = 0; i < hy.size(); i += 128u)
            hy[i] = __builtin_bit_cast(float, patterns[(i / 128u) % 8u]);
    }
    for (auto& v : hs) v = static_cast<uint16_t>(0x3800u + rng() % 2048u);
    if (special) {
        constexpr uint16_t patterns[] = {0u, 0x8000u, 0x7c00u, 0xfc00u,
                                         0x7e01u, 0xfe02u, 1u, 0x8001u};
        for (size_t i = 0; i < hs.size(); ++i) hs[i] = patterns[i % 8u];
    }
    float* yp = y.p + (unaligned ? 1u : 0u);
    void* ref_out = reinterpret_cast<uint8_t*>(reference.p) + (unaligned ? (bf16 ? 2u : 4u) : 0u);
    void* candidate_out = reinterpret_cast<uint8_t*>(candidate.p) + (unaligned ? (bf16 ? 2u : 4u) : 0u);
    check(hipMemcpy(yp, hy.data(), hy.size() * sizeof(float), hipMemcpyHostToDevice));
    check(hipMemcpy(svh.p, hs.data(), hs.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    check(hipMemset(reference.p, 0xa5, (values + 1u) * sizeof(uint32_t)));
    check(hipMemset(candidate.p, 0xa5, (values + 1u) * sizeof(uint32_t)));
    const dim3 grid(rows, n / 128u);
    if (bf16) scalar_rotate_reference<true><<<grid, 128>>>(yp, svh.p, ref_out, rows, n, stride, splits);
    else scalar_rotate_reference<false><<<grid, 128>>>(yp, svh.p, ref_out, rows, n, stride, splits);
    check(hipGetLastError());
    check(shelf::kernel::launch_exl3_mul24_rotate_out(
        yp, svh.p, candidate_out, bf16 ? shelf::kernel::Exl3OutputDType::BF16 : shelf::kernel::Exl3OutputDType::F32,
        rows, n, stride, splits, nullptr));
    std::vector<uint32_t> before(values + 1u), after(values + 1u);
    check(hipMemcpy(before.data(), reference.p, (values + 1u) * sizeof(uint32_t), hipMemcpyDeviceToHost));
    check(hipMemcpy(after.data(), candidate.p, (values + 1u) * sizeof(uint32_t), hipMemcpyDeviceToHost));
    if (before != after)
        std::fprintf(stderr, "rotate rows=%u n=%u splits=%u bf16=%d padding=%u mismatch\n",
                     rows, n, splits, bf16, padding);
    return before == after;
}

bool run(uint32_t rows, uint32_t bits, uint32_t count, uint32_t k = 1024u, uint32_t n = 384u,
         uint32_t splits = 0u, bool allow_shape_fusion = true) {
    const uint32_t input_stride = k + 4, out_stride = n + 4;
    const uint32_t ss = k / 128 * 4;
    const size_t weights = size_t(n) * k * bits / 16;
    const size_t plane = size_t(rows) * out_stride;
    Buffer<uint16_t> x(size_t(rows) * input_stride), suh(k * count), svh(n * count), t(weights * count);
    Buffer<int8_t> codes(size_t(rows) * k * count);
    Buffer<float> scales(size_t(rows) * ss / 4 * count), y(size_t(rows) * n * std::max(3u, splits) * count);
    Buffer<uint16_t> out(plane * count);
    std::mt19937 rng(17 + rows + bits * count);
    std::vector<uint16_t> hx(size_t(rows) * input_stride), ht(weights * count), hu(k * count),
        hv(n * count);
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
        p.n = i == 0u ? n : 128u * (i + 1u);
        p.splits = splits ? splits : count == 1u ? 1u : i + 1u;
        p.out_stride = out_stride;
        p.trellis = t.p + weights * i;
        p.suh = suh.p + k * i;
        p.svh = svh.p + n * i;
        p.codes = codes.p + size_t(rows) * k * i;
        p.scales = scales.p + size_t(rows) * ss / 4 * i;
        p.yprime = y.p + size_t(rows) * n * std::max(3u, splits) * i;
        p.out = reinterpret_cast<shelf::bf16_t *>(out.p + plane * i);
        check(shelf::kernel::launch_exl3_mul24_rotate_quantize(args.input, input_stride, p.suh, p.codes,
                                                               p.scales, rows, k, k, ss, nullptr));
        check(shelf::kernel::launch_exl3_mul24_gemm(p.codes, p.scales, k, ss, p.trellis, bits, p.yprime,
                                                    rows, p.n, k, p.splits, nullptr));
        check(shelf::kernel::launch_exl3_mul24_rotate_out(p.yprime, p.svh, p.out,
                                                          shelf::kernel::Exl3OutputDType::BF16, rows, p.n,
                                                          out_stride, p.splits, nullptr));
    }
    std::vector<uint16_t> baseline(plane * count), candidate(baseline.size());
    check(hipMemcpy(baseline.data(), out.p, baseline.size() * 2, hipMemcpyDeviceToHost));
    check(hipMemset(out.p, 0x55, plane * count * 2));
    uint32_t launches = 99u;
    check(shelf::kernel::launch_plamo_exl3_projection(args, bits, nullptr, &launches, allow_shape_fusion));
    const bool input_fused = rows == 1u && bits == 4u &&
                             ((k == 2560u && args.p[0].n == 3072u) || (k == 3072u && args.p[0].n == 2560u));
    const bool output_fused = rows <= 16u && args.p[0].n >= 10240u && count == 1u;
    if (launches != (allow_shape_fusion && (input_fused || output_fused) ? 2u : 3u)) {
        std::fprintf(stderr, "launch count rows=%u n=%u k=%u bits=%u actual=%u\n", rows, n, k, bits,
                     launches);
        return false;
    }
    check(hipMemcpy(candidate.data(), out.p, candidate.size() * 2, hipMemcpyDeviceToHost));
    if (baseline != candidate) {
        size_t diffs = 0;
        for (size_t i = 0; i < baseline.size(); ++i)
            diffs += baseline[i] != candidate[i];
        std::fprintf(stderr, "rows=%u bits=%u count=%u diffs=%zu\n", rows, bits, count, diffs);
        return false;
    }
    args.count = 4;
    if (shelf::kernel::launch_plamo_exl3_projection(args, bits, nullptr, &launches) !=
            hipErrorInvalidValue ||
        launches != 0u)
        return false;
    return true;
}
bool run_flag(uint32_t rows, uint32_t active, uint32_t bits, uint32_t n = 384u, uint32_t k = 1024u) {
    const uint32_t ss = (k / 128u * 4u + 15u) & ~15u;
    Buffer<int8_t> codes(size_t(rows) * k);
    Buffer<float> scales(size_t(rows) * ss / 4u), y0(size_t(rows) * n), y1(size_t(rows) * n);
    Buffer<uint16_t> trellis(size_t(n) * k * bits / 16u);
    Buffer<uint32_t> count(1u);
    std::mt19937 rng(31u + rows + active + bits);
    std::vector<int8_t> hc(size_t(rows) * k);
    std::vector<float> hs(size_t(rows) * ss / 4u);
    std::vector<uint16_t> ht(size_t(n) * k * bits / 16u);
    for (auto &v : hc)
        v = static_cast<int8_t>(rng());
    for (auto &v : hs)
        v = float(1u + rng() % 256u) / 65536.0f;
    for (auto &v : ht)
        v = static_cast<uint16_t>(rng());
    check(hipMemcpy(codes.p, hc.data(), hc.size(), hipMemcpyHostToDevice));
    check(hipMemcpy(scales.p, hs.data(), hs.size() * 4u, hipMemcpyHostToDevice));
    check(hipMemcpy(trellis.p, ht.data(), ht.size() * 2u, hipMemcpyHostToDevice));
    check(hipMemcpy(count.p, &active, 4u, hipMemcpyHostToDevice));
    check(hipMemset(y0.p, 0x55, size_t(rows) * n * 4u));
    check(hipMemset(y1.p, 0x55, size_t(rows) * n * 4u));
    const uint32_t* active_rows = active == rows && rows <= 64u ? nullptr : count.p;
    check(shelf::kernel::launch_exl3_mul24_gemm(codes.p, scales.p, k, ss, trellis.p, bits, y0.p, rows, n, k,
                                                1u, nullptr, active_rows));
    check(shelf::kernel::launch_plamo_exl3_flagged_gemm(codes.p, scales.p, k, ss, trellis.p, bits, y1.p,
                                                        rows, count.p, n, k, nullptr));
    std::vector<uint32_t> h0(size_t(rows) * n), h1(h0.size());
    check(hipMemcpy(h0.data(), y0.p, h0.size() * 4u, hipMemcpyDeviceToHost));
    check(hipMemcpy(h1.data(), y1.p, h1.size() * 4u, hipMemcpyDeviceToHost));
    if (h0 != h1)
        std::fprintf(stderr, "flag rows=%u active=%u bits=%u mismatch\n", rows, active, bits);
    return h0 == h1;
}

bool run_i32(uint32_t rows, bool bf16) {
    constexpr uint32_t n = 1152, stride = n + 4u, cap = 48u;
    const size_t plane = size_t(rows) * stride * (bf16 ? 1u : 2u);
    Buffer<int32_t> acc(size_t(rows) * n), slots(rows);
    Buffer<float> scale(rows), yp(size_t(cap) * n);
    Buffer<uint16_t> svh(n), out0(plane), out1(plane);
    std::mt19937 rng(73u + rows);
    std::vector<int32_t> ha(size_t(rows) * n), hl(rows);
    std::vector<float> hs(rows), hy(size_t(cap) * n);
    std::vector<uint16_t> hv(n);
    for (auto &v : ha)
        v = static_cast<int32_t>(rng() % 200000u) - 100000;
    for (auto &v : hs)
        v = float(1u + rng() % 256u) / 65536.0f;
    for (auto &v : hy)
        v = float(static_cast<int32_t>(rng() % 20000u) - 10000) / 65536.0f;
    for (auto &v : hv)
        v = uint16_t(0x3800u + rng() % 2048u);
    for (uint32_t i = 0; i < rows; ++i)
        hl[i] = i % 3u == 0u ? int32_t(i % cap) : -1;
    check(hipMemcpy(acc.p, ha.data(), ha.size() * 4u, hipMemcpyHostToDevice));
    check(hipMemcpy(slots.p, hl.data(), hl.size() * 4u, hipMemcpyHostToDevice));
    check(hipMemcpy(scale.p, hs.data(), hs.size() * 4u, hipMemcpyHostToDevice));
    check(hipMemcpy(yp.p, hy.data(), hy.size() * 4u, hipMemcpyHostToDevice));
    check(hipMemcpy(svh.p, hv.data(), hv.size() * 2u, hipMemcpyHostToDevice));
    check(hipMemset(out0.p, 0x55, plane * 2u));
    check(hipMemset(out1.p, 0x55, plane * 2u));
    check(shelf::kernel::launch_exl3_mul24_rotate_out_i32(acc.p, scale.p, slots.p, yp.p, svh.p, out0.p,
                                                          bf16 ? shelf::kernel::Exl3OutputDType::BF16
                                                               : shelf::kernel::Exl3OutputDType::F32,
                                                          rows, n, stride, nullptr));
    check(shelf::kernel::launch_plamo_exl3_rotate_out_i32(acc.p, scale.p, slots.p, yp.p, svh.p, out1.p,
                                                          bf16, rows, n, stride, nullptr));
    std::vector<uint16_t> h0(plane), h1(plane);
    check(hipMemcpy(h0.data(), out0.p, plane * 2u, hipMemcpyDeviceToHost));
    check(hipMemcpy(h1.data(), out1.p, plane * 2u, hipMemcpyDeviceToHost));
    if (h0 != h1)
        std::fprintf(stderr, "i32 rows=%u bf16=%d mismatch\n", rows, bf16);
    return h0 == h1;
}
}

int main() {
    for (uint32_t rows : {1u, 7u, 8u, 10u, 16u, 32u, 38u, 40u, 64u, 255u, 256u, 257u, 258u, 272u, 288u, 319u, 320u, 384u, 448u, 511u, 512u, 513u})
        for (bool bf16 : {false, true})
            for (uint32_t splits : {1u, 2u, 5u})
                if (!run_out_exact(rows, 5120u, splits, bf16, 4u)) return 1;
    for (uint32_t rows : {1u, 8u, 32u})
        for (bool bf16 : {false, true})
            if (!run_out_exact(rows, 65536u, 2u, bf16, 4u) ||
                !run_out_exact(rows, 5120u, 2u, bf16, 1u) ||
                !run_out_exact(rows, 65536u, 2u, bf16, 4u, true) ||
                !run_out_exact(rows, 65536u, 2u, bf16, 4u, false, true)) return 1;
    for (uint32_t rows : {511u, 512u, 513u})
        for (bool bf16 : {false, true})
            if (!run_out_exact(rows, 1024u, 2u, bf16, 4u, false, true)) return 1;
    for (uint32_t rows : {256u, 511u})
        for (bool bf16 : {false, true})
            for (uint32_t splits : {1u, 5u})
                if (!run_out_exact(rows, 17408u, splits, bf16, 4u)) return 1;
    for (uint32_t rows : {256u, 511u})
        for (bool bf16 : {false, true})
            if (!run_out_exact(rows, 65536u, 1u, bf16, 4u)) return 1;
    for (uint32_t n : {6144u, 10240u, 12288u})
        for (uint32_t rows : {256u, 511u})
            for (bool bf16 : {false, true})
                if (!run_out_exact(rows, n, 2u, bf16, 4u)) return 1;
    for (bool bf16 : {false, true})
        for (uint32_t splits : {1u, 5u})
            if (!run_out_exact(128u, 10240u, splits, bf16, 4u) ||
                !run_out_exact(16u, 81920u, splits, bf16, 4u) ||
                !run_out_exact(128u, 12288u, splits, bf16, 4u) ||
                !run_out_exact(16u, 98304u, splits, bf16, 4u) ||
                !run_out_exact(128u, 8192u, splits, bf16, 4u) ||
                !run_out_exact(16u, 65536u, splits, bf16, 4u)) return 1;
    for (uint32_t rows : {1u, 4u, 16u, 17u, 32u, 33u, 48u, 49u, 64u, 70u, 128u, 192u, 255u})
        for (uint32_t bits : {4u, 5u, 8u})
            for (uint32_t count : {1u, 2u, 3u})
                if (!run(rows, bits, count))
                    return 1;
    for (uint32_t rows : {1u, 8u, 16u, 17u, 32u, 40u, 64u})
        for (uint32_t count : {2u, 3u})
            if (!run(rows, 4u, count, 1024u, 384u, 0u, false))
                return 1;
    if (!run(1u, 4u, 2u, 1024u, 10240u, 1u, false))
        return 1;
    for (uint32_t rows : {1u, 4u, 16u})
        if (!run(rows, 4u, 1u, 1024u, 10240u) || !run(rows, 8u, 1u, 256u, 107520u))
            return 1;
    if (!run(1u, 4u, 3u, 2560u, 3072u) || !run(1u, 4u, 1u, 3072u, 2560u) ||
        !run(1u, 4u, 1u, 10240u, 2560u, 9u))
        return 1;
    for (uint32_t n : {512u, 1024u})
        for (uint32_t k : {1024u, 5120u}) {
            for (uint32_t rows : {49u, 56u, 64u})
                if (!run_flag(rows, rows, 4u, n, k)) return 1;
            for (uint32_t rows : {17u, 32u, 40u, 64u})
                if (!run_flag(rows, rows, 5u, n, k)) return 1;
        }
    for (uint32_t rows : {49u, 56u, 64u})
        if (!run_flag(rows, rows, 4u, 1024u, 8192u)) return 1;
    for (uint32_t k : {8192u, 17408u})
        for (uint32_t rows : {17u, 32u, 33u, 48u, 49u, 64u})
            if (!run_flag(rows, rows, 5u, 1024u, k)) return 1;
    for (uint32_t rows : {16u, 32u, 48u, 64u, 128u, 256u})
        for (uint32_t active : {0u, 1u, 17u, 63u, 256u})
            for (uint32_t bits : {4u, 5u, 8u})
                if (!run_flag(rows, active, bits))
                    return 1;
    for (uint32_t rows : {1u, 7u, 64u, 256u, 320u, 512u, 1024u})
        for (bool bf16 : {false, true})
            if (!run_i32(rows, bf16))
                return 1;
    std::puts("PLaMo projection, flagged GEMM and BF16/F32 epilogue: outputs and padding match");
}
