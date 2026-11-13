#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <random>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <vector>
#include <shelf/inference/execution/exl3_dense_gemm.h>

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

bool run(uint32_t rows, uint32_t bits, uint32_t count, uint32_t k = 1024u, uint32_t n = 384u,
         bool cached = false, bool rowscaled = false) {
    const uint32_t splits = 1u;
    const uint32_t cap = rowscaled ? rows : shelf::kernel::exl3_mul24_flag_cap(rows);
    const uint32_t input_stride = k + 4, out_stride = n + 4;
    const uint32_t ss = k / 128 * 4;
    const size_t weights = size_t(n) * k * bits / 16;
    const size_t plane = size_t(rows) * out_stride;
    Buffer<uint16_t> x(size_t(rows) * input_stride), suh(k * count), svh(n * count),
        t(weights * count);
    Buffer<int8_t> codes(size_t(rows) * k * count);
    Buffer<float> scales(size_t(rows) * ss / 4 * count),
        y(size_t(rows) * n * std::max(3u, splits) * count);
    Buffer<uint16_t> out(plane * count);
    Buffer<int8_t> packed(size_t(n) * k * count);
    Buffer<int8_t> dense(rowscaled ? size_t(n) * k : 1u), cf(size_t(cap) * k * count);
    Buffer<float> rs(size_t(rows) * count), sf(size_t(cap) * ss / 4u * count),
        yf(size_t(cap) * n * count);
    Buffer<int32_t> slots(size_t(rows) * count);
    Buffer<uint32_t> active(1u);
    std::mt19937 rng(17 + rows + bits * count);
    std::vector<uint16_t> hx(size_t(rows) * input_stride), ht(weights * count), hu(k * count),
        hv(n * count);
    for (auto &v : hx)
        v = static_cast<uint16_t>(0x3f00 + rng() % 256);
    if (rowscaled) {
        for (uint32_t r = 0u; r < rows; ++r)
            if (r % 3u == 0u || r >= rows / 2u)
                for (uint32_t c = 0u; c < 128u; ++c)
                    hx[size_t(r) * input_stride + c] += 512u;
    }
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
        p.n = i == 0u                      ? n
              : n == 3072u && count == 3u  ? 512u
              : n == 10240u && count == 2u ? n
                                           : 128u * (i + 1u);
        p.splits = splits;
        p.out_stride = out_stride;
        p.trellis = t.p + weights * i;
        p.suh = suh.p + k * i;
        p.svh = svh.p + n * i;
        p.codes = codes.p + size_t(rows) * k * i;
        p.scales = scales.p + size_t(rows) * ss / 4 * i;
        p.yprime = y.p + size_t(rows) * n * std::max(3u, splits) * i;
        p.out = reinterpret_cast<shelf::bf16_t *>(out.p + plane * i);
        check(shelf::kernel::launch_exl3_mul24_rotate_quantize(
            args.input, input_stride, p.suh, p.codes, p.scales, rows, k, k, ss, nullptr));
        if (rowscaled) {
            check(shelf::kernel::launch_exl3_mul24_rowscale(
                p.codes, p.scales, rs.p + size_t(rows) * i, slots.p + size_t(rows) * i, active.p,
                cf.p + size_t(cap) * k * i, sf.p + size_t(cap) * ss / 4u * i, rows, cap, k, k, ss,
                0.5f, nullptr));
            check(
                shelf::kernel::launch_exl3_mul24_recon(p.trellis, bits, dense.p, p.n, k, nullptr));
            check(shelf::execution::exl3_dense_gemm_int8(rows, p.n, k, dense.p, p.codes, p.yprime,
                                                         nullptr));
            check(shelf::kernel::launch_plamo_exl3_flagged_gemm(
                cf.p + size_t(cap) * k * i, sf.p + size_t(cap) * ss / 4u * i, k, ss, p.trellis,
                bits, yf.p + size_t(cap) * n * i, cap, active.p, p.n, k, nullptr));
            check(shelf::kernel::launch_plamo_exl3_rotate_out_i32(
                reinterpret_cast<int32_t *>(p.yprime), rs.p + size_t(rows) * i,
                slots.p + size_t(rows) * i, yf.p + size_t(cap) * n * i, p.svh, p.out, true, rows,
                p.n, out_stride, nullptr));
        } else {
            check(shelf::kernel::launch_exl3_mul24_gemm(p.codes, p.scales, k, ss, p.trellis, bits,
                                                        p.yprime, rows, p.n, k, p.splits, nullptr));
            check(shelf::kernel::launch_exl3_mul24_rotate_out(
                p.yprime, p.svh, p.out, shelf::kernel::Exl3OutputDType::BF16, rows, p.n, out_stride,
                p.splits, nullptr));
        }
    }
    std::vector<uint16_t> baseline(plane * count), candidate(baseline.size());
    check(hipMemcpy(baseline.data(), out.p, baseline.size() * 2, hipMemcpyDeviceToHost));
    std::vector<float> gold_rs(size_t(rows) * count);
    std::vector<int32_t> gold_slots(size_t(rows) * count);
    std::vector<int8_t> gold_codes(size_t(rows) * k * count);
    if (rowscaled) {
        check(hipMemcpy(gold_rs.data(), rs.p, gold_rs.size() * 4u, hipMemcpyDeviceToHost));
        check(hipMemcpy(gold_slots.data(), slots.p, gold_slots.size() * 4u, hipMemcpyDeviceToHost));
        check(hipMemcpy(gold_codes.data(), codes.p, gold_codes.size(), hipMemcpyDeviceToHost));
    }
    check(hipMemset(out.p, 0x55, plane * count * 2));
    shelf::kernel::PlamoExl3PrefillGroup prefill{};
    prefill.projection = args;
    prefill.cached = cached;
    prefill.rowscaled = rowscaled;
    for (uint32_t i = 0u; i < count; ++i) {
        prefill.rowscale[i] = rs.p + size_t(rows) * i;
        prefill.row_flags[i] = slots.p + size_t(rows) * i;
    }
    for (uint32_t i = 0u; i < count; ++i)
        prefill.packed[i] = packed.p + size_t(n) * k * i;
    check(shelf::kernel::launch_plamo_exl3_projection_rotate(args, nullptr));
    if (rowscaled) {
        check(shelf::kernel::launch_plamo_exl3_prefill_rowscale(prefill, nullptr));
        std::vector<float> got_rs(gold_rs.size());
        std::vector<int32_t> got_slots(gold_slots.size());
        std::vector<int8_t> got_codes(gold_codes.size());
        check(hipMemcpy(got_rs.data(), rs.p, got_rs.size() * 4u, hipMemcpyDeviceToHost));
        check(hipMemcpy(got_slots.data(), slots.p, got_slots.size() * 4u, hipMemcpyDeviceToHost));
        check(hipMemcpy(got_codes.data(), codes.p, got_codes.size(), hipMemcpyDeviceToHost));
        bool flags_equal = true;
        for (size_t i = 0u; i < gold_slots.size(); ++i)
            flags_equal = flags_equal && ((gold_slots[i] >= 0) == (got_slots[i] >= 0));
        if (gold_rs != got_rs || !flags_equal || gold_codes != got_codes) {
            std::fprintf(stderr, "rowscale metadata mismatch\n");
            return false;
        }
    }
    if (cached)
        check(shelf::kernel::launch_plamo_exl3_prefill_recon(prefill, bits, nullptr));
    check(shelf::kernel::launch_plamo_exl3_prefill_gemm(prefill, bits, nullptr));
    check(hipMemcpy(candidate.data(), out.p, candidate.size() * 2, hipMemcpyDeviceToHost));
    if (baseline != candidate) {
        size_t diffs = 0;
        for (size_t i = 0; i < baseline.size(); ++i)
            diffs += baseline[i] != candidate[i];
        std::fprintf(stderr, "rows=%u bits=%u count=%u diffs=%zu\n", rows, bits, count, diffs);
        return false;
    }
    prefill.projection.count = 4u;
    if (shelf::kernel::launch_plamo_exl3_prefill_gemm(prefill, bits, nullptr) !=
        hipErrorInvalidValue)
        return false;
    prefill.projection = args;
    if (shelf::kernel::launch_plamo_exl3_prefill_gemm(prefill, 8u, nullptr) != hipErrorInvalidValue)
        return false;
    return true;
}
}
int main() {
    for (uint32_t rows : {65u, 96u, 129u, 192u, 256u, 257u, 320u, 511u, 512u})
        for (uint32_t bits : {4u, 5u})
            for (uint32_t count : {1u, 2u, 3u})
                for (bool cached : {false, true})
                    if (!run(rows, bits, count, 1024u, 384u, cached))
                        return 1;
    for (uint32_t rows : {257u, 512u})
        for (bool cached : {false, true}) {
            for (uint32_t bits : {4u, 5u})
                if (!run(rows, bits, 3u, 2560u, 3072u, cached) ||
                    !run(rows, bits, 2u, 2560u, 10240u, cached) ||
                    !run(rows, bits, 1u, 3072u, 2560u, cached))
                    return 1;
            if (!run(rows, 4u, 1u, 10240u, 2560u, cached))
                return 1;
        }
    for (uint32_t rows : {256u, 257u, 320u, 511u, 512u, 1025u, 2048u})
        for (uint32_t bits : {4u, 5u})
            for (uint32_t count : {1u, 2u, 3u})
                for (bool cached : {false, true})
                    if (!run(rows, bits, count, 1024u, 384u, cached, true))
                        return 1;
    for (uint32_t rows : {257u, 512u})
        for (bool cached : {false, true}) {
            for (uint32_t bits : {4u, 5u})
                if (!run(rows, bits, 3u, 2560u, 3072u, cached, true) ||
                    !run(rows, bits, 2u, 2560u, 10240u, cached, true) ||
                    !run(rows, bits, 1u, 3072u, 2560u, cached, true))
                    return 1;
            if (!run(rows, 4u, 1u, 10240u, 2560u, cached, true))
                return 1;
        }
    std::puts("PASS plamo exl3 prefill");
    return 0;
}
