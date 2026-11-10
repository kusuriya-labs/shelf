// EXL3-INT8 under tensor parallel, simulated on one GPU: the real rank-local partition of a mul24 tensor
// (shelf::weights::materialize_rank_local_canonical), the same INT8 kernels each rank would run, and the
// combine a TP step does (concatenate for an n split, sum for a k split). The combined result is compared
// with the unsharded run of the same path. A real multi-GPU run only adds the collective on top of this.
#include <shelf/core/memory/types.h>
#include <shelf/inference/execution/exl3_dense_gemm.h>
#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <shelf/weights/canonical_partition.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define HIP_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %d at %s:%d\n", static_cast<int>(e_), __FILE__, __LINE__); return false; } } while (0)

namespace {

using shelf::quantization::manifest::QuantizedEncoding;
using shelf::quantization::bundle::QuantizedTensorView;
using shelf::weights::OwnedCanonicalTensor;
using shelf::weights::TensorPartitionDesc;
using shelf::weights::TensorPartitionRange;

template <typename T>
struct Dev {
    T* p = nullptr;
    explicit Dev(size_t n) { (void)hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
};

// One INT8 linear of the shape [rows][k] x [n][k]^T through either the per-group path (decode) or the dense path
// (prefill: row scale, flagged rows on the fused kernel, hipBLASLt int32 GEMM). `canon` is the trellis in the
// canonical [kt][nt][16*bits] order. Returns false when the shape has no hipBLASLt int8 algorithm (dense only).
bool run_linear(bool dense, const std::vector<uint16_t>& canon, uint32_t bits, const std::vector<uint16_t>& suh_h,
                const std::vector<uint16_t>& svh_h, const std::vector<uint16_t>& x, uint32_t rows, uint32_t n,
                uint32_t k, std::vector<float>& out, bool& unsupported) {
    unsupported = false;
    const uint64_t kt = k / 16u, nt = n / 16u;
    std::vector<uint16_t> resident(canon.size());
    if (!shelf::quantization::exl3::exl3_tiles_to_nmajor(canon.data(), kt, nt, bits, resident.data(), 2)) return false;
    const uint32_t code_stride = k;
    const uint32_t groups = k / 128u;
    const uint32_t scale_stride_bytes = (groups * 4u + 15u) & ~15u;
    const uint32_t rows_padded = (rows + 15u) & ~15u;
    Dev<uint16_t> d_x(x.size()), d_suh(k), d_svh(n), d_trellis(resident.size());
    Dev<int8_t> d_codes(static_cast<size_t>(rows_padded) * code_stride);
    Dev<float> d_scales(static_cast<size_t>(rows) * scale_stride_bytes / 4u), d_y(static_cast<size_t>(rows) * n),
        d_out(static_cast<size_t>(rows) * n);
    HIP_OK(hipMemcpy(d_x.p, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_suh.p, suh_h.data(), k * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_svh.p, svh_h.data(), n * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_trellis.p, resident.data(), resident.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_codes.p, 0, static_cast<size_t>(rows_padded) * code_stride));
    HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_quantize(reinterpret_cast<const shelf::bf16_t*>(d_x.p), k, d_suh.p,
                                                           d_codes.p, d_scales.p, rows, k, code_stride,
                                                           scale_stride_bytes, nullptr));
    if (!dense) {
        HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(d_codes.p, d_scales.p, code_stride, scale_stride_bytes,
                                                   d_trellis.p, bits, d_y.p, rows, n, k, 1u, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_out(d_y.p, d_svh.p, d_out.p,
                                                         shelf::kernel::Exl3OutputDType::F32, rows, n, n, 1u,
                                                         nullptr));
    } else {
        const uint32_t cap = static_cast<uint32_t>(shelf::kernel::exl3_mul24_flag_cap(rows));
        Dev<int8_t> d_dense(static_cast<size_t>(n) * k), d_codes_f(static_cast<size_t>(cap) * code_stride);
        Dev<float> d_rs(rows), d_scales_f(static_cast<size_t>(cap) * scale_stride_bytes / 4u), d_y2(static_cast<size_t>(cap) * n);
        Dev<int32_t> d_slots(rows), d_acc(static_cast<size_t>(rows) * n);
        Dev<uint32_t> d_count(1);
        HIP_OK(shelf::kernel::launch_exl3_mul24_recon(d_trellis.p, bits, d_dense.p, n, k, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_rowscale(d_codes.p, d_scales.p, d_rs.p, d_slots.p, d_count.p,
                                                        d_codes_f.p, d_scales_f.p, rows, cap, k, code_stride,
                                                        scale_stride_bytes, 0.5f, nullptr));
        const hipError_t ge = shelf::execution::exl3_dense_gemm_int8(rows, n, k, d_dense.p, d_codes.p, d_acc.p, nullptr);
        if (ge == hipErrorNotSupported) {
            unsupported = true;
            return true;
        }
        HIP_OK(ge);
        HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(d_codes_f.p, d_scales_f.p, code_stride, scale_stride_bytes,
                                                   d_trellis.p, bits, d_y2.p, cap, n, k, 1u, nullptr, d_count.p));
        HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_out_i32(d_acc.p, d_rs.p, d_slots.p, d_y2.p, d_svh.p, d_out.p,
                                                             shelf::kernel::Exl3OutputDType::F32, rows, n, n,
                                                             nullptr));
    }
    HIP_OK(hipDeviceSynchronize());
    out.resize(static_cast<size_t>(rows) * n);
    HIP_OK(hipMemcpy(out.data(), d_out.p, out.size() * 4, hipMemcpyDeviceToHost));
    return true;
}

uint16_t f16_bits(float v) {
    const _Float16 h = static_cast<_Float16>(v);
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

std::vector<uint8_t> bytes_of(const std::vector<uint16_t>& v) {
    std::vector<uint8_t> b(v.size() * 2);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

std::vector<uint16_t> words_of(const std::vector<uint8_t>& b) {
    std::vector<uint16_t> v(b.size() / 2);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0, r = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        r += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(d / std::max(1e-30, r));
}

// axis 0 splits n (column parallel: concatenate), axis 1 splits k (row parallel: sum the partials).
bool run_case(bool dense, int axis, uint32_t tp, uint32_t rows, uint32_t n, uint32_t k, uint32_t bits, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * k);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = shelf::f32_to_bf16_rne(nd(rng) * (1.0f + 0.1f * static_cast<float>(i % 7)));
    // Massive-activation rows (one 300x group) so the flagged-row path is part of the comparison.
    for (uint32_t m = 1; m < rows; m += 5u)
        for (uint32_t c = 128; c < 256 && c < k; ++c) {
            const size_t i = static_cast<size_t>(m) * k + c;
            uint32_t bits32 = static_cast<uint32_t>(x[i]) << 16;
            float f;
            std::memcpy(&f, &bits32, 4);
            x[i] = shelf::f32_to_bf16_rne(f * 300.0f);
        }
    std::vector<uint16_t> suh(k), svh(n);
    for (auto& s : suh) s = f16_bits((rng() & 1u) ? 1.0f : -1.0f);
    for (auto& s : svh) s = f16_bits(0.5f + std::fabs(nd(rng)));
    const uint64_t kt = k / 16u, nt = n / 16u;
    std::vector<uint16_t> trellis(kt * nt * 16u * bits);
    for (auto& w : trellis) w = static_cast<uint16_t>(rng());

    std::vector<float> ref;
    bool unsupported = false;
    if (!run_linear(dense, trellis, bits, suh, svh, x, rows, n, k, ref, unsupported)) return false;
    if (unsupported) {
        std::printf("SKIP dense axis=%d tp=%u: no hipBLASLt int8 algorithm for rows=%u n=%u k=%u\n", axis, tp, rows, n, k);
        return true;
    }

    const std::vector<uint8_t> codes_b = bytes_of(trellis), suh_b = bytes_of(suh), svh_b = bytes_of(svh);
    const std::vector<uint8_t> marker = {0x2D, 0xD1, 0xDC, 0x00};  // 0x00DCD12D, little endian
    QuantizedTensorView g;
    g.name = "w";
    g.encoding = QuantizedEncoding::Exl3;
    g.logical_shape = {static_cast<int64_t>(n), static_cast<int64_t>(k)};
    g.k_padded = k;
    g.exl3_ka = bits;
    g.exl3_half = false;
    g.codes = shelf::quantization::bundle::ByteSpan{codes_b.data(), codes_b.size()};
    g.suh = shelf::quantization::bundle::ByteSpan{suh_b.data(), suh_b.size()};
    g.svh = shelf::quantization::bundle::ByteSpan{svh_b.data(), svh_b.size()};
    g.marker = shelf::quantization::bundle::ByteSpan{marker.data(), marker.size()};

    std::vector<float> combined(static_cast<size_t>(rows) * n, 0.0f);
    const uint32_t extent = (axis == 0 ? n : k) / tp;
    for (uint32_t r = 0; r < tp; ++r) {
        TensorPartitionDesc d;
        d.axis = axis;
        d.index = r;
        d.count = tp;
        d.global_shape = {static_cast<int64_t>(n), static_cast<int64_t>(k)};
        TensorPartitionRange pr;
        pr.global_offset = static_cast<uint64_t>(r) * extent;
        pr.extent = extent;
        d.ranges = {pr};
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        const shelf::Status st = shelf::weights::materialize_rank_local_canonical(d, g, owned, local);
        if (!st.ok()) {
            std::printf("partition failed: %s\n", st.message().c_str());
            return false;
        }
        const uint32_t ln = static_cast<uint32_t>(local.logical_shape[0]), lk = static_cast<uint32_t>(local.logical_shape[1]);
        std::vector<uint16_t> lx(static_cast<size_t>(rows) * lk);
        for (uint32_t m = 0; m < rows; ++m)
            for (uint32_t c = 0; c < lk; ++c)
                lx[static_cast<size_t>(m) * lk + c] =
                    x[static_cast<size_t>(m) * k + (axis == 1 ? static_cast<size_t>(r) * extent : 0u) + c];
        std::vector<float> lout;
        if (!run_linear(dense, words_of(owned.codes), bits, words_of(owned.suh), words_of(owned.svh), lx, rows, ln, lk,
                        lout, unsupported))
            return false;
        if (unsupported) {
            std::printf("SKIP dense axis=%d tp=%u: rank-local shape has no hipBLASLt int8 algorithm\n", axis, tp);
            return true;
        }
        for (uint32_t m = 0; m < rows; ++m)
            for (uint32_t c = 0; c < ln; ++c) {
                const float v = lout[static_cast<size_t>(m) * ln + c];
                if (axis == 0) combined[static_cast<size_t>(m) * n + static_cast<size_t>(r) * extent + c] = v;
                else combined[static_cast<size_t>(m) * n + c] += v;
            }
    }
    // n split: every output column is computed by one rank from the same activations, so the result is the same
    // up to nothing at all. k split: partials sum in a different order (group path), and the dense path takes its
    // row scale from the rank-local K slice, so it differs by activation rounding only.
    const double tol = axis == 0 ? 1e-6 : (dense ? 0.03 : 1e-5);
    const double e = rel_l2(combined, ref);
    const bool ok = e <= tol;
    std::printf("%s %s %s tp=%u rows=%u n=%u k=%u bits=%u: rel_l2=%.3g (tol %.3g)\n", ok ? "PASS" : "FAIL",
                dense ? "dense" : "group", axis == 0 ? "n-split" : "k-split", tp, rows, n, k, bits, e, tol);
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    for (bool dense : {false, true})
        for (uint32_t tp : {2u, 4u}) {
            ok &= run_case(dense, 0, tp, 300, 512, 1024, 4, 11u + tp);
            ok &= run_case(dense, 1, tp, 300, 512, 1024, 4, 21u + tp);
        }
    // Shapes of the real model: 5120 -> 17408 (gate/up, n split) and 17408 -> 5120 (down, k split).
    ok &= run_case(true, 0, 4, 300, 17408, 5120, 4, 31u);
    ok &= run_case(true, 1, 4, 300, 5120, 17408, 4, 32u);
    std::printf(ok ? "ALL PASS\n" : "SOME FAILED\n");
    return ok ? 0 : 1;
}
