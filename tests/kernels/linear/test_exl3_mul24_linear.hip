#include <shelf/inference/execution/exl3_dense_gemm.h>
#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <shelf/core/memory/types.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %d at %s:%d\n", static_cast<int>(e_), __FILE__, __LINE__); return false; } } while (0)

namespace {

constexpr float kInvSqrt128 = 0.08838834764831845f;

float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

void fwht128_f32(float* v) {
    for (int h = 1; h < 128; h <<= 1)
        for (int t = 0; t < 128; ++t)
            if (!(t & h)) {
                const float a = v[t], b = v[t | h];
                v[t] = a + b;
                v[t | h] = a - b;
            }
}

template <typename T>
struct Dev {
    T* p = nullptr;
    explicit Dev(size_t n) { (void)hipMalloc(&p, n * sizeof(T)); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
};

bool run_case(uint32_t rows, uint32_t n, uint32_t k, uint32_t bits, uint32_t seed, uint32_t splits = 1u) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> qd(-128, 127);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * k);
    std::vector<float> xf(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = ::shelf::f32_to_bf16_rne(nd(rng) * (1.0f + 0.1f * static_cast<float>(i % 7)));
        xf[i] = bf16_to_f32(x[i]);
    }
    // Massive-activation rows: a few rows get one group 300x larger, so their group scales are far apart.
    if (k >= 256u) {
        for (uint32_t m = 1; m < rows; m += 3u)
            for (uint32_t c = 128; c < 256; ++c) {
                const size_t i = static_cast<size_t>(m) * k + c;
                x[i] = ::shelf::f32_to_bf16_rne(xf[i] * 300.0f);
                xf[i] = bf16_to_f32(x[i]);
            }
    }
    std::vector<float> suh(k), svh(n);
    for (auto& s : suh) s = (rng() & 1u) ? 1.0f : -1.0f;
    for (auto& s : svh) s = static_cast<float>(static_cast<_Float16>(0.5f + std::fabs(nd(rng))));  // fp16-exact, as stored
    // Random trellis words; the expected q comes from the CPU reference decoder (itself checked against
    // the real rocm_exl3 decoder in test_exl3_mul24_codec_decode).
    const uint64_t kt = k / 16u, nt = n / 16u;
    std::vector<uint16_t> trellis(kt * nt * 16u * bits);
    for (auto& w : trellis) w = static_cast<uint16_t>(rng());
    std::vector<int8_t> q(static_cast<size_t>(n) * k);
    if (!shelf::quantization::exl3::exl3_decode_int8(trellis.data(), kt, nt, bits, q.data(), 4)) return false;
    (void)qd;

    const uint32_t code_stride = (k + 15u) & ~15u;
    const uint32_t groups = k / 128u;
    const uint32_t scale_stride_bytes = ((groups * 4u) + 15u) & ~15u;
    const uint32_t rows_padded = (rows + 15u) & ~15u;
    Dev<uint16_t> d_x(x.size());
    Dev<uint16_t> d_suh(k), d_svh(n);
    Dev<uint16_t> d_trellis(trellis.size());
    Dev<int8_t> d_codes(static_cast<size_t>(rows_padded) * code_stride);
    Dev<float> d_scales(static_cast<size_t>(rows) * scale_stride_bytes / 4u), d_y(static_cast<size_t>(rows) * n * splits), d_out(static_cast<size_t>(rows) * n);
    HIP_OK(hipMemcpy(d_x.p, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    auto f16_bits = [](const std::vector<float>& v) {
        std::vector<uint16_t> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            const _Float16 h = static_cast<_Float16>(v[i]);
            std::memcpy(&out[i], &h, sizeof(uint16_t));
        }
        return out;
    };
    const std::vector<uint16_t> suh_bits = f16_bits(suh), svh_bits = f16_bits(svh);
    HIP_OK(hipMemcpy(d_suh.p, suh_bits.data(), k * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_svh.p, svh_bits.data(), n * 2, hipMemcpyHostToDevice));
    std::vector<uint16_t> resident(trellis.size());
    if (!shelf::quantization::exl3::exl3_tiles_to_nmajor(trellis.data(), kt, nt, bits, resident.data(), 2)) return false;
    HIP_OK(hipMemcpy(d_trellis.p, resident.data(), resident.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_codes.p, 0x55, static_cast<size_t>(rows_padded) * code_stride));

    HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_quantize(
        reinterpret_cast<const shelf::bf16_t*>(d_x.p), k, d_suh.p, d_codes.p, d_scales.p, rows, k,
        code_stride, scale_stride_bytes, nullptr));
    HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(d_codes.p, d_scales.p, code_stride, scale_stride_bytes,
                                               d_trellis.p, bits, d_y.p, rows, n, k, splits, nullptr));
    HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_out(d_y.p, d_svh.p, d_out.p,
                                                     shelf::kernel::Exl3OutputDType::F32, rows, n, n, splits, nullptr));
    HIP_OK(hipDeviceSynchronize());

    std::vector<int8_t> codes(static_cast<size_t>(rows_padded) * code_stride);
    std::vector<float> scales(static_cast<size_t>(rows) * scale_stride_bytes / 4u), y(static_cast<size_t>(rows) * n), out(y.size());
    std::vector<float> y_parts(y.size() * splits);
    HIP_OK(hipMemcpy(codes.data(), d_codes.p, codes.size(), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(scales.data(), d_scales.p, scales.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(y_parts.data(), d_y.p, y_parts.size() * 4, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < y.size(); ++i) { float a = 0.0f; for (uint32_t z = 0; z < splits; ++z) a += y_parts[z * y.size() + i]; y[i] = a; }
    HIP_OK(hipMemcpy(out.data(), d_out.p, out.size() * 4, hipMemcpyDeviceToHost));

    // Mirrored float32 reference of each stage.
    size_t code_mismatch = 0;
    double worst_scale = 0.0, max_y_diff = 0.0, max_y_ref = 0.0;
    std::vector<float> ref_codes_f(static_cast<size_t>(rows) * k);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t g = 0; g < groups; ++g) {
            float v[128];
            for (int t = 0; t < 128; ++t) v[t] = xf[static_cast<size_t>(m) * k + g * 128 + t] * suh[g * 128 + t];
            fwht128_f32(v);
            float peak = 0.0f;
            for (int t = 0; t < 128; ++t) { v[t] *= kInvSqrt128; peak = std::max(peak, std::fabs(v[t])); }
            const float scale = peak / 127.0f;
            worst_scale = std::max(worst_scale, std::fabs(static_cast<double>(scale) - scales[static_cast<size_t>(m) * scale_stride_bytes / 4u + g]));
            for (int t = 0; t < 128; ++t) {
                float qq = 0.0f;
                if (scale > 0.0f) qq = std::min(std::max(std::nearbyintf(v[t] / scale), -128.0f), 127.0f);
                const int8_t got = codes[static_cast<size_t>(m) * code_stride + g * 128 + t];
                if (static_cast<int>(qq) != got) ++code_mismatch;
                ref_codes_f[static_cast<size_t>(m) * k + g * 128 + t] = static_cast<float>(got);
            }
        }
    }
    // GEMM reference from the GPU codes and scales (exact int32 per group, float accumulation in group order).
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t nn = 0; nn < n; ++nn) {
            float acc = 0.0f;
            for (uint32_t g = 0; g < groups; ++g) {
                int32_t gi = 0;
                for (int t = 0; t < 128; ++t)
                    gi += static_cast<int32_t>(codes[static_cast<size_t>(m) * code_stride + g * 128 + t]) *
                          static_cast<int32_t>(q[static_cast<size_t>(nn) * k + g * 128 + t]);
                acc += static_cast<float>(gi) * scales[static_cast<size_t>(m) * scale_stride_bytes / 4u + g];
            }
            const float ref = acc * shelf::quantization::exl3::kExl3Int8GridStep;
            max_y_diff = std::max(max_y_diff, std::fabs(static_cast<double>(ref) - y[static_cast<size_t>(m) * n + nn]));
            max_y_ref = std::max(max_y_ref, static_cast<double>(std::fabs(ref)));
        }
    }
    const double worst_y = max_y_diff / std::max(1e-30, max_y_ref);
    // Rotate-out reference from the GPU y' and compare end to end against the ideal math.
    double max_o_diff = 0.0, max_o_ref = 0.0, worst_ideal = 0.0, ideal_norm = 0.0;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t b = 0; b < n / 128u; ++b) {
            float v[128];
            for (int t = 0; t < 128; ++t) v[t] = y[static_cast<size_t>(m) * n + b * 128 + t];
            fwht128_f32(v);
            for (int t = 0; t < 128; ++t) {
                const float ref = v[t] * kInvSqrt128 * svh[b * 128 + t];
                max_o_diff = std::max(max_o_diff, std::fabs(static_cast<double>(ref) - out[static_cast<size_t>(m) * n + b * 128 + t]));
                max_o_ref = std::max(max_o_ref, static_cast<double>(std::fabs(ref)));
            }
        }
    }
    const double worst_out = max_o_diff / std::max(1e-30, max_o_ref);
    // Ideal (float64, unquantised activations): y = svh * H( S * Q . H(suh * x) ).
    std::vector<double> ideal(static_cast<size_t>(rows) * n);
    for (uint32_t m = 0; m < rows; ++m) {
        std::vector<double> xh(k);
        for (uint32_t g = 0; g < groups; ++g) {
            for (int i = 0; i < 128; ++i) {
                double s = 0.0;
                for (int j = 0; j < 128; ++j)
                    s += static_cast<double>(xf[static_cast<size_t>(m) * k + g * 128 + j]) * suh[g * 128 + j] *
                         ((__builtin_popcount(i & j) & 1) ? -1.0 : 1.0);
                xh[g * 128 + i] = s / std::sqrt(128.0);
            }
        }
        std::vector<double> yy(n);
        for (uint32_t nn = 0; nn < n; ++nn) {
            double s = 0.0;
            for (uint32_t kk = 0; kk < k; ++kk) s += xh[kk] * q[static_cast<size_t>(nn) * k + kk];
            yy[nn] = s * shelf::quantization::exl3::kExl3Int8GridStep;
        }
        for (uint32_t b = 0; b < n / 128u; ++b)
            for (int i = 0; i < 128; ++i) {
                double s = 0.0;
                for (int j = 0; j < 128; ++j) s += yy[b * 128 + j] * ((__builtin_popcount(i & j) & 1) ? -1.0 : 1.0);
                ideal[static_cast<size_t>(m) * n + b * 128 + i] = s / std::sqrt(128.0) * svh[b * 128 + i];
            }
    }
    double err2 = 0.0;
    for (size_t i = 0; i < ideal.size(); ++i) {
        const double d = ideal[i] - out[i];
        err2 += d * d;
        ideal_norm += ideal[i] * ideal[i];
    }
    worst_ideal = std::sqrt(err2 / std::max(1e-30, ideal_norm));


    // B prefill (reconstructed dense weights + per-128-K scales): must reproduce the fused per-group
    // accumulator, because the only difference is the weight storage (trellis decode vs dense int8).
    bool wmma_pass = true;
    {
        Dev<int8_t> d_dense_w(static_cast<size_t>(n) * k);
        Dev<float> d_yw(static_cast<size_t>(rows) * n);
        HIP_OK(shelf::kernel::launch_exl3_mul24_recon_wmma(d_trellis.p, bits, d_dense_w.p, n, k, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_prefill_wmma(d_codes.p, d_scales.p, code_stride,
                                                            scale_stride_bytes, d_dense_w.p, d_yw.p, rows, n, k,
                                                            nullptr));
        HIP_OK(hipDeviceSynchronize());
        std::vector<float> yw(static_cast<size_t>(rows) * n);
        HIP_OK(hipMemcpy(yw.data(), d_yw.p, yw.size() * 4, hipMemcpyDeviceToHost));
        double d2 = 0.0, r2 = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            d2 += (yw[i] - y[i]) * static_cast<double>(yw[i] - y[i]);
            r2 += static_cast<double>(y[i]) * y[i];
        }
        const double rel = std::sqrt(d2 / std::max(1e-30, r2));
        wmma_pass = rel < 1e-6;
        std::printf("  prefill_wmma: vs_fused_rel_l2=%.3g %s\n", rel, wmma_pass ? "PASS" : "FAIL");
    }

    // Prefill (dense) path: recon must reproduce the CPU decode exactly; for the rows without a slot the row-scale
    // re-rounding and the int32 GEMM must match the CPU integer math exactly; rows with a slot must give the
    // per-group result above; rotate_out_i32 must match the float reference. The whole result is also compared with
    // the per-group path (the row scale costs a little precision).
    bool dense_pass = true;
    {
        const uint32_t cap = static_cast<uint32_t>(shelf::kernel::exl3_mul24_flag_cap(rows));
        Dev<int8_t> d_dense(static_cast<size_t>(n) * k);
        Dev<float> d_rs(rows);
        Dev<int32_t> d_slots(rows);
        Dev<uint32_t> d_count(1);
        Dev<int8_t> d_codes_f(static_cast<size_t>(cap) * code_stride);
        Dev<float> d_scales_f(static_cast<size_t>(cap) * scale_stride_bytes / 4u);
        Dev<int32_t> d_acc(static_cast<size_t>(rows) * n);
        Dev<float> d_y2(static_cast<size_t>(cap) * n);
        Dev<float> d_out2(static_cast<size_t>(rows) * n);
        HIP_OK(shelf::kernel::launch_exl3_mul24_recon(d_trellis.p, bits, d_dense.p, n, k, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_rowscale(d_codes.p, d_scales.p, d_rs.p, d_slots.p, d_count.p, d_codes_f.p,
                                                        d_scales_f.p, rows, cap, k, code_stride, scale_stride_bytes, 0.5f,
                                                        nullptr));
        const hipError_t ge = shelf::execution::exl3_dense_gemm_int8(rows, n, k, d_dense.p, d_codes.p, d_acc.p, nullptr);
        if (ge == hipErrorNotSupported) {
            std::printf("  dense int8 GEMM: no hipBLASLt algorithm for this shape, skipped\n");
        } else {
            HIP_OK(ge);
            HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(d_codes_f.p, d_scales_f.p, code_stride, scale_stride_bytes,
                                                       d_trellis.p, bits, d_y2.p, cap, n, k, 1u, nullptr, d_count.p));
            HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_out_i32(
                d_acc.p, d_rs.p, d_slots.p, d_y2.p, d_svh.p, d_out2.p, shelf::kernel::Exl3OutputDType::F32, rows, n, n,
                nullptr));
            HIP_OK(hipDeviceSynchronize());
            std::vector<int8_t> dense(static_cast<size_t>(n) * k), codes2(codes.size());
            std::vector<int32_t> acc(static_cast<size_t>(rows) * n), slots(rows);
            std::vector<float> rs(rows), out2(static_cast<size_t>(rows) * n);
            uint32_t count = 0;
            HIP_OK(hipMemcpy(dense.data(), d_dense.p, dense.size(), hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(codes2.data(), d_codes.p, codes2.size(), hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(acc.data(), d_acc.p, acc.size() * 4, hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(rs.data(), d_rs.p, rs.size() * 4, hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(slots.data(), d_slots.p, slots.size() * 4, hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(&count, d_count.p, 4, hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(out2.data(), d_out2.p, out2.size() * 4, hipMemcpyDeviceToHost));
            size_t recon_bad = (dense == q) ? 0 : 1, code_bad = 0, acc_bad = 0, slot_bad = 0, flagged_bad = 0;
            std::vector<float> y2(static_cast<size_t>(rows) * n);
            uint32_t next_slot = 0;
            for (uint32_t m = 0; m < rows; ++m) {
                float peak = 0.0f, sq = 0.0f;
                for (uint32_t g = 0; g < groups; ++g) {
                    const float sg = scales[static_cast<size_t>(m) * scale_stride_bytes / 4u + g];
                    peak = std::max(peak, sg);
                    sq += sg * sg;
                }
                if (rs[m] != peak) ++code_bad;
                const bool want_flag = peak > 0.0f && peak * peak > 0.5f * sq;
                const int32_t want_slot = want_flag && next_slot < cap ? static_cast<int32_t>(next_slot++) : -1;
                if (want_flag && want_slot < 0) ++next_slot;
                if (slots[m] != want_slot) ++slot_bad;
                if (want_slot >= 0) {
                    // Flagged: the same fused kernel as the per-group path (split-K changes only the float summation order).
                    float row_max = 0.0f;
                    for (uint32_t nn = 0; nn < n; ++nn) row_max = std::max(row_max, std::fabs(out[static_cast<size_t>(m) * n + nn]));
                    for (uint32_t nn = 0; nn < n; ++nn)
                        if (std::fabs(out2[static_cast<size_t>(m) * n + nn] - out[static_cast<size_t>(m) * n + nn]) > 1e-4f * row_max) ++flagged_bad;
                    continue;
                }
                for (uint32_t i = 0; i < k; ++i) {
                    const int8_t c0 = codes[static_cast<size_t>(m) * code_stride + i];
                    int8_t want = c0;
                    if (peak > 0.0f) {
                        const float ratio = scales[static_cast<size_t>(m) * scale_stride_bytes / 4u + i / 128u] / peak;
                        if (ratio != 1.0f) want = static_cast<int8_t>(static_cast<int>(std::nearbyintf(static_cast<float>(c0) * ratio)));
                    }
                    // GPU codes were re-rounded in place; the reference must match them.
                    if (codes2[static_cast<size_t>(m) * code_stride + i] != want) ++code_bad;
                }
                for (uint32_t nn = 0; nn < n; ++nn) {
                    int32_t a = 0;
                    for (uint32_t i = 0; i < k; ++i)
                        a += static_cast<int32_t>(codes2[static_cast<size_t>(m) * code_stride + i]) * q[static_cast<size_t>(nn) * k + i];
                    if (a != acc[static_cast<size_t>(m) * n + nn]) ++acc_bad;
                    y2[static_cast<size_t>(m) * n + nn] = static_cast<float>(a) * rs[m] * shelf::quantization::exl3::kExl3Int8GridStep;
                }
            }
            const uint32_t want_count = std::min(next_slot, cap);
            if (count != want_count) ++slot_bad;
            double d2 = 0.0, r2 = 0.0, g2 = 0.0, gn = 0.0;
            for (uint32_t m = 0; m < rows; ++m) {
                for (uint32_t b = 0; b < n / 128u; ++b) {
                    float v[128];
                    for (int t = 0; t < 128; ++t) v[t] = y2[static_cast<size_t>(m) * n + b * 128 + t];
                    fwht128_f32(v);
                    for (int t = 0; t < 128; ++t) {
                        const size_t idx = static_cast<size_t>(m) * n + b * 128 + t;
                        g2 += (out[idx] - out2[idx]) * static_cast<double>(out[idx] - out2[idx]);
                        gn += static_cast<double>(out[idx]) * out[idx];
                        if (slots[m] >= 0) continue;
                        const float ref = v[t] * kInvSqrt128 * svh[b * 128 + t];
                        d2 += (ref - out2[idx]) * static_cast<double>(ref - out2[idx]);
                        r2 += static_cast<double>(ref) * ref;
                    }
                }
            }
            const double out_rel = std::sqrt(d2 / std::max(1e-30, r2));
            const double vs_group = std::sqrt(g2 / std::max(1e-30, gn));
            dense_pass = recon_bad == 0 && code_bad == 0 && acc_bad == 0 && slot_bad == 0 && flagged_bad == 0 &&
                         out_rel < 1e-5 && vs_group < 0.05;
            std::printf("  dense: recon_bad=%zu code_bad=%zu acc_bad=%zu slot_bad=%zu flagged_bad=%zu flagged_rows=%u out_rel=%.3g vs_group_rel_l2=%.4f %s\n",
                        recon_bad, code_bad, acc_bad, slot_bad, flagged_bad, count, out_rel, vs_group, dense_pass ? "PASS" : "FAIL");
        }
    }

    const bool ok = dense_pass && wmma_pass && code_mismatch == 0 && worst_scale == 0.0 && worst_y < 1e-5 && worst_out < 1e-5 && worst_ideal < 0.02;
    std::printf("%s rows=%u n=%u k=%u: code_mismatch=%zu scale_err=%.3g gemm_rel=%.3g out_rel=%.3g ideal_rel_l2=%.4f\n",
                ok ? "PASS" : "FAIL", rows, n, k, code_mismatch, worst_scale, worst_y, worst_out, worst_ideal);
    return ok;
}



bool row_parity(uint32_t bits, uint32_t n = 1024u, uint32_t k = 5120u, uint32_t capacity = 1024u) {
    constexpr uint32_t rows = 8u;
    for (uint32_t scratch_rows : {8u, 16u, 32u, 48u, 64u, 1024u}) {
        const uint32_t one = shelf::kernel::exl3_mul24_gemm_splits(1u, n, k, scratch_rows, bits);
        for (uint32_t count = 2u; count <= std::min(16u, scratch_rows); ++count) {
            if (one != shelf::kernel::exl3_mul24_gemm_splits(count, n, k, scratch_rows, bits)) return false;
        }
    }
    const uint32_t scale_stride = ((k / 128u * 4u) + 15u) & ~15u;
    std::mt19937 rng(123456u + bits);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<uint16_t> input(rows * k), suh(k), svh(n);
    for (auto& x : input) x = shelf::f32_to_bf16_rne(dist(rng));
    for (auto& x : suh) x = (rng() & 1u) ? 0x3c00u : 0xbc00u;
    for (auto& x : svh) x = (rng() & 1u) ? 0x3c00u : 0xbc00u;
    std::vector<uint16_t> weights(static_cast<size_t>(n) * k * bits / 16u);
    for (auto& x : weights) x = static_cast<uint16_t>(rng());
    const uint32_t splits = shelf::kernel::exl3_mul24_gemm_splits(rows, n, k, capacity, bits);
    if (splits != shelf::kernel::exl3_mul24_gemm_splits(1u, n, k, capacity, bits)) return false;
    Dev<uint16_t> dx(rows * k), dsuh(k), dsvh(n), dw(weights.size());
    Dev<int8_t> codes(rows * k);
    Dev<float> scales(rows * scale_stride / 4u), accum(static_cast<size_t>(splits) * rows * n), out(rows * n);
    HIP_OK(hipMemcpy(dx.p, input.data(), input.size() * 2u, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dsuh.p, suh.data(), k * 2u, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dsvh.p, svh.data(), n * 2u, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dw.p, weights.data(), weights.size() * 2u, hipMemcpyHostToDevice));
    auto launch = [&](uint32_t count, const uint16_t* x) {
        HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_quantize(reinterpret_cast<const shelf::bf16_t*>(x), k, dsuh.p, codes.p, scales.p, count, k, k, scale_stride, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(codes.p, scales.p, k, scale_stride, dw.p, bits, accum.p, count, n, k, splits, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_out(accum.p, dsvh.p, out.p, shelf::kernel::Exl3OutputDType::F32, count, n, n, splits, nullptr));
        return true;
    };
    if (!launch(rows, dx.p)) return false;
    std::vector<float> batch(rows * n), single(n);
    HIP_OK(hipMemcpy(batch.data(), out.p, batch.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (uint32_t row = 0; row < rows; ++row) {
        if (!launch(1u, dx.p + static_cast<size_t>(row) * k)) return false;
        HIP_OK(hipMemcpy(single.data(), out.p, n * sizeof(float), hipMemcpyDeviceToHost));
        if (std::memcmp(single.data(), batch.data() + static_cast<size_t>(row) * n, n * sizeof(float)) != 0) {
            std::printf("FAIL row parity bits=%u row=%u\n", bits, row);
            return false;
        }
    }
    std::printf("PASS row parity bits=%u splits=%u\n", bits, splits);
    return true;
}


__global__ void fill_head_trellis(uint32_t* words, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        uint32_t x = static_cast<uint32_t>(i) * 747796405u + 2891336453u;
        x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
        words[i] = (x >> 22u) ^ x;
    }
}

// Rows 17..64 share one split count per shape (so a row's K-sum order does not depend on how many rows are batched),
// the count fits the scratch, and wider launches stay unsplit.
bool mid_row_splits_constant(uint32_t bits, uint32_t n, uint32_t k) {
    const uint32_t first = shelf::kernel::exl3_mul24_gemm_splits(17u, n, k, 1u << 20, bits);
    if (first < 2u) return false;
    for (uint32_t rows = 18u; rows <= 64u; ++rows)
        if (shelf::kernel::exl3_mul24_gemm_splits(rows, n, k, 1u << 20, bits) != first) return false;
    for (uint32_t rows = 17u; rows <= 64u; ++rows) {
        const uint32_t fit = shelf::kernel::exl3_mul24_gemm_splits(rows, n, k, 3u * rows, bits);
        if (fit > 3u) return false;
    }
    if (shelf::kernel::exl3_mul24_gemm_splits(65u, n, k, 1u << 20, bits) != 1u) return false;
    std::printf("PASS mid-row splits n=%u k=%u splits=%u\n", n, k, first);
    return true;
}

bool head_row_parity() {
    constexpr uint32_t rows = 8u, n = 248320u, k = 5120u, bits = 6u, scale_stride = 160u;
    shelf::weights::MatrixWeight weight;
    weight.encoding = shelf::weights::MatrixEncoding::Exl3;
    weight.quant_spec = shelf::quantization::QuantSpecId::EXL3_MUL24_INT8_V1;
    weight.rows = n;
    weight.cols = k;
    shelf::weights::Exl3LinearShape shape;
    shelf::weights::note_exl3_int8_decode_accum(weight, shape);
    const uint32_t scratch_rows = static_cast<uint32_t>(shape.max_int8_decode_accum_bytes / (n * sizeof(float)));
    const uint32_t splits = shelf::kernel::exl3_mul24_gemm_splits(1u, n, k, scratch_rows);
    if (splits != shelf::kernel::exl3_mul24_gemm_splits(rows, n, k, scratch_rows)) return false;
    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<uint16_t> input(rows * k), ones(std::max(k, n), 0x3c00u);
    for (uint32_t i = 0; i < k; ++i) input[i] = shelf::f32_to_bf16_rne(dist(rng));
    for (uint32_t row = 1; row < rows; ++row)
        std::copy_n(input.data(), k, input.data() + static_cast<size_t>(row) * k);
    const size_t words = static_cast<size_t>(n) * k * bits / 32u;
    Dev<uint16_t> dx(input.size()), ds(ones.size());
    Dev<uint32_t> dw(words);
    Dev<int8_t> codes(rows * k);
    Dev<float> scales(rows * scale_stride / 4u), accum(static_cast<size_t>(rows) * n * splits), out(static_cast<size_t>(rows) * n);
    HIP_OK(hipMemcpy(dx.p, input.data(), input.size() * 2u, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(ds.p, ones.data(), ones.size() * 2u, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(fill_head_trellis, dim3(4096), dim3(256), 0, nullptr, dw.p, words);
    HIP_OK(hipGetLastError());
    auto launch = [&](uint32_t count) {
        HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_quantize(reinterpret_cast<const shelf::bf16_t*>(dx.p), k, ds.p, codes.p, scales.p, count, k, k, scale_stride, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(codes.p, scales.p, k, scale_stride, reinterpret_cast<uint16_t*>(dw.p), bits, accum.p, count, n, k, splits, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_out(accum.p, ds.p, out.p, shelf::kernel::Exl3OutputDType::F32, count, n, n, splits, nullptr));
        return true;
    };
    std::vector<float> one(n), batch(static_cast<size_t>(rows) * n);
    if (!launch(1u)) return false;
    HIP_OK(hipMemcpy(one.data(), out.p, n * sizeof(float), hipMemcpyDeviceToHost));
    if (!launch(rows)) return false;
    HIP_OK(hipMemcpy(batch.data(), out.p, batch.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (uint32_t row = 0; row < rows; ++row)
        if (std::memcmp(one.data(), batch.data() + static_cast<size_t>(row) * n, n * sizeof(float)) != 0) return false;
    std::printf("PASS head row parity splits=%u scratch_rows=%u\n", splits, scratch_rows);
    return true;
}

bool bench_case(uint32_t rows, uint32_t n, uint32_t k, uint32_t bits) {
    const uint32_t kt = k / 16u, nt = n / 16u;
    const uint32_t code_stride = k, groups = k / 128u;
    const uint32_t scale_stride_bytes = ((groups * 4u) + 15u) & ~15u;
    Dev<uint16_t> d_trellis(static_cast<size_t>(kt) * nt * 16u * bits);
    Dev<int8_t> d_codes(static_cast<size_t>(((rows + 15) / 16) * 16) * code_stride);
    Dev<float> d_scales(static_cast<size_t>(rows) * scale_stride_bytes / 4u), d_y(static_cast<size_t>(rows) * n * 16u);
    const uint32_t splits = shelf::kernel::exl3_mul24_gemm_splits(rows, n, k, 1u << 20, bits);
    HIP_OK(hipMemset(d_trellis.p, 0x5A, static_cast<size_t>(kt) * nt * 16u * bits * 2));
    HIP_OK(hipMemset(d_codes.p, 1, static_cast<size_t>(((rows + 15) / 16) * 16) * code_stride));
    HIP_OK(hipMemset(d_scales.p, 0, static_cast<size_t>(rows) * scale_stride_bytes));
    hipEvent_t a, b;
    (void)hipEventCreate(&a); (void)hipEventCreate(&b);
    for (int i = 0; i < 5; ++i)
        HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(d_codes.p, d_scales.p, code_stride, scale_stride_bytes, d_trellis.p, bits, d_y.p, rows, n, k, splits, nullptr));
    (void)hipEventRecord(a);
    const int iters = 50;
    for (int i = 0; i < iters; ++i)
        HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(d_codes.p, d_scales.p, code_stride, scale_stride_bytes, d_trellis.p, bits, d_y.p, rows, n, k, splits, nullptr));
    (void)hipEventRecord(b); (void)hipEventSynchronize(b);
    float ms = 0; (void)hipEventElapsedTime(&ms, a, b); ms /= iters;
    const double bytes = static_cast<double>(kt) * nt * 32.0 * bits;
    const double wps = static_cast<double>(n) * k;
    std::printf("BENCH splits=%u rows=%4u n=%6u k=%5u bits=%u: %.3f ms  weights %.1f GB/s  %.1f Gweights/s  %.2f TOPS(int8)\n",
                splits, rows, n, k, bits, ms, bytes / ms / 1e6, wps / ms / 1e6, 2.0 * wps * rows / ms / 1e9);
    return true;
}
}

int main() {
    bool ok = true;
    if (std::getenv("EXL3_BENCH")) {
        for (uint32_t rows : {1u, 4u, 16u, 64u, 256u, 600u}) { bench_case(rows, 17408, 5120, 4); }
        bench_case(1, 5120, 17408, 4); bench_case(600, 5120, 17408, 4); bench_case(1, 12288, 5120, 4); bench_case(1, 248320, 5120, 6);
        return 0;
    }
    ok &= row_parity(4u);
    ok &= row_parity(6u);
    ok &= row_parity(4u, 5120u, 17408u);
    ok &= row_parity(4u, 17408u, 5120u);
    ok &= row_parity(4u, 17408u, 5120u, 16u);
    ok &= head_row_parity();
    for (uint32_t rows : {1u, 5u, 16u, 17u, 33u, 70u}) ok &= run_case(rows, 256, 256, 4, 1234u + rows);
    for (uint32_t bits = 1; bits <= 8; ++bits) ok &= run_case(19, 128, 256, bits, 77u + bits);
    ok &= run_case(40, 384, 640, 6, 99u);
    // More flagged rows than slots (rows / 8 rounded up to 16): the surplus rows stay on the row-scale path.
    ok &= run_case(100, 128, 768, 4, 98u);
    // Full-row CTAs and the swizzled grid: 1024 rows is a multiple of the 128-row CTA, 129 is not.
    ok &= run_case(128, 128, 256, 4, 200u);
    ok &= run_case(129, 128, 256, 4, 201u);
    ok &= run_case(1024, 128, 256, 4, 202u);
    // Mid-size batches (17..64 rows) run K-split too: MT = 2, 3 and 4 row tiles, with even / uneven / tail splits.
    ok &= run_case(17, 256, 1024, 4, 11u, 3u);
    ok &= run_case(32, 384, 640, 3, 12u, 5u);
    ok &= run_case(33, 256, 1024, 4, 13u, 4u);
    ok &= run_case(48, 384, 896, 5, 14u, 7u);
    ok &= run_case(49, 256, 1280, 4, 15u, 5u);
    ok &= run_case(64, 384, 1024, 6, 16u, 2u);
    ok &= mid_row_splits_constant(4u, 17408u, 5120u);
    ok &= mid_row_splits_constant(4u, 5120u, 17408u);
    ok &= mid_row_splits_constant(4u, 5120u, 5120u);
    // split-K (decode-sized rows only): even and uneven splits, including a split that owns a short tail.
    ok &= run_case(1, 256, 256, 4, 5u, 2u);
    ok &= run_case(7, 384, 640, 4, 6u, 3u);
    ok &= run_case(16, 256, 1024, 6, 7u, 8u);
    ok &= run_case(5, 128, 896, 5, 8u, 4u);
    std::printf(ok ? "ALL PASS\n" : "SOME FAILED\n");
    return ok ? 0 : 1;
}
