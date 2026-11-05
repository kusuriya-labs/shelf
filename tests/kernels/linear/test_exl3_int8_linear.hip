#include <shelf/kernels/optimized/linear/exl3_int8.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <shelf/core/memory/types.h>
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
    std::vector<float> suh(k), svh(n);
    for (auto& s : suh) s = (rng() & 1u) ? 1.0f : -1.0f;
    for (auto& s : svh) s = static_cast<float>(static_cast<_Float16>(0.5f + std::fabs(nd(rng))));  // fp16-exact, as stored
    // Random trellis words; the expected q comes from the CPU reference decoder (itself checked against
    // the real rocm_exl3 decoder in test_exl3_int8_decode).
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

    HIP_OK(shelf::kernel::launch_exl3_int8_rotate_quantize(
        reinterpret_cast<const shelf::bf16_t*>(d_x.p), k, d_suh.p, d_codes.p, d_scales.p, rows, k,
        code_stride, scale_stride_bytes, nullptr));
    HIP_OK(shelf::kernel::launch_exl3_int8_gemm(d_codes.p, d_scales.p, code_stride, scale_stride_bytes,
                                               d_trellis.p, bits, d_y.p, rows, n, k, splits, nullptr));
    HIP_OK(shelf::kernel::launch_exl3_int8_rotate_out(d_y.p, d_svh.p, d_out.p,
                                                     shelf::kernel::Exl3Int8OutputDType::F32, rows, n, n, splits, nullptr));
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

    const bool ok = code_mismatch == 0 && worst_scale == 0.0 && worst_y < 1e-5 && worst_out < 1e-5 && worst_ideal < 0.02;
    std::printf("%s rows=%u n=%u k=%u: code_mismatch=%zu scale_err=%.3g gemm_rel=%.3g out_rel=%.3g ideal_rel_l2=%.4f\n",
                ok ? "PASS" : "FAIL", rows, n, k, code_mismatch, worst_scale, worst_y, worst_out, worst_ideal);
    return ok;
}


bool bench_case(uint32_t rows, uint32_t n, uint32_t k, uint32_t bits) {
    const uint32_t kt = k / 16u, nt = n / 16u;
    const uint32_t code_stride = k, groups = k / 128u;
    const uint32_t scale_stride_bytes = ((groups * 4u) + 15u) & ~15u;
    Dev<uint16_t> d_trellis(static_cast<size_t>(kt) * nt * 16u * bits);
    Dev<int8_t> d_codes(static_cast<size_t>(((rows + 15) / 16) * 16) * code_stride);
    Dev<float> d_scales(static_cast<size_t>(rows) * scale_stride_bytes / 4u), d_y(static_cast<size_t>(rows) * n * 16u);
    const uint32_t splits = shelf::kernel::exl3_int8_gemm_splits(rows, n, k, 1u << 20);
    HIP_OK(hipMemset(d_trellis.p, 0x5A, static_cast<size_t>(kt) * nt * 16u * bits * 2));
    HIP_OK(hipMemset(d_codes.p, 1, static_cast<size_t>(((rows + 15) / 16) * 16) * code_stride));
    HIP_OK(hipMemset(d_scales.p, 0, static_cast<size_t>(rows) * scale_stride_bytes));
    hipEvent_t a, b;
    (void)hipEventCreate(&a); (void)hipEventCreate(&b);
    for (int i = 0; i < 5; ++i)
        HIP_OK(shelf::kernel::launch_exl3_int8_gemm(d_codes.p, d_scales.p, code_stride, scale_stride_bytes, d_trellis.p, bits, d_y.p, rows, n, k, splits, nullptr));
    (void)hipEventRecord(a);
    const int iters = 50;
    for (int i = 0; i < iters; ++i)
        HIP_OK(shelf::kernel::launch_exl3_int8_gemm(d_codes.p, d_scales.p, code_stride, scale_stride_bytes, d_trellis.p, bits, d_y.p, rows, n, k, splits, nullptr));
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
    for (uint32_t rows : {1u, 5u, 16u, 17u, 33u, 70u}) ok &= run_case(rows, 256, 256, 4, 1234u + rows);
    for (uint32_t bits = 1; bits <= 8; ++bits) ok &= run_case(19, 128, 256, bits, 77u + bits);
    ok &= run_case(40, 384, 640, 6, 99u);
    // split-K (decode-sized rows only): even and uneven splits, including a split that owns a short tail.
    ok &= run_case(1, 256, 256, 4, 5u, 2u);
    ok &= run_case(7, 384, 640, 4, 6u, 3u);
    ok &= run_case(16, 256, 1024, 6, 7u, 8u);
    ok &= run_case(5, 128, 896, 5, 8u, 4u);
    std::printf(ok ? "ALL PASS\n" : "SOME FAILED\n");
    return ok ? 0 : 1;
}
