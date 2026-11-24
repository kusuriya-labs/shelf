#include <shelf/kernels/fused/exl3/group_body.h>
#include <shelf/kernels/fused/exl3/rotate_quantize.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <shelf/core/memory/types.h>
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

int g_failures = 0;

void check(const char* name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++g_failures;
    }
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

template <int BITS, int MT, bool PIPE, bool K_MAJOR, bool SHUFFLE>
__global__ void group_body_test_kernel(
    const int8_t* codes, const float* scales, uint32_t code_stride, uint32_t scale_stride_floats,
    const uint16_t* trellis, float* yprime, uint32_t rows, uint32_t n, uint32_t k) {
    shelf::kernel::exl3_detail::mul24::group_body<BITS, MT, PIPE, 4, false, K_MAJOR, SHUFFLE>(
        codes, scales, code_stride, scale_stride_floats, reinterpret_cast<const uint32_t*>(trellis),
        yprime, rows, nullptr, n, k, k / 128u, 1u, blockIdx.x, 0u);
}

template <typename T>
struct Dev {
    T* p = nullptr;
    explicit Dev(size_t n) { (void)hipMalloc(&p, n * sizeof(T)); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

struct Fixture {
    uint32_t rows = 0, n = 0, k = 0, bits = 0;
    uint32_t code_stride = 0, scale_stride_bytes = 0;
    std::vector<uint16_t> x, suh_bits, svh_bits, trellis, resident;
    std::vector<float> suh, svh;
    Dev<uint16_t> d_x{0}, d_suh{0}, d_svh{0}, d_trellis{0}, d_resident{0};
    explicit Fixture(uint32_t r, uint32_t nn, uint32_t kk, uint32_t b, uint32_t seed)
        : rows(r), n(nn), k(kk), bits(b),
          code_stride(kk), scale_stride_bytes(((kk / 128u * 4u) + 15u) & ~15u),
          d_x(static_cast<size_t>(r) * kk), d_suh(kk), d_svh(nn),
          d_trellis(static_cast<size_t>(nn / 16u) * (kk / 16u) * 16u * b),
          d_resident(static_cast<size_t>(nn / 16u) * (kk / 16u) * 16u * b) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        x.resize(static_cast<size_t>(rows) * k);
        suh.resize(k);
        svh.resize(n);
        suh_bits.resize(k);
        svh_bits.resize(n);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = shelf::f32_to_bf16_rne(nd(rng));
        for (uint32_t i = 0; i < k; ++i) {
            suh[i] = (rng() & 1u) ? 1.0f : -1.0f;
            const _Float16 h = static_cast<_Float16>(suh[i]);
            std::memcpy(&suh_bits[i], &h, sizeof(uint16_t));
        }
        for (uint32_t i = 0; i < n; ++i) {
            svh[i] = static_cast<float>(static_cast<_Float16>(0.5f + std::fabs(nd(rng))));
            const _Float16 h = static_cast<_Float16>(svh[i]);
            std::memcpy(&svh_bits[i], &h, sizeof(uint16_t));
        }
        trellis.resize(d_trellis_size());
        for (auto& w : trellis) w = static_cast<uint16_t>(rng());
        resident.resize(trellis.size());
        if (!shelf::quantization::exl3::exl3_tiles_to_nmajor(
                trellis.data(), k / 16u, n / 16u, bits, resident.data(), 2)) {
            std::printf("trellis repack failed\n");
            ++g_failures;
        }
        if (!valid()) {
            std::printf("device allocation failed\n");
            ++g_failures;
            return;
        }
        (void)hipMemcpy(d_x.p, x.data(), x.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_suh.p, suh_bits.data(), suh_bits.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_svh.p, svh_bits.data(), svh_bits.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_trellis.p, trellis.data(), trellis.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_resident.p, resident.data(), resident.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
    }
    bool valid() const {
        return d_x.p != nullptr && d_suh.p != nullptr && d_svh.p != nullptr &&
               d_trellis.p != nullptr && d_resident.p != nullptr;
    }
    size_t d_trellis_size() const {
        return static_cast<size_t>(n / 16u) * (k / 16u) * 16u * bits;
    }
};

bool compare_bf16(const char* name, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    if (a.size() != b.size()) return false;
    double worst = 0.0, norm = 1e-6;
    for (size_t i = 0; i < a.size(); ++i) {
        const double ea = std::fabs(bf16_to_f32(a[i]) - bf16_to_f32(b[i]));
        worst = std::max(worst, ea);
        norm = std::max(norm, static_cast<double>(std::fabs(bf16_to_f32(b[i]))));
    }
    const bool ok = worst / norm < 2e-2;
    std::printf("  %s: max_abs=%.3g norm=%.3g %s\n", name, worst, norm, ok ? "PASS" : "FAIL");
    if (!ok) ++g_failures;
    return ok;
}

bool compare_f32(const char* name, const std::vector<float>& a, const std::vector<float>& b, double tol) {
    if (a.size() != b.size()) return false;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        worst = std::max(worst, e / std::max(1.0, std::fabs(static_cast<double>(b[i]))));
    }
    const bool ok = worst < tol;
    std::printf("  %s: rel_inf=%.3g %s\n", name, worst, ok ? "PASS" : "FAIL");
    if (!ok) ++g_failures;
    return ok;
}

bool test_fused_linear(uint32_t rows, uint32_t n, uint32_t k, uint32_t bits) {
    std::printf("fused_linear rows=%u n=%u k=%u bits=%u\n", rows, n, k, bits);
    Fixture f(rows, n, k, bits, rows * 97u + bits);
    Dev<int8_t> d_codes(static_cast<size_t>(rows) * f.code_stride);
    Dev<float> d_scales(static_cast<size_t>(rows) * f.scale_stride_bytes / 4u);
    Dev<float> d_y(static_cast<size_t>(rows) * n);
    Dev<uint16_t> d_base(static_cast<size_t>(rows) * n), d_fused(static_cast<size_t>(rows) * n);
    HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_quantize(
        reinterpret_cast<const shelf::bf16_t*>(f.d_x.p), k, f.d_suh.p, d_codes.p, d_scales.p,
        rows, k, f.code_stride, f.scale_stride_bytes, nullptr));
    HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(
        d_codes.p, d_scales.p, f.code_stride, f.scale_stride_bytes, f.d_resident.p, bits,
        d_y.p, rows, n, k, 1u, nullptr));
    HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_out(
        d_y.p, f.d_svh.p, d_base.p, shelf::kernel::Exl3OutputDType::BF16, rows, n, n, 1u, nullptr));
    HIP_OK(shelf::kernel::launch_exl3_mul24_fused_linear(
        f.d_x.p, k, f.d_suh.p, f.d_svh.p, f.d_resident.p, d_fused.p, n, rows, n, k, bits, nullptr));
    HIP_OK(hipDeviceSynchronize());
    std::vector<uint16_t> base(static_cast<size_t>(rows) * n), fused(base.size());
    HIP_OK(hipMemcpy(base.data(), d_base.p, base.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(fused.data(), d_fused.p, fused.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    compare_bf16("fused vs 3-launch", fused, base);
    return true;
}

bool test_fused_linear_guard() {
    std::printf("fused_linear guards\n");
    uint16_t dummy = 0;
    check("rows > 16 rejected",
          shelf::kernel::launch_exl3_mul24_fused_linear(&dummy, 5120u, &dummy, &dummy, &dummy,
                                                        &dummy, 512u, 17u, 512u, 5120u, 4u, nullptr) ==
              hipErrorInvalidValue);
    check("unsupported bits rejected",
          shelf::kernel::launch_exl3_mul24_fused_linear(&dummy, 5120u, &dummy, &dummy, &dummy,
                                                        &dummy, 512u, 1u, 512u, 5120u, 5u, nullptr) ==
              hipErrorInvalidValue);
    check("misaligned n rejected",
          shelf::kernel::launch_exl3_mul24_fused_linear(&dummy, 5120u, &dummy, &dummy, &dummy,
                                                        &dummy, 512u, 1u, 192u, 5120u, 4u, nullptr) ==
              hipErrorInvalidValue);
    check("oversized k rejected by LDS budget",
          shelf::kernel::launch_exl3_mul24_fused_linear(&dummy, 65536u, &dummy, &dummy, &dummy,
                                                        &dummy, 512u, 1u, 512u, 65536u, 4u, nullptr) ==
              hipErrorInvalidValue);
    return true;
}

bool test_prefill(uint32_t rows, uint32_t n, uint32_t k, uint32_t bits, uint32_t splits) {
    std::printf("prefill_wmma rows=%u n=%u k=%u bits=%u splits=%u\n", rows, n, k, bits, splits);
    Fixture f(rows, n, k, bits, rows * 31u + splits);
    Dev<int8_t> d_codes(static_cast<size_t>(rows) * f.code_stride);
    Dev<float> d_scales(static_cast<size_t>(rows) * f.scale_stride_bytes / 4u);
    Dev<int8_t> d_dense(static_cast<size_t>(n) * k);
    Dev<float> d_y1(static_cast<size_t>(rows) * n);
    Dev<float> d_ys(static_cast<size_t>(rows) * n * splits);
    HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_quantize(
        reinterpret_cast<const shelf::bf16_t*>(f.d_x.p), k, f.d_suh.p, d_codes.p, d_scales.p,
        rows, k, f.code_stride, f.scale_stride_bytes, nullptr));
    HIP_OK(shelf::kernel::launch_exl3_mul24_recon_wmma(f.d_resident.p, bits, d_dense.p, n, k, nullptr));
    HIP_OK(shelf::kernel::launch_exl3_mul24_prefill_wmma(
        d_codes.p, d_scales.p, f.code_stride, f.scale_stride_bytes, d_dense.p, d_y1.p,
        rows, n, k, nullptr, 1u));
    HIP_OK(shelf::kernel::launch_exl3_mul24_prefill_wmma(
        d_codes.p, d_scales.p, f.code_stride, f.scale_stride_bytes, d_dense.p, d_ys.p,
        rows, n, k, nullptr, splits));
    HIP_OK(hipDeviceSynchronize());
    std::vector<float> y1(static_cast<size_t>(rows) * n), ys(static_cast<size_t>(rows) * n * splits);
    HIP_OK(hipMemcpy(y1.data(), d_y1.p, y1.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(ys.data(), d_ys.p, ys.size() * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<float> sum(y1.size(), 0.0f);
    for (uint32_t z = 0u; z < splits; ++z)
        for (size_t i = 0; i < y1.size(); ++i) sum[i] += ys[static_cast<size_t>(z) * y1.size() + i];
    compare_f32("split sum vs single", sum, y1, 1e-4);
    return true;
}

bool test_grouped_mixed() {
    const uint32_t rows = 1u, n = 512u, k = 5120u;
    std::printf("grouped projection 3-bit + mixed 3/4\n");
    Fixture f3(rows, n, k, 3u, 77u), f4(rows, n, k, 4u, 77u);
    Dev<int8_t> c3(static_cast<size_t>(rows) * k), c4(static_cast<size_t>(rows) * k);
    Dev<float> s3(static_cast<size_t>(rows) * k / 128u), s4(static_cast<size_t>(rows) * k / 128u);
    Dev<float> y3(static_cast<size_t>(rows) * n), y4(static_cast<size_t>(rows) * n);
    Dev<uint16_t> g3(static_cast<size_t>(rows) * n), g4(static_cast<size_t>(rows) * n);
    Dev<uint16_t> b3(static_cast<size_t>(rows) * n), b4(static_cast<size_t>(rows) * n);
    shelf::kernel::PlamoExl3ProjectionGroup group{};
    group.input = reinterpret_cast<const shelf::bf16_t*>(f4.d_x.p);
    group.rows = rows;
    group.k = k;
    group.input_stride = k;
    group.scale_stride_bytes = k / 128u * 4u;
    group.count = 2u;
    auto fill = [&](shelf::kernel::PlamoExl3Projection& p, Fixture& f, int8_t* codes, float* scales,
                    float* yprime, uint16_t* out) {
        p.trellis = f.d_resident.p;
        p.suh = f.d_suh.p;
        p.svh = f.d_svh.p;
        p.codes = codes;
        p.scales = scales;
        p.yprime = yprime;
        p.out = reinterpret_cast<shelf::bf16_t*>(out);
        p.n = n;
        p.splits = 1u;
        p.out_stride = n;
    };
    fill(group.p[0], f3, c3.p, s3.p, y3.p, g3.p);
    fill(group.p[1], f4, c4.p, s4.p, y4.p, g4.p);
    const uint32_t bpw[2] = {3u, 4u};
    uint32_t launches = 0u;
    hipError_t err = hipSuccess;
    for (uint32_t j = 0u; j < 2u; ++j) {
        shelf::kernel::PlamoExl3ProjectionGroup sub = group;
        sub.count = 1u;
        sub.p[0] = group.p[j];
        uint32_t sub_launches = 0u;
        err = shelf::kernel::launch_plamo_exl3_projection(sub, bpw[j], nullptr, &sub_launches, false, false);
        launches += sub_launches;
        if (err != hipSuccess) break;
    }
    check("mixed grouped launches", err == hipSuccess && launches == 6u);
    auto baseline = [&](Fixture& f, int8_t* codes, float* scales, float* yprime, uint16_t* out, uint32_t bits) {
        HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_quantize(
            reinterpret_cast<const shelf::bf16_t*>(f.d_x.p), k, f.d_suh.p, codes, scales, rows, k, k,
            k / 128u * 4u, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_gemm(
            codes, scales, k, k / 128u * 4u, f.d_resident.p, bits, yprime, rows, n, k, 1u, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_out(
            yprime, f.d_svh.p, out, shelf::kernel::Exl3OutputDType::BF16, rows, n, n, 1u, nullptr));
    };
    baseline(f3, c3.p, s3.p, y3.p, b3.p, 3u);
    baseline(f4, c4.p, s4.p, y4.p, b4.p, 4u);
    HIP_OK(hipDeviceSynchronize());
    std::vector<uint16_t> go3(static_cast<size_t>(rows) * n), bo3(go3.size());
    std::vector<uint16_t> go4(static_cast<size_t>(rows) * n), bo4(go4.size());
    HIP_OK(hipMemcpy(go3.data(), g3.p, go3.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(bo3.data(), b3.p, bo3.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(go4.data(), g4.p, go4.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(bo4.data(), b4.p, bo4.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    compare_bf16("grouped 3-bit vs baseline", go3, bo3);
    compare_bf16("grouped 4-bit vs baseline", go4, bo4);
    return true;
}

bool test_group_body(int mt) {
    const uint32_t rows = static_cast<uint32_t>(16 * mt);
    const uint32_t n = 512u, k = 5120u, bits = 4u;
    std::printf("group_body mt=%d k_major+shuffle\n", mt);
    Fixture f(rows, n, k, bits, 1234u + static_cast<uint32_t>(mt));
    Dev<int8_t> d_codes(static_cast<size_t>(rows) * f.code_stride);
    Dev<float> d_scales(static_cast<size_t>(rows) * f.scale_stride_bytes / 4u);
    Dev<float> d_a(static_cast<size_t>(rows) * n), d_b(static_cast<size_t>(rows) * n);
    HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_quantize(
        reinterpret_cast<const shelf::bf16_t*>(f.d_x.p), k, f.d_suh.p, d_codes.p, d_scales.p,
        rows, k, f.code_stride, f.scale_stride_bytes, nullptr));
    if (mt <= 3) {
        group_body_test_kernel<4, 1, true, false, false><<<n / 64u, 128>>>(
            d_codes.p, d_scales.p, f.code_stride, f.scale_stride_bytes / 4u, f.d_resident.p, d_a.p,
            rows, n, k);
        group_body_test_kernel<4, 1, false, true, true><<<n / 64u, 128>>>(
            d_codes.p, d_scales.p, f.code_stride, f.scale_stride_bytes / 4u, f.d_trellis.p, d_b.p,
            rows, n, k);
    } else {
        group_body_test_kernel<4, 4, true, false, false><<<n / 64u, 128>>>(
            d_codes.p, d_scales.p, f.code_stride, f.scale_stride_bytes / 4u, f.d_resident.p, d_a.p,
            rows, n, k);
        group_body_test_kernel<4, 4, false, true, true><<<n / 64u, 128>>>(
            d_codes.p, d_scales.p, f.code_stride, f.scale_stride_bytes / 4u, f.d_trellis.p, d_b.p,
            rows, n, k);
    }
    HIP_OK(hipGetLastError());
    HIP_OK(hipDeviceSynchronize());
    std::vector<float> a(static_cast<size_t>(rows) * n), b(a.size());
    HIP_OK(hipMemcpy(a.data(), d_a.p, a.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(b.data(), d_b.p, b.size() * sizeof(float), hipMemcpyDeviceToHost));
    compare_f32("k-major shuffle vs n-major", a, b, 1e-6);
    return true;
}

}  // namespace

int main() {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return 1;
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess) return 1;
    std::printf("device=%s\n", props.gcnArchName);

    test_fused_linear_guard();
    test_fused_linear(1u, 512u, 5120u, 4u);
    test_fused_linear(4u, 512u, 5120u, 4u);
    test_fused_linear(16u, 512u, 5120u, 4u);
    test_fused_linear(3u, 512u, 5120u, 3u);
    test_fused_linear(16u, 512u, 5120u, 3u);
    test_prefill(256u, 512u, 5120u, 4u, 1u);
    test_prefill(256u, 512u, 5120u, 4u, 4u);
    test_prefill(264u, 512u, 5120u, 3u, 2u);
    test_grouped_mixed();
    test_group_body(1);
    test_group_body(4);

    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
