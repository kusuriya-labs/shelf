#include <shelf/core/memory/types.h>
#include <shelf/kernels/fused/exl3/rotate_quantize.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/quantization/exl3/exl3_int8.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void check(const char* name, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++g_failures;
}

template <typename T>
struct Dev {
    T* p = nullptr;
    explicit Dev(std::size_t n) { (void)hipMalloc(&p, n * sizeof(T)); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

struct Fixture {
    std::uint32_t rows = 0, n = 0, k = 0, bits = 0;
    std::uint32_t code_stride = 0, scale_stride_bytes = 0;
    std::vector<std::uint16_t> x, suh_bits, svh_bits, trellis, resident;
    Dev<std::uint16_t> d_x{0}, d_suh{0}, d_svh{0}, d_resident{0};

    Fixture(std::uint32_t r, std::uint32_t nn, std::uint32_t kk, std::uint32_t b, std::uint32_t seed)
        : rows(r), n(nn), k(kk), bits(b), code_stride(kk),
          scale_stride_bytes(((kk / 128u * 4u) + 15u) & ~15u),
          d_x(static_cast<std::size_t>(r) * kk), d_suh(kk), d_svh(nn),
          d_resident(static_cast<std::size_t>(nn / 16u) * (kk / 16u) * 16u * b) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        x.resize(static_cast<std::size_t>(rows) * k);
        suh_bits.resize(k);
        svh_bits.resize(n);
        trellis.resize(static_cast<std::size_t>(n / 16u) * (k / 16u) * 16u * bits);
        resident.resize(trellis.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = shelf::f32_to_bf16_rne(nd(rng));
        for (std::uint32_t i = 0; i < k; ++i) {
            const _Float16 h = static_cast<_Float16>((rng() & 1u) ? 1.0f : -1.0f);
            std::memcpy(&suh_bits[i], &h, sizeof(std::uint16_t));
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            const _Float16 h = static_cast<_Float16>(0.5f + std::fabs(nd(rng)));
            std::memcpy(&svh_bits[i], &h, sizeof(std::uint16_t));
        }
        for (auto& w : trellis) w = static_cast<std::uint16_t>(rng());
        if (!shelf::quantization::exl3::exl3_tiles_to_nmajor(
                trellis.data(), k / 16u, n / 16u, bits, resident.data(), 2)) {
            std::printf("FAIL trellis repack\n");
            ++g_failures;
        }
        (void)hipMemcpy(d_x.p, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_suh.p, suh_bits.data(), suh_bits.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(d_svh.p, svh_bits.data(), svh_bits.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(d_resident.p, resident.data(), resident.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice);
    }

    bool valid() const {
        return d_x.p && d_suh.p && d_svh.p && d_resident.p;
    }
};

bool same_bf16(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

bool baseline(Fixture& f, Dev<std::int8_t>& codes, Dev<float>& scales, Dev<float>& yprime,
              Dev<std::uint16_t>& out) {
    if (shelf::kernel::launch_exl3_mul24_rotate_quantize(
            reinterpret_cast<const shelf::bf16_t*>(f.d_x.p), f.k, f.d_suh.p, codes.p, scales.p,
            f.rows, f.k, f.code_stride, f.scale_stride_bytes, nullptr) != hipSuccess)
        return false;
    if (shelf::kernel::launch_exl3_mul24_gemm(
            codes.p, scales.p, f.code_stride, f.scale_stride_bytes, f.d_resident.p, f.bits,
            yprime.p, f.rows, f.n, f.k, 1u, nullptr) != hipSuccess)
        return false;
    if (shelf::kernel::launch_exl3_mul24_rotate_out(
            yprime.p, f.d_svh.p, out.p, shelf::kernel::Exl3OutputDType::BF16, f.rows, f.n,
            f.n, 1u, nullptr) != hipSuccess)
        return false;
    return hipDeviceSynchronize() == hipSuccess;
}

bool grouped_case(std::uint32_t rows, std::uint32_t bits0, std::uint32_t bits1,
                  std::uint32_t mask, const char* name) {
    constexpr std::uint32_t n = 512u;
    constexpr std::uint32_t k = 5120u;
    Fixture f0(rows, n, k, bits0, rows * 97u + bits0);
    Fixture f1(rows, n, k, bits1, rows * 97u + bits0);
    if (!f0.valid() || !f1.valid()) return false;

    Dev<std::int8_t> c0(static_cast<std::size_t>(rows) * k), c1(static_cast<std::size_t>(rows) * k);
    Dev<float> s0(static_cast<std::size_t>(rows) * k / 128u), s1(static_cast<std::size_t>(rows) * k / 128u);
    Dev<float> y0(static_cast<std::size_t>(rows) * n), y1(static_cast<std::size_t>(rows) * n);
    Dev<std::uint16_t> g0(static_cast<std::size_t>(rows) * n), g1(static_cast<std::size_t>(rows) * n);
    Dev<std::uint16_t> b0(static_cast<std::size_t>(rows) * n), b1(static_cast<std::size_t>(rows) * n);

    shelf::kernel::PlamoExl3ProjectionGroup group{};
    group.input = reinterpret_cast<const shelf::bf16_t*>(f0.d_x.p);
    group.rows = rows;
    group.k = k;
    group.input_stride = k;
    group.scale_stride_bytes = k / 128u * 4u;
    group.count = 2u;
    group.three_bit_mask = mask;
    auto fill = [&](shelf::kernel::PlamoExl3Projection& p, Fixture& f, std::int8_t* codes,
                    float* scales, float* yprime, std::uint16_t* out) {
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
    fill(group.p[0], f0, c0.p, s0.p, y0.p, g0.p);
    fill(group.p[1], f1, c1.p, s1.p, y1.p, g1.p);

    std::uint32_t launches = 0u;
    const hipError_t err = shelf::kernel::launch_plamo_exl3_projection(
        group, bits0, nullptr, &launches, false, false);
    check(name, err == hipSuccess && launches == 3u);
    if (err != hipSuccess) return false;

    if (!baseline(f0, c0, s0, y0, b0) || !baseline(f1, c1, s1, y1, b1)) {
        check("baseline", false);
        return false;
    }
    std::vector<std::uint16_t> got0(static_cast<std::size_t>(rows) * n), got1(got0.size());
    std::vector<std::uint16_t> want0(got0.size()), want1(got1.size());
    (void)hipMemcpy(got0.data(), g0.p, got0.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(got1.data(), g1.p, got1.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(want0.data(), b0.p, want0.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(want1.data(), b1.p, want1.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    return same_bf16(got0, want0) && same_bf16(got1, want1);
}

bool grouped_case3(std::uint32_t rows, std::uint32_t bits0, std::uint32_t bits1,
                   std::uint32_t bits2, std::uint32_t mask, const char* name) {
    constexpr std::uint32_t n = 512u;
    constexpr std::uint32_t k = 5120u;
    Fixture f0(rows, n, k, bits0, rows * 97u + bits0);
    Fixture f1(rows, n, k, bits1, rows * 97u + bits0);
    Fixture f2(rows, n, k, bits2, rows * 97u + bits0);
    if (!f0.valid() || !f1.valid() || !f2.valid()) return false;

    const std::size_t code_count = static_cast<std::size_t>(rows) * k;
    const std::size_t scale_count = code_count / 128u;
    const std::size_t out_count = static_cast<std::size_t>(rows) * n;
    Dev<std::int8_t> c0(code_count), c1(code_count), c2(code_count);
    Dev<float> s0(scale_count), s1(scale_count), s2(scale_count);
    Dev<float> y0(out_count), y1(out_count), y2(out_count);
    Dev<std::uint16_t> g0(out_count), g1(out_count), g2(out_count);
    Dev<std::uint16_t> b0(out_count), b1(out_count), b2(out_count);

    shelf::kernel::PlamoExl3ProjectionGroup group{};
    group.input = reinterpret_cast<const shelf::bf16_t*>(f0.d_x.p);
    group.rows = rows;
    group.k = k;
    group.input_stride = k;
    group.scale_stride_bytes = k / 128u * 4u;
    group.count = 3u;
    group.three_bit_mask = mask;
    auto fill = [&](shelf::kernel::PlamoExl3Projection& p, Fixture& f, std::int8_t* codes,
                    float* scales, float* yprime, std::uint16_t* out) {
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
    fill(group.p[0], f0, c0.p, s0.p, y0.p, g0.p);
    fill(group.p[1], f1, c1.p, s1.p, y1.p, g1.p);
    fill(group.p[2], f2, c2.p, s2.p, y2.p, g2.p);

    std::uint32_t launches = 0u;
    const hipError_t err = shelf::kernel::launch_plamo_exl3_projection(
        group, bits0, nullptr, &launches, false, false);
    check(name, err == hipSuccess && launches == 3u);
    if (err != hipSuccess) return false;

    if (!baseline(f0, c0, s0, y0, b0) || !baseline(f1, c1, s1, y1, b1) ||
        !baseline(f2, c2, s2, y2, b2)) {
        check("baseline3", false);
        return false;
    }
    std::vector<std::uint16_t> got0(static_cast<std::size_t>(rows) * n), got1(got0), got2(got0);
    std::vector<std::uint16_t> want0(got0), want1(got0), want2(got0);
    (void)hipMemcpy(got0.data(), g0.p, got0.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(got1.data(), g1.p, got1.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(got2.data(), g2.p, got2.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(want0.data(), b0.p, want0.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(want1.data(), b1.p, want1.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(want2.data(), b2.p, want2.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    return same_bf16(got0, want0) && same_bf16(got1, want1) && same_bf16(got2, want2);
}

bool guard_case(std::uint32_t rows, std::uint32_t bits0,
                std::uint32_t mask, const char* name) {
    constexpr std::uint32_t n = 512u;
    constexpr std::uint32_t k = 5120u;
    Fixture f(rows, n, k, bits0, 1u);
    Dev<std::int8_t> c(static_cast<std::size_t>(rows) * k);
    Dev<float> s(static_cast<std::size_t>(rows) * k / 128u);
    Dev<float> y(static_cast<std::size_t>(rows) * n);
    Dev<std::uint16_t> out(static_cast<std::size_t>(rows) * n);
    shelf::kernel::PlamoExl3ProjectionGroup group{};
    group.input = reinterpret_cast<const shelf::bf16_t*>(f.d_x.p);
    group.rows = rows;
    group.k = k;
    group.input_stride = k;
    group.scale_stride_bytes = k / 128u * 4u;
    group.count = 2u;
    group.three_bit_mask = mask;
    for (std::uint32_t i = 0; i < 2u; ++i) {
        auto& p = group.p[i];
        p.trellis = f.d_resident.p;
        p.suh = f.d_suh.p;
        p.svh = f.d_svh.p;
        p.codes = c.p;
        p.scales = s.p;
        p.yprime = y.p;
        p.out = reinterpret_cast<shelf::bf16_t*>(out.p);
        p.n = n;
        p.splits = 1u;
        p.out_stride = n;
    }
    const bool rejected = shelf::kernel::launch_plamo_exl3_projection(
        group, bits0, nullptr, nullptr, false, false) == hipErrorInvalidValue;
    check(name, rejected);
    return rejected;
}

}  // namespace

int main() {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return 77;
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess) return 77;
    if (std::strncmp(props.gcnArchName, "gfx1201", 7) != 0) return 77;

    bool ok = true;
    ok = grouped_case(1u, 3u, 4u, 1u, "rows=1 3/4 grouped") && ok;
    ok = grouped_case(8u, 3u, 4u, 1u, "rows=8 3/4 grouped") && ok;
    ok = grouped_case(16u, 3u, 4u, 1u, "rows=16 3/4 grouped") && ok;
    ok = grouped_case(1u, 4u, 3u, 2u, "rows=1 4/3 grouped") && ok;
    ok = grouped_case(16u, 4u, 3u, 2u, "rows=16 4/3 grouped") && ok;
    ok = grouped_case3(1u, 3u, 4u, 4u, 1u, "rows=1 3/4/4 grouped") && ok;
    ok = grouped_case3(1u, 4u, 3u, 4u, 2u, "rows=1 4/3/4 grouped") && ok;
    ok = grouped_case3(16u, 4u, 4u, 3u, 4u, "rows=16 4/4/3 grouped") && ok;
    ok = grouped_case3(16u, 3u, 4u, 3u, 5u, "rows=16 3/4/3 grouped") && ok;
    ok = guard_case(17u, 3u, 1u, "rows=17 mixed rejected") && ok;
    ok = guard_case(17u, 4u, 2u, "rows=17 4/3 rejected") && ok;
    ok = guard_case(1u, 3u, 4u, "mask out of range rejected") && ok;

    if (!ok || g_failures != 0) {
        std::printf("%d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("ALL PASS\n");
    return 0;
}
