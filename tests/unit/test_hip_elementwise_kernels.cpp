// shelf unit test — vendored PhaseShift elementwise kernel family (gfx1201).
// Parity of every kernel against an independent host reference (double), plus
// a small launch-timing sample. Tolerances: 3e-2 abs for bf16 in/out paths,
// 1e-5 for paths whose output (and intermediates) stay in f32.
#include <hip/hip_runtime.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <shelf/backend/hip/kernels/elementwise.h>
#include <shelf/backend/hip/kernels/embedding.h>
#include <shelf/backend/hip/kernels/l2_normalize.h>
#include <shelf/backend/hip/kernels/rmsnorm.h>
#include <shelf/foundation/numeric.hpp>

namespace {

using shelf::bf16_t;
using shelf::bf16_to_f32;
using shelf::f32_to_bf16;

using shelf::hipk::ElementwiseSplitLayout;
using shelf::hipk::ProgramStatus;
using shelf::hipk::RmsNormDataType;
using shelf::hipk::RmsNormWeightLayout;
using shelf::hipk::RmsNormWeightMode;
using namespace shelf::hipk;  // launch_* host APIs

int failures = 0;
int checks = 0;

void expect_ok(hipError_t e, const char* what) {
    ++checks;
    if (e != hipSuccess) {
        ++failures;
        std::printf("FAIL hip: %s: %s\n", what, hipGetErrorString(e));
    }
}

// NOTE on the 3e-2 bf16 tolerance: one bf16 ulp is 2^-8 relative, so for
// |value| >= 8 a single correct rounding already exceeds 3e-2. The bf16-output
// cases below keep their inputs scaled so |result| < 8 (largest ulp ~0.031/2),
// which is what makes a fixed absolute tolerance meaningful. Relative error is
// printed alongside so a rounding-level deviation stays visible.
void report(double max_abs, double tol, const char* name, double max_rel = -1.0) {
    ++checks;
    const bool bad = !(max_abs <= tol) || !(max_abs == max_abs);
    if (bad) {
        ++failures;
        std::printf("FAIL %-46s max_abs=%.3e (tol %.0e)", name, max_abs, tol);
    } else {
        std::printf("ok   %-46s max_abs=%.3e (tol %.0e)", name, max_abs, tol);
    }
    if (max_rel >= 0.0) std::printf(" max_rel=%.2e", max_rel);
    std::printf("\n");
}

// relative error helper, guarding against tiny reference magnitudes
double rel_err(double got, double ref) {
    const double m = std::abs(ref);
    return (m < 1e-3) ? 0.0 : std::abs(got - ref) / m;
}

struct Buf {
    void* p = nullptr;
    size_t bytes = 0;
    explicit Buf(size_t b) : bytes(b) {
        if (hipMalloc(&p, b) != hipSuccess) {
            p = nullptr;
            ++failures;
        }
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() {
        if (p) hipFree(p);
    }
    void fill(int byte) {
        if (p) expect_ok(hipMemset(p, byte, bytes), "memset");
    }
};

template <class F>
double bench_ms(F&& fn, int iters) {
    expect_ok(hipDeviceSynchronize(), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        const hipError_t e = fn();
        (void)e;
    }
    expect_ok(hipDeviceSynchronize(), "sync");
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count() /
        static_cast<double>(iters);
}

std::mt19937 rng(0xC0FFEEu);
float frand(float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(rng);
}

// ---------------------------------------------------------------- rmsnorm
struct RmsCase {
    RmsNormDataType in_t;
    RmsNormDataType out_t;
    RmsNormWeightLayout lay;
    RmsNormWeightMode mode;
    uint32_t rows, features, group, in_stride, out_stride;
    float eps;
    double tol;
    const char* name;
};

void run_rmsnorm(const RmsCase& c) {
    const size_t in_n = static_cast<size_t>(c.rows) * c.in_stride;
    const size_t out_n = static_cast<size_t>(c.rows) * c.out_stride;
    const uint32_t w_n = (c.lay == RmsNormWeightLayout::PER_FEATURE)
                             ? c.features
                             : (c.lay == RmsNormWeightLayout::PER_GROUP ? c.group : 1u);

    std::vector<bf16_t> hIn_b(in_n, bf16_t{0});
    std::vector<float> hIn_f(in_n, 0.0f);
    for (size_t i = 0; i < in_n; ++i) {
        const float v = frand(-1.5f, 1.5f);
        hIn_b[i] = f32_to_bf16(v);
        hIn_f[i] = v;
    }
    std::vector<bf16_t> hW(w_n);
    for (auto& v : hW) v = f32_to_bf16(frand(-0.6f, 0.6f));

    // Fresh double reference; mirrors only the *semantics* of the launch config
    // (which weight index, ONE_PLUS or DIRECT, and — for f32 inputs — the
    // kernel's documented intermediate round of x*inv to bf16).
    const bool round_mid = (c.in_t == RmsNormDataType::F32);
    std::vector<double> ref(out_n, 0.0);
    for (uint32_t r = 0; r < c.rows; ++r) {
        for (uint32_t gi = 0; gi < c.features / c.group; ++gi) {
            const uint32_t g0 = gi * c.group;
            double ss = 0.0;
            for (uint32_t h = 0; h < c.group; ++h) {
                const double xv = (c.in_t == RmsNormDataType::BF16)
                                      ? static_cast<double>(
                                            bf16_to_f32(hIn_b[static_cast<size_t>(r) * c.in_stride + g0 + h]))
                                      : static_cast<double>(hIn_f[static_cast<size_t>(r) * c.in_stride + g0 + h]);
                ss += xv * xv;
            }
            const double inv = 1.0 / std::sqrt(ss / static_cast<double>(c.group) +
                                               static_cast<double>(c.eps));
            for (uint32_t h = 0; h < c.group; ++h) {
                const size_t i = static_cast<size_t>(r) * c.in_stride + g0 + h;
                const double xv = (c.in_t == RmsNormDataType::BF16)
                                      ? static_cast<double>(bf16_to_f32(hIn_b[i]))
                                      : static_cast<double>(hIn_f[i]);
                double y = xv * inv;
                if (round_mid) y = static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(y))));
                if (c.lay != RmsNormWeightLayout::NONE) {
                    const uint32_t wi = (c.lay == RmsNormWeightLayout::PER_FEATURE) ? (g0 + h) : h;
                    const double wv = static_cast<double>(bf16_to_f32(hW[wi]));
                    y *= (c.mode == RmsNormWeightMode::ONE_PLUS) ? (1.0 + wv) : wv;
                }
                ref[static_cast<size_t>(r) * c.out_stride + g0 + h] = y;
            }
        }
    }

    const size_t in_bytes = (c.in_t == RmsNormDataType::BF16) ? in_n * 2u : in_n * 4u;
    const size_t out_bytes = (c.out_t == RmsNormDataType::BF16) ? out_n * 2u : out_n * 4u;
    Buf dIn(in_bytes), dW(w_n * 2u), dOut(out_n * 4u);
    if (!dIn.p || !dW.p || !dOut.p) return;
    dIn.fill(0);
    dOut.fill(0);
    expect_ok(hipMemcpy(dIn.p, (c.in_t == RmsNormDataType::BF16) ? (void*)hIn_b.data() : (void*)hIn_f.data(),
                        in_bytes, hipMemcpyHostToDevice),
              "rms in");
    expect_ok(hipMemcpy(dW.p, hW.data(), w_n * 2u, hipMemcpyHostToDevice), "rms w");

    const hipError_t err = launch_rmsnorm(
        dIn.p, c.in_t, dW.p, RmsNormDataType::BF16, c.lay, c.mode, dOut.p, c.out_t,
        c.rows, c.features, c.group, c.in_stride, c.out_stride, c.eps, 0);
    expect_ok(err, c.name);
    if (err != hipSuccess) return;

    std::vector<char> hOut(out_bytes, 0);
    expect_ok(hipMemcpy(hOut.data(), dOut.p, out_bytes, hipMemcpyDeviceToHost), "rms out");
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (uint32_t r = 0; r < c.rows; ++r) {
        for (uint32_t f = 0; f < c.features; ++f) {
            const size_t o = static_cast<size_t>(r) * c.out_stride + f;
            const double got = (c.out_t == RmsNormDataType::BF16)
                                   ? static_cast<double>(bf16_to_f32(
                                         *reinterpret_cast<const bf16_t*>(hOut.data() + o * 2u)))
                                   : static_cast<double>(*reinterpret_cast<const float*>(hOut.data() + o * 4u));
            max_abs = std::max(max_abs, std::abs(got - ref[o]));
            max_rel = std::max(max_rel, rel_err(got, ref[o]));
        }
    }
    report(max_abs, c.tol, c.name, max_rel);
}

// ------------------------------------------------------------- elementwise
constexpr double kTolBf16 = 3e-2;
constexpr double kTolF32 = 1e-5;

void run_silu_bf16_f32() {
    const uint32_t rows = 5, feat = 100, stride = 103;  // stride exercises the scalar path
    const size_t n = static_cast<size_t>(rows) * stride;
    std::vector<bf16_t> in(n);
    for (auto& v : in) v = f32_to_bf16(frand(-8.0f, 8.0f));
    Buf dIn(n * 2u), dOut(n * 4u);
    dIn.fill(0);
    dOut.fill(0);
    expect_ok(hipMemcpy(dIn.p, in.data(), n * 2u, hipMemcpyHostToDevice), "silu in");
    const hipError_t err = launch_silu_bf16_f32(static_cast<const bf16_t*>(dIn.p), stride,
                                                static_cast<float*>(dOut.p), stride, rows, feat, 0);
    expect_ok(err, "silu bf16->f32 launch");
    if (err != hipSuccess) return;
    std::vector<float> out(n, 0.0f);
    expect_ok(hipMemcpy(out.data(), dOut.p, n * 4u, hipMemcpyDeviceToHost), "silu out");
    double mx = 0.0;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < feat; ++c) {
            const size_t i = static_cast<size_t>(r) * stride + c;
            const double x = bf16_to_f32(in[i]);
            const double ref = x / (1.0 + std::exp(-x));
            mx = std::max(mx, std::abs(static_cast<double>(out[i]) - ref));
        }
    report(mx, kTolF32, "silu_bf16_f32 (x/(1+e^-x))");
    (void)bench_ms([&] { return launch_silu_bf16_f32(static_cast<const bf16_t*>(dIn.p), stride,
                                                     static_cast<float*>(dOut.p), stride, rows, feat, 0); },
                   100);
}

void run_silu_f32_bf16() {
    const uint32_t rows = 4, feat = 256, stride = 256;
    const size_t n = static_cast<size_t>(rows) * stride;
    std::vector<float> in(n);
    for (auto& v : in) v = frand(-5.5f, 5.5f);  // keeps |out| < 6: one bf16 ulp < 3e-2
    Buf dIn(n * 4u), dOut(n * 2u);
    dIn.fill(0);
    dOut.fill(0);
    expect_ok(hipMemcpy(dIn.p, in.data(), n * 4u, hipMemcpyHostToDevice), "silu2 in");
    const hipError_t err = launch_silu_f32_bf16(static_cast<const float*>(dIn.p), stride,
                                                static_cast<bf16_t*>(dOut.p), stride, rows, feat, 0);
    expect_ok(err, "silu f32->bf16 launch");
    if (err != hipSuccess) return;
    std::vector<bf16_t> out(n, bf16_t{0});
    expect_ok(hipMemcpy(out.data(), dOut.p, n * 2u, hipMemcpyDeviceToHost), "silu2 out");
    double mx = 0.0, mr = 0.0;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < feat; ++c) {
            const size_t i = static_cast<size_t>(r) * stride + c;
            const double x = in[i];
            const double ref = x / (1.0 + std::exp(-x));
            const double got = static_cast<double>(bf16_to_f32(out[i]));
            mx = std::max(mx, std::abs(got - ref));
            mr = std::max(mr, rel_err(got, ref));
        }
    report(mx, kTolBf16, "silu_f32_bf16 (x/(1+e^-x)) -> bf16", mr);
}

void run_mul_f32_f32_bf16() {
    const uint32_t rows = 4, feat = 130, stride = 133;
    const size_t n = static_cast<size_t>(rows) * stride;
    std::vector<float> a(n), b(n);
    for (auto& v : a) v = frand(-2.2f, 2.2f);  // |a*b| < 5: one bf16 ulp < 3e-2
    for (auto& v : b) v = frand(-2.2f, 2.2f);
    Buf dA(n * 4u), dB(n * 4u), dOut(n * 2u);
    dA.fill(0);
    dB.fill(0);
    dOut.fill(0);
    expect_ok(hipMemcpy(dA.p, a.data(), n * 4u, hipMemcpyHostToDevice), "mul a");
    expect_ok(hipMemcpy(dB.p, b.data(), n * 4u, hipMemcpyHostToDevice), "mul b");
    const hipError_t err = launch_mul_f32_f32_bf16(static_cast<const float*>(dA.p), stride,
                                                   static_cast<const float*>(dB.p), stride,
                                                   static_cast<bf16_t*>(dOut.p), stride, rows, feat, 0);
    expect_ok(err, "mul f32*f32->bf16 launch");
    if (err != hipSuccess) return;
    std::vector<bf16_t> out(n, bf16_t{0});
    expect_ok(hipMemcpy(out.data(), dOut.p, n * 2u, hipMemcpyDeviceToHost), "mul out");
    double mx = 0.0, mr = 0.0;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < feat; ++c) {
            const size_t i = static_cast<size_t>(r) * stride + c;
            const double ref = static_cast<double>(a[i]) * static_cast<double>(b[i]);
            const double got = static_cast<double>(bf16_to_f32(out[i]));
            mx = std::max(mx, std::abs(got - ref));
            mr = std::max(mr, rel_err(got, ref));
        }
    report(mx, kTolBf16, "mul_f32_f32_bf16 (a*b) -> bf16", mr);
}

void run_residual_add() {
    const uint32_t rows = 6, feat = 96, stride = 96;
    const size_t n = static_cast<size_t>(rows) * stride;
    std::vector<bf16_t> a(n), b(n);
    for (auto& v : a) v = f32_to_bf16(frand(-3.0f, 3.0f));  // |a+b| < 6: one bf16 ulp < 3e-2
    for (auto& v : b) v = f32_to_bf16(frand(-3.0f, 3.0f));
    Buf dA(n * 2u), dB(n * 2u), dOut(n * 2u);
    dA.fill(0);
    dB.fill(0);
    dOut.fill(0);
    expect_ok(hipMemcpy(dA.p, a.data(), n * 2u, hipMemcpyHostToDevice), "add a");
    expect_ok(hipMemcpy(dB.p, b.data(), n * 2u, hipMemcpyHostToDevice), "add b");
    const hipError_t err = launch_residual_add_bf16(static_cast<const bf16_t*>(dA.p), stride,
                                                    static_cast<const bf16_t*>(dB.p), stride,
                                                    static_cast<bf16_t*>(dOut.p), stride, rows, feat, 0);
    expect_ok(err, "add bf16 launch");
    if (err != hipSuccess) return;
    std::vector<bf16_t> out(n, bf16_t{0});
    expect_ok(hipMemcpy(out.data(), dOut.p, n * 2u, hipMemcpyDeviceToHost), "add out");
    double mx = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double ref = static_cast<double>(bf16_to_f32(a[i])) + static_cast<double>(bf16_to_f32(b[i]));
        const double got = static_cast<double>(bf16_to_f32(out[i]));
        mx = std::max(mx, std::abs(got - ref));
        mr = std::max(mr, rel_err(got, ref));
    }
    report(mx, kTolBf16, "residual_add_bf16 (a+b)", mr);
    (void)bench_ms([&] {
        return launch_residual_add_bf16(static_cast<const bf16_t*>(dA.p), stride,
                                        static_cast<const bf16_t*>(dB.p), stride,
                                        static_cast<bf16_t*>(dOut.p), stride, rows, feat, 0);
    }, 100);
}

void run_sigmoid_bf16_f32() {
    const uint32_t rows = 3, feat = 333, stride = 336;  // feat not a multiple of 4
    const size_t n = static_cast<size_t>(rows) * stride;
    std::vector<bf16_t> in(n);
    for (auto& v : in) v = f32_to_bf16(frand(-20.0f, 20.0f));
    Buf dIn(n * 2u), dOut(n * 4u);
    dIn.fill(0);
    dOut.fill(0);
    expect_ok(hipMemcpy(dIn.p, in.data(), n * 2u, hipMemcpyHostToDevice), "sig in");
    const hipError_t err = launch_sigmoid_bf16_f32(static_cast<const bf16_t*>(dIn.p), stride,
                                                   static_cast<float*>(dOut.p), stride, rows, feat, 0);
    expect_ok(err, "sigmoid launch");
    if (err != hipSuccess) return;
    std::vector<float> out(n, 0.0f);
    expect_ok(hipMemcpy(out.data(), dOut.p, n * 4u, hipMemcpyDeviceToHost), "sig out");
    double mx = 0.0;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < feat; ++c) {
            const size_t i = static_cast<size_t>(r) * stride + c;
            const double x = bf16_to_f32(in[i]);
            const double ref = 1.0 / (1.0 + std::exp(-x));
            mx = std::max(mx, std::abs(static_cast<double>(out[i]) - ref));
        }
    report(mx, kTolF32, "sigmoid_bf16_f32 (1/(1+e^-x))");
}

void run_swiglu_bf16() {
    const uint32_t rows = 4, feat = 256, stride = 256;
    const size_t n = static_cast<size_t>(rows) * stride;
    std::vector<bf16_t> g(n), u(n);
    for (auto& v : g) v = f32_to_bf16(frand(-4.0f, 4.0f));
    for (auto& v : u) v = f32_to_bf16(frand(-1.5f, 1.5f));  // |out| < 6: bf16 ulp < 3e-2
    Buf dG(n * 2u), dU(n * 2u), dOut(n * 2u);
    dG.fill(0);
    dU.fill(0);
    dOut.fill(0);
    expect_ok(hipMemcpy(dG.p, g.data(), n * 2u, hipMemcpyHostToDevice), "swiglu g");
    expect_ok(hipMemcpy(dU.p, u.data(), n * 2u, hipMemcpyHostToDevice), "swiglu u");
    const hipError_t err = launch_swiglu_bf16(static_cast<const bf16_t*>(dG.p), stride,
                                              static_cast<const bf16_t*>(dU.p), stride,
                                              static_cast<bf16_t*>(dOut.p), stride, rows, feat, 0);
    expect_ok(err, "swiglu launch");
    if (err != hipSuccess) return;
    std::vector<bf16_t> out(n, bf16_t{0});
    expect_ok(hipMemcpy(out.data(), dOut.p, n * 2u, hipMemcpyDeviceToHost), "swiglu out");
    double mx = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double gv = bf16_to_f32(g[i]), uv = bf16_to_f32(u[i]);
        const double ref = uv * (gv / (1.0 + std::exp(-gv)));
        const double got = static_cast<double>(bf16_to_f32(out[i]));
        mx = std::max(mx, std::abs(got - ref));
        mr = std::max(mr, rel_err(got, ref));
    }
    report(mx, kTolBf16, "swiglu_bf16 (up*gate*sigmoid(gate))", mr);
}

void run_scale_f32() {
    const uint32_t rows = 2, feat = 70, stride = 71;
    const size_t n = static_cast<size_t>(rows) * stride;
    std::vector<float> in(n);
    for (auto& v : in) v = frand(-3.0f, 3.0f);
    const float scale = 0.375f;
    Buf dIn(n * 4u), dOut(n * 4u);
    dIn.fill(0);
    dOut.fill(0);
    expect_ok(hipMemcpy(dIn.p, in.data(), n * 4u, hipMemcpyHostToDevice), "scale in");
    const hipError_t err = launch_scale_f32(static_cast<const float*>(dIn.p), stride,
                                            static_cast<float*>(dOut.p), stride, rows, feat, scale, 0);
    expect_ok(err, "scale launch");
    if (err != hipSuccess) return;
    std::vector<float> out(n, 0.0f);
    expect_ok(hipMemcpy(out.data(), dOut.p, n * 4u, hipMemcpyDeviceToHost), "scale out");
    double mx = 0.0;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < feat; ++c) {
            const size_t i = static_cast<size_t>(r) * stride + c;
            mx = std::max(mx, std::abs(static_cast<double>(out[i]) - static_cast<double>(in[i]) * scale));
        }
    report(mx, kTolF32, "scale_f32 (x*scalar)");
}

void run_split_concat() {
    // split: halves layout, features = 64 (o0 = cols [0,64), o1 = cols [64,128))
    const uint32_t rows = 3, feat = 64, in_stride = 128;
    const size_t in_n = static_cast<size_t>(rows) * in_stride;
    std::vector<bf16_t> in(in_n);
    for (size_t i = 0; i < in_n; ++i) in[i] = f32_to_bf16(frand(-4.0f, 4.0f));
    Buf dIn(in_n * 2u), dO0(static_cast<size_t>(rows) * feat * 2u), dO1(static_cast<size_t>(rows) * feat * 2u);
    dIn.fill(0);
    dO0.fill(0);
    dO1.fill(0);
    expect_ok(hipMemcpy(dIn.p, in.data(), in_n * 2u, hipMemcpyHostToDevice), "split in");
    const hipError_t err = launch_split_bf16(static_cast<const bf16_t*>(dIn.p), in_stride,
                                             static_cast<bf16_t*>(dO0.p), feat,
                                             static_cast<bf16_t*>(dO1.p), feat, rows, feat,
                                             ElementwiseSplitLayout::Halves, 0u, 0);
    expect_ok(err, "split launch");
    std::vector<bf16_t> o0(static_cast<size_t>(rows) * feat, bf16_t{0}), o1(o0);
    expect_ok(hipMemcpy(o0.data(), dO0.p, o0.size() * 2u, hipMemcpyDeviceToHost), "split o0");
    expect_ok(hipMemcpy(o1.data(), dO1.p, o1.size() * 2u, hipMemcpyDeviceToHost), "split o1");
    double mx = 0.0;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < feat; ++c) {
            mx = std::max(mx, std::abs(static_cast<double>(o0[static_cast<size_t>(r) * feat + c].data) -
                                       static_cast<double>(in[static_cast<size_t>(r) * in_stride + c].data)));
            mx = std::max(mx, std::abs(static_cast<double>(o1[static_cast<size_t>(r) * feat + c].data) -
                                       static_cast<double>(in[static_cast<size_t>(r) * in_stride + feat + c].data)));
        }
    report(mx, 0.0, "split_bf16 (halves, exact bits)");

    // concat back: [o0 | o1] == row of `in` (first 128 cols)
    Buf dCat(static_cast<size_t>(rows) * in_stride * 2u);
    dCat.fill(0);
    const hipError_t cerr = launch_concat_bf16(static_cast<const bf16_t*>(dO0.p), feat,
                                               static_cast<const bf16_t*>(dO1.p), feat,
                                               static_cast<bf16_t*>(dCat.p), in_stride, rows, feat,
                                               feat, 0);
    expect_ok(cerr, "concat launch");
    if (cerr != hipSuccess) return;
    std::vector<bf16_t> cat(static_cast<size_t>(rows) * in_stride, bf16_t{0});
    expect_ok(hipMemcpy(cat.data(), dCat.p, cat.size() * 2u, hipMemcpyDeviceToHost), "concat out");
    double mx2 = 0.0;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < in_stride; ++c)
            mx2 = std::max(mx2, std::abs(static_cast<double>(cat[static_cast<size_t>(r) * in_stride + c].data) -
                                         static_cast<double>(in[static_cast<size_t>(r) * in_stride + c].data)));
    report(mx2, 0.0, "concat_bf16 (round-trip, exact bits)");
}

// ----------------------------------------------------------- l2_normalize
void run_l2_normalize() {
    struct Cfg {
        uint32_t rows, feat, group, stride;
        const char* name;
    };
    const Cfg cfgs[] = {
        {4, 64, 64, 64, "l2_normalize_bf16_f32 g=64 aligned (+zero row)"},
        // stride > features and not a multiple of 4 -> exercises the
        // unaligned/scalar path of the warp head body
        {3, 300, 100, 301, "l2_normalize_bf16_f32 g=100 strided/unaligned"},
    };
    const float eps = 1e-6f;
    for (const auto& c : cfgs) {
        const size_t n = static_cast<size_t>(c.rows) * c.stride;
        std::vector<bf16_t> in(n);
        for (auto& v : in) v = f32_to_bf16(frand(-2.0f, 2.0f));
        // last row of the first config is all-zero -> pins the eps semantics
        if (c.group == 64)
            for (uint32_t f = 0; f < c.feat; ++f)
                in[static_cast<size_t>(c.rows - 1) * c.stride + f] = bf16_t{0};
        Buf dIn(n * 2u), dOut(n * 4u);
        dIn.fill(0);
        dOut.fill(0);
        expect_ok(hipMemcpy(dIn.p, in.data(), n * 2u, hipMemcpyHostToDevice), "l2 in");
        const hipError_t err = launch_l2_normalize_bf16_f32(
            static_cast<const bf16_t*>(dIn.p), c.stride, static_cast<float*>(dOut.p), c.stride,
            c.rows, c.feat, c.group, eps, 0);
        expect_ok(err, c.name);
        if (err != hipSuccess) continue;
        std::vector<float> out(n, 0.0f);
        expect_ok(hipMemcpy(out.data(), dOut.p, n * 4u, hipMemcpyDeviceToHost), "l2 out");
        double mx = 0.0;
        for (uint32_t r = 0; r < c.rows; ++r) {
            for (uint32_t gi = 0; gi < c.feat / c.group; ++gi) {
                const uint32_t g0 = gi * c.group;
                double ss = 0.0;
                for (uint32_t h = 0; h < c.group; ++h) {
                    const double x = bf16_to_f32(in[static_cast<size_t>(r) * c.stride + g0 + h]);
                    ss += x * x;
                }
                const double inv = 1.0 / std::sqrt(ss + static_cast<double>(eps));
                for (uint32_t h = 0; h < c.group; ++h) {
                    const size_t i = static_cast<size_t>(r) * c.stride + g0 + h;
                    const double ref = static_cast<double>(bf16_to_f32(in[i])) * inv;
                    mx = std::max(mx, std::abs(static_cast<double>(out[i]) - ref));
                }
            }
        }
        report(mx, kTolF32, c.name);
    }
}

// --------------------------------------------------------------- embedding
void run_embedding() {
    const uint32_t vocab_local = 6;      // rows stored in this (sharded) table
    const uint32_t vocab_offset = 16;    // global ids [16, 22)
    const uint32_t hidden = 13;          // exercises the non-vector path
    const uint32_t out_stride = 16;
    const uint32_t rows = 6;
    const std::vector<int32_t> ids = {16, 21, 18, 99, -1, 22};

    std::vector<bf16_t> table(static_cast<size_t>(vocab_local) * hidden);
    for (size_t i = 0; i < table.size(); ++i) table[i] = f32_to_bf16(frand(-4.0f, 4.0f));

    Buf dTable(table.size() * 2u), dIds(ids.size() * 4u), dOut(static_cast<size_t>(rows) * out_stride * 2u),
        dErr(4);
    expect_ok(hipMemcpy(dTable.p, table.data(), table.size() * 2u, hipMemcpyHostToDevice), "emb table");
    expect_ok(hipMemcpy(dIds.p, ids.data(), ids.size() * 4u, hipMemcpyHostToDevice), "emb ids");

    // ---- partial_vocab = 0: out-of-shard tokens flag INVALID_BINDING, leave bytes alone
    dOut.fill(0xAA);
    uint32_t zero = 0u;
    expect_ok(hipMemcpy(dErr.p, &zero, 4, hipMemcpyHostToDevice), "emb err init");
    hipError_t err = launch_embedding_bf16(static_cast<const bf16_t*>(dTable.p),
                                           static_cast<const int32_t*>(dIds.p),
                                           static_cast<bf16_t*>(dOut.p), rows, vocab_local, hidden,
                                           out_stride, static_cast<uint32_t*>(dErr.p), 0, vocab_offset,
                                           0u);
    expect_ok(err, "embedding launch (partial_vocab=0)");
    std::vector<bf16_t> out(static_cast<size_t>(rows) * out_stride, bf16_t{0});
    expect_ok(hipMemcpy(out.data(), dOut.p, out.size() * 2u, hipMemcpyDeviceToHost), "emb out");
    uint32_t err_word = 0u;
    expect_ok(hipMemcpy(&err_word, dErr.p, 4, hipMemcpyDeviceToHost), "emb err read");
    ++checks;
    if (err_word != static_cast<uint32_t>(ProgramStatus::INVALID_BINDING)) {
        ++failures;
        std::printf("FAIL embedding error_word=%u (expected %u)\n", err_word,
                    static_cast<uint32_t>(ProgramStatus::INVALID_BINDING));
    } else {
        std::printf("ok   embedding error_word=INVALID_BINDING\n");
    }
    double mx = 0.0;
    bool poison = true;
    for (uint32_t r = 0; r < rows; ++r) {
        const int32_t tok = ids[r];
        const bool valid = tok >= 0 && static_cast<uint32_t>(tok) >= vocab_offset &&
                           static_cast<uint32_t>(tok) < vocab_offset + vocab_local;
        for (uint32_t h = 0; h < hidden; ++h) {
            const size_t o = static_cast<size_t>(r) * out_stride + h;
            if (valid) {
                const double got = bf16_to_f32(out[o]);
                const double ref = bf16_to_f32(table[static_cast<size_t>(tok - vocab_offset) * hidden + h]);
                mx = std::max(mx, std::abs(got - ref));
            } else {
                const uint16_t bits = out[o].data;
                if (bits != 0xAAAAu) poison = false;
            }
        }
    }
    report(mx, 0.0, "embedding_bf16 row gather w/ vocab_offset (exact)");
    ++checks;
    if (!poison) {
        ++failures;
        std::printf("FAIL embedding: out-of-shard rows were written\n");
    }

    // ---- partial_vocab != 0: out-of-shard rows are zero-filled
    dOut.fill(0xAA);
    err = launch_embedding_bf16(static_cast<const bf16_t*>(dTable.p),
                                static_cast<const int32_t*>(dIds.p), static_cast<bf16_t*>(dOut.p), rows,
                                vocab_local, hidden, out_stride, static_cast<uint32_t*>(dErr.p), 0,
                                vocab_offset, 1u);
    expect_ok(err, "embedding launch (partial_vocab=1)");
    expect_ok(hipMemcpy(out.data(), dOut.p, out.size() * 2u, hipMemcpyDeviceToHost), "emb out2");
    double mx2 = 0.0;
    for (uint32_t r = 0; r < rows; ++r) {
        const int32_t tok = ids[r];
        const bool valid = tok >= 0 && static_cast<uint32_t>(tok) >= vocab_offset &&
                           static_cast<uint32_t>(tok) < vocab_offset + vocab_local;
        for (uint32_t h = 0; h < hidden; ++h) {
            const size_t o = static_cast<size_t>(r) * out_stride + h;
            const double ref = valid ? bf16_to_f32(table[static_cast<size_t>(tok - vocab_offset) * hidden + h])
                                     : 0.0;
            mx2 = std::max(mx2, std::abs(static_cast<double>(bf16_to_f32(out[o])) - ref));
        }
    }
    report(mx2, 0.0, "embedding_bf16 partial_vocab zero-fill (exact)");

    // timing sample on a wider hidden size
    const uint32_t hid2 = 2560, rows2 = 64;
    std::vector<bf16_t> big_tbl(static_cast<size_t>(vocab_local) * hid2);
    for (auto& v : big_tbl) v = f32_to_bf16(frand(-1.0f, 1.0f));
    Buf dBig(big_tbl.size() * 2u), dBigOut(static_cast<size_t>(rows2) * hid2 * 2u);
    expect_ok(hipMemcpy(dBig.p, big_tbl.data(), big_tbl.size() * 2u, hipMemcpyHostToDevice), "emb big");
    std::vector<int32_t> ids2(rows2, 17);
    Buf dIds2(ids2.size() * 4u);
    expect_ok(hipMemcpy(dIds2.p, ids2.data(), ids2.size() * 4u, hipMemcpyHostToDevice), "emb ids2");
    const double ms = bench_ms([&] {
        return launch_embedding_bf16(static_cast<const bf16_t*>(dBig.p),
                                     static_cast<const int32_t*>(dIds2.p),
                                     static_cast<bf16_t*>(dBigOut.p), rows2, vocab_local, hid2, hid2,
                                     nullptr, 0, vocab_offset, 1u);
    }, 100);
    std::printf("time embedding_bf16 rows=%u hidden=%u: %.2f us\n", rows2, hid2, ms * 1000.0);
}

// launch-latency / bandwidth sanity numbers on decode-sized shapes
void run_timings() {
    const uint32_t rows = 1, feat = 5120;
    const size_t n = static_cast<size_t>(rows) * feat;
    std::vector<bf16_t> hb(n);
    for (auto& v : hb) v = f32_to_bf16(frand(-1.0f, 1.0f));
    std::vector<float> hf(n, 0.5f);
    Buf dA(n * 2u), dAf(n * 4u), dB(n * 2u), dBf(n * 4u), dW(feat * 2u), dOut(n * 4u);
    expect_ok(hipMemcpy(dA.p, hb.data(), n * 2u, hipMemcpyHostToDevice), "t a");
    expect_ok(hipMemcpy(dAf.p, hf.data(), n * 4u, hipMemcpyHostToDevice), "t af");
    expect_ok(hipMemcpy(dB.p, hb.data(), n * 2u, hipMemcpyHostToDevice), "t b");
    expect_ok(hipMemcpy(dBf.p, hf.data(), n * 4u, hipMemcpyHostToDevice), "t bf");
    expect_ok(hipMemcpy(dW.p, hb.data(), feat * 2u, hipMemcpyHostToDevice), "t w");

    std::printf("time rmsnorm bf16 rows=%u feat=%u:            %.2f us\n", rows, feat,
                bench_ms([&] {
                    return launch_rmsnorm(dA.p, RmsNormDataType::BF16, dW.p, RmsNormDataType::BF16,
                                          RmsNormWeightLayout::PER_FEATURE, RmsNormWeightMode::ONE_PLUS,
                                          dOut.p, RmsNormDataType::BF16, rows, feat, feat, feat, feat,
                                          1e-6f, 0);
                }, 200) *
                    1000.0);
    std::printf("time silu_bf16_f32 feat=%u:                   %.2f us\n", feat,
                bench_ms([&] {
                    return launch_silu_bf16_f32(static_cast<const bf16_t*>(dA.p), feat,
                                                static_cast<float*>(dOut.p), feat, rows, feat, 0);
                }, 200) *
                    1000.0);
    std::printf("time sigmoid_bf16_f32 feat=%u:                %.2f us\n", feat,
                bench_ms([&] {
                    return launch_sigmoid_bf16_f32(static_cast<const bf16_t*>(dA.p), feat,
                                                   static_cast<float*>(dOut.p), feat, rows, feat, 0);
                }, 200) *
                    1000.0);
    std::printf("time mul_f32_f32_bf16 feat=%u:                %.2f us\n", feat,
                bench_ms([&] {
                    return launch_mul_f32_f32_bf16(static_cast<const float*>(dAf.p), feat,
                                                   static_cast<const float*>(dBf.p), feat,
                                                   static_cast<bf16_t*>(dOut.p), feat, rows, feat, 0);
                }, 200) *
                    1000.0);
    std::printf("time residual_add_bf16 feat=%u:               %.2f us\n", feat,
                bench_ms([&] {
                    return launch_residual_add_bf16(static_cast<const bf16_t*>(dA.p), feat,
                                                    static_cast<const bf16_t*>(dB.p), feat,
                                                    static_cast<bf16_t*>(dOut.p), feat, rows, feat, 0);
                }, 200) *
                    1000.0);
    std::printf("time l2_normalize_bf16_f32 feat=%u g=128:     %.2f us\n", feat,
                bench_ms([&] {
                    return launch_l2_normalize_bf16_f32(static_cast<const bf16_t*>(dA.p), feat,
                                                        static_cast<float*>(dOut.p), feat, rows, feat,
                                                        128u, 1e-6f, 0);
                }, 200) *
                    1000.0);
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: no HIP device\n");
        return 1;
    }

    // ---- rmsnorm: plain gamma (DIRECT) and zero-centered (1+w) (ONE_PLUS),
    //      per-feature / per-group / no weight, bf16 and f32 paths, eps=1e-6.
    const RmsCase cases[] = {
        {RmsNormDataType::BF16, RmsNormDataType::BF16, RmsNormWeightLayout::PER_FEATURE,
         RmsNormWeightMode::DIRECT, 5, 128, 128, 128, 128, 1e-6f, kTolBf16,
         "rmsnorm bf16->bf16 PER_FEATURE DIRECT"},
        {RmsNormDataType::BF16, RmsNormDataType::BF16, RmsNormWeightLayout::PER_FEATURE,
         RmsNormWeightMode::ONE_PLUS, 5, 128, 128, 128, 128, 1e-6f, kTolBf16,
         "rmsnorm bf16->bf16 PER_FEATURE ONE_PLUS (1+w)"},
        {RmsNormDataType::BF16, RmsNormDataType::BF16, RmsNormWeightLayout::PER_GROUP,
         RmsNormWeightMode::ONE_PLUS, 3, 256, 64, 256, 256, 1e-6f, kTolBf16,
         "rmsnorm bf16->bf16 PER_GROUP ONE_PLUS (1+w)"},
        {RmsNormDataType::BF16, RmsNormDataType::F32, RmsNormWeightLayout::PER_GROUP,
         RmsNormWeightMode::ONE_PLUS, 4, 192, 64, 192, 192, 1e-6f, kTolF32,
         "rmsnorm bf16->f32 PER_GROUP ONE_PLUS (1+w)"},
        {RmsNormDataType::BF16, RmsNormDataType::F32, RmsNormWeightLayout::PER_FEATURE,
         RmsNormWeightMode::DIRECT, 4, 64, 64, 64, 64, 1e-6f, kTolF32,
         "rmsnorm bf16->f32 PER_FEATURE DIRECT"},
        {RmsNormDataType::F32, RmsNormDataType::F32, RmsNormWeightLayout::PER_GROUP,
         RmsNormWeightMode::DIRECT, 3, 128, 128, 128, 128, 1e-6f, 3e-2,
         "rmsnorm f32->f32 PER_GROUP DIRECT (bf16 mid-round)"},
        {RmsNormDataType::F32, RmsNormDataType::F32, RmsNormWeightLayout::PER_FEATURE,
         RmsNormWeightMode::ONE_PLUS, 3, 96, 32, 96, 96, 1e-6f, 3e-2,
         "rmsnorm f32->f32 PER_FEATURE ONE_PLUS (1+w)"},
        {RmsNormDataType::BF16, RmsNormDataType::BF16, RmsNormWeightLayout::NONE,
         RmsNormWeightMode::ONE_PLUS, 2, 64, 64, 64, 64, 1e-6f, kTolBf16,
         "rmsnorm bf16->bf16 NONE (unweighted)"},
        {RmsNormDataType::BF16, RmsNormDataType::BF16, RmsNormWeightLayout::PER_FEATURE,
         RmsNormWeightMode::ONE_PLUS, 3, 12, 6, 13, 13, 1e-6f, kTolBf16,
         "rmsnorm bf16->bf16 odd group/strides"},
        {RmsNormDataType::BF16, RmsNormDataType::BF16, RmsNormWeightLayout::PER_GROUP,
         RmsNormWeightMode::DIRECT, 2, 512, 512, 512, 512, 1e-6f, kTolBf16,
         "rmsnorm bf16->bf16 PER_GROUP DIRECT g=512"},
    };
    for (const auto& c : cases) run_rmsnorm(c);

    run_silu_bf16_f32();
    run_silu_f32_bf16();
    run_mul_f32_f32_bf16();
    run_residual_add();
    run_sigmoid_bf16_f32();
    run_swiglu_bf16();
    run_scale_f32();
    run_split_concat();
    run_l2_normalize();
    run_embedding();
    run_timings();

    std::printf("test_hip_elementwise_kernels: failures=%d (checks=%d)\n", failures, checks);
    return failures == 0 ? 0 : 1;
}
