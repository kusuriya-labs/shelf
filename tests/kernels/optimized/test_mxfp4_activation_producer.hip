#include <shelf/kernels/fused/mxfp4_activation_producer.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/optimized/rmsnorm.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/program/int8_activation_workspace.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
        std::printf("PASS: %s\n", name);
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

uint32_t hash32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}

float value(uint64_t seed, uint32_t i) {
    const int32_t n = static_cast<int32_t>(hash32(seed * 131u + i) % 4001u) - 2000;
    return static_cast<float>(n) / 500.0f;
}

shelf::bf16_t to_bf16(float x) {
    return shelf::bf16_t{shelf::f32_to_bf16_rne(x)};
}

template <typename T>
bool copy_to(void* dst, const std::vector<T>& src) {
    return hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice) ==
           hipSuccess;
}

template <typename T>
bool copy_from(std::vector<T>& dst, const void* src) {
    return hipMemcpy(dst.data(), src, dst.size() * sizeof(T), hipMemcpyDeviceToHost) ==
           hipSuccess;
}

struct Outputs {
    uint8_t* codes = nullptr;
    float* scales = nullptr;
};

bool equal_plane(const Outputs& ref, const Outputs& fused, uint32_t rows,
                 uint32_t rows_padded, uint32_t code_stride, uint32_t scale_stride) {
    std::vector<uint8_t> a(rows_padded * code_stride), b(rows_padded * code_stride);
    std::vector<float> c(rows * scale_stride / 4u), d(rows * scale_stride / 4u);
    if (!copy_from(a, ref.codes) || !copy_from(b, fused.codes) || !copy_from(c, ref.scales) ||
        !copy_from(d, fused.scales))
        return false;
    return std::memcmp(a.data(), b.data(), a.size()) == 0 &&
           std::memcmp(c.data(), d.data(), c.size() * sizeof(float)) == 0;
}

bool run_case(uint32_t rows, uint32_t k, uint32_t k_padded,
              const std::function<hipError_t(Outputs)>& reference,
              const std::function<hipError_t(Outputs)>& fused) {
    const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(k, rows);
    const uint32_t code_stride = layout.code_row_stride_bytes;
    const uint32_t scale_stride = layout.scale_row_stride_bytes;
    const uint32_t rows_padded = (rows + 15u) & ~15u;
    Outputs a{};
    Outputs b{};
    auto cleanup = [&]() {
        if (a.codes != nullptr) shelf::gpu::discard_cleanup_result(hipFree(a.codes));
        if (a.scales != nullptr) shelf::gpu::discard_cleanup_result(hipFree(a.scales));
        if (b.codes != nullptr) shelf::gpu::discard_cleanup_result(hipFree(b.codes));
        if (b.scales != nullptr) shelf::gpu::discard_cleanup_result(hipFree(b.scales));
    };
    if (hipMalloc(&a.codes, rows_padded * code_stride) != hipSuccess ||
        hipMalloc(&a.scales, rows * scale_stride) != hipSuccess ||
        hipMalloc(&b.codes, rows_padded * code_stride) != hipSuccess ||
        hipMalloc(&b.scales, rows * scale_stride) != hipSuccess) {
        cleanup();
        return false;
    }
    shelf::gpu::discard_cleanup_result(
        hipMemset(a.codes, 0xAB, rows_padded * code_stride));
    shelf::gpu::discard_cleanup_result(
        hipMemset(b.codes, 0xAB, rows_padded * code_stride));
    shelf::gpu::discard_cleanup_result(hipMemset(a.scales, 0xAB, rows * scale_stride));
    shelf::gpu::discard_cleanup_result(hipMemset(b.scales, 0xAB, rows * scale_stride));
    const bool ok = reference(a) == hipSuccess && fused(b) == hipSuccess &&
                    hipDeviceSynchronize() == hipSuccess &&
                    equal_plane(a, b, rows, rows_padded, code_stride, scale_stride);
    cleanup();
    return ok;
}

bool run_residual(uint32_t rows, uint32_t features) {
    std::vector<shelf::bf16_t> ha(rows * features), hb(rows * features);
    std::vector<shelf::bf16_t> hw(features);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t i = 0; i < features; ++i) {
            ha[r * features + i] = to_bf16(value(11u + r, i));
            hb[r * features + i] = to_bf16(value(101u + r, i) * 0.5f);
        }
    for (uint32_t i = 0; i < features; ++i) hw[i] = to_bf16(value(7u, i) * 0.1f);
    void *da = nullptr, *db = nullptr, *dw = nullptr, *dres = nullptr, *dnorm = nullptr;
    auto cleanup = [&]() {
        void* all[] = {da, db, dw, dres, dnorm};
        for (void* p : all)
            if (p != nullptr) shelf::gpu::discard_cleanup_result(hipFree(p));
    };
    const size_t elem = sizeof(shelf::bf16_t);
    if (hipMalloc(&da, ha.size() * elem) != hipSuccess ||
        hipMalloc(&db, hb.size() * elem) != hipSuccess ||
        hipMalloc(&dw, hw.size() * elem) != hipSuccess ||
        hipMalloc(&dres, ha.size() * elem) != hipSuccess ||
        hipMalloc(&dnorm, ha.size() * elem) != hipSuccess || !copy_to(da, ha) ||
        !copy_to(db, hb) || !copy_to(dw, hw)) {
        cleanup();
        return false;
    }
    auto* a = static_cast<const shelf::bf16_t*>(da);
    auto* b = static_cast<const shelf::bf16_t*>(db);
    auto* w = static_cast<const shelf::bf16_t*>(dw);
    auto* res = static_cast<shelf::bf16_t*>(dres);
    auto* norm = static_cast<shelf::bf16_t*>(dnorm);
    const bool ok = run_case(
        rows, features, features,
        [&](Outputs o) {
            hipError_t err = shelf::kernel::launch_residual_add_bf16(
                a, features, b, features, res, features, rows, features, nullptr);
            if (err == hipSuccess)
                err = shelf::kernel::launch_rmsnorm(
                    res, shelf::kernel::RmsNormDataType::BF16, w,
                    shelf::kernel::RmsNormDataType::BF16,
                    shelf::kernel::RmsNormWeightLayout::PER_FEATURE,
                    shelf::kernel::RmsNormWeightMode::ONE_PLUS, norm,
                    shelf::kernel::RmsNormDataType::BF16, rows, features, features,
                    features, features, 1e-6f, nullptr);
            if (err == hipSuccess)
                err = shelf::kernel::launch_activation_quantize_mxfp4_rowref(
                    norm, o.codes, o.scales, rows, features, features, features,
                    static_cast<uint32_t>(shelf::runtime::Int8ActivationWorkspaceLayout::make(
                        features, rows).code_row_stride_bytes),
                    static_cast<uint32_t>(shelf::runtime::Int8ActivationWorkspaceLayout::make(
                        features, rows).scale_row_stride_bytes),
                    nullptr, nullptr);
            return err;
        },
        [&](Outputs o) {
            const auto layout =
                shelf::runtime::Int8ActivationWorkspaceLayout::make(features, rows);
            shelf::execution::Mxfp4ResidualNormArgs args{};
            args.a = a;
            args.b = b;
            args.residual = res;
            args.weight = w;
            args.codes = o.codes;
            args.scales = o.scales;
            args.rows = rows;
            args.k = features;
            args.k_padded = features;
            args.a_stride = features;
            args.b_stride = features;
            args.residual_stride = features;
            args.code_stride = layout.code_row_stride_bytes;
            args.scale_stride = layout.scale_row_stride_bytes;
            args.eps = 1e-6f;
            return shelf::execution::launch_mxfp4_residual_norm_quantize(args, nullptr);
        });
    cleanup();
    return ok;
}

bool run_norm(uint32_t rows, uint32_t features) {
    std::vector<shelf::bf16_t> hx(rows * features);
    std::vector<shelf::bf16_t> hw(features);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t i = 0; i < features; ++i)
            hx[r * features + i] = to_bf16(value(31u + r, i));
    for (uint32_t i = 0; i < features; ++i) hw[i] = to_bf16(value(17u, i) * 0.2f);
    void *dx = nullptr, *dw = nullptr, *dnorm = nullptr;
    auto cleanup = [&]() {
        void* all[] = {dx, dw, dnorm};
        for (void* p : all)
            if (p != nullptr) shelf::gpu::discard_cleanup_result(hipFree(p));
    };
    const size_t elem = sizeof(shelf::bf16_t);
    if (hipMalloc(&dx, hx.size() * elem) != hipSuccess ||
        hipMalloc(&dw, hw.size() * elem) != hipSuccess ||
        hipMalloc(&dnorm, hx.size() * elem) != hipSuccess || !copy_to(dx, hx) ||
        !copy_to(dw, hw)) {
        cleanup();
        return false;
    }
    auto* x = static_cast<const shelf::bf16_t*>(dx);
    auto* w = static_cast<const shelf::bf16_t*>(dw);
    auto* norm = static_cast<shelf::bf16_t*>(dnorm);
    const bool ok = run_case(
        rows, features, features,
        [&](Outputs o) {
            hipError_t err = shelf::kernel::launch_rmsnorm(
                x, shelf::kernel::RmsNormDataType::BF16, w,
                shelf::kernel::RmsNormDataType::BF16,
                shelf::kernel::RmsNormWeightLayout::PER_FEATURE,
                shelf::kernel::RmsNormWeightMode::ONE_PLUS, norm,
                shelf::kernel::RmsNormDataType::BF16, rows, features, features, features,
                features, 1e-6f, nullptr);
            if (err == hipSuccess) {
                const auto layout =
                    shelf::runtime::Int8ActivationWorkspaceLayout::make(features, rows);
                err = shelf::kernel::launch_activation_quantize_mxfp4_rowref(
                    norm, o.codes, o.scales, rows, features, features, features,
                    layout.code_row_stride_bytes, layout.scale_row_stride_bytes, nullptr,
                    nullptr);
            }
            return err;
        },
        [&](Outputs o) {
            const auto layout =
                shelf::runtime::Int8ActivationWorkspaceLayout::make(features, rows);
            shelf::execution::Mxfp4NormArgs args{};
            args.input = x;
            args.weight = w;
            args.codes = o.codes;
            args.scales = o.scales;
            args.rows = rows;
            args.k = features;
            args.k_padded = features;
            args.input_stride = features;
            args.code_stride = layout.code_row_stride_bytes;
            args.scale_stride = layout.scale_row_stride_bytes;
            args.eps = 1e-6f;
            return shelf::execution::launch_mxfp4_norm_quantize(args, nullptr);
        });
    cleanup();
    return ok;
}

bool run_swiglu(uint32_t rows, uint32_t features) {
    std::vector<shelf::bf16_t> hg(rows * features), hu(rows * features);
    std::vector<shelf::bf16_t> hy(rows * features);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t i = 0; i < features; ++i) {
            hg[r * features + i] = to_bf16(value(41u + r, i));
            hu[r * features + i] = to_bf16(value(43u + r, i) * 0.7f);
        }
    void *dg = nullptr, *du = nullptr, *dy = nullptr;
    auto cleanup = [&]() {
        void* all[] = {dg, du, dy};
        for (void* p : all)
            if (p != nullptr) shelf::gpu::discard_cleanup_result(hipFree(p));
    };
    const size_t elem = sizeof(shelf::bf16_t);
    if (hipMalloc(&dg, hg.size() * elem) != hipSuccess ||
        hipMalloc(&du, hu.size() * elem) != hipSuccess ||
        hipMalloc(&dy, hy.size() * elem) != hipSuccess || !copy_to(dg, hg) ||
        !copy_to(du, hu)) {
        cleanup();
        return false;
    }
    auto* gate = static_cast<const shelf::bf16_t*>(dg);
    auto* up = static_cast<const shelf::bf16_t*>(du);
    auto* y = static_cast<shelf::bf16_t*>(dy);
    const bool ok = run_case(
        rows, features, features,
        [&](Outputs o) {
            hipError_t err = shelf::kernel::launch_swiglu_bf16(
                gate, features, up, features, y, features, rows, features, nullptr);
            if (err == hipSuccess) {
                const auto layout =
                    shelf::runtime::Int8ActivationWorkspaceLayout::make(features, rows);
                err = shelf::kernel::launch_activation_quantize_mxfp4_rowref(
                    y, o.codes, o.scales, rows, features, features, features,
                    layout.code_row_stride_bytes, layout.scale_row_stride_bytes, nullptr,
                    nullptr);
            }
            return err;
        },
        [&](Outputs o) {
            const auto layout =
                shelf::runtime::Int8ActivationWorkspaceLayout::make(features, rows);
            shelf::execution::Mxfp4SwigluArgs args{};
            args.gate = gate;
            args.up = up;
            args.codes = o.codes;
            args.scales = o.scales;
            args.rows = rows;
            args.k = features;
            args.k_padded = features;
            args.gate_stride = features;
            args.up_stride = features;
            args.code_stride = layout.code_row_stride_bytes;
            args.scale_stride = layout.scale_row_stride_bytes;
            args.bf16_intermediate = false;
            return shelf::execution::launch_mxfp4_swiglu_quantize(args, nullptr);
        });
    cleanup();
    return ok;
}

bool run_gdn_gated(uint32_t rows) {
    constexpr uint32_t kFeatures = 6144;
    constexpr uint32_t kGroup = 128;
    constexpr float kEps = 1e-6f;
    std::vector<float> hr(rows * kFeatures);
    std::vector<shelf::bf16_t> hz(rows * kFeatures), hw(kGroup);
    std::vector<float> hsilu(rows * kFeatures);
    std::vector<shelf::bf16_t> hmul(rows * kFeatures);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t i = 0; i < kFeatures; ++i) {
            hr[r * kFeatures + i] = value(61u + r, i);
            hz[r * kFeatures + i] = to_bf16(value(67u + r, i));
        }
    for (uint32_t i = 0; i < kGroup; ++i) hw[i] = to_bf16(value(71u, i) * 0.05f);
    void *dr = nullptr, *dz = nullptr, *dw = nullptr, *dnorm = nullptr, *dsilu = nullptr,
         *dmul = nullptr;
    auto cleanup = [&]() {
        void* all[] = {dr, dz, dw, dnorm, dsilu, dmul};
        for (void* p : all)
            if (p != nullptr) shelf::gpu::discard_cleanup_result(hipFree(p));
    };
    if (hipMalloc(&dr, hr.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&dz, hz.size() * sizeof(shelf::bf16_t)) != hipSuccess ||
        hipMalloc(&dw, hw.size() * sizeof(shelf::bf16_t)) != hipSuccess ||
        hipMalloc(&dnorm, hr.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&dsilu, hsilu.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&dmul, hmul.size() * sizeof(shelf::bf16_t)) != hipSuccess ||
        hipMemcpy(dr, hr.data(), hr.size() * sizeof(float), hipMemcpyHostToDevice) !=
            hipSuccess ||
        !copy_to(dz, hz) || !copy_to(dw, hw)) {
        cleanup();
        return false;
    }
    auto* rec = static_cast<const float*>(dr);
    auto* z = static_cast<const shelf::bf16_t*>(dz);
    auto* w = static_cast<const shelf::bf16_t*>(dw);
    auto* norm = static_cast<float*>(dnorm);
    auto* silu = static_cast<float*>(dsilu);
    auto* mul = static_cast<shelf::bf16_t*>(dmul);
    const bool ok = run_case(
        rows, kFeatures, kFeatures,
        [&](Outputs o) {
            hipError_t err = shelf::kernel::launch_rmsnorm(
                rec, shelf::kernel::RmsNormDataType::F32, w,
                shelf::kernel::RmsNormDataType::BF16,
                shelf::kernel::RmsNormWeightLayout::PER_GROUP,
                shelf::kernel::RmsNormWeightMode::DIRECT, norm,
                shelf::kernel::RmsNormDataType::F32, rows, kFeatures, kGroup, kFeatures,
                kFeatures, kEps, nullptr);
            if (err == hipSuccess)
                err = shelf::kernel::launch_silu_bf16_f32(
                    z, kFeatures, silu, kFeatures, rows, kFeatures, nullptr);
            if (err == hipSuccess)
                err = shelf::kernel::launch_mul_f32_f32_bf16(
                    norm, kFeatures, silu, kFeatures, mul, kFeatures, rows, kFeatures,
                    nullptr);
            if (err == hipSuccess) {
                const auto layout =
                    shelf::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, rows);
                err = shelf::kernel::launch_activation_quantize_mxfp4_rowref(
                    mul, o.codes, o.scales, rows, kFeatures, kFeatures, kFeatures,
                    layout.code_row_stride_bytes, layout.scale_row_stride_bytes, nullptr,
                    nullptr);
            }
            return err;
        },
        [&](Outputs o) {
            const auto layout =
                shelf::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, rows);
            shelf::execution::Mxfp4GdnGatedArgs args{};
            args.rec = rec;
            args.z = z;
            args.weight = w;
            args.codes = o.codes;
            args.scales = o.scales;
            args.rows = rows;
            args.features = kFeatures;
            args.group_size = kGroup;
            args.rec_stride = kFeatures;
            args.z_stride = kFeatures;
            args.code_stride = layout.code_row_stride_bytes;
            args.scale_stride = layout.scale_row_stride_bytes;
            args.eps = kEps;
            return shelf::execution::launch_mxfp4_gdn_gated_quantize(args, nullptr);
        });
    cleanup();
    return ok;
}

}  // namespace

int main() {
    for (uint32_t rows : {1u, 5u, 16u, 128u}) {
        char name[96];
        std::snprintf(name, sizeof(name), "residual_norm rows=%u", rows);
        check(run_residual(rows, 5120u), name);
        std::snprintf(name, sizeof(name), "norm rows=%u", rows);
        check(run_norm(rows, 5120u), name);
        std::snprintf(name, sizeof(name), "swiglu rows=%u", rows);
        check(run_swiglu(rows, 17408u), name);
        std::snprintf(name, sizeof(name), "gdn_gated rows=%u", rows);
        check(run_gdn_gated(rows), name);
    }
    std::printf("%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
