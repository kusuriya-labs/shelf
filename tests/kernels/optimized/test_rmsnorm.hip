#include <shelf/kernels/optimized/rmsnorm.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        std::printf("PASS: %s\n", name);
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

uint32_t h32(uint64_t x) {
    uint32_t h = static_cast<uint32_t>(x * 2654435761u + 12345u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

float f32_from_bf16(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

float bf16_rne_f32(float x) {
    return f32_from_bf16(shelf::f32_to_bf16_rne(x));
}

struct Config {
    bool in_f32;
    bool out_f32;
    shelf::kernel::RmsNormWeightLayout layout;
    shelf::kernel::RmsNormWeightMode mode;
    uint32_t weight_elements;
};

void gen_values(uint32_t rows, uint32_t F, uint32_t in_stride, uint32_t in_off_seed,
                std::vector<double>& in_vals) {
    in_vals.assign(static_cast<size_t>(rows) * in_stride, 0.0);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t c = 0; c < F; ++c) {
            double v = (static_cast<double>(h32(static_cast<uint64_t>(m) * 7919u + c * 31u + in_off_seed) % 2001u) - 1000.0) / 1000.0;
            in_vals[static_cast<size_t>(m) * in_stride + c] = v;
        }
    }
}

void cpu_ref(const Config& cfg, const std::vector<double>& in_vals,
             const std::vector<double>& wt_vals,
             uint32_t rows, uint32_t F, uint32_t G,
             uint32_t in_stride, uint32_t out_stride, double eps,
             std::vector<double>& out_vals) {
    out_vals.assign(static_cast<size_t>(rows) * out_stride, 0.0);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t g0 = 0; g0 < F; g0 += G) {
            double sum = 0.0;
            for (uint32_t h = 0; h < G; ++h) {
                const double x = cfg.in_f32
                    ? static_cast<double>(static_cast<float>(in_vals[static_cast<size_t>(m) * in_stride + g0 + h]))
                    : static_cast<double>(f32_from_bf16(shelf::f32_to_bf16_rne(
                        static_cast<float>(in_vals[static_cast<size_t>(m) * in_stride + g0 + h]))));
                sum += x * x;
            }
            const double inv = 1.0 / std::sqrt(sum / static_cast<double>(G) + eps);
            for (uint32_t h = 0; h < G; ++h) {
                double x;
                if (cfg.in_f32) {
                    x = static_cast<double>(static_cast<float>(in_vals[static_cast<size_t>(m) * in_stride + g0 + h]));
                } else {
                    x = static_cast<double>(f32_from_bf16(shelf::f32_to_bf16_rne(
                        static_cast<float>(in_vals[static_cast<size_t>(m) * in_stride + g0 + h]))));
                }
                double y = x * inv;
                if (cfg.in_f32) y = bf16_rne_f32(static_cast<float>(y));
                if (cfg.layout != shelf::kernel::RmsNormWeightLayout::NONE) {
                    const uint32_t widx =
                        (cfg.layout == shelf::kernel::RmsNormWeightLayout::PER_FEATURE) ? (g0 + h) : h;
                    const double w = f32_from_bf16(
                        shelf::f32_to_bf16_rne(static_cast<float>(wt_vals[widx])));
                    if (cfg.mode == shelf::kernel::RmsNormWeightMode::ONE_PLUS) y *= 1.0 + w;
                    else y *= w;
                }
                out_vals[static_cast<size_t>(m) * out_stride + g0 + h] = y;
            }
        }
    }
}

struct UploadResult {
    void* in_dev;
    void* wt_dev;
    void* out_dev;
};

bool upload(shelf::gpu::GpuArena& arena, const Config& cfg,
            const std::vector<double>& in_vals, uint32_t in_stride,
            const std::vector<double>& wt_vals,
            uint32_t rows, uint32_t out_stride, UploadResult& res,
            bool sentinel) {
    const uint32_t in_elem = cfg.in_f32 ? 4u : 2u;
    const uint32_t out_elem = cfg.out_f32 ? 4u : 2u;
    const uint32_t wt_elem = 2u;
    const size_t in_bytes = static_cast<size_t>(rows) * in_stride * in_elem;
    const size_t out_bytes = static_cast<size_t>(rows) * out_stride * out_elem;
    const size_t wt_bytes = wt_vals.empty() ? 256u : static_cast<size_t>(wt_vals.size()) * wt_elem;
    auto ir = arena.allocate_aligned(in_bytes, 256);
    auto orr = arena.allocate_aligned(out_bytes, 256);
    auto wtr = arena.allocate_aligned(wt_bytes, 256);
    if (!ir.ok() || !orr.ok() || !wtr.ok()) return false;
    res.in_dev = ir.release().data();
    res.out_dev = orr.release().data();
    res.wt_dev = wtr.release().data();
    if (sentinel) {
        if (cfg.out_f32) {
            std::vector<float> snt(static_cast<size_t>(rows) * out_stride, 9999.0f);
            if (hipMemcpy(res.out_dev, snt.data(), out_bytes, hipMemcpyHostToDevice) != hipSuccess) return false;
        } else {
            std::vector<uint16_t> snt(static_cast<size_t>(rows) * out_stride, shelf::f32_to_bf16_rne(9999.0f));
            if (hipMemcpy(res.out_dev, snt.data(), out_bytes, hipMemcpyHostToDevice) != hipSuccess) return false;
        }
    }
    if (cfg.in_f32) {
        std::vector<float> tmp(in_vals.size());
        for (size_t i = 0; i < in_vals.size(); ++i) tmp[i] = static_cast<float>(in_vals[i]);
        if (hipMemcpy(res.in_dev, tmp.data(), in_bytes, hipMemcpyHostToDevice) != hipSuccess) return false;
    } else {
        std::vector<uint16_t> tmp(in_vals.size());
        for (size_t i = 0; i < in_vals.size(); ++i) tmp[i] = shelf::f32_to_bf16_rne(static_cast<float>(in_vals[i]));
        if (hipMemcpy(res.in_dev, tmp.data(), in_bytes, hipMemcpyHostToDevice) != hipSuccess) return false;
    }
    if (!wt_vals.empty()) {
        std::vector<uint16_t> tmp(wt_vals.size());
        for (size_t i = 0; i < wt_vals.size(); ++i) tmp[i] = shelf::f32_to_bf16_rne(static_cast<float>(wt_vals[i]));
        if (hipMemcpy(res.wt_dev, tmp.data(), wt_bytes, hipMemcpyHostToDevice) != hipSuccess) return false;
    }
    return true;
}

bool run_case(const Config& cfg, uint32_t rows, uint32_t F, uint32_t G,
              uint32_t in_stride, uint32_t out_stride, double eps,
              uint32_t seed, shelf::gpu::GpuArena& arena, hipStream_t stream,
              const char* label, bool sentinel) {
    if (F % G != 0u) return true;
    std::vector<double> in_vals;
    gen_values(rows, F, in_stride, seed, in_vals);
    if (sentinel) {
        for (uint32_t m = 0; m < rows; ++m) {
            for (uint32_t c = F; c < in_stride; ++c) {
                in_vals[static_cast<size_t>(m) * in_stride + c] = 9999.0;
            }
        }
    }
    std::vector<double> wt_vals;
    if (cfg.layout != shelf::kernel::RmsNormWeightLayout::NONE) {
        wt_vals.assign(cfg.weight_elements, 0.0);
        for (uint32_t i = 0; i < cfg.weight_elements; ++i) {
            wt_vals[i] = (static_cast<double>(h32(static_cast<uint64_t>(i) * 6151u + seed + 1u) % 1001u) - 500.0) / 1000.0;
        }
    }

    std::vector<double> out_vals;
    cpu_ref(cfg, in_vals, wt_vals, rows, F, G, in_stride, out_stride, eps, out_vals);

    UploadResult up;
    if (!upload(arena, cfg, in_vals, in_stride, wt_vals, rows, out_stride, up, sentinel)) {
        std::printf("FAIL: %s alloc\n", label);
        return false;
    }

    const shelf::kernel::RmsNormDataType in_t = cfg.in_f32 ? shelf::kernel::RmsNormDataType::F32 : shelf::kernel::RmsNormDataType::BF16;
    const shelf::kernel::RmsNormDataType out_t = cfg.out_f32 ? shelf::kernel::RmsNormDataType::F32 : shelf::kernel::RmsNormDataType::BF16;
    const shelf::kernel::RmsNormDataType wt_t = shelf::kernel::RmsNormDataType::BF16;
    hipError_t h = shelf::kernel::launch_rmsnorm(
        up.in_dev, in_t,
        cfg.layout == shelf::kernel::RmsNormWeightLayout::NONE ? nullptr : up.wt_dev, wt_t,
        cfg.layout, cfg.mode,
        up.out_dev, out_t,
        rows, F, G, in_stride, out_stride, static_cast<float>(eps), stream);
    if (h != hipSuccess) {
        std::printf("FAIL: %s launch (%s)\n", label, hipGetErrorString(h));
        return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: %s sync\n", label);
        return false;
    }
    const uint32_t out_elem = cfg.out_f32 ? 4u : 2u;
    std::vector<double> got(static_cast<size_t>(rows) * out_stride, 0.0);
    if (cfg.out_f32) {
        std::vector<float> tmp(got.size());
        if (hipMemcpy(tmp.data(), up.out_dev, tmp.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
        for (size_t i = 0; i < tmp.size(); ++i) got[i] = tmp[i];
    } else {
        std::vector<uint16_t> tmp(got.size());
        if (hipMemcpy(tmp.data(), up.out_dev, tmp.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) return false;
        for (size_t i = 0; i < tmp.size(); ++i) got[i] = f32_from_bf16(tmp[i]);
    }

    bool ok = true;
    double max_abs = 0.0, max_rel = 0.0;
    size_t arg = 0;
    const double f32_abs_tol = 2e-4;
    const double f32_rel_tol = std::max(5e-4, 1e-6 * static_cast<double>(G));
    const double bf16_eps = 0.00390625;
    for (uint32_t m = 0; m < rows && ok; ++m) {
        for (uint32_t c = 0; c < out_stride; ++c) {
            const size_t gi = static_cast<size_t>(m) * out_stride + c;
            const double g = got[gi];
            const double r = out_vals[gi];
            if (std::isnan(static_cast<float>(g)) || std::isinf(static_cast<float>(g))) {
                ok = false; arg = gi; break;
            }
            if (c >= F) {
                if (sentinel && std::fabs(g) < 9000.0) { ok = false; arg = gi; break; }
                continue;
            }
            const double da = std::fabs(g - r);
            const double dr = da / std::max(1.0, std::fabs(r));
            if (da > max_abs) max_abs = da;
            if (dr > max_rel) max_rel = dr;
            if (cfg.out_f32 && !cfg.in_f32) {
                if (!(da <= f32_abs_tol || dr <= f32_rel_tol)) { ok = false; arg = gi; break; }
            } else {
                const double tol = 2.0 * bf16_eps * std::max(1.0, std::fabs(r));
                if (!(da <= tol)) { ok = false; arg = gi; break; }
            }
        }
    }
    if (!ok)
        std::printf("  mismatch at %zu: got=%.9g ref=%.9g max_abs=%.3g max_rel=%.3g\n",
                    arg, got[arg], out_vals[arg], max_abs, max_rel);
    if (ok)
        std::printf("  [%s] max_abs=%.3g max_rel=%.3g\n", label, max_abs, max_rel);
    check(ok, label);
    return ok;
}

}  // namespace

int main() {
    std::printf("=== RMSNorm (optimized) Tests ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto arena_result = shelf::gpu::GpuArena::create(0, 512 * 1024 * 1024);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    shelf::gpu::GpuArena arena = arena_result.release();

    using L = shelf::kernel::RmsNormWeightLayout;
    using M = shelf::kernel::RmsNormWeightMode;

    const Config bf16_bf16_pf_op { false, false, L::PER_FEATURE, M::ONE_PLUS, 0 };
    const Config bf16_bf16_pf_d  { false, false, L::PER_FEATURE, M::DIRECT, 0 };
    const Config bf16_bf16_pg_op { false, false, L::PER_GROUP, M::ONE_PLUS, 0 };
    const Config bf16_f32_pg_op  { false, true, L::PER_GROUP, M::ONE_PLUS, 0 };
    const Config bf16_f32_pf_op  { false, true, L::PER_FEATURE, M::ONE_PLUS, 0 };
    const Config f32_f32_pg_d    { true, true, L::PER_GROUP, M::DIRECT, 0 };
    const Config f32_f32_pf_d    { true, true, L::PER_FEATURE, M::DIRECT, 0 };
    const Config f32_f32_pg_op   { true, true, L::PER_GROUP, M::ONE_PLUS, 0 };
    const Config none_bf16_op    { false, false, L::NONE, M::ONE_PLUS, 0 };

    const uint32_t rows_list[] = {1, 2, 4, 16, 33};
    const uint32_t fg_list[][2] = {
        {1, 1}, {7, 1}, {31, 1}, {32, 32}, {33, 33},
        {32, 1}, {128, 16}, {256, 64}, {256, 256}, {257, 257},
        {2560, 2560}, {2560, 128}, {2560, 32}, {4096, 256}, {512, 32},
    };

    auto try_combo = [&](const Config& base, bool in_f32, bool out_f32, L layout, M mode,
                         uint32_t F, uint32_t G, uint32_t rows) {
        Config cfg = base;
        cfg.in_f32 = in_f32;
        cfg.out_f32 = out_f32;
        cfg.layout = layout;
        cfg.mode = mode;
        if (layout == L::PER_FEATURE) cfg.weight_elements = F;
        else if (layout == L::PER_GROUP) cfg.weight_elements = G;
        else cfg.weight_elements = 0;
        char label[200];
        std::snprintf(label, sizeof(label), "rows=%u F=%u G=%u %s->%s %s %s",
                      rows, F, G, in_f32 ? "F32" : "BF16", out_f32 ? "F32" : "BF16",
                      layout == L::PER_FEATURE ? "PF" : layout == L::PER_GROUP ? "PG" : "NONE",
                      mode == M::ONE_PLUS ? "ONE_PLUS" : "DIRECT");
        run_case(cfg, rows, F, G, F, F, 1e-6, rows * 131u + F, arena, stream, label, false);
    };

    for (uint32_t rows : rows_list) {
        for (auto& fg : fg_list) {
            const uint32_t F = fg[0], G = fg[1];
            if (static_cast<size_t>(rows) * F > 4000000u) continue;
            try_combo(bf16_bf16_pf_op, false, false, L::PER_FEATURE, M::ONE_PLUS, F, G, rows);
            try_combo(bf16_bf16_pf_d, false, false, L::PER_FEATURE, M::DIRECT, F, G, rows);
            try_combo(bf16_bf16_pg_op, false, false, L::PER_GROUP, M::ONE_PLUS, F, G, rows);
            try_combo(bf16_f32_pg_op, false, true, L::PER_GROUP, M::ONE_PLUS, F, G, rows);
            try_combo(bf16_f32_pf_op, false, true, L::PER_FEATURE, M::ONE_PLUS, F, G, rows);
            try_combo(f32_f32_pg_d, true, true, L::PER_GROUP, M::DIRECT, F, G, rows);
            try_combo(f32_f32_pf_d, true, true, L::PER_FEATURE, M::DIRECT, F, G, rows);
            try_combo(f32_f32_pg_op, true, true, L::PER_GROUP, M::ONE_PLUS, F, G, rows);
        }
    }

    (void)none_bf16_op;

    // 27B variants: GDN norm (F32->F32 PG DIRECT F=6144 G=128) and
    // q_norm (BF16->F32 PG ONE_PLUS F=6144 G=256).
    {
        const uint32_t rows27[] = {1, 128, 2048};
        for (uint32_t rows : rows27)
            try_combo(f32_f32_pg_d, true, true, L::PER_GROUP, M::DIRECT, 6144, 128, rows);
        for (uint32_t rows : rows27)
            try_combo(bf16_f32_pg_op, false, true, L::PER_GROUP, M::ONE_PLUS, 6144, 256, rows);
    }

    for (uint32_t rows : rows_list) {
        const uint32_t F = 256, G = 64;
        run_case({false, false, L::NONE, M::ONE_PLUS, 0}, rows, F, G, F, F, 1e-6, rows * 31u, arena, stream,
                 std::to_string(rows).c_str(), false);
    }

    const Config stride_cfg { false, false, L::PER_FEATURE, M::ONE_PLUS, 0 };
    Config scfg = stride_cfg;
    scfg.weight_elements = 256;
    {
        char label[200];
        std::snprintf(label, sizeof(label), "stride in=320 out=300 F=256 G=64 sentinel");
        run_case(scfg, 5, 256, 64, 320, 300, 1e-6, 999u, arena, stream, label, true);
    }
    {
        Config fc = stride_cfg;
        fc.in_f32 = true; fc.out_f32 = true; fc.layout = L::PER_GROUP; fc.mode = M::DIRECT; fc.weight_elements = 64;
        char label[200];
        std::snprintf(label, sizeof(label), "stride F32 in=272 out=288 F=256 G=64 sentinel");
        run_case(fc, 5, 256, 64, 272, 288, 1e-6, 1000u, arena, stream, label, true);
    }

    const Config align_cfg { false, false, L::PER_GROUP, M::ONE_PLUS, 0 };
    Config acfg = align_cfg;
    acfg.weight_elements = 11;
    {
        char label[200];
        std::snprintf(label, sizeof(label), "unaligned F=33 in_stride=35 G=11");
        run_case(acfg, 4, 33, 11, 35, 35, 1e-6, 555u, arena, stream, label, false);
    }
    {
        Config tc = align_cfg;
        tc.weight_elements = 7;
        char label[200];
        std::snprintf(label, sizeof(label), "tail F=21 G=7");
        run_case(tc, 3, 21, 7, 21, 21, 1e-6, 777u, arena, stream, label, false);
    }

    {
        Config rc { true, true, L::PER_GROUP, M::DIRECT, 0 };
        rc.weight_elements = 2;
        const uint32_t F = 2, G = 2, rows = 1;
        std::vector<double> in_vals = {0.3, 0.7};
        std::vector<double> wt_vals = {3.0, 3.0};
        const double sum = 0.3 * 0.3 + 0.7 * 0.7;
        const double inv = 1.0 / std::sqrt(sum / G + 1e-6);
        const double y0 = 0.3 * inv;
        const double correct0 = bf16_rne_f32(static_cast<float>(y0)) * 3.0;
        const double wrong0 = y0 * 3.0;
        UploadResult up;
        if (!upload(arena, rc, in_vals, F, wt_vals, rows, F, up, false)) { check(false, "f32round alloc"); }
        else {
            hipError_t h = shelf::kernel::launch_rmsnorm(
                up.in_dev, shelf::kernel::RmsNormDataType::F32, up.wt_dev,
                shelf::kernel::RmsNormDataType::BF16, L::PER_GROUP, M::DIRECT,
                up.out_dev, shelf::kernel::RmsNormDataType::F32, rows, F, G, F, F, 1e-6f, stream);
            (void)hipStreamSynchronize(stream);
            float got = 0.f;
            (void)hipMemcpy(&got, up.out_dev, 4, hipMemcpyDeviceToHost);
            const double diff_correct = std::fabs(static_cast<double>(got) - correct0);
            const double diff_wrong = std::fabs(static_cast<double>(got) - wrong0);
            std::printf("  [f32-round] got=%.9g correct=%.9g wrong=%.9g |dc|=%.3g |dw|=%.3g\n",
                        static_cast<double>(got), correct0, wrong0, diff_correct, diff_wrong);
            check(h == hipSuccess && diff_correct <= 1e-5 && diff_wrong > 1e-4,
                  "f32 intermediate rounding isolation");
        }
    }

    {
        char label[200];
        Config tp2_gdn = f32_f32_pg_d;
        tp2_gdn.weight_elements = 128;
        Config tp2_pg256 = bf16_f32_pg_op;
        tp2_pg256.weight_elements = 256;

        std::snprintf(label, sizeof(label), "tp2 GDN norm F=3072 G=128 rows=1");
        run_case(tp2_gdn, 1, 3072, 128, 3072, 3072, 1e-6, 900u, arena, stream, label,
                 false);
        std::snprintf(label, sizeof(label), "tp2 GDN norm F=3072 G=128 rows=64");
        run_case(tp2_gdn, 64, 3072, 128, 3072, 3072, 1e-6, 901u, arena, stream, label,
                 false);
        std::snprintf(label, sizeof(label), "tp2 K norm F=512 G=256 rows=1");
        run_case(tp2_pg256, 1, 512, 256, 512, 512, 1e-6, 902u, arena, stream, label,
                 false);
        std::snprintf(label, sizeof(label), "tp2 q_norm F=3072 G=256 rows=1");
        run_case(tp2_pg256, 1, 3072, 256, 3072, 3072, 1e-6, 903u, arena, stream, label,
                 false);
        std::snprintf(label, sizeof(label), "tp2 q_norm F=3072 G=256 rows=64");
        run_case(tp2_pg256, 64, 3072, 256, 3072, 3072, 1e-6, 904u, arena, stream, label,
                 false);
    }

    // Check the offset and both BF16 rounding points across heads and padded rows.
    for (bool per_feature : {false, true}) {
        constexpr uint32_t F = 256, G = 128, rows = 2, stride = 272;
        Config cfg{false, false, per_feature ? L::PER_FEATURE : L::PER_GROUP,
                   M::DIRECT, per_feature ? F : G};
        std::vector<double> in(static_cast<size_t>(rows) * stride, 1.0);
        std::vector<double> weights(cfg.weight_elements);
        for (uint32_t i = 0; i < cfg.weight_elements; ++i)
            weights[i] = (static_cast<int>(i % 17) - 8) / 64.0;
        UploadResult up{};
        if (!upload(arena, cfg, in, stride, weights, rows, stride, up, true)) {
            check(false, "offset RMSNorm allocation");
            continue;
        }
        constexpr float offset = 0.2f;
        const auto err = shelf::kernel::launch_rmsnorm_offset_bf16(
            up.in_dev, up.wt_dev, up.out_dev, rows, F, G, stride, stride,
            per_feature, 0.f, offset, stream);
        (void)hipStreamSynchronize(stream);
        std::vector<uint16_t> got(static_cast<size_t>(rows) * stride);
        (void)hipMemcpy(got.data(), up.out_dev, got.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
        bool ok = err == hipSuccess;
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t i = 0; i < F; ++i) {
                const float w = bf16_rne_f32(static_cast<float>(weights[per_feature ? i : i % G]));
                const uint16_t expected = shelf::f32_to_bf16_rne(offset + w);
                ok = ok && got[static_cast<size_t>(r) * stride + i] == expected;
            }
        }
        check(ok, per_feature ? "offset RMSNorm per-feature" : "offset RMSNorm per-head");
    }

    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
