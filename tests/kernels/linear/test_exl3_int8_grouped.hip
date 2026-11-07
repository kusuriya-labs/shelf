// MoE routed expert (grouped EXL3) with the mul24 codebook: the grouped kernel decodes q * kExl3Int8GridStep and runs
// an FP16 GEMM. A synthetic rank-3 expert plane (canonical [kt][nt] trellis, as the Qwen4Exp loader leaves it) is
// checked against a CPU reference built on exl3_decode_int8. Covers several bit rates, experts with no / few / many
// tokens (fused groups of one expert), token-row gather, and BF16 / F32 output.
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/exl3_grouped.h>
#include <shelf/quantization/exl3/exl3_int8.h>

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

namespace kernel = shelf::kernel;
namespace exl3 = shelf::quantization::exl3;

constexpr float kInvSqrt128 = 0.08838834764831845f;
constexpr uint32_t kRowTile = kernel::kExl3W16A16RowTile;

float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t f16_bits(float v) {
    const _Float16 h = static_cast<_Float16>(v);
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

float f16_value(uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

void fwht128(float* v) {
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
    explicit Dev(size_t n) { (void)hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
};

struct Group {
    uint32_t expert;
    uint32_t rows;
};

struct Case {
    const char* name;
    uint32_t experts, k, n, bits;
    std::vector<Group> groups;  // sorted by expert
    bool fuse;                  // pass expert_group_begin (fused groups of one expert)
    bool token_rows;            // gather the input through row_token
    bool bf16_out;  // n >= 1024 and n % 256 == 0 selects the 2-tile kernel
};

bool run_case(const Case& c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const uint32_t k = c.k, n = c.n, bits = c.bits;
    const uint64_t kt = k / 16u, nt = n / 16u;
    const uint64_t trellis_words = kt * nt * 16u * bits;  // uint16 per expert
    const uint32_t group_count = static_cast<uint32_t>(c.groups.size());

    std::vector<uint16_t> trellis(trellis_words * c.experts), suh(static_cast<size_t>(c.experts) * k),
        svh(static_cast<size_t>(c.experts) * n);
    for (auto& w : trellis) w = static_cast<uint16_t>(rng());
    for (auto& s : suh) s = f16_bits((rng() & 1u) ? 1.0f : -1.0f);
    for (auto& s : svh) s = f16_bits(0.5f + std::fabs(nd(rng)));

    // Rows: token plane (token_rows) or group-row plane. Padding rows hold NaN-free garbage that must not leak.
    const uint32_t plane_rows = group_count * kRowTile;
    std::vector<uint16_t> input(static_cast<size_t>(plane_rows) * k);
    for (auto& v : input) v = shelf::f32_to_bf16_rne(nd(rng));
    std::vector<int32_t> row_token(plane_rows, -1);
    std::vector<uint32_t> group_expert(group_count), group_rows(group_count);
    std::vector<uint32_t> expert_begin(c.experts + 1u, 0u);
    // With token rows the plane is token order; scramble the mapping so the gather is exercised.
    std::vector<uint32_t> perm(plane_rows);
    for (uint32_t i = 0; i < plane_rows; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (uint32_t g = 0; g < group_count; ++g) {
        group_expert[g] = c.groups[g].expert;
        group_rows[g] = c.groups[g].rows;
        for (uint32_t r = 0; r < kRowTile; ++r)
            if (r < group_rows[g]) row_token[g * kRowTile + r] = static_cast<int32_t>(perm[g * kRowTile + r]);
        ++expert_begin[c.groups[g].expert + 1u];
    }
    for (uint32_t e = 0; e < c.experts; ++e) expert_begin[e + 1u] += expert_begin[e];

    const size_t out_elems = static_cast<size_t>(plane_rows) * n;
    Dev<uint16_t> d_trellis(trellis.size()), d_suh(suh.size()), d_svh(svh.size()), d_in(input.size());
    Dev<uint32_t> d_ge(group_count), d_gr(group_count), d_gc(1), d_eb(expert_begin.size());
    Dev<int32_t> d_rt(row_token.size());
    Dev<uint8_t> d_out(out_elems * (c.bf16_out ? 2u : 4u));
    HIP_OK(hipMemcpy(d_trellis.p, trellis.data(), trellis.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_suh.p, suh.data(), suh.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_svh.p, svh.data(), svh.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_in.p, input.data(), input.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_ge.p, group_expert.data(), group_count * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_gr.p, group_rows.data(), group_count * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_gc.p, &group_count, 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_eb.p, expert_begin.data(), expert_begin.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_rt.p, row_token.data(), row_token.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_out.p, 0xFF, out_elems * (c.bf16_out ? 2u : 4u)));  // untouched rows stay 0xFF

    kernel::Exl3GroupedArgs args;
    args.trellis = d_trellis.p;
    args.suh = d_suh.p;
    args.svh = d_svh.p;
    args.input = d_in.p;
    args.output = d_out.p;
    args.out_features = n;
    args.k = k;
    args.input_row_stride = k;
    args.output_row_stride = n;
    args.output_dtype = c.bf16_out ? 0u : 1u;
    args.ka = bits;
    args.half = 0;
    args.mul24 = 1;
    args.trellis_batch_stride = trellis_words * 2u;
    args.suh_batch_stride = static_cast<uint64_t>(k) * 2u;
    args.svh_batch_stride = static_cast<uint64_t>(n) * 2u;
    args.group_expert = d_ge.p;
    args.group_rows = d_gr.p;
    args.group_count = d_gc.p;
    args.expert_group_begin = c.fuse ? d_eb.p : nullptr;
    args.row_token = c.token_rows ? d_rt.p : nullptr;
    args.group_capacity = group_count;
    HIP_OK(kernel::launch_gemm_exl3_w16a16_grouped(args, nullptr));
    HIP_OK(hipDeviceSynchronize());

    std::vector<uint8_t> raw(out_elems * (c.bf16_out ? 2u : 4u));
    HIP_OK(hipMemcpy(raw.data(), d_out.p, raw.size(), hipMemcpyDeviceToHost));

    // CPU reference per expert: q from the canonical trellis.
    std::vector<std::vector<int8_t>> q(c.experts);
    for (uint32_t e = 0; e < c.experts; ++e) {
        q[e].assign(static_cast<size_t>(n) * k, 0);
        if (!exl3::exl3_decode_int8(trellis.data() + e * trellis_words, kt, nt, bits, q[e].data(), 2)) return false;
    }
    double max_ref = 0.0, max_err = 0.0;
    size_t untouched_bad = 0;
    std::vector<float> ref(out_elems, 0.0f);
    std::vector<uint8_t> row_written(plane_rows, 0);
    for (uint32_t g = 0; g < group_count; ++g) {
        const uint32_t e = group_expert[g];
        for (uint32_t r = 0; r < group_rows[g]; ++r) {
            const uint32_t out_row = g * kRowTile + r;
            row_written[out_row] = 1;
            const uint32_t in_row = c.token_rows ? static_cast<uint32_t>(row_token[out_row]) : out_row;
            std::vector<float> xrot(k);
            for (uint32_t b = 0; b < k / 128u; ++b) {
                float v[128];
                for (int t = 0; t < 128; ++t)
                    v[t] = bf16_to_f32(input[static_cast<size_t>(in_row) * k + b * 128 + t]) *
                           f16_value(suh[static_cast<size_t>(e) * k + b * 128 + t]);
                fwht128(v);
                for (int t = 0; t < 128; ++t) xrot[b * 128 + t] = f16_value(f16_bits(v[t] * kInvSqrt128));
            }
            std::vector<float> yp(n);
            for (uint32_t col = 0; col < n; ++col) {
                double acc = 0.0;
                const int8_t* qrow = q[e].data() + static_cast<size_t>(col) * k;
                for (uint32_t i = 0; i < k; ++i) acc += static_cast<double>(qrow[i]) * xrot[i];
                yp[col] = static_cast<float>(acc * exl3::kExl3Int8GridStep);
            }
            for (uint32_t b = 0; b < n / 128u; ++b) {
                fwht128(yp.data() + b * 128);
                for (int t = 0; t < 128; ++t) {
                    const uint32_t col = b * 128 + t;
                    const float y = yp[col] * kInvSqrt128 * f16_value(svh[static_cast<size_t>(e) * n + col]);
                    ref[static_cast<size_t>(out_row) * n + col] = y;
                    max_ref = std::max(max_ref, static_cast<double>(std::fabs(y)));
                }
            }
        }
    }
    for (uint32_t row = 0; row < plane_rows; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            const size_t i = static_cast<size_t>(row) * n + col;
            if (!row_written[row]) {
                // The kernel must leave rows beyond group_rows alone (still 0xFF).
                const size_t w = c.bf16_out ? 2u : 4u;
                for (size_t b = 0; b < w; ++b)
                    if (raw[i * w + b] != 0xFFu) { ++untouched_bad; break; }
                continue;
            }
            float got;
            if (c.bf16_out) {
                uint16_t b;
                std::memcpy(&b, &raw[i * 2], 2);
                got = bf16_to_f32(b);
            } else {
                std::memcpy(&got, &raw[i * 4], 4);
            }
            max_err = std::max(max_err, static_cast<double>(std::fabs(got - ref[i])));
        }
    }
    // The CPU mirror rounds xrot to FP16 like the kernel; the rest is FP32 summation order (and bf16 output rounding).
    const double tol = (c.bf16_out ? 1.0e-2 : 1.0e-3) * std::max(max_ref, 1e-6);
    const bool ok = max_err <= tol && untouched_bad == 0;
    std::printf("%s: bits=%u k=%u n=%u groups=%u max_ref=%g max_err=%g tol=%g untouched_bad=%zu %s\n", c.name, bits, k,
                n, group_count, max_ref, max_err, tol, untouched_bad, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    int failed = 0;
    uint32_t seed = 1;
    auto run = [&](const Case& c) {
        if (!run_case(c, seed++)) ++failed;
    };
    // Experts: 0 has a full group, 1 has no token, 2 has 3 rows, 3 has three groups (16 / 16 / 5 rows, fused on request).
    const std::vector<Group> mixed = {{0, 16}, {2, 3}, {3, 16}, {3, 16}, {3, 5}};
    for (uint32_t bits : {1u, 2u, 3u, 4u, 5u, 6u, 8u}) {
        run({"nt1", 4, 256, 512, bits, mixed, false, false, false});
        run({"nt1-fuse", 4, 256, 512, bits, mixed, true, false, false});
    }
    run({"nt2", 4, 384, 1024, 3, mixed, false, false, false});
    run({"nt2-fuse", 4, 384, 1024, 3, mixed, true, false, false});
    run({"nt2-fuse-tokenrows", 4, 384, 1024, 4, mixed, true, true, false});
    run({"nt1-tokenrows", 4, 256, 512, 2, mixed, false, true, false});
    run({"bf16-out", 4, 256, 512, 3, mixed, true, true, true});
    run({"single-row", 2, 128, 128, 4, {{1, 1}}, false, false, false});
    if (failed != 0) {
        std::printf("FAILED: %d case(s)\n", failed);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
