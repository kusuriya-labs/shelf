#include <shelf/kernels/fused/exl3/decode.h>
#include <shelf/kernels/fused/exl3/grouped.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
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
    kernel::Exl3GroupedArgs args;
    args.trellis = d_trellis.p;
    args.suh = d_suh.p;
    args.svh = d_svh.p;
    args.input = d_in.p;
    args.input_row_stride = k;
    args.output_row_stride = n;
    args.output_dtype = c.bf16_out ? 0u : 1u;
    args.out_features = n;
    args.k = k;
    args.ka = bits;
    args.half = 0;
    args.trellis_batch_stride = trellis_words * 2u;
    args.suh_batch_stride = static_cast<uint64_t>(k) * 2u;
    args.svh_batch_stride = static_cast<uint64_t>(n) * 2u;
    args.group_expert = d_ge.p;
    args.group_rows = d_gr.p;
    args.group_count = d_gc.p;
    args.expert_group_begin = c.fuse ? d_eb.p : nullptr;
    args.row_token = c.token_rows ? d_rt.p : nullptr;
    args.group_capacity = group_count;

    const size_t out_bytes = out_elems * (c.bf16_out ? 2u : 4u);
    std::vector<uint8_t> raw(out_bytes);
    HIP_OK(hipMemset(d_out.p, 0xFF, out_bytes));
    args.output = d_out.p;
    HIP_OK(kernel::launch_gemm_exl3_int8_grouped(args, nullptr));
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(raw.data(), d_out.p, out_bytes, hipMemcpyDeviceToHost));

    // CPU reference per expert: q from the canonical trellis.
    std::vector<std::vector<int8_t>> q(c.experts);
    for (uint32_t e = 0; e < c.experts; ++e) {
        q[e].assign(static_cast<size_t>(n) * k, 0);
        if (!exl3::exl3_decode_int8(trellis.data() + e * trellis_words, kt, nt, bits, q[e].data(), 2)) return false;
    }
    std::vector<float> refs(out_elems, 0.0f);
    std::vector<uint8_t> row_written(plane_rows, 0);
    double max_ref = 0.0;
    for (uint32_t g = 0; g < group_count; ++g) {
        const uint32_t e = group_expert[g];
        for (uint32_t r = 0; r < group_rows[g]; ++r) {
            const uint32_t out_row = g * kRowTile + r;
            row_written[out_row] = 1;
            const uint32_t in_row = c.token_rows ? static_cast<uint32_t>(row_token[out_row]) : out_row;
            std::vector<float> xq(k), scale(k / 128u);
            for (uint32_t b = 0; b < k / 128u; ++b) {
                float v[128];
                for (int t = 0; t < 128; ++t)
                    v[t] = bf16_to_f32(input[static_cast<size_t>(in_row) * k + b * 128 + t]) *
                           f16_value(suh[static_cast<size_t>(e) * k + b * 128 + t]);
                fwht128(v);
                float peak = 0.0f;
                for (int t = 0; t < 128; ++t) {
                    v[t] *= kInvSqrt128;
                    peak = std::max(peak, std::fabs(v[t]));
                }
                scale[b] = peak / 127.0f;
                const float inv = scale[b] > 0.0f ? 1.0f / scale[b] : 0.0f;
                for (int t = 0; t < 128; ++t) {
                    xq[b * 128 + t] = std::min(std::max(std::nearbyintf(v[t] * inv), -128.0f), 127.0f);
                }
            }
            std::vector<float> yp(n);
            for (uint32_t col = 0; col < n; ++col) {
                const int8_t* qrow = q[e].data() + static_cast<size_t>(col) * k;
                float accf = 0.0f;
                for (uint32_t b = 0; b < k / 128u; ++b) {
                    int64_t isum = 0;
                    for (uint32_t i = b * 128; i < (b + 1) * 128; ++i)
                        isum += static_cast<int64_t>(qrow[i]) * static_cast<int64_t>(xq[i]);
                    accf += static_cast<float>(isum) * scale[b];
                }
                yp[col] = accf * exl3::kExl3Int8GridStep;
            }
            for (uint32_t b = 0; b < n / 128u; ++b) {
                fwht128(yp.data() + b * 128);
                for (int t = 0; t < 128; ++t) {
                    const uint32_t col = b * 128 + t;
                    const float y = yp[col] * kInvSqrt128 * f16_value(svh[static_cast<size_t>(e) * n + col]);
                    refs[static_cast<size_t>(out_row) * n + col] = y;
                    max_ref = std::max(max_ref, static_cast<double>(std::fabs(y)));
                }
            }
        }
    }
    auto element = [&](const std::vector<uint8_t>& bytes, size_t i) {
        float got;
        if (c.bf16_out) {
            uint16_t b;
            std::memcpy(&b, &bytes[i * 2], 2);
            got = bf16_to_f32(b);
        } else {
            std::memcpy(&got, &bytes[i * 4], 4);
        }
        return got;
    };
    // The CPU mirrors round like the kernel; the rest is FP32 summation order (and bf16 output rounding). The INT8
    // mirror may differ from the kernel by a flipped rint tie (one activation code by 1) and the order of the FHT.
    const double tol = (c.bf16_out ? 1.0e-2 : 1.0e-3) * std::max(max_ref, 1e-6);
    double err = 0.0;
    size_t untouched_bad = 0;
    for (uint32_t row = 0; row < plane_rows; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            const size_t i = static_cast<size_t>(row) * n + col;
            if (!row_written[row]) {
                const size_t w = c.bf16_out ? 2u : 4u;
                for (size_t b = 0; b < w; ++b)
                    if (raw[i * w + b] != 0xFFu) { ++untouched_bad; break; }
                continue;
            }
            err = std::max(err, static_cast<double>(std::fabs(element(raw, i) - refs[i])));
        }
    }
    // The pre-rotated plane must reproduce the inline rotation bit for bit: the producer and
    // the grouped kernel run the same rotation, and the GEMM then only skips redoing it.
    std::vector<uint8_t> pre_raw(out_bytes, 0);
    const size_t elem_bytes = c.bf16_out ? 2u : 4u;
    bool pre_exact = true;
    {
        Dev<uint8_t> d_codes(static_cast<size_t>(plane_rows) * k);
        Dev<float> d_scales(static_cast<size_t>(plane_rows) * (k / 128u));
        HIP_OK(kernel::launch_exl3_grouped_activate(
            d_in.p, d_suh.p, d_codes.p, d_scales.p, d_ge.p, d_gr.p, d_gc.p, nullptr,
            c.token_rows ? d_rt.p : nullptr, plane_rows, k, k,
            static_cast<uint64_t>(k) * 2u, nullptr));
        kernel::Exl3GroupedArgs pre_args = args;
        pre_args.pre_rotated_codes = d_codes.p;
        pre_args.pre_rotated_scales = d_scales.p;
        HIP_OK(hipMemset(d_out.p, 0xFF, out_bytes));
        HIP_OK(kernel::launch_gemm_exl3_int8_grouped(pre_args, nullptr));
        HIP_OK(hipDeviceSynchronize());
        HIP_OK(hipMemcpy(pre_raw.data(), d_out.p, out_bytes, hipMemcpyDeviceToHost));
        size_t bad = 0;
        for (uint32_t row = 0; row < plane_rows; ++row) {
            const size_t off = static_cast<size_t>(row) * n * elem_bytes;
            if (row_written[row]) {
                if (std::memcmp(raw.data() + off, pre_raw.data() + off,
                                static_cast<size_t>(n) * elem_bytes) != 0)
                    ++bad;
            } else {
                for (size_t b = 0; b < static_cast<size_t>(n) * elem_bytes; ++b)
                    if (pre_raw[off + b] != 0xFFu) { ++bad; break; }
            }
        }
        pre_exact = bad == 0;
        if (!pre_exact) std::printf("%s: pre-rotated rows differ (%zu rows)\n", c.name, bad);
    }
    // The SwiGLU producer must land on the same codes/scales as writing SwiGLU to a plane and
    // rotating that plane, for both token-row and group-row gathers.
    bool swiglu_exact = true;
    {
        std::mt19937 srng(seed * 7u + 3u);
        std::normal_distribution<float> snd(0.0f, 1.0f);
        std::vector<uint16_t> gate(input.size()), up(input.size()), swiglu(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            gate[i] = shelf::f32_to_bf16_rne(snd(srng));
            up[i] = shelf::f32_to_bf16_rne(snd(srng));
        }
        Dev<uint16_t> d_gate(gate.size()), d_up(up.size()), d_swiglu(swiglu.size());
        HIP_OK(hipMemcpy(d_gate.p, gate.data(), gate.size() * 2, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_up.p, up.data(), up.size() * 2, hipMemcpyHostToDevice));
        HIP_OK(kernel::launch_swiglu_bf16(
            reinterpret_cast<const shelf::bf16_t*>(d_gate.p), k,
            reinterpret_cast<const shelf::bf16_t*>(d_up.p), k,
            reinterpret_cast<shelf::bf16_t*>(d_swiglu.p), k, plane_rows, k, nullptr, false));
        Dev<uint8_t> d_codes_a(static_cast<size_t>(plane_rows) * k);
        Dev<float> d_scales_a(static_cast<size_t>(plane_rows) * (k / 128u));
        Dev<uint8_t> d_codes_b(static_cast<size_t>(plane_rows) * k);
        Dev<float> d_scales_b(static_cast<size_t>(plane_rows) * (k / 128u));
        HIP_OK(kernel::launch_exl3_grouped_activate(
            d_swiglu.p, d_suh.p, d_codes_a.p, d_scales_a.p, d_ge.p, d_gr.p, d_gc.p, nullptr,
            c.token_rows ? d_rt.p : nullptr, plane_rows, k, k,
            static_cast<uint64_t>(k) * 2u, nullptr));
        HIP_OK(kernel::launch_exl3_grouped_swiglu_activate(
            d_gate.p, d_up.p, d_suh.p, d_codes_b.p, d_scales_b.p, d_ge.p, d_gr.p, d_gc.p,
            nullptr, c.token_rows ? d_rt.p : nullptr, plane_rows, k, k, k, 0u,
            static_cast<uint64_t>(k) * 2u, nullptr));
        HIP_OK(hipDeviceSynchronize());
        std::vector<uint8_t> ca(static_cast<size_t>(plane_rows) * k), cb(ca.size());
        std::vector<float> sa(static_cast<size_t>(plane_rows) * (k / 128u)), sb(sa.size());
        HIP_OK(hipMemcpy(ca.data(), d_codes_a.p, ca.size(), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(cb.data(), d_codes_b.p, cb.size(), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(sa.data(), d_scales_a.p, sa.size() * 4, hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(sb.data(), d_scales_b.p, sb.size() * 4, hipMemcpyDeviceToHost));
        bool same_codes = true, same_scales = true;
        for (uint32_t row = 0; row < plane_rows; ++row) {
            if (!row_written[row]) continue;
            if (std::memcmp(ca.data() + static_cast<size_t>(row) * k,
                            cb.data() + static_cast<size_t>(row) * k, k) != 0)
                same_codes = false;
            const size_t so = static_cast<size_t>(row) * (k / 128u);
            if (std::memcmp(sa.data() + so, sb.data() + so, (k / 128u) * 4u) != 0)
                same_scales = false;
        }
        swiglu_exact = same_codes && same_scales;
        if (!swiglu_exact)
            std::printf("%s: swiglu producer differs (codes=%d scales=%d)\n", c.name,
                        static_cast<int>(same_codes), static_cast<int>(same_scales));
    }
    const bool ok = err <= tol && untouched_bad == 0 && pre_exact && swiglu_exact;
    std::printf("%s: bits=%u k=%u n=%u groups=%u max_ref=%g err_int8=%g tol=%g untouched_bad=%zu %s\n",
                c.name, bits, k, n, group_count, max_ref, err, tol, untouched_bad,
                ok ? "ok" : "FAIL");
    return ok;
}


// Split plan: hits and misses are compacted subsequences of the full group list, and each
// launch addresses its output rows through group_row_base. The pre-rotated codes must land
// on those same full rows.
bool run_split_case(uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const uint32_t experts = 6, k = 1280, n = 512, bits = 3;
    const uint32_t kt = k / 16u, nt = n / 16u;
    const uint64_t trellis_words = kt * nt * 16u * bits;
    const std::vector<Group> groups = {{0, 16}, {1, 3}, {2, 16}, {2, 16}, {2, 5}, {4, 1}, {5, 16}};
    const uint32_t group_count = static_cast<uint32_t>(groups.size());
    const uint32_t plane_rows = group_count * kRowTile;

    std::vector<uint16_t> trellis(trellis_words * experts), suh(static_cast<size_t>(experts) * k),
        svh(static_cast<size_t>(experts) * n);
    for (auto& w : trellis) w = static_cast<uint16_t>(rng());
    for (auto& s : suh) s = f16_bits((rng() & 1u) ? 1.0f : -1.0f);
    for (auto& s : svh) s = f16_bits(0.5f + std::fabs(nd(rng)));
    std::vector<uint16_t> input(static_cast<size_t>(plane_rows) * k);
    for (auto& v : input) v = shelf::f32_to_bf16_rne(nd(rng));
    std::vector<uint32_t> perm(plane_rows);
    for (uint32_t i = 0; i < plane_rows; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> row_token(plane_rows, -1);
    std::vector<uint32_t> group_expert(group_count), group_rows(group_count);
    for (uint32_t g = 0; g < group_count; ++g) {
        group_expert[g] = groups[g].expert;
        group_rows[g] = groups[g].rows;
        for (uint32_t r = 0; r < groups[g].rows; ++r)
            row_token[g * kRowTile + r] = static_cast<int32_t>(perm[g * kRowTile + r]);
    }
    // Even experts are resident (hits), odd experts are offloaded (misses).
    std::vector<uint32_t> hit_slot, hit_rows_v, hit_base, miss_slot, miss_rows_v, miss_base;
    for (uint32_t g = 0; g < group_count; ++g) {
        if ((group_expert[g] & 1u) == 0u) {
            hit_slot.push_back(group_expert[g]);
            hit_rows_v.push_back(group_rows[g]);
            hit_base.push_back(g);
        } else {
            miss_slot.push_back(group_expert[g]);
            miss_rows_v.push_back(group_rows[g]);
            miss_base.push_back(g);
        }
    }
    auto begin_for = [&](const std::vector<uint32_t>& slot, uint32_t e) {
        uint32_t at = 0u;
        for (uint32_t i = 0; i < slot.size(); ++i)
            if (slot[i] < e) ++at;
        return at;
    };
    std::vector<uint32_t> hit_eb(experts + 1u), miss_eb(experts + 1u);
    for (uint32_t e = 0; e < experts; ++e) {
        hit_eb[e] = begin_for(hit_slot, e);
        miss_eb[e] = begin_for(miss_slot, e);
    }
    hit_eb[experts] = static_cast<uint32_t>(hit_slot.size());
    miss_eb[experts] = static_cast<uint32_t>(miss_slot.size());

    Dev<uint16_t> d_trellis(trellis.size()), d_suh(suh.size()), d_svh(svh.size()), d_in(input.size());
    Dev<uint32_t> d_hslot(hit_slot.size()), d_hrows(hit_rows_v.size()), d_hbase(hit_base.size());
    Dev<uint32_t> d_mslot(miss_slot.size()), d_mrows(miss_rows_v.size()), d_mbase(miss_base.size());
    Dev<uint32_t> d_hcount(1), d_mcount(1), d_heb(hit_eb.size()), d_meb(miss_eb.size());
    Dev<int32_t> d_rt(row_token.size());
    const uint32_t hc = static_cast<uint32_t>(hit_slot.size());
    const uint32_t mc = static_cast<uint32_t>(miss_slot.size());
    HIP_OK(hipMemcpy(d_trellis.p, trellis.data(), trellis.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_suh.p, suh.data(), suh.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_svh.p, svh.data(), svh.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_in.p, input.data(), input.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_hslot.p, hit_slot.data(), hc * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_hrows.p, hit_rows_v.data(), hc * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_hbase.p, hit_base.data(), hc * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_mslot.p, miss_slot.data(), mc * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_mrows.p, miss_rows_v.data(), mc * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_mbase.p, miss_base.data(), mc * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_hcount.p, &hc, 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_mcount.p, &mc, 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_heb.p, hit_eb.data(), hit_eb.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_meb.p, miss_eb.data(), miss_eb.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_rt.p, row_token.data(), row_token.size() * 4, hipMemcpyHostToDevice));

    const size_t out_elems = static_cast<size_t>(plane_rows) * n;
    const size_t out_bytes = out_elems * 2u;
    Dev<uint8_t> d_out(out_bytes);
    Dev<uint8_t> d_codes(static_cast<size_t>(plane_rows) * k);
    Dev<float> d_scales(static_cast<size_t>(plane_rows) * (k / 128u));

    kernel::Exl3GroupedArgs base;
    base.trellis = d_trellis.p; base.suh = d_suh.p; base.svh = d_svh.p;
    base.input = d_in.p; base.output = d_out.p; base.out_features = n; base.k = k;
    base.input_row_stride = k;
    base.output_row_stride = n; base.output_dtype = 0u; base.ka = bits;
    base.trellis_batch_stride = trellis_words * 2u;
    base.suh_batch_stride = static_cast<uint64_t>(k) * 2u;
    base.svh_batch_stride = static_cast<uint64_t>(n) * 2u;
    base.row_token = d_rt.p; base.group_capacity = group_count;

    auto launch = [&](const uint32_t* ge, const uint32_t* gr, const uint32_t* gc,
                      const uint32_t* gb, const uint32_t* eb, bool pre) {
        kernel::Exl3GroupedArgs a = base;
        a.group_expert = ge; a.group_rows = gr; a.group_count = gc;
        a.group_row_base = gb; a.expert_group_begin = eb;
        if (pre) { a.pre_rotated_codes = d_codes.p; a.pre_rotated_scales = d_scales.p; }
        return kernel::launch_gemm_exl3_int8_grouped(a, nullptr);
    };
    Dev<uint32_t> d_ge(group_count), d_gr(group_count), d_gc7(1);
    HIP_OK(hipMemcpy(d_ge.p, group_expert.data(), group_count * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_gr.p, group_rows.data(), group_count * 4, hipMemcpyHostToDevice));
    const uint32_t g7 = group_count;
    HIP_OK(hipMemcpy(d_gc7.p, &g7, 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_out.p, 0xFF, out_bytes));
    HIP_OK(launch(d_ge.p, d_gr.p, d_gc7.p, nullptr, nullptr, false));
    HIP_OK(hipDeviceSynchronize());
    std::vector<uint8_t> full(out_bytes);
    HIP_OK(hipMemcpy(full.data(), d_out.p, out_bytes, hipMemcpyDeviceToHost));
    HIP_OK(hipMemset(d_out.p, 0xFF, out_bytes));
    hipError_t le = launch(d_hslot.p, d_hrows.p, d_hcount.p, d_hbase.p, d_heb.p, false);
    if (le != hipSuccess) { std::printf("hit launch err %d\n", (int)le); return false; }
    le = launch(d_mslot.p, d_mrows.p, d_mcount.p, d_mbase.p, d_meb.p, false);
    if (le != hipSuccess) { std::printf("miss launch err %d hc=%u mc=%u\n", (int)le, hc, mc); return false; }
    HIP_OK(hipDeviceSynchronize());
    std::vector<uint8_t> fused(out_bytes);
    HIP_OK(hipMemcpy(fused.data(), d_out.p, out_bytes, hipMemcpyDeviceToHost));

    HIP_OK(hipMemset(d_out.p, 0xFF, out_bytes));
    HIP_OK(kernel::launch_exl3_grouped_activate(
        d_in.p, d_suh.p, d_codes.p, d_scales.p, d_hslot.p, d_hrows.p, d_hcount.p,
        d_hbase.p, d_rt.p, plane_rows, k, k, static_cast<uint64_t>(k) * 2u, nullptr));
    HIP_OK(launch(d_hslot.p, d_hrows.p, d_hcount.p, d_hbase.p, d_heb.p, true));
    HIP_OK(kernel::launch_exl3_grouped_activate(
        d_in.p, d_suh.p, d_codes.p, d_scales.p, d_mslot.p, d_mrows.p, d_mcount.p,
        d_mbase.p, d_rt.p, plane_rows, k, k, static_cast<uint64_t>(k) * 2u, nullptr));
    HIP_OK(launch(d_mslot.p, d_mrows.p, d_mcount.p, d_mbase.p, d_meb.p, true));
    HIP_OK(hipDeviceSynchronize());
    std::vector<uint8_t> pre(out_bytes);
    HIP_OK(hipMemcpy(pre.data(), d_out.p, out_bytes, hipMemcpyDeviceToHost));
    bool exact = true;
    for (uint32_t row = 0; row < plane_rows && exact; ++row) {
        const size_t off = static_cast<size_t>(row) * n * 2u;
        if (std::memcmp(fused.data() + off, pre.data() + off, static_cast<size_t>(n) * 2u) != 0) {
            exact = false;
            uint32_t col = 0;
            for (uint32_t c = 0; c < n; ++c)
                if (std::memcmp(fused.data() + off + c * 2u, pre.data() + off + c * 2u, 2) != 0) {
                    col = c;
                    break;
                }
            std::printf("split-base first diff row=%u col=%u\n", row, col);
        }
    }
    std::printf("split-base: hits=%u misses=%u %s full_vs_split=%s pre_vs_full=%s\n", hc, mc,
                exact ? "ok" : "FAIL",
                std::memcmp(full.data(), fused.data(), out_bytes) == 0 ? "same" : "DIFF",
                std::memcmp(full.data(), pre.data(), out_bytes) == 0 ? "same" : "DIFF");
    return exact;
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
    // Zero-row groups, one-token experts and a heavily loaded expert (six full groups plus a short one).
    run({"zero-and-one", 3, 256, 512, 3, {{0, 0}, {1, 1}, {2, 16}, {2, 0}}, true, true, false});
    run({"heavy-expert", 2, 384, 512, 4, {{0, 16}, {0, 16}, {0, 16}, {0, 16}, {0, 16}, {0, 16}, {0, 7}, {1, 2}},
         true, true, false});
    run({"heavy-expert-nt2-bf16", 2, 384, 1024, 2, {{0, 16}, {0, 16}, {0, 16}, {0, 9}, {1, 2}}, true, true, true});
    // Long K selects the paired-stage kernel (8+ K chunks), including an odd chunk count.
    run({"pair-k10-nt1", 2, 1280, 512, 3, {{0, 16}, {0, 16}, {0, 16}, {0, 9}, {1, 2}}, false, false, false});
    run({"pair-k10-nt1-fuse", 2, 1280, 512, 3, {{0, 16}, {0, 16}, {0, 16}, {0, 9}, {1, 2}}, true, true, false});
    run({"pair-k9-nt1", 2, 1152, 512, 3, {{0, 16}, {0, 16}, {0, 16}, {0, 9}, {1, 2}}, true, true, false});
    run({"pair-k8-nt2", 2, 1024, 1024, 3, {{0, 16}, {0, 16}, {0, 16}, {0, 9}, {1, 2}},
         false, false, false});
    run({"pair-k9-nt2", 2, 1152, 1024, 3, {{0, 16}, {0, 16}, {0, 16}, {0, 9}, {1, 2}},
         true, true, false});
    run({"pair-k11-nt2", 2, 1408, 1024, 3, {{0, 16}, {0, 16}, {0, 16}, {0, 9}, {1, 2}},
         true, true, false});
    if (!run_split_case(seed++)) ++failed;
    if (failed != 0) {
        std::printf("FAILED: %d case(s)\n", failed);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
