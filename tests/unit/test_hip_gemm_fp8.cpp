// shelf unit test — FP8 W8A8 のカーネル（gfx1201 E4M3 行列命令）と活性化量子化。
//
// 検証の三段:
//   1. 活性化量子化: デバイスが出すコード/スケールを host の quant/fp8 実装で再現した
//      ものと一致させる（**デバイス e4m3 コーデックと host コーデックの一致**、
//      **断片順レイアウトの一致**、**粒度（128 / 行全体）どおり**を同時に確認する）。
//   2. GEMM（小・境界形状）: 量子化後の値に対する f64 golden と突き合わせる。
//   3. GEMM（実モデル形状）: 同じ量子化値を bf16 で積んだ既存 GEMM と突き合わせる
//      （E4M3 の値は bf16 で厳密に表せるため、差は累積順だけ）。
//
// 粒度の組合せを総当りで見る:
//   act_group ∈ {128, 0} × weight ∈ {per-channel, block128}
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <shelf/backend/hip/kernels/elementwise/activation_quantize_fp8.h>
#include <shelf/backend/hip/kernels/gemm/linear_bf16.h>
#include <shelf/backend/hip/kernels/gemm/linear_fp8.h>
#include <shelf/foundation/numeric.hpp>
#include <shelf/quant/fp8_w8a8.hpp>

namespace {

using shelf::bf16_t;
using shelf::quant::fp8::act_scale_groups;
using shelf::quant::fp8::amax_scale;
using shelf::quant::fp8::e4m3_decode;
using shelf::quant::fp8::e4m3_encode;
using shelf::quant::fp8::kActGroupK;
using shelf::quant::fp8::pad_k;
using shelf::quant::fp8::scale_k_blocks;
using shelf::quant::fp8::tiled_bytes;
using shelf::quant::fp8::tiled_offset;
using shelf::hipk::Fp8GemmConfig;
using shelf::hipk::Fp8GemmConfigId;
using shelf::hipk::Fp8GemmOutputDType;
using shelf::hipk::Fp8GemmScaleArgs;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

#define HIP_OK(call)                                                          \
    do {                                                                      \
        const hipError_t e_ = (call);                                         \
        if (e_ != hipSuccess) {                                               \
            std::printf("FAIL hip: %s @%d: %s\n", #call, __LINE__,            \
                        hipGetErrorString(e_));                               \
            return 1;                                                         \
        }                                                                     \
    } while (0)

std::uint32_t g_seed = 20261029u;

// ---- 1. 活性化量子化 -------------------------------------------------------

int test_activation_quantize(std::uint32_t rows, std::uint32_t k, std::uint32_t act_group) {
    const std::uint32_t kp = pad_k(k);
    if (!shelf::hipk::activation_quantize_fp8_supported(k, kp, act_group)) {
        std::printf("FAIL: supported(%u,%u)\n", k, act_group);
        ++failures;
        return 1;
    }
    std::mt19937 rng(g_seed++ + rows * 131u + k * 3u + act_group);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<bf16_t> x(static_cast<std::size_t>(rows) * k);
    for (auto& v : x) v = shelf::f32_to_bf16(dist(rng));
    // 外れ値を意図的に入れて、粒度の違いが実際に効くことを確認する足場
    if (act_group == kActGroupK && rows > 0u && k > 300u)
        x[static_cast<std::size_t>(0) * k + 7u] = shelf::f32_to_bf16(6000.0f);

    const std::size_t code_bytes = tiled_bytes(rows, kp);
    const std::uint32_t kblocks = act_scale_groups(kp, act_group);
    const std::size_t scale_count = static_cast<std::size_t>(rows) * kblocks;

    bf16_t* dx = nullptr;
    std::uint8_t* dcodes = nullptr;
    float* dscales = nullptr;
    HIP_OK(hipMalloc(&dx, x.size() * sizeof(bf16_t)));
    HIP_OK(hipMalloc(&dcodes, code_bytes));
    HIP_OK(hipMalloc(&dscales, sizeof(float) * scale_count));
    HIP_OK(hipMemcpy(dx, x.data(), x.size() * sizeof(bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dcodes, 0, code_bytes));           // 詰め行は書かない前提
    HIP_OK(hipMemset(dscales, 0xFF, sizeof(float) * scale_count));

    HIP_OK(shelf::hipk::launch_activation_quantize_fp8(dx, dcodes, dscales, rows, k, kp, k,
                                                       act_group, nullptr));
    std::vector<std::uint8_t> codes(code_bytes, 0u);
    std::vector<float> gscales(scale_count, 0.0f);
    HIP_OK(hipMemcpy(codes.data(), dcodes, code_bytes, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(gscales.data(), dscales, sizeof(float) * scale_count,
                     hipMemcpyDeviceToHost));
    (void)hipFree(dx); (void)hipFree(dcodes); (void)hipFree(dscales);

    // host 側で同じ手順（範囲別 amax -> scale -> 逆数を掛けて encode）を再現する。
    std::vector<std::uint8_t> expect(code_bytes, 0u);
    std::vector<float> host_scales(scale_count, 0.0f);
    bool scales_ok = true;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t kb = 0; kb < kblocks; ++kb) {
            const std::uint32_t b0 = (act_group == 0u) ? 0u : kb * kActGroupK;
            const std::uint32_t b1 = (act_group == 0u) ? kp : std::min(kp, b0 + kActGroupK);
            float amax = 0.0f;
            for (std::uint32_t kk = b0; kk < b1 && kk < k; ++kk)
                amax = std::fmax(amax, std::fabs(shelf::bf16_to_f32(x[std::size_t(r) * k + kk])));
            const float sc = amax_scale(amax);
            const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
            host_scales[std::size_t(r) * kblocks + kb] = sc;
            if (gscales[std::size_t(r) * kblocks + kb] != sc) scales_ok = false;
            for (std::uint32_t kk = b0; kk < b1; ++kk) {
                const float v = (kk < k) ? shelf::bf16_to_f32(x[std::size_t(r) * k + kk]) : 0.0f;
                expect[tiled_offset(r, kk, kp)] = (sc > 0.0f) ? e4m3_encode(v * inv) : 0u;
            }
        }
    }
    check(scales_ok, "活性化スケールが粒度どおりに一致");
    if (codes != expect) {
        int shown = 0;
        for (std::size_t i = 0; i < codes.size() && shown < 4; ++i) {
            if (codes[i] == expect[i]) continue;
            ++shown;
            std::printf("  code[%zu] dev=%02X host=%02X\n", i, codes[i], expect[i]);
        }
    }
    check(codes == expect, "活性化コードの量子化値と断片順配置が一致");

    if (act_group == kActGroupK && rows > 0u && k > 300u) {
        // 外れ値があるブロックだけスケールが膨らみ、他ブロックは小さいまま。
        const float sc0 = host_scales[0];
        const float sc1 = host_scales[kblocks - 1];
        check(sc0 > 10.0f * (sc1 > 0.0f ? sc1 : 1e-9f),
              "128 グループでは外れ値が他のブロックへ移らない");
    }
    return 0;
}

// ---- 2/3. GEMM ------------------------------------------------------------

struct Case {
    std::uint32_t rows, features, k;
    std::uint32_t act_group;
    bool block128;
};

struct QuantizedWeights {
    std::vector<std::uint8_t> codes_rowmajor;   // [features][k]
    std::vector<float> scales;                  // per-channel [features] / block128 [nb][kb]
    std::vector<bf16_t> as_bf16;                // 量子化値を bf16 にしたもの（突合用）
    std::uint32_t n_blocks = 0, k_blocks = 0;
};

// 値 -> (code, scale) を per-channel または block128 で作る。
QuantizedWeights make_weights(std::uint32_t features, std::uint32_t k, bool block128,
                              std::uint32_t kp) {
    std::mt19937 rng(g_seed++ + features * 71u + k * 5u + (block128 ? 3u : 0u));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    QuantizedWeights qw;
    qw.codes_rowmajor.assign(std::size_t(features) * k, 0u);
    qw.as_bf16.assign(std::size_t(features) * k, bf16_t{0u});
    qw.k_blocks = scale_k_blocks(kp);
    qw.n_blocks = block128 ? ((features + kActGroupK - 1u) / kActGroupK) : features;
    qw.scales.assign(std::size_t(qw.n_blocks) * (block128 ? qw.k_blocks : 1u), 0.0f);

    std::vector<float> w(std::size_t(features) * k);
    for (auto& v : w) v = dist(rng);

    if (!block128) {
        for (std::uint32_t n = 0; n < features; ++n) {
            float amax = 0.0f;
            for (std::uint32_t kk = 0; kk < k; ++kk)
                amax = std::fmax(amax, std::fabs(w[std::size_t(n) * k + kk]));
            const float sc = amax_scale(amax);
            const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
            qw.scales[n] = sc;
            for (std::uint32_t kk = 0; kk < k; ++kk) {
                const std::uint8_t c = (sc > 0.0f) ? e4m3_encode(w[std::size_t(n) * k + kk] * inv)
                                                   : 0u;
                qw.codes_rowmajor[std::size_t(n) * k + kk] = c;
                qw.as_bf16[std::size_t(n) * k + kk] = shelf::f32_to_bf16(e4m3_decode(c));
            }
        }
        return qw;
    }
    for (std::uint32_t nb = 0; nb < qw.n_blocks; ++nb) {
        for (std::uint32_t kb = 0; kb < qw.k_blocks; ++kb) {
            const std::uint32_t r0 = nb * kActGroupK, r1 = std::min(features, r0 + kActGroupK);
            const std::uint32_t c0 = kb * kActGroupK, c1 = std::min(k, c0 + kActGroupK);
            float amax = 0.0f;
            for (std::uint32_t n = r0; n < r1; ++n)
                for (std::uint32_t kk = c0; kk < c1; ++kk)
                    amax = std::fmax(amax, std::fabs(w[std::size_t(n) * k + kk]));
            const float sc = amax_scale(amax);
            const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
            qw.scales[std::size_t(nb) * qw.k_blocks + kb] = sc;
            for (std::uint32_t n = r0; n < r1; ++n)
                for (std::uint32_t kk = c0; kk < c1; ++kk) {
                    const std::uint8_t c =
                        (sc > 0.0f) ? e4m3_encode(w[std::size_t(n) * k + kk] * inv) : 0u;
                    qw.codes_rowmajor[std::size_t(n) * k + kk] = c;
                    qw.as_bf16[std::size_t(n) * k + kk] = shelf::f32_to_bf16(e4m3_decode(c));
                }
        }
    }
    return qw;
}

int run_gemm_case(const Case& c, bool golden_f64) {
    const std::uint32_t kp = pad_k(c.k);
    const QuantizedWeights qw = make_weights(c.features, c.k, c.block128, kp);

    std::mt19937 rng(g_seed++ + c.rows * 13u + c.act_group);
    std::uniform_real_distribution<float> adist(-4.0f, 4.0f);
    std::vector<bf16_t> x(std::size_t(c.rows) * c.k);
    for (auto& v : x) v = shelf::f32_to_bf16(adist(rng));

    std::vector<std::uint8_t> w_tiled(tiled_bytes(c.features, kp), 0u);
    if (!shelf::quant::fp8::preshuffle_codes(qw.codes_rowmajor.data(), c.features, c.k, kp,
                                             w_tiled.data()).ok()) {
        std::printf("FAIL: preshuffle\n");
        ++failures;
        return 1;
    }

    const std::uint32_t kblocks = act_scale_groups(kp, c.act_group);
    const std::size_t a_bytes = shelf::hipk::fp8_activation_bytes(c.rows, kp);
    const std::size_t a_scale_count = std::size_t(c.rows) * kblocks;

    std::uint8_t* dw = nullptr;
    float* dws = nullptr;
    bf16_t* dx = nullptr;
    std::uint8_t* dac = nullptr;
    float* das = nullptr;
    float* dout = nullptr;
    const std::size_t out_elems = std::size_t(c.rows) * c.features;
    HIP_OK(hipMalloc(&dw, w_tiled.size()));
    HIP_OK(hipMalloc(&dws, sizeof(float) * qw.scales.size()));
    HIP_OK(hipMalloc(&dx, x.size() * sizeof(bf16_t)));
    HIP_OK(hipMalloc(&dac, a_bytes));
    HIP_OK(hipMalloc(&das, sizeof(float) * a_scale_count));
    HIP_OK(hipMalloc(&dout, out_elems * sizeof(float)));
    HIP_OK(hipMemcpy(dw, w_tiled.data(), w_tiled.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dws, qw.scales.data(), sizeof(float) * qw.scales.size(),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dx, x.data(), x.size() * sizeof(bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dac, 0, a_bytes));
    HIP_OK(hipMemset(das, 0xFF, sizeof(float) * a_scale_count));
    HIP_OK(hipMemset(dout, 0, out_elems * sizeof(float)));

    HIP_OK(shelf::hipk::launch_activation_quantize_fp8(dx, dac, das, c.rows, c.k, kp, c.k,
                                                       c.act_group, nullptr));
    Fp8GemmScaleArgs sa;
    sa.act_group = c.act_group;
    sa.weight_block128 = c.block128;
    sa.activation_scale_stride = kblocks;
    sa.weight_scale_k_blocks = qw.k_blocks;

    Fp8GemmConfig cfg;
    const bool use_decode1 =
        (c.rows == 1u) && shelf::hipk::fp8_decode1_supported(1u, c.features, kp);
    if (use_decode1)
        cfg.id = Fp8GemmConfigId::Decode1;
    else
        cfg.id = (c.rows <= 16u) ? Fp8GemmConfigId::RowBlock1
                                 : (c.rows <= 32u ? Fp8GemmConfigId::RowBlock2
                                                  : Fp8GemmConfigId::RowBlock4);
    HIP_OK(shelf::hipk::launch_gemm_fp8_w8a8(cfg, dw, dws, dac, das, dout,
                                             Fp8GemmOutputDType::F32, c.rows, c.features, kp,
                                             c.features, sa, nullptr));
    std::vector<float> out(out_elems, 0.0f);
    HIP_OK(hipMemcpy(out.data(), dout, out_elems * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<std::uint8_t> a_codes(a_bytes, 0u);
    HIP_OK(hipMemcpy(a_codes.data(), dac, a_bytes, hipMemcpyDeviceToHost));
    std::vector<float> ascales(a_scale_count, 0.0f);
    HIP_OK(hipMemcpy(ascales.data(), das, sizeof(float) * a_scale_count,
                     hipMemcpyDeviceToHost));

    auto w_scale_of = [&](std::uint32_t n, std::uint32_t kk) -> double {
        if (!c.block128) return double(qw.scales[n]);
        return double(qw.scales[std::size_t(n / kActGroupK) * qw.k_blocks +
                                std::min(qw.k_blocks - 1u, kk / kActGroupK)]);
    };
    auto a_scale_of = [&](std::uint32_t r, std::uint32_t kk) -> double {
        const std::uint32_t kb = (c.act_group == 0u) ? 0u : std::min(kblocks - 1u, kk / kActGroupK);
        return double(ascales[std::size_t(r) * kblocks + kb]);
    };

    double worst_abs = 0.0, max_mag = 0.0, worst_work = 0.0;
    double worst_ref = 0.0, worst_got = 0.0;
    std::uint32_t worst_r = 0, worst_n = 0;
    if (golden_f64) {
        // スケールが切り替わる粒度でブロックを切り、ブロック内を f64 で累積する
        // （カーネルもブロック単位で掛けるため、比較の形が揃う）。
        //
        // 判定は **項の絶対値の総和に対する比**で見る。内積は打ち消しで最終値が
        // 項の総和より桁違いに小さくなることがあり（このテストの形状で 20-30 桁差
        // ではなく 1-2 桁ではなく、最大で約 25 倍の差）、そのとき要素の絶対値に対する
        // 相対誤差は f32 の累積誤差でも必然的に大きくなる。よって
        //   |got - ref| <= 1.0e-4 × Σ_k |a_code·w_code| × scales
        // （1.0e-4 ≈ 840 × f32 eps。K=4096 のランダムな符号の和で √K≈64、
        //   1 ブロック内の累積・ブロック間の累積で数百 eps を見て余裕を持たせた値）
        // を使う。要素ごとの相対誤差は参考表示にとどめる。
        for (std::uint32_t r = 0; r < c.rows; ++r) {
            for (std::uint32_t n = 0; n < c.features; ++n) {
                double acc = 0.0, sum_abs = 0.0;
                for (std::uint32_t kb = 0; kb < kblocks; ++kb) {
                    const std::uint32_t b0 = (c.act_group == 0u) ? 0u : kb * kActGroupK;
                    const std::uint32_t b1 = (c.act_group == 0u)
                                                 ? c.k
                                                 : std::min(c.k, (kb + 1u) * kActGroupK);
                    if (b0 >= b1) continue;
                    double part = 0.0, part_abs = 0.0;
                    for (std::uint32_t kk = b0; kk < b1; ++kk) {
                        const double aq = e4m3_decode(a_codes[tiled_offset(r, kk, kp)]);
                        const double wq =
                            e4m3_decode(qw.codes_rowmajor[std::size_t(n) * c.k + kk]);
                        part += aq * wq;
                        part_abs += std::fabs(aq * wq);
                    }
                    const double s = a_scale_of(r, b0) * w_scale_of(n, b0);
                    acc += part * s;
                    sum_abs += part_abs * s;
                }
                const double got = out[std::size_t(r) * c.features + n];
                const double err = std::fabs(acc - got);
                max_mag = std::max(max_mag, std::fabs(acc));
                if (sum_abs > 0.0 && err / sum_abs > worst_work) worst_work = err / sum_abs;
                if (err > worst_abs) {
                    worst_abs = err;
                    worst_ref = acc;
                    worst_got = got;
                    worst_r = r;
                    worst_n = n;
                }
            }
        }
        check(max_mag > 0.0 && worst_work <= 1.0e-4,
              "f64 golden に対する誤差が項の総和基準（f32 累積の範囲）内");
        if (!(max_mag > 0.0 && worst_work <= 1.0e-4)) {
            std::printf("  rows=%u N=%u K=%u act=%u blk=%d worst_abs=%.3e max|ref|=%.1f "
                        "worst_work=%.3e\n",
                        c.rows, c.features, c.k, c.act_group, int(c.block128), worst_abs,
                        max_mag, worst_work);
            std::printf("    最悪要素 r=%u n=%u ref=%.4f got=%.4f 相対=%.3e "
                        "（a_scale=%g w_scale=%g）\n",
                        worst_r, worst_n, worst_ref, worst_got,
                        std::fabs(worst_ref) > 1e-12 ? worst_abs / std::fabs(worst_ref) : 0.0,
                        a_scale_of(worst_r, 0), w_scale_of(worst_n, 0));
        }
    }

    // 既存 bf16 GEMM との突合（同じ量子化値を bf16 で積む -> 差は累積順のみ）
    {
        std::vector<bf16_t> axq(std::size_t(c.rows) * c.k, bf16_t{0u});
        for (std::uint32_t r = 0; r < c.rows; ++r)
            for (std::uint32_t kk = 0; kk < c.k; ++kk)
                axq[std::size_t(r) * c.k + kk] =
                    shelf::f32_to_bf16(e4m3_decode(a_codes[tiled_offset(r, kk, kp)]));
        bf16_t* dbw = nullptr;
        bf16_t* dbx = nullptr;
        float* dbo = nullptr;
        HIP_OK(hipMalloc(&dbw, qw.as_bf16.size() * sizeof(bf16_t)));
        HIP_OK(hipMalloc(&dbx, axq.size() * sizeof(bf16_t)));
        HIP_OK(hipMalloc(&dbo, out_elems * sizeof(float)));
        HIP_OK(hipMemcpy(dbw, qw.as_bf16.data(), qw.as_bf16.size() * sizeof(bf16_t),
                         hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(dbx, axq.data(), axq.size() * sizeof(bf16_t), hipMemcpyHostToDevice));
        HIP_OK(hipMemset(dbo, 0, out_elems * sizeof(float)));
        shelf::hipk::Bf16GemmConfig bcfg{
            c.rows >= 2u ? shelf::hipk::Bf16GemmConfigId::ExactRowsDot2
                         : shelf::hipk::Bf16GemmConfigId::ExactRows,
            static_cast<std::uint8_t>(c.rows)};   // exact_rows == rows（<= 64）が条件
        HIP_OK(shelf::hipk::launch_gemm_bf16(bcfg, dbw, dbx, dbo,
                                             shelf::hipk::Bf16GemmOutputDType::F32, c.rows,
                                             c.features, c.k, c.k, c.features, nullptr));
        std::vector<float> bout(out_elems, 0.0f);
        HIP_OK(hipMemcpy(bout.data(), dbo, out_elems * sizeof(float), hipMemcpyDeviceToHost));
        double w_abs = 0.0, mag = 0.0;
        if (kblocks == 1u && !c.block128) {
            // ブロック数が 1 のときだけ bf16 経路と素直に突合できる
            // （スケールが K に依らないので、fp8 側を割って比べられる）。
            for (std::uint32_t r = 0; r < c.rows; ++r)
                for (std::uint32_t n = 0; n < c.features; ++n) {
                    const double ref = double(bout[std::size_t(r) * c.features + n]);
                    const double got = out[std::size_t(r) * c.features + n] /
                                       (a_scale_of(r, 0) * w_scale_of(n, 0));
                    w_abs = std::max(w_abs, std::fabs(ref - got));
                    mag = std::max(mag, std::fabs(ref));
                }
            check(mag > 0.0 && w_abs <= 1.0e-4 * mag, "bf16 GEMM（同じ量子化値）との一致");
            if (!(mag > 0.0 && w_abs <= 1.0e-4 * mag))
                std::printf("  rows=%u N=%u K=%u bf16 突合 worst_abs=%.3e max=%.1f\n", c.rows,
                            c.features, c.k, w_abs, mag);
        }
        (void)hipFree(dbw); (void)hipFree(dbx); (void)hipFree(dbo);
    }

    // BF16 出力経路（rows=1）も見る
    if (c.rows == 1u) {
        bf16_t* dout_bf = nullptr;
        HIP_OK(hipMalloc(&dout_bf, sizeof(bf16_t) * c.features));
        HIP_OK(hipMemset(dout_bf, 0, sizeof(bf16_t) * c.features));
        HIP_OK(shelf::hipk::launch_gemm_fp8_w8a8(cfg, dw, dws, dac, das, dout_bf,
                                                 Fp8GemmOutputDType::BF16, 1u, c.features, kp,
                                                 c.features, sa, nullptr));
        std::vector<bf16_t> obf(c.features, bf16_t{0u});
        HIP_OK(hipMemcpy(obf.data(), dout_bf, sizeof(bf16_t) * c.features,
                         hipMemcpyDeviceToHost));
        double mag = 0.0;
        for (const float v : out) mag = std::max(mag, std::fabs(double(v)));
        bool ok = mag > 0.0;
        for (std::uint32_t n = 0; n < c.features; ++n) {
            const double ref = out[n];
            const double got = shelf::bf16_to_f32(obf[n]);
            if (std::fabs(ref - got) > 1.6e-2 * std::fabs(ref) + 1.0e-5 * mag) ok = false;
        }
        check(ok, "decode1 の BF16 出力が f32 出力と丸め範囲で一致");
        (void)hipFree(dout_bf);
    }

    (void)hipFree(dw); (void)hipFree(dws); (void)hipFree(dx); (void)hipFree(dac);
    (void)hipFree(das); (void)hipFree(dout);
    std::printf("  case rows=%u N=%u K=%u act=%u blk=%d ok (worst_abs=%.2e max=%.1f)\n", c.rows,
                c.features, c.k, c.act_group, int(c.block128), worst_abs, max_mag);
    return 0;
}

int test_supported_predicates() {
    using namespace shelf::quant::fp8;
    check(shelf::hipk::fp8_gemm_shape_supported(1u, 4096u, 4096u), "rows=1 受理");
    check(shelf::hipk::fp8_gemm_shape_supported(64u, 12288u, 4096u), "rows=64 受理");
    check(!shelf::hipk::fp8_gemm_shape_supported(65u, 4096u, 4096u), "rows>64 は不受理");
    check(!shelf::hipk::fp8_gemm_shape_supported(4u, 4096u, 4100u), "K が 32 倍数でないと不受理");
    check(shelf::hipk::fp8_decode1_supported(1u, 1024u, 4096u), "decode1 受理");
    check(!shelf::hipk::fp8_decode1_supported(2u, 1024u, 4096u), "decode1 は rows=1 のみ");
    check(!shelf::hipk::fp8_decode1_supported(1u, 1020u, 4096u), "decode1 は N が 16 の倍数必要");
    check(shelf::hipk::fp8_activation_bytes(1u, 32u) == 512u, "活性化バッファは 16 行単位で確保");

    Fp8GemmScaleArgs sa;
    sa.act_group = kActGroupK; sa.weight_block128 = false;
    sa.activation_scale_stride = act_scale_groups(4096u, kActGroupK);
    sa.weight_scale_k_blocks = scale_k_blocks(4096u);
    check(shelf::hipk::fp8_scale_args_supported(sa, 1u, 4096u, 4096u), "act=128 / chan 受理");
    sa.activation_scale_stride = 1u;
    check(!shelf::hipk::fp8_scale_args_supported(sa, 1u, 4096u, 4096u),
          "stride が粒度と不一致なら不受理");
    sa.activation_scale_stride = act_scale_groups(4096u, kActGroupK);
    sa.weight_block128 = true;
    check(shelf::hipk::fp8_scale_args_supported(sa, 1u, 4096u, 4096u), "act=128 / block128 受理");
    sa.weight_scale_k_blocks = 1u;
    check(!shelf::hipk::fp8_scale_args_supported(sa, 1u, 4096u, 4096u),
          "block128 の k_blocks 不一致は不受理");
    sa.weight_scale_k_blocks = scale_k_blocks(4096u);
    check(!shelf::hipk::fp8_scale_args_supported(sa, 1u, 4096u, 4128u),
          "K が 128 倍数でない場所で act=128 は不受理");

    check(shelf::hipk::activation_quantize_fp8_supported(4096u, 4096u, kActGroupK),
          "量子化カーネル act=128 受理");
    check(shelf::hipk::activation_quantize_fp8_supported(96u, 96u, 0u), "量子化カーネル act=0 受理");
    check(!shelf::hipk::activation_quantize_fp8_supported(96u, 96u, kActGroupK),
          "K が 128 未満で act=128 は不受理");
    check(!shelf::hipk::activation_quantize_fp8_supported(4096u, 4096u, 64u),
          "粒度は 0 か 128 のみ");
    return 0;
}

}  // namespace

int main() {
    HIP_OK(hipSetDevice(0));
    if (test_supported_predicates() != 0) return 1;
    for (std::uint32_t ag : {128u, 0u}) {
        if (test_activation_quantize(1u, 4096u, ag) != 0) return 1;
        if (test_activation_quantize(3u, 12288u, ag) != 0) return 1;
        if (test_activation_quantize(64u, 4096u, ag) != 0) return 1;
    }

    // 境界形状（f64 golden）: 粒度の 4 組合せを.rows の境界も交えて見る
    for (Case c : {Case{1u, 1024u, 4096u, 128u, false}, Case{1u, 1024u, 4096u, 0u, false},
                   Case{1u, 1024u, 4096u, 128u, true},  Case{1u, 1024u, 4096u, 0u, true},
                   Case{1u, 16u, 32u, 0u, false},       Case{1u, 271u, 96u, 0u, true},
                   Case{2u, 64u, 256u, 128u, false},    Case{15u, 33u, 96u, 0u, false},
                   Case{16u, 256u, 256u, 128u, false},  Case{16u, 256u, 256u, 128u, true},
                   Case{17u, 256u, 12288u, 128u, false}, Case{33u, 100u, 256u, 128u, true},
                   Case{64u, 1024u, 4096u, 128u, false}, Case{64u, 1024u, 4096u, 128u, true},
                   Case{64u, 1024u, 4096u, 0u, true}})
        if (run_gemm_case(c, /*golden_f64=*/true) != 0) return 1;

    // 実モデル形状（bf16 GEMM 突合は効かないので golden は軽量なものだけ、
    // ここでは decode1 と MB4 の速度形状を f64 なしで通す）
    for (Case c : {Case{1u, 12288u, 4096u, 128u, false}, Case{1u, 4096u, 12288u, 128u, false},
                   Case{1u, 8192u, 4096u, 128u, false}, Case{64u, 12288u, 4096u, 128u, false},
                   Case{64u, 4096u, 12288u, 128u, false}})
        if (run_gemm_case(c, /*golden_f64=*/false) != 0) return 1;

    std::printf("test_hip_gemm_fp8: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
