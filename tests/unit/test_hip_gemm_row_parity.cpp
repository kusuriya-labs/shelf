// shelf unit test — decode1（GEMV）とバッチ GEMM の数値差（N01）。
//
// decode の並列数が 1 のときは GEMV（decode1）、2 以上のときはバッチ GEMM を通る。
// 同じ 1 行の活性化を両方で計算し、**加算順の違いだけで説明できる差**に収まるかを
// 測る。ここが大きいと greedy の生成列が並列数で分岐する（N01 の動機）。
//
// 実際の実行器は N01 で「少数並列も行ごとに decode1」へ揃えたため、この差は
// 既定では現れない。このテストは (a) 揃える前の差の大きさを記録し、(b) 将来
// バッチ経路を触ったときに差が広がっていないことを検出するための基準になる。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <shelf/backend/hip/kernels/elementwise/activation_quantize_fp8.h>
#include <shelf/backend/hip/kernels/gemm/linear_fp8.h>
#include <shelf/backend/hip/kernels/gemm/linear_mxfp4.h>
#include <shelf/quant/fp8_finegrained.hpp>
#include <shelf/quant/fp8_perchannel.hpp>
#include <shelf/quant/mxfp4.hpp>

namespace {

using namespace shelf;
using namespace shelf::hipk;

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

void check_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        ++failures;
        std::printf("FAIL: %s (%s)\n", what, hipGetErrorString(e));
    }
}

struct Diff {
    double worst_abs = 0.0;
    double worst_rel = 0.0;
    double mag = 0.0;
    int argmax_mismatch = 0;
};

// バッチ GEMM の出力（row ごと）と 1 行 decode1 の出力を比べる。
Diff compare_rows(const std::vector<float>& single, const std::vector<float>& batch,
                  std::uint32_t rows, std::uint32_t features) {
    Diff d;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::uint32_t best_s = 0, best_b = 0;
        for (std::uint32_t n = 0; n < features; ++n) {
            const std::size_t i = std::size_t(r) * features + n;
            const double s = single[i], b = batch[i];
            const double diff = std::fabs(s - b);
            d.mag = std::fmax(d.mag, std::fabs(b));
            if (diff > d.worst_abs) d.worst_abs = diff;
            if (std::fabs(single[i]) > std::fabs(single[std::size_t(r) * features + best_s]))
                best_s = n;
            if (std::fabs(batch[i]) > std::fabs(batch[std::size_t(r) * features + best_b]))
                best_b = n;
        }
        if (best_s != best_b) ++d.argmax_mismatch;
    }
    d.worst_rel = d.mag > 0.0 ? d.worst_abs / d.mag : 0.0;
    return d;
}

void report(const char* label, const Diff& d, double tol) {
    std::printf("    %s: worst abs=%.4e rel=%.4e（|ref|max=%.3e）argmax 不一致 %d 行\n", label,
                d.worst_abs, d.worst_rel, d.mag, d.argmax_mismatch);
    check(d.worst_rel < tol, "decode1 とバッチ GEMM の差が丸めの範囲");
    check(d.argmax_mismatch == 0, "decode1 とバッチ GEMM の argmax が一致");
}

// ---- FP8 W8A8（per-channel スケール・活性化 128 K グループ）----
void run_fp8(std::uint32_t features, std::uint32_t k, std::uint32_t seed) {
    const std::uint32_t kp = quant::fp8::pad_k(k);
    const std::uint32_t rows = 4;
    std::printf("  fp8: features=%u k=%u kp=%u\n", features, k, kp);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> w(std::size_t(features) * k);
    for (auto& v : w) v = dist(rng);
    std::vector<std::uint8_t> wc(kp * features, 0u);
    std::vector<float> ws(features, 0.0f);
    for (std::uint32_t n = 0; n < features; ++n) {
        float amax = 0.0f;
        for (std::uint32_t c = 0; c < k; ++c)
            amax = std::fmax(amax, std::fabs(w[std::size_t(n) * k + c]));
        const float sc = quant::fp8::amax_scale(amax);
        ws[n] = sc;
        const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
        for (std::uint32_t c = 0; c < k; ++c) {
            const std::size_t i = std::size_t(n) * kp + c;
            wc[i] = sc > 0.0f ? quant::fp8::e4m3_encode(w[std::size_t(n) * k + c] * inv) : 0u;
        }
    }
    std::vector<std::uint8_t> w_tiled(quant::fp8::tiled_bytes(features, kp), 0u);
    check(quant::fp8::preshuffle_codes(wc.data(), features, k, kp, w_tiled.data()).ok(),
          "fp8 preshuffle");

    std::vector<bf16_t> x(std::size_t(rows) * k);
    for (auto& v : x) v = f32_to_bf16(dist(rng));

    std::uint8_t* dw = nullptr;
    float* dws = nullptr;
    bf16_t* dx = nullptr;
    std::uint8_t* dac = nullptr;
    float* das = nullptr;
    float* dout = nullptr;
    const std::size_t out_elems = std::size_t(rows) * features;
    check_hip(hipMalloc(&dw, w_tiled.size()), "malloc w");
    check_hip(hipMalloc(&dws, sizeof(float) * ws.size()), "malloc ws");
    check_hip(hipMalloc(&dx, x.size() * sizeof(bf16_t)), "malloc x");
    const std::size_t row_bytes = quant::fp8::tiled_bytes(1u, kp);
    const std::size_t groups = quant::fp8::fine::act_scale_groups(kp, 128u);
    check_hip(hipMalloc(&dac, row_bytes * rows), "malloc ac");
    check_hip(hipMalloc(&das, sizeof(float) * groups * rows), "malloc as");
    check_hip(hipMalloc(&dout, sizeof(float) * out_elems), "malloc out");
    check_hip(hipMemcpy(dw, w_tiled.data(), w_tiled.size(), hipMemcpyHostToDevice), "copy w");
    check_hip(hipMemcpy(dws, ws.data(), sizeof(float) * ws.size(), hipMemcpyHostToDevice),
              "copy ws");
    check_hip(hipMemcpy(dx, x.data(), x.size() * sizeof(bf16_t), hipMemcpyHostToDevice),
              "copy x");

    Fp8GemmScaleArgs sa;
    sa.act_group = 128u;
    sa.weight_block128 = false;
    sa.activation_scale_stride = static_cast<std::uint32_t>(groups);
    sa.weight_scale_k_blocks = quant::fp8::fine::scale_k_blocks(kp);

    // 1 行ずつ decode1（出力はデバイス。まとめて host へ戻す）。
    std::vector<float> single(out_elems, 0.0f);
    for (std::uint32_t r = 0; r < rows; ++r) {
        check_hip(launch_activation_quantize_fp8(
                      dx + std::size_t(r) * k, dac + r * row_bytes, das + r * groups, 1u, k, kp,
                      k, 128u, nullptr),
                  "quantize 1 row (fp8)");
        check_hip(launch_gemm_fp8_w8a8(Fp8GemmConfig{Fp8GemmConfigId::Decode1}, dw, dws,
                                       dac + r * row_bytes, das + r * groups,
                                       dout + std::size_t(r) * features,
                                       Fp8GemmOutputDType::F32, 1u, features, kp, features, sa,
                                       nullptr),
                  "decode1 (fp8)");
    }
    check_hip(hipMemcpy(single.data(), dout, sizeof(float) * out_elems,
                        hipMemcpyDeviceToHost),
              "copy single (fp8)");

    // 4 行をバッチ GEMM（prefill と同じ RowBlock1）。
    check_hip(launch_activation_quantize_fp8(dx, dac, das, rows, k, kp, k, 128u, nullptr),
              "quantize rows (fp8)");
    std::vector<float> batch(out_elems, 0.0f);
    check_hip(launch_gemm_fp8_w8a8(Fp8GemmConfig{Fp8GemmConfigId::RowBlock1}, dw, dws, dac, das,
                                   dout, Fp8GemmOutputDType::F32, rows, features, kp,
                                   features, sa, nullptr),
              "batch gemm (fp8)");
    check_hip(hipMemcpy(batch.data(), dout, sizeof(float) * out_elems,
                        hipMemcpyDeviceToHost),
              "copy batch (fp8)");
    report("fp8 W8A8", compare_rows(single, batch, rows, features), 1.0e-3);
}

// ---- MXFP4 W4A8（E2M1 重み × E4M3 活性化）----
void run_mxfp4(std::uint32_t features, std::uint32_t k, std::uint32_t seed,
               std::uint32_t rows = 4u) {
    const std::uint32_t kp = quant::mxfp4::pad_k(quant::fp8::pad_k(k));
    std::printf("  mxfp4: features=%u k=%u kp=%u\n", features, k, kp);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> w(std::size_t(features) * k);
    for (auto& v : w) v = dist(rng);
    std::vector<std::uint8_t> wc_canon(quant::mxfp4::canonical_code_bytes(features, kp), 0u);
    std::vector<std::uint8_t> ws(quant::mxfp4::scale_bytes(features, kp), 0u);
    for (std::uint32_t n = 0; n < features; ++n) {
        check(quant::mxfp4::quantize_row(w.data() + std::size_t(n) * k, k, kp,
                                         wc_canon.data() + std::size_t(n) * (kp / 2u),
                                         ws.data() + std::size_t(n) * (kp / 32u))
                  .ok(),
              "mxfp4 quantize_row");
    }
    std::vector<std::uint8_t> w_tiled(quant::mxfp4::tiled_bytes(features, kp), 0u);
    check(quant::mxfp4::preshuffle_codes(wc_canon.data(), features, k, kp, w_tiled.data()).ok(),
          "mxfp4 preshuffle");

    std::vector<bf16_t> x(std::size_t(rows) * k);
    for (auto& v : x) v = f32_to_bf16(dist(rng));

    std::uint8_t* dw = nullptr;
    std::uint8_t* dws = nullptr;
    bf16_t* dx = nullptr;
    std::uint8_t* dac = nullptr;
    float* das = nullptr;
    float* dout = nullptr;
    const std::size_t out_elems = std::size_t(rows) * features;
    const std::size_t row_bytes = quant::fp8::tiled_bytes(1u, kp);
    const std::size_t groups = quant::fp8::fine::act_scale_groups(kp, 128u);
    check_hip(hipMalloc(&dw, w_tiled.size()), "malloc w");
    check_hip(hipMalloc(&dws, ws.size()), "malloc ws");
    check_hip(hipMalloc(&dx, x.size() * sizeof(bf16_t)), "malloc x");
    check_hip(hipMalloc(&dac, row_bytes * rows), "malloc ac");
    check_hip(hipMalloc(&das, sizeof(float) * groups * rows), "malloc as");
    check_hip(hipMalloc(&dout, sizeof(float) * out_elems), "malloc out");
    check_hip(hipMemcpy(dw, w_tiled.data(), w_tiled.size(), hipMemcpyHostToDevice), "copy w");
    check_hip(hipMemcpy(dws, ws.data(), ws.size(), hipMemcpyHostToDevice), "copy ws");
    check_hip(hipMemcpy(dx, x.data(), x.size() * sizeof(bf16_t), hipMemcpyHostToDevice),
              "copy x");

    Mxfp4GemmScaleArgs sa;
    sa.act_group = 128u;
    sa.activation_scale_stride = static_cast<std::uint32_t>(groups);
    sa.weight_scale_stride = kp / quant::mxfp4::kBlockK;

    std::vector<float> single(out_elems, 0.0f);
    for (std::uint32_t r = 0; r < rows; ++r) {
        check_hip(launch_activation_quantize_fp8(
                      dx + std::size_t(r) * k, dac + r * row_bytes, das + r * groups, 1u, k, kp,
                      k, 128u, nullptr),
                  "quantize 1 row (mxfp4)");
        check_hip(launch_gemm_mxfp4_w4a8_decode1(
                      dw, dws, dac + r * row_bytes, das + r * groups,
                      dout + std::size_t(r) * features, Mxfp4GemmOutputDType::F32, 1u,
                      features, kp, features, sa, nullptr),
                  "decode1 (mxfp4)");
    }
    check_hip(hipMemcpy(single.data(), dout, sizeof(float) * out_elems,
                        hipMemcpyDeviceToHost),
              "copy single (mxfp4)");

    check_hip(launch_activation_quantize_fp8(dx, dac, das, rows, k, kp, k, 128u, nullptr),
              "quantize rows (mxfp4)");
    // 量子化そのものの実行間決定性（同じ入力で 2 回 → ビット一致）。
    {
        const std::size_t code_bytes = quant::fp8::tiled_bytes(rows, kp);
        std::vector<std::uint8_t> c1(code_bytes), c2(code_bytes);
        std::vector<float> s1(groups * rows), s2(groups * rows);
        check_hip(hipMemcpy(c1.data(), dac, c1.size(), hipMemcpyDeviceToHost), "q copy c1");
        check_hip(hipMemcpy(s1.data(), das, sizeof(float) * s1.size(), hipMemcpyDeviceToHost),
                  "q copy s1");
        check_hip(launch_activation_quantize_fp8(dx, dac, das, rows, k, kp, k, 128u, nullptr),
                  "quantize rows again (mxfp4)");
        check_hip(hipMemcpy(c2.data(), dac, c2.size(), hipMemcpyDeviceToHost), "q copy c2");
        check_hip(hipMemcpy(s2.data(), das, sizeof(float) * s2.size(), hipMemcpyDeviceToHost),
                  "q copy s2");
        check(c1 == c2, "活性化量子化（rows>1）は 2 回でコードがビット一致");
        check(std::memcmp(s1.data(), s2.data(), sizeof(float) * s1.size()) == 0,
              "活性化量子化（rows>1）は 2 回でスケールがビット一致");
    }
    std::vector<float> batch(out_elems, 0.0f);
    check_hip(launch_gemm_mxfp4_w4a8_prefill(dw, dws, dac, das, dout,
                                             Mxfp4GemmOutputDType::F32, rows, features, kp,
                                             features, sa, nullptr),
              "batch gemm (mxfp4)");
    check_hip(hipMemcpy(batch.data(), dout, sizeof(float) * out_elems,
                        hipMemcpyDeviceToHost),
              "copy batch (mxfp4)");
    // 同じ入力で 2 回目（実行間の決定性）。
    std::vector<float> again(out_elems, 0.0f);
    check_hip(launch_gemm_mxfp4_w4a8_prefill(dw, dws, dac, das, dout,
                                             Mxfp4GemmOutputDType::F32, rows, features, kp,
                                             features, sa, nullptr),
              "batch gemm again (mxfp4)");
    check_hip(hipMemcpy(again.data(), dout, sizeof(float) * out_elems,
                        hipMemcpyDeviceToHost),
              "copy batch again (mxfp4)");
    if (std::memcmp(batch.data(), again.data(), sizeof(float) * out_elems) != 0) {
        double w = 0.0;
        for (std::size_t i = 0; i < batch.size(); ++i)
            w = std::fmax(w, std::fabs(double(batch[i]) - double(again[i])));
        std::printf("    !! mxfp4 prefill が 2 回で不一致（rows=%u 最大差 %.3e）\n", rows, w);
    }
    check(std::memcmp(batch.data(), again.data(), sizeof(float) * out_elems) == 0,
          "mxfp4 prefill は 2 回実行でビット一致");
    report("mxfp4 W4A8", compare_rows(single, batch, rows, features), 1.0e-3);
}

// ---- N07: FP8 の 2〜4 行 decode（重み共有）が decode1 とビット一致するか ----
void run_fp8_decode_n(std::uint32_t features, std::uint32_t k, std::uint32_t seed,
                      bool block128 = false) {
    const std::uint32_t kp = quant::fp8::pad_k(k);
    const std::uint32_t max_rows = 4u;
    std::printf("  fp8 decode_n: features=%u k=%u kp=%u block128=%d\n", features, k, kp,
                block128 ? 1 : 0);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> w(std::size_t(features) * k);
    for (auto& v : w) v = dist(rng);
    std::vector<std::uint8_t> wc(kp * features, 0u);
    std::vector<float> ws;
    const std::uint32_t kblk = quant::fp8::fine::scale_k_blocks(kp);
    if (block128) {
        const std::uint32_t nblk = (features + 127u) / 128u;
        ws.assign(std::size_t(nblk) * kblk, 0.0f);
        for (std::uint32_t nb = 0; nb < nblk; ++nb) {
            for (std::uint32_t kb = 0; kb < kblk; ++kb) {
                float amax = 0.0f;
                for (std::uint32_t n = nb * 128u; n < std::min(features, (nb + 1u) * 128u); ++n)
                    for (std::uint32_t c = kb * 128u; c < std::min(k, (kb + 1u) * 128u); ++c)
                        amax = std::fmax(amax, std::fabs(w[std::size_t(n) * k + c]));
                const float sc = quant::fp8::amax_scale(amax);
                ws[std::size_t(nb) * kblk + kb] = sc;
                const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
                for (std::uint32_t n = nb * 128u; n < std::min(features, (nb + 1u) * 128u); ++n)
                    for (std::uint32_t c = kb * 128u; c < std::min(k, (kb + 1u) * 128u); ++c)
                        wc[std::size_t(n) * kp + c] =
                            sc > 0.0f ? quant::fp8::e4m3_encode(w[std::size_t(n) * k + c] * inv)
                                      : 0u;
            }
        }
    } else {
        ws.assign(features, 0.0f);
        for (std::uint32_t n = 0; n < features; ++n) {
            float amax = 0.0f;
            for (std::uint32_t c = 0; c < k; ++c)
                amax = std::fmax(amax, std::fabs(w[std::size_t(n) * k + c]));
            const float sc = quant::fp8::amax_scale(amax);
            ws[n] = sc;
            const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
            for (std::uint32_t c = 0; c < k; ++c)
                wc[std::size_t(n) * kp + c] =
                    sc > 0.0f ? quant::fp8::e4m3_encode(w[std::size_t(n) * k + c] * inv) : 0u;
        }
    }
    std::vector<std::uint8_t> w_tiled(quant::fp8::tiled_bytes(features, kp), 0u);
    check(quant::fp8::preshuffle_codes(wc.data(), features, k, kp, w_tiled.data()).ok(),
          "fp8 preshuffle (decode_n)");

    std::vector<bf16_t> x(std::size_t(max_rows) * k);
    for (auto& v : x) v = f32_to_bf16(dist(rng));

    std::uint8_t* dw = nullptr;
    float* dws = nullptr;
    bf16_t* dx = nullptr;
    std::uint8_t* dac = nullptr;
    float* das = nullptr;
    float* dout = nullptr;
    const std::size_t row_bytes = quant::fp8::tiled_bytes(1u, kp);
    const std::size_t groups = quant::fp8::fine::act_scale_groups(kp, 128u);
    const std::size_t out_elems = std::size_t(max_rows) * features;
    check_hip(hipMalloc(&dw, w_tiled.size()), "malloc w (fp8 decode_n)");
    check_hip(hipMalloc(&dws, sizeof(float) * ws.size()), "malloc ws (fp8 decode_n)");
    check_hip(hipMalloc(&dx, x.size() * sizeof(bf16_t)), "malloc x (fp8 decode_n)");
    check_hip(hipMalloc(&dac, row_bytes * max_rows), "malloc ac (fp8 decode_n)");
    check_hip(hipMalloc(&das, sizeof(float) * groups * max_rows), "malloc as (fp8 decode_n)");
    check_hip(hipMalloc(&dout, sizeof(float) * out_elems), "malloc out (fp8 decode_n)");
    check_hip(hipMemcpy(dw, w_tiled.data(), w_tiled.size(), hipMemcpyHostToDevice),
              "copy w (fp8 decode_n)");
    check_hip(hipMemcpy(dws, ws.data(), sizeof(float) * ws.size(), hipMemcpyHostToDevice),
              "copy ws (fp8 decode_n)");
    check_hip(hipMemcpy(dx, x.data(), x.size() * sizeof(bf16_t), hipMemcpyHostToDevice),
              "copy x (fp8 decode_n)");

    Fp8GemmScaleArgs sa;
    sa.act_group = 128u;
    sa.weight_block128 = block128;
    sa.activation_scale_stride = static_cast<std::uint32_t>(groups);
    sa.weight_scale_k_blocks = kblk;

    for (std::uint32_t r = 0; r < max_rows; ++r)
        check_hip(launch_activation_quantize_fp8(dx + std::size_t(r) * k, dac + r * row_bytes,
                                                 das + r * groups, 1u, k, kp, k, 128u,
                                                 nullptr),
                  "quantize 1 row (fp8 decode_n)");

    for (std::uint32_t rows = 2u; rows <= max_rows; ++rows) {
        std::vector<float> ref(out_elems, 0.0f);
        std::vector<float> got(out_elems, 0.0f);
        for (std::uint32_t r = 0; r < rows; ++r)
            check_hip(launch_gemm_fp8_w8a8(Fp8GemmConfig{Fp8GemmConfigId::Decode1}, dw, dws,
                                           dac + r * row_bytes, das + r * groups,
                                           dout + std::size_t(r) * features,
                                           Fp8GemmOutputDType::F32, 1u, features, kp, features,
                                           sa, nullptr),
                      "decode1 (fp8 decode_n ref)");
        check_hip(hipMemcpy(ref.data(), dout, sizeof(float) * out_elems,
                            hipMemcpyDeviceToHost),
                  "copy ref (fp8 decode_n)");
        check_hip(launch_gemm_fp8_w8a8_decode_n(dw, dws, dac, das, dout,
                                                Fp8GemmOutputDType::F32, rows, features, kp,
                                                features, sa, nullptr),
                  "fp8 decode_n");
        check_hip(hipMemcpy(got.data(), dout, sizeof(float) * out_elems,
                            hipMemcpyDeviceToHost),
                  "copy got (fp8 decode_n)");
        const std::size_t n = std::size_t(rows) * features;
        if (std::memcmp(ref.data(), got.data(), sizeof(float) * n) != 0) {
            double wmax = 0.0;
            std::size_t at = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = std::fabs(double(ref[i]) - double(got[i]));
                if (d > wmax) { wmax = d; at = i; }
            }
            std::printf("    !! fp8 decode_n rows=%u が decode1 と不一致（最大差 %.3e @%zu）\n",
                        rows, wmax, at);
        }
        check(std::memcmp(ref.data(), got.data(), sizeof(float) * n) == 0,
              "fp8 decode_n は行ごとの decode1 とビット一致");
    }
}

// ---- N07: 2〜4 行の decode GEMV が「行ごとの decode1」とビット一致するか ----
void run_mxfp4_decode_n(std::uint32_t features, std::uint32_t k, std::uint32_t seed) {
    const std::uint32_t kp = quant::mxfp4::pad_k(quant::fp8::pad_k(k));
    std::printf("  mxfp4 decode_n: features=%u k=%u kp=%u\n", features, k, kp);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> w(std::size_t(features) * k);
    for (auto& v : w) v = dist(rng);
    std::vector<std::uint8_t> wc_canon(quant::mxfp4::canonical_code_bytes(features, kp), 0u);
    std::vector<std::uint8_t> ws(quant::mxfp4::scale_bytes(features, kp), 0u);
    for (std::uint32_t n = 0; n < features; ++n)
        check(quant::mxfp4::quantize_row(w.data() + std::size_t(n) * k, k, kp,
                                         wc_canon.data() + std::size_t(n) * (kp / 2u),
                                         ws.data() + std::size_t(n) * (kp / 32u))
                  .ok(),
              "mxfp4 quantize_row (decode_n)");
    std::vector<std::uint8_t> w_tiled(quant::mxfp4::tiled_bytes(features, kp), 0u);
    check(quant::mxfp4::preshuffle_codes(wc_canon.data(), features, k, kp, w_tiled.data()).ok(),
          "mxfp4 preshuffle (decode_n)");

    const std::uint32_t max_rows = 4u;
    std::vector<bf16_t> x(std::size_t(max_rows) * k);
    for (auto& v : x) v = f32_to_bf16(dist(rng));

    std::uint8_t* dw = nullptr;
    std::uint8_t* dws = nullptr;
    bf16_t* dx = nullptr;
    std::uint8_t* dac = nullptr;
    float* das = nullptr;
    float* dout = nullptr;
    const std::size_t row_bytes = quant::fp8::tiled_bytes(1u, kp);
    const std::size_t groups = quant::fp8::fine::act_scale_groups(kp, 128u);
    const std::size_t out_elems = std::size_t(max_rows) * features;
    check_hip(hipMalloc(&dw, w_tiled.size()), "malloc w (decode_n)");
    check_hip(hipMalloc(&dws, ws.size()), "malloc ws (decode_n)");
    check_hip(hipMalloc(&dx, x.size() * sizeof(bf16_t)), "malloc x (decode_n)");
    check_hip(hipMalloc(&dac, row_bytes * max_rows), "malloc ac (decode_n)");
    check_hip(hipMalloc(&das, sizeof(float) * groups * max_rows), "malloc as (decode_n)");
    check_hip(hipMalloc(&dout, sizeof(float) * out_elems), "malloc out (decode_n)");
    check_hip(hipMemcpy(dw, w_tiled.data(), w_tiled.size(), hipMemcpyHostToDevice),
              "copy w (decode_n)");
    check_hip(hipMemcpy(dws, ws.data(), ws.size(), hipMemcpyHostToDevice),
              "copy ws (decode_n)");
    check_hip(hipMemcpy(dx, x.data(), x.size() * sizeof(bf16_t), hipMemcpyHostToDevice),
              "copy x (decode_n)");

    Mxfp4GemmScaleArgs sa;
    sa.act_group = 128u;
    sa.activation_scale_stride = static_cast<std::uint32_t>(groups);
    sa.weight_scale_stride = kp / quant::mxfp4::kBlockK;

    // 行ごとに量子化して 1 行ぶんの断片順ブロックを並べる（step_executor と同じ形）。
    for (std::uint32_t r = 0; r < max_rows; ++r)
        check_hip(launch_activation_quantize_fp8(dx + std::size_t(r) * k, dac + r * row_bytes,
                                                 das + r * groups, 1u, k, kp, k, 128u,
                                                 nullptr),
                  "quantize 1 row (decode_n)");

    for (std::uint32_t rows = 2u; rows <= max_rows; ++rows) {
        std::vector<float> ref(out_elems, 0.0f);
        std::vector<float> got(out_elems, 0.0f);
        for (std::uint32_t r = 0; r < rows; ++r)
            check_hip(launch_gemm_mxfp4_w4a8_decode1(
                          dw, dws, dac + r * row_bytes, das + r * groups,
                          dout + std::size_t(r) * features, Mxfp4GemmOutputDType::F32, 1u,
                          features, kp, features, sa, nullptr),
                      "decode1 (decode_n ref)");
        check_hip(hipMemcpy(ref.data(), dout, sizeof(float) * out_elems,
                            hipMemcpyDeviceToHost),
                  "copy ref (decode_n)");
        check_hip(launch_gemm_mxfp4_w4a8_decode_n(
                      dw, dws, dac, das, dout, Mxfp4GemmOutputDType::F32, rows, features, kp,
                      features, sa, nullptr),
                  "decode_n");
        check_hip(hipMemcpy(got.data(), dout, sizeof(float) * out_elems,
                            hipMemcpyDeviceToHost),
                  "copy got (decode_n)");
        const std::size_t n = std::size_t(rows) * features;
        if (std::memcmp(ref.data(), got.data(), sizeof(float) * n) != 0) {
            double wmax = 0.0;
            std::size_t at = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = std::fabs(double(ref[i]) - double(got[i]));
                if (d > wmax) { wmax = d; at = i; }
            }
            std::printf("    !! decode_n rows=%u が decode1 と不一致（最大差 %.3e @%zu）\n",
                        rows, wmax, at);
        }
        check(std::memcmp(ref.data(), got.data(), sizeof(float) * n) == 0,
              "decode_n は行ごとの decode1 とビット一致");
    }
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }
    std::printf("test_hip_gemm_row_parity: decode1(GEMV) とバッチ GEMM の差（N01）\n");
    run_fp8(1024u, 2048u, 1u);
    run_fp8(3584u, 4096u, 2u);
    run_mxfp4(1024u, 2048u, 3u);
    run_mxfp4(3584u, 4096u, 4u);
    run_mxfp4(10240u, 5120u, 5u, 64u);   // 27B の qkv 相当（rows=64）
    run_mxfp4(17408u, 5120u, 6u, 64u);   // 27B の gate/up 相当
    run_mxfp4(5120u, 17408u, 7u, 64u);   // 27B の down 相当
    run_mxfp4_decode_n(1024u, 2048u, 8u);
    run_mxfp4_decode_n(3584u, 4096u, 9u);
    run_mxfp4_decode_n(17408u, 5120u, 10u);   // 27B の gate/up 相当
    run_fp8_decode_n(1024u, 2048u, 11u);
    run_fp8_decode_n(3584u, 4096u, 12u);
    run_fp8_decode_n(4096u, 4096u, 13u, true);   // block128 スケール
    if (failures == 0) std::printf("test_hip_gemm_row_parity: OK\n");
    return failures == 0 ? 0 : 1;
}
