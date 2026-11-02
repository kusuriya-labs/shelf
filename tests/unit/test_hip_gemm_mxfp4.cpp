// shelf unit test — backend/hip の MXFP4 W4A8 GEMV（E2M1 重み × E4M3 活性化）。
//
// 参照は CPU 側の素朴な実装（mxfp4 の逆量子化 × fp8 の逆量子化の内積、f64 で累加）。
// カーネルは FP8 WMMA で f32 累加し、32 K ごとに E8M0・128 K ごとに活性化スケールを
// 掛ける。差は累加順だけなので相対誤差は 1e-5 程度に収まるはず。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <shelf/backend/hip/kernels/gemm/linear_mxfp4.h>
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

// 1 行の活性化を fp8 へ量子化する（128 K グループの動的スケール。既存の fp8 経路と同じ）。
void quantize_activation_row(const std::vector<float>& x, std::uint32_t k_padded,
                             std::uint32_t act_group, std::vector<std::uint8_t>& codes,
                             std::vector<float>& scales) {
    const std::uint32_t groups = act_group ? (k_padded / act_group) : 1u;
    scales.assign(groups, 0.0f);
    codes.assign(k_padded, 0u);
    for (std::uint32_t g = 0; g < groups; ++g) {
        const std::uint32_t begin = g * (act_group ? act_group : k_padded);
        const std::uint32_t end = begin + (act_group ? act_group : k_padded);
        float amax = 0.0f;
        for (std::uint32_t k = begin; k < end && k < x.size(); ++k) {
            amax = std::fmax(amax, std::fabs(x[k]));
        }
        const float s = quant::fp8::amax_scale(amax);
        scales[g] = s;
        for (std::uint32_t k = begin; k < end && k < x.size(); ++k) {
            codes[k] = quant::fp8::e4m3_encode(s > 0.0f ? x[k] / s : 0.0f);
        }
    }
}

void run_case(std::uint32_t features, std::uint32_t k, std::uint32_t act_group,
              std::uint32_t seed) {
    const std::uint32_t kp = quant::fp8::pad_k(k);
    const std::uint32_t kp4 = quant::mxfp4::pad_k(kp);   // 32 の倍数（fp8 は 64 の倍数）
    std::printf("  case features=%u k=%u act_group=%u\n", features, k, act_group);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    // ---- 重み: f32 → MXFP4（行優先）→ 断片順 ----
    std::vector<float> w(static_cast<std::size_t>(features) * k);
    for (auto& v : w) v = dist(rng);
    std::vector<std::uint8_t> wc_canon(quant::mxfp4::canonical_code_bytes(features, kp4));
    std::vector<std::uint8_t> ws(quant::mxfp4::scale_bytes(features, kp4));
    for (std::uint32_t n = 0; n < features; ++n) {
        const auto st = quant::mxfp4::quantize_row(
            w.data() + static_cast<std::size_t>(n) * k, k, kp4,
            wc_canon.data() + static_cast<std::size_t>(n) * (kp4 / 2u),
            ws.data() + static_cast<std::size_t>(n) * (kp4 / 32u));
        check(st.ok(), "mxfp4::quantize_row");
    }
    std::vector<std::uint8_t> wc(quant::mxfp4::tiled_bytes(features, kp4));
    check(quant::mxfp4::preshuffle_codes(wc_canon.data(), features, k, kp4, wc.data()).ok(),
          "mxfp4::preshuffle_codes");

    // ---- 活性化: f32 → E4M3（1 行）→ 断片順 ----
    std::vector<float> a(k);
    for (auto& v : a) v = dist(rng);
    std::vector<std::uint8_t> ac_canon(kp4, 0u);
    std::vector<float> as;
    quantize_activation_row(a, kp4, act_group, ac_canon, as);
    std::vector<std::uint8_t> ac(quant::fp8::tiled_bytes(1u, kp4));
    check(quant::fp8::preshuffle_codes(ac_canon.data(), 1u, kp4, kp4, ac.data()).ok(),
          "fp8::preshuffle_codes");

    // ---- 実行 ----
    float* dev_out = nullptr;
    std::uint8_t* dev_wc = nullptr;
    std::uint8_t* dev_ws = nullptr;
    std::uint8_t* dev_ac = nullptr;
    float* dev_as = nullptr;
    check_hip(hipMalloc(&dev_out, sizeof(float) * features), "hipMalloc out");
    check_hip(hipMalloc(&dev_wc, wc.size()), "hipMalloc wc");
    check_hip(hipMalloc(&dev_ws, ws.size()), "hipMalloc ws");
    check_hip(hipMalloc(&dev_ac, ac.size()), "hipMalloc ac");
    check_hip(hipMalloc(&dev_as, sizeof(float) * as.size()), "hipMalloc as");
    check_hip(hipMemcpy(dev_wc, wc.data(), wc.size(), hipMemcpyHostToDevice), "copy wc");
    check_hip(hipMemcpy(dev_ws, ws.data(), ws.size(), hipMemcpyHostToDevice), "copy ws");
    check_hip(hipMemcpy(dev_ac, ac.data(), ac.size(), hipMemcpyHostToDevice), "copy ac");
    check_hip(hipMemcpy(dev_as, as.data(), sizeof(float) * as.size(), hipMemcpyHostToDevice),
              "copy as");

    Mxfp4GemmScaleArgs sc;
    sc.act_group = act_group;
    sc.activation_scale_stride = static_cast<std::uint32_t>(as.size());
    sc.weight_scale_stride = kp4 / 32u;
    check_hip(launch_gemm_mxfp4_w4a8_decode1(dev_wc, dev_ws, dev_ac, dev_as, dev_out,
                                             Mxfp4GemmOutputDType::F32, 1u, features, kp4,
                                             1u, sc, nullptr),
              "launch_gemm_mxfp4_w4a8_decode1");
    std::vector<float> out(features, 0.0f);
    check_hip(hipMemcpy(out.data(), dev_out, sizeof(float) * features, hipMemcpyDeviceToHost),
              "copy out");
    check_hip(hipFree(dev_out), "free out");
    check_hip(hipFree(dev_wc), "free wc");
    check_hip(hipFree(dev_ws), "free ws");
    check_hip(hipFree(dev_ac), "free ac");
    check_hip(hipFree(dev_as), "free as");

    // ---- CPU 参照（f64 で内積）----
    double worst_rel = 0.0;
    double mag = 0.0;
    for (std::uint32_t n = 0; n < features; ++n) {
        double ref = 0.0;
        for (std::uint32_t c = 0; c < kp4; ++c) {
            const float wv = quant::mxfp4::read_element(wc.data(), ws.data(), n, c, kp4);
            const std::uint8_t code = ac_canon[c];
            const std::uint32_t g = act_group ? (c / act_group) : 0u;
            const float av = quant::fp8::e4m3_decode(code) * as[g];
            ref += static_cast<double>(wv) * static_cast<double>(av);
        }
        mag = std::fmax(mag, std::fabs(ref));
        const double d = std::fabs(ref - static_cast<double>(out[n]));
        if (d > worst_rel) worst_rel = d;
    }
    const double rel = (mag > 0.0) ? worst_rel / mag : 0.0;
    std::printf("    worst abs=%.4e rel=%.4e（|ref|max=%.3e）\n", worst_rel, rel, mag);
    check(rel < 2.0e-5, "カーネル出力が CPU 参照と一致（相対 2e-5 以内）");
}

// prefill（rows > 1）の検証。活性化は 128 K グループのスケールで量子化する。
void run_prefill_case(std::uint32_t rows, std::uint32_t features, std::uint32_t k,
                      std::uint32_t seed) {
    const std::uint32_t kp = quant::fp8::pad_k(k);
    const std::uint32_t kp4 = quant::mxfp4::pad_k(kp);
    std::printf("  prefill case rows=%u features=%u k=%u\n", rows, features, k);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> w(static_cast<std::size_t>(features) * k);
    for (auto& v : w) v = dist(rng);
    std::vector<std::uint8_t> wc_canon(quant::mxfp4::canonical_code_bytes(features, kp4));
    std::vector<std::uint8_t> ws(quant::mxfp4::scale_bytes(features, kp4));
    for (std::uint32_t n = 0; n < features; ++n) {
        check(quant::mxfp4::quantize_row(
                  w.data() + static_cast<std::size_t>(n) * k, k, kp4,
                  wc_canon.data() + static_cast<std::size_t>(n) * (kp4 / 2u),
                  ws.data() + static_cast<std::size_t>(n) * (kp4 / 32u)).ok(),
              "mxfp4::quantize_row (prefill)");
    }
    std::vector<std::uint8_t> wc(quant::mxfp4::tiled_bytes(features, kp4));
    check(quant::mxfp4::preshuffle_codes(wc_canon.data(), features, k, kp4, wc.data()).ok(),
          "mxfp4::preshuffle_codes (prefill)");

    std::vector<float> a(static_cast<std::size_t>(rows) * k);
    for (auto& v : a) v = dist(rng);
    const std::uint32_t ngroups = kp4 / 128u;
    std::vector<std::uint8_t> ac_canon(static_cast<std::size_t>(rows) * kp4, 0u);
    std::vector<float> as(static_cast<std::size_t>(rows) * ngroups, 0.0f);
    for (std::uint32_t m = 0; m < rows; ++m) {
        std::vector<std::uint8_t> codes(kp4, 0u);
        std::vector<float> scales;
        quantize_activation_row(
            std::vector<float>(a.begin() + static_cast<std::size_t>(m) * k,
                               a.begin() + static_cast<std::size_t>(m) * k + k),
            kp4, 128u, codes, scales);
        for (std::uint32_t kk = 0; kk < kp4; ++kk)
            ac_canon[static_cast<std::size_t>(m) * kp4 + kk] = codes[kk];
        for (std::uint32_t g = 0; g < ngroups; ++g)
            as[static_cast<std::size_t>(m) * ngroups + g] = scales[g];
    }
    std::vector<std::uint8_t> ac(quant::fp8::tiled_bytes(rows, kp4));
    check(quant::fp8::preshuffle_codes(ac_canon.data(), rows, kp4, kp4, ac.data()).ok(),
          "fp8::preshuffle_codes (prefill)");

    std::uint8_t* dev_wc = nullptr;
    std::uint8_t* dev_ws = nullptr;
    std::uint8_t* dev_ac = nullptr;
    float* dev_as = nullptr;
    float* dev_out = nullptr;
    check_hip(hipMalloc(&dev_wc, wc.size()), "malloc wc");
    check_hip(hipMalloc(&dev_ws, ws.size()), "malloc ws");
    check_hip(hipMalloc(&dev_ac, ac.size()), "malloc ac");
    check_hip(hipMalloc(&dev_as, sizeof(float) * as.size()), "malloc as");
    check_hip(hipMalloc(&dev_out, sizeof(float) * rows * features), "malloc out");
    check_hip(hipMemcpy(dev_wc, wc.data(), wc.size(), hipMemcpyHostToDevice), "copy wc");
    check_hip(hipMemcpy(dev_ws, ws.data(), ws.size(), hipMemcpyHostToDevice), "copy ws");
    check_hip(hipMemcpy(dev_ac, ac.data(), ac.size(), hipMemcpyHostToDevice), "copy ac");
    check_hip(hipMemcpy(dev_as, as.data(), sizeof(float) * as.size(), hipMemcpyHostToDevice),
              "copy as");

    Mxfp4GemmScaleArgs sc;
    sc.act_group = 128u;
    sc.activation_scale_stride = ngroups;
    sc.weight_scale_stride = kp4 / 32u;
    check_hip(launch_gemm_mxfp4_w4a8_prefill(dev_wc, dev_ws, dev_ac, dev_as, dev_out,
                                             Mxfp4GemmOutputDType::F32, rows, features, kp4,
                                             features, sc, nullptr),
              "launch_gemm_mxfp4_w4a8_prefill");
    std::vector<float> out(static_cast<std::size_t>(rows) * features, 0.0f);
    check_hip(hipMemcpy(out.data(), dev_out, sizeof(float) * out.size(),
                        hipMemcpyDeviceToHost), "copy out");
    check_hip(hipFree(dev_wc), "free wc");
    check_hip(hipFree(dev_ws), "free ws");
    check_hip(hipFree(dev_ac), "free ac");
    check_hip(hipFree(dev_as), "free as");
    check_hip(hipFree(dev_out), "free out");

    double worst = 0.0, mag = 0.0;
    std::uint32_t worst_m = 0, worst_n = 0;
    for (std::uint32_t m = 0; m < rows; ++m) {
        for (std::uint32_t n = 0; n < features; ++n) {
            double ref = 0.0;
            for (std::uint32_t c = 0; c < kp4; ++c) {
                const float wv = quant::mxfp4::read_element(wc.data(), ws.data(), n, c, kp4);
                const std::uint8_t code = ac_canon[static_cast<std::size_t>(m) * kp4 + c];
                const float av = quant::fp8::e4m3_decode(code) *
                                 as[static_cast<std::size_t>(m) * ngroups + c / 128u];
                ref += static_cast<double>(wv) * static_cast<double>(av);
            }
            mag = std::fmax(mag, std::fabs(ref));
            const double d = std::fabs(ref - static_cast<double>(out[static_cast<std::size_t>(m) * features + n]));
            if (d > worst) { worst = d; worst_m = m; worst_n = n; }
        }
    }
    const double rel = (mag > 0.0) ? worst / mag : 0.0;
    std::printf("    worst abs=%.4e rel=%.4e\n", worst, rel);
    if (rel >= 2.0e-5)
        std::printf("    最悪要素 row=%u col=%u（rel=%.3e）\n", worst_m, worst_n, rel);
    check(rel < 2.0e-5, "prefill 出力が CPU 参照と一致（相対 2e-5 以内）");
}

// 速度計測（同期を取って 20 回の中央値）。fp8 の wide 版と比べるための道具。
void timed_case(std::uint32_t rows, std::uint32_t features, std::uint32_t k) {
    const std::uint32_t kp4 = quant::mxfp4::pad_k(quant::fp8::pad_k(k));
    const std::uint32_t ngroups = kp4 / 128u;
    std::vector<std::uint8_t> wc(quant::mxfp4::tiled_bytes(features, kp4), 0u);
    std::vector<std::uint8_t> ws(quant::mxfp4::scale_bytes(features, kp4), 128u);
    std::vector<std::uint8_t> ac(quant::fp8::tiled_bytes(rows, kp4), 0u);
    std::vector<float> as(static_cast<std::size_t>(rows) * ngroups, 1.0f);

    std::uint8_t* dwc = nullptr;
    std::uint8_t* dws = nullptr;
    std::uint8_t* dac = nullptr;
    float* das = nullptr;
    float* dout = nullptr;
    check_hip(hipMalloc(&dwc, wc.size()), "t wc");
    check_hip(hipMalloc(&dws, ws.size()), "t ws");
    check_hip(hipMalloc(&dac, ac.size()), "t ac");
    check_hip(hipMalloc(&das, sizeof(float) * as.size()), "t as");
    check_hip(hipMalloc(&dout, sizeof(float) * rows * features), "t out");
    check_hip(hipMemcpy(dwc, wc.data(), wc.size(), hipMemcpyHostToDevice), "t copy wc");
    check_hip(hipMemcpy(dws, ws.data(), ws.size(), hipMemcpyHostToDevice), "t copy ws");
    check_hip(hipMemcpy(dac, ac.data(), ac.size(), hipMemcpyHostToDevice), "t copy ac");
    check_hip(hipMemcpy(das, as.data(), sizeof(float) * as.size(), hipMemcpyHostToDevice),
              "t copy as");
    Mxfp4GemmScaleArgs sc;
    sc.act_group = 128u;
    sc.activation_scale_stride = ngroups;
    sc.weight_scale_stride = kp4 / 32u;

    auto run = [&]() {
        (void)launch_gemm_mxfp4_w4a8_prefill(dwc, dws, dac, das, dout,
                                            Mxfp4GemmOutputDType::BF16, rows, features, kp4,
                                            features, sc, nullptr);
    };
    for (int i = 0; i < 5; ++i) run();
    check_hip(hipDeviceSynchronize(), "t sync");
    std::vector<double> samples;
    for (int i = 0; i < 20; ++i) {
        hipEvent_t a{}, b{};
        check_hip(hipEventCreate(&a), "ev a");
        check_hip(hipEventCreate(&b), "ev b");
        check_hip(hipEventRecord(a), "rec a");
        run();
        check_hip(hipEventRecord(b), "rec b");
        check_hip(hipEventSynchronize(b), "sync b");
        float ms = 0.0f;
        check_hip(hipEventElapsedTime(&ms, a, b), "elapsed");
        samples.push_back(static_cast<double>(ms) * 1000.0);
        hipEventDestroy(a);
        hipEventDestroy(b);
    }
    std::sort(samples.begin(), samples.end());
    const double us = samples[samples.size() / 2];
    const double tflops = 2.0 * rows * features * k / (us * 1e6);
    std::printf("  mxfp4 prefill rows=%u features=%u k=%u: %.2f us = %.1f TFLOP/s\n",
                rows, features, k, us, tflops);
    check_hip(hipFree(dwc), "t free");
    check_hip(hipFree(dws), "t free");
    check_hip(hipFree(dac), "t free");
    check_hip(hipFree(das), "t free");
    check_hip(hipFree(dout), "t free");
}

// decode1 GEMV の速度（重みが DRAM に載る形で測る）。
void timed_decode1(std::uint32_t features, std::uint32_t k) {
    const std::uint32_t kp4 = quant::mxfp4::pad_k(quant::fp8::pad_k(k));
    const std::uint32_t ngroups = kp4 / 128u;
    std::vector<std::uint8_t> wc(quant::mxfp4::tiled_bytes(features, kp4), 0u);
    std::vector<std::uint8_t> ws(quant::mxfp4::scale_bytes(features, kp4), 128u);
    std::vector<std::uint8_t> ac(quant::fp8::tiled_bytes(1u, kp4), 0u);
    std::vector<float> as(ngroups, 1.0f);
    std::uint8_t* dwc = nullptr;
    std::uint8_t* dws = nullptr;
    std::uint8_t* dac = nullptr;
    float* das = nullptr;
    float* dout = nullptr;
    check_hip(hipMalloc(&dwc, wc.size()), "d1 wc");
    check_hip(hipMalloc(&dws, ws.size()), "d1 ws");
    check_hip(hipMalloc(&dac, ac.size()), "d1 ac");
    check_hip(hipMalloc(&das, sizeof(float) * as.size()), "d1 as");
    check_hip(hipMalloc(&dout, sizeof(float) * features), "d1 out");
    check_hip(hipMemcpy(dwc, wc.data(), wc.size(), hipMemcpyHostToDevice), "d1 c1");
    check_hip(hipMemcpy(dws, ws.data(), ws.size(), hipMemcpyHostToDevice), "d1 c2");
    check_hip(hipMemcpy(dac, ac.data(), ac.size(), hipMemcpyHostToDevice), "d1 c3");
    check_hip(hipMemcpy(das, as.data(), sizeof(float) * as.size(), hipMemcpyHostToDevice), "d1 c4");
    Mxfp4GemmScaleArgs sc;
    sc.act_group = 128u;
    sc.activation_scale_stride = ngroups;
    sc.weight_scale_stride = kp4 / 32u;
    auto run = [&]() {
        (void)launch_gemm_mxfp4_w4a8_decode1(dwc, dws, dac, das, dout,
                                            Mxfp4GemmOutputDType::BF16, 1u, features, kp4, 1u,
                                            sc, nullptr);
    };
    for (int i = 0; i < 5; ++i) run();
    check_hip(hipDeviceSynchronize(), "d1 sync");
    std::vector<double> s;
    for (int i = 0; i < 20; ++i) {
        hipEvent_t a{}, b{};
        hipEventCreate(&a);
        hipEventCreate(&b);
        hipEventRecord(a);
        run();
        hipEventRecord(b);
        hipEventSynchronize(b);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, a, b);
        s.push_back(static_cast<double>(ms) * 1000.0);
        hipEventDestroy(a);
        hipEventDestroy(b);
    }
    std::sort(s.begin(), s.end());
    const double us = s[s.size() / 2];
    const double mb = static_cast<double>(features) * k * 17.0 / 32.0 / 1e6;
    std::printf("  mxfp4 decode1 features=%u k=%u: %.2f us, %.1f MB, %.0f GB/s\n", features, k,
                us, mb, mb / (us * 1e-6) / 1e3);
    hipFree(dwc); hipFree(dws); hipFree(dac); hipFree(das); hipFree(dout);
}

// ---- 実モデル（Quark MXFP4）のテンソルで確認する ------------------------------
// SHELF_MXFP4_MODEL=<dir> のときだけ走る。safetensors のヘッダだけ自前で読み、
// <layer>.weight (U8 [N, K/2]) と <layer>.weight_scale (U8 [N, K/32]) を取る。
bool read_tensor(const std::string& path, const std::string& name,
                 std::vector<std::uint8_t>& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::uint64_t hlen = 0;
    if (std::fread(&hlen, 8, 1, f) != 1) { std::fclose(f); return false; }
    std::string hdr(static_cast<std::size_t>(hlen), '\0');
    if (std::fread(hdr.data(), 1, hdr.size(), f) != hdr.size()) { std::fclose(f); return false; }
    const std::string key = "\"" + name + "\"";
    const std::size_t k = hdr.find(key);
    if (k == std::string::npos) { std::fclose(f); return false; }
    const std::size_t off = hdr.find("\"data_offsets\"", k);
    if (off == std::string::npos) { std::fclose(f); return false; }
    const std::size_t lb = hdr.find('[', off), comma = hdr.find(',', lb);
    const std::size_t rb = hdr.find(']', comma);
    if (lb == std::string::npos || comma == std::string::npos || rb == std::string::npos) {
        std::fclose(f); return false;
    }
    const std::uint64_t a = std::strtoull(hdr.substr(lb + 1, comma - lb - 1).c_str(), nullptr, 10);
    const std::uint64_t b = std::strtoull(hdr.substr(comma + 1, rb - comma - 1).c_str(), nullptr, 10);
    out.resize(static_cast<std::size_t>(b - a));
    std::fseek(f, static_cast<long>(8 + hlen + a), SEEK_SET);
    const bool ok = std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

void real_model_case(const std::string& dir, const std::string& layer,
                     std::uint32_t rows_prefill) {
    const std::string path = dir + "/model.safetensors";
    std::vector<std::uint8_t> codes, scales;
    if (!read_tensor(path, layer + ".weight", codes) ||
        !read_tensor(path, layer + ".weight_scale", scales)) {
        std::printf("FAIL: %s を読めない\n", layer.c_str());
        ++failures;
        return;
    }
    // U8 [N, K/2] と [N, K/32] から N と K を出す（K/32 の要素数 = N*(K/32)）。
    const std::size_t total_scales = scales.size();
    const std::size_t total_codes = codes.size();
    if (total_codes == 0 || total_scales == 0 || total_codes % total_scales != 0) {
        std::printf("FAIL: テンソルサイズが不整合\n");
        ++failures;
        return;
    }
    // codes/scales = (N*K/2)/(N*K/32) = 16 なので、N*K/32 = total_scales から
    // 実形状はファイル名から既知（呼び手が渡す）。ここでは K=5120 固定で検査する。
    const std::uint32_t k = 5120u;
    const std::uint32_t kp = quant::mxfp4::pad_k(k);
    const std::size_t n = total_codes / (kp / 2u);
    std::printf("  real %s: N=%zu K=%u, codes=%zu B, scales=%zu B\n", layer.c_str(), n, k,
                total_codes, total_scales);
    check(n * (kp / 32u) == total_scales, "スケール数が N*(K/32) と一致");

    std::vector<std::uint8_t> tiled(quant::mxfp4::tiled_bytes(static_cast<std::uint32_t>(n), kp));
    check(quant::mxfp4::preshuffle_codes(codes.data(), static_cast<std::uint32_t>(n), k, kp,
                                         tiled.data()).ok(),
          "実テンソルの preshuffle");
    // 断片順から読み戻した値が、元のニブルと一致すること（無損失 relayout の確認）。
    std::size_t mism = 0;
    for (std::uint32_t r = 0; r < n; r += 997u) {
        for (std::uint32_t c = 0; c < kp; c += 131u) {
            const std::uint8_t want = static_cast<std::uint8_t>(
                (codes[static_cast<std::size_t>(r) * (kp / 2u) + (c >> 1u)] >> (4u * (c & 1u))) & 0x0Fu);
            const std::uint8_t byte = tiled[quant::mxfp4::tiled_offset(r, c, kp)];
            const std::uint8_t got =
                static_cast<std::uint8_t>((byte >> quant::mxfp4::tiled_nibble_shift(c)) & 0x0Fu);
            if (got != want) ++mism;
        }
    }
    check(mism == 0, "実テンソルで relayout が無損失");

    // decode1 と prefill を実行して、CPU 参照（生ニブルから作る）と比べる。
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> a(static_cast<std::size_t>(rows_prefill) * k);
    for (auto& v : a) v = dist(rng);
    const std::uint32_t ngroups = kp / 128u;
    std::vector<std::uint8_t> ac_canon(static_cast<std::size_t>(rows_prefill) * kp, 0u);
    std::vector<float> as(static_cast<std::size_t>(rows_prefill) * ngroups, 0.0f);
    for (std::uint32_t m = 0; m < rows_prefill; ++m) {
        std::vector<std::uint8_t> c(kp, 0u);
        std::vector<float> s;
        quantize_activation_row(std::vector<float>(a.begin() + static_cast<std::size_t>(m) * k,
                                                   a.begin() + static_cast<std::size_t>(m) * k + k),
                                kp, 128u, c, s);
        for (std::uint32_t i = 0; i < kp; ++i)
            ac_canon[static_cast<std::size_t>(m) * kp + i] = c[i];
        for (std::uint32_t g = 0; g < ngroups; ++g)
            as[static_cast<std::size_t>(m) * ngroups + g] = s[g];
    }
    std::vector<std::uint8_t> ac(quant::fp8::tiled_bytes(rows_prefill, kp));
    check(quant::fp8::preshuffle_codes(ac_canon.data(), rows_prefill, kp, kp, ac.data()).ok(),
          "活性化の preshuffle");

    std::uint8_t* dt = nullptr;
    std::uint8_t* ds = nullptr;
    std::uint8_t* da = nullptr;
    float* dfa = nullptr;
    float* dout = nullptr;
    const std::size_t out_elems = static_cast<std::size_t>(rows_prefill) * n;
    check_hip(hipMalloc(&dt, tiled.size()), "r dt");
    check_hip(hipMalloc(&ds, total_scales), "r ds");
    check_hip(hipMalloc(&da, ac.size()), "r da");
    check_hip(hipMalloc(&dfa, sizeof(float) * as.size()), "r dfa");
    check_hip(hipMalloc(&dout, sizeof(float) * out_elems), "r dout");
    check_hip(hipMemcpy(dt, tiled.data(), tiled.size(), hipMemcpyHostToDevice), "r ct");
    check_hip(hipMemcpy(ds, scales.data(), total_scales, hipMemcpyHostToDevice), "r cs");
    check_hip(hipMemcpy(da, ac.data(), ac.size(), hipMemcpyHostToDevice), "r ca");
    check_hip(hipMemcpy(dfa, as.data(), sizeof(float) * as.size(), hipMemcpyHostToDevice), "r cfa");
    Mxfp4GemmScaleArgs sc;
    sc.act_group = 128u;
    sc.activation_scale_stride = ngroups;
    sc.weight_scale_stride = kp / 32u;
    check_hip(launch_gemm_mxfp4_w4a8_prefill(dt, ds, da, dfa, dout,
                                             Mxfp4GemmOutputDType::F32, rows_prefill,
                                             static_cast<std::uint32_t>(n), kp,
                                             static_cast<std::uint32_t>(n), sc, nullptr),
              "real prefill launch");
    std::vector<float> out(out_elems, 0.0f);
    check_hip(hipMemcpy(out.data(), dout, sizeof(float) * out_elems, hipMemcpyDeviceToHost),
              "r cout");

    double worst = 0.0, mag = 0.0;
    for (std::uint32_t m = 0; m < rows_prefill; ++m) {
        for (std::uint32_t nn = 0; nn < 64u; ++nn) {   // 先頭 64 出力だけ照合
            double ref = 0.0;
            for (std::uint32_t c = 0; c < kp; ++c) {
                const float wv = quant::mxfp4::read_element(tiled.data(), scales.data(), nn, c, kp);
                const float av = quant::fp8::e4m3_decode(ac_canon[static_cast<std::size_t>(m) * kp + c]) *
                                 as[static_cast<std::size_t>(m) * ngroups + c / 128u];
                ref += static_cast<double>(wv) * static_cast<double>(av);
            }
            mag = std::fmax(mag, std::fabs(ref));
            worst = std::fmax(worst, std::fabs(ref - static_cast<double>(out[static_cast<std::size_t>(m) * n + nn])));
        }
    }
    const double rel = mag > 0.0 ? worst / mag : 0.0;
    std::printf("    real weight での相対誤差 = %.3e\n", rel);
    // 折りたたみを使わない（32 K ごとに E8M0 を f32 で掛ける）ので、実テンソルでも
    // CPU 参照と厳密に一致する（差は f32 の累加順のみ）。
    check(rel < 2.0e-5, "実テンソルでカーネルが CPU 参照と一致");

    // 速度（実形状）
    auto bench = [&](const char* what, auto&& fn) {
        for (int i = 0; i < 5; ++i) fn();
        hipDeviceSynchronize();
        std::vector<double> s;
        for (int i = 0; i < 15; ++i) {
            hipEvent_t e0{}, e1{};
            hipEventCreate(&e0); hipEventCreate(&e1);
            hipEventRecord(e0); fn(); hipEventRecord(e1); hipEventSynchronize(e1);
            float ms = 0.0f; hipEventElapsedTime(&ms, e0, e1);
            s.push_back(static_cast<double>(ms) * 1000.0);
            hipEventDestroy(e0); hipEventDestroy(e1);
        }
        std::sort(s.begin(), s.end());
        std::printf("    %s: %.2f us\n", what, s[s.size() / 2]);
    };
    bench("prefill rows=512（実形状）", [&]() {
        (void)launch_gemm_mxfp4_w4a8_prefill(dt, ds, da, dfa, dout,
                                             Mxfp4GemmOutputDType::BF16, rows_prefill,
                                             static_cast<std::uint32_t>(n), kp,
                                             static_cast<std::uint32_t>(n), sc, nullptr);
    });
    // decode1 は 1 行なので活性化バッファの先頭だけ使う
    Mxfp4GemmScaleArgs sc1 = sc;
    bench("decode1（実形状）", [&]() {
        (void)launch_gemm_mxfp4_w4a8_decode1(dt, ds, da, dfa, dout,
                                             Mxfp4GemmOutputDType::BF16, 1u,
                                             static_cast<std::uint32_t>(n), kp,
                                             static_cast<std::uint32_t>(n), sc1, nullptr);
    });
    hipFree(dt); hipFree(ds); hipFree(da); hipFree(dfa); hipFree(dout);
}

}  // namespace

int main() {
    if (!mxfp4_decode1_supported(1u, 256u, 256u)) {
        std::printf("FAIL: mxfp4_decode1_supported が false\n");
        return 1;
    }
    check(!mxfp4_decode1_supported(2u, 256u, 256u), "rows != 1 は不受理");
    check(!mxfp4_decode1_supported(1u, 256u, 32u), "K が 64 の倍数でないと不受理");
    check(mxfp4_weight_bytes(256u, 256u) == quant::mxfp4::tiled_bytes(256u, 256u),
          "mxfp4_weight_bytes が tiled_bytes と一致");

    run_case(256u, 256u, 128u, 11u);
    run_case(256u, 512u, 128u, 12u);
    run_case(64u, 256u, 0u, 13u);
    run_case(128u, 1024u, 128u, 14u);
    run_prefill_case(128u, 128u, 256u, 21u);
    run_prefill_case(128u, 256u, 512u, 22u);
    // 128 行タイル未満の端数（ロード側 clamp の回帰）。以前は範囲外読みでフォルト。
    run_prefill_case(16u, 128u, 256u, 23u);
    run_prefill_case(33u, 128u, 256u, 27u);
    run_prefill_case(64u, 128u, 256u, 28u);
    run_prefill_case(32u, 128u, 256u, 24u);
    run_prefill_case(64u, 256u, 512u, 25u);
    run_prefill_case(96u, 128u, 512u, 26u);
    // 実機のエキスパート形（I=512, H=2048 と down の H=2048, I=512）
    if (const char* md = std::getenv("SHELF_MXFP4_MODEL")) {
        real_model_case(md, "model.language_model.layers.0.mlp.gate_proj", 512u);
    }
    if (std::getenv("SHELF_MXFP4_TIMING") != nullptr) {
        timed_case(512u, 5120u, 5120u);
        timed_case(2048u, 5120u, 5120u);
        timed_case(512u, 17408u, 5120u);
        timed_decode1(5120u, 5120u);
        timed_decode1(5120u, 17408u);
    }

    if (failures == 0) {
        std::printf("unit.hip_gemm_mxfp4: OK\n");
        return 0;
    }
    std::printf("unit.hip_gemm_mxfp4: %d 件失敗\n", failures);
    return 1;
}
