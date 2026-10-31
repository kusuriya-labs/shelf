// shelf unit test — 活性化の MXFP4 量子化（E2M1 + E8M0、断片順）。
//
// 参照は `quant::mxfp4::quantize_row` + `preshuffle_codes`（ホスト実装）。
// カーネルの出力はコード・スケールとも**バイト一致**することを要求する
// （同じ式を device に写したものなので、ずれは実装の誤り）。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <shelf/backend/hip/kernels/elementwise/activation_quantize_mxfp4.h>
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

void run_case(std::uint32_t rows, std::uint32_t k, std::uint32_t seed) {
    const std::uint32_t kp = quant::mxfp4::pad_k(k);
    std::printf("  case rows=%u k=%u (k_padded=%u)\n", rows, k, kp);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    // 入力（bf16 に丸めた値を使う。カーネルは bf16 を読むため）
    std::vector<std::uint16_t> x(static_cast<std::size_t>(rows) * k);
    for (auto& v : x) v = __bfloat16_as_ushort(__float2bfloat16(dist(rng)));

    // 参照（ホスト）
    std::vector<std::uint8_t> ref_canon(quant::mxfp4::canonical_code_bytes(rows, kp));
    std::vector<std::uint8_t> ref_scales(quant::mxfp4::scale_bytes(rows, kp));
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<float> row(k);
        for (std::uint32_t i = 0; i < k; ++i) {
            row[i] = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&x[static_cast<std::size_t>(r) * k + i]));
        }
        check(quant::mxfp4::quantize_row(row.data(), k, kp,
                                         ref_canon.data() + static_cast<std::size_t>(r) * (kp / 2u),
                                         ref_scales.data() + static_cast<std::size_t>(r) * (kp / 32u)).ok(),
              "quantize_row");
    }
    std::vector<std::uint8_t> ref_codes(quant::mxfp4::tiled_bytes(rows, kp));
    check(quant::mxfp4::preshuffle_codes(ref_canon.data(), rows, k, kp, ref_codes.data()).ok(),
          "preshuffle_codes");

    // カーネル
    std::vector<std::uint8_t> got_codes(quant::mxfp4::tiled_bytes(rows, kp), 0u);   // パディング行は 0 のまま（参照も 0）
    std::vector<std::uint8_t> got_scales(quant::mxfp4::scale_bytes(rows, kp), 0u);
    shelf::bf16_t* dx = nullptr;
    std::uint8_t* dc = nullptr;
    std::uint8_t* ds = nullptr;
    check_hip(hipMalloc(&dx, sizeof(std::uint16_t) * x.size()), "malloc x");
    check_hip(hipMalloc(&dc, got_codes.size()), "malloc codes");
    check_hip(hipMalloc(&ds, got_scales.size()), "malloc scales");
    // hipMalloc はゼロ初期化しない。カーネルは有効行しか書かないので、
    // パディング行を参照（0 埋め）と比べるには呼び手がゼロ化する必要がある。
    check_hip(hipMemset(dc, 0, got_codes.size()), "zero codes");
    check_hip(hipMemset(ds, 0, got_scales.size()), "zero scales");
    check_hip(hipMemcpy(dx, x.data(), sizeof(std::uint16_t) * x.size(), hipMemcpyHostToDevice), "copy x");
    check_hip(launch_activation_quantize_mxfp4(dx, dc, ds, rows, k, kp, k, nullptr),
              "launch_activation_quantize_mxfp4");
    check_hip(hipMemcpy(got_codes.data(), dc, got_codes.size(), hipMemcpyDeviceToHost), "copy codes");
    check_hip(hipMemcpy(got_scales.data(), ds, got_scales.size(), hipMemcpyDeviceToHost), "copy scales");
    hipFree(dx);
    hipFree(dc);
    hipFree(ds);

    std::size_t cmis = 0, smis = 0;
    std::size_t shown = 0;
    for (std::size_t i = 0; i < ref_codes.size(); ++i) {
        if (ref_codes[i] != got_codes[i]) {
            if (shown < 6) {
                std::printf("    code[%zu] ref=%02x got=%02x (row=%zu block=%zu byte=%zu)\n", i,
                            ref_codes[i], got_codes[i], i / (kp / 2u),
                            (i % (kp / 2u)) / 256u, (i % (kp / 2u)) % 256u);
                ++shown;
            }
            ++cmis;
        }
    }
    for (std::size_t i = 0; i < ref_scales.size(); ++i) {
        if (ref_scales[i] != got_scales[i]) ++smis;
    }
    std::printf("    codes 不一致 %zu / %zu、scales 不一致 %zu / %zu\n", cmis, ref_codes.size(), smis,
                ref_scales.size());
    check(cmis == 0, "コードがホスト実装とバイト一致");
    check(smis == 0, "スケールがホスト実装とバイト一致");

    // 逆量子化して元の値との誤差も確認（格子の半歩以内）
    double worst = 0.0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t i = 0; i < k; ++i) {
            const float want =
                __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&x[static_cast<std::size_t>(r) * k + i]));
            const float got = quant::mxfp4::read_element(got_codes.data(), got_scales.data(), r, i, kp);
            const float s = quant::mxfp4::e8m0_decode_u8(
                got_scales[static_cast<std::size_t>(r) * (kp / 32u) + i / 32u]);
            if (std::fabs(got - want) > s) worst = std::fabs(got - want);
        }
    }
    check(worst == 0.0, "逆量子化誤差が格子の 1 歩以内");
    std::printf("    最大誤差超過 = %.3e\n", worst);
}

void run_swiglu_case(std::uint32_t rows, std::uint32_t k, std::uint32_t seed) {
    const std::uint32_t kp = quant::mxfp4::pad_k(k);
    std::printf("  swiglu case rows=%u k=%u\n", rows, k);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<std::uint16_t> g(static_cast<std::size_t>(rows) * k), u(g.size());
    for (auto& v : g) v = __bfloat16_as_ushort(__float2bfloat16(dist(rng)));
    for (auto& v : u) v = __bfloat16_as_ushort(__float2bfloat16(dist(rng)));

    // 参照: silu(g)*u を bf16 に丸めてから量子化
    std::vector<std::uint8_t> ref_canon(quant::mxfp4::canonical_code_bytes(rows, kp));
    std::vector<std::uint8_t> ref_scales(quant::mxfp4::scale_bytes(rows, kp));
    std::vector<std::uint16_t> ref_out(static_cast<std::size_t>(rows) * k);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<float> row(k);
        for (std::uint32_t i = 0; i < k; ++i) {
            const float gv = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&g[static_cast<std::size_t>(r) * k + i]));
            const float uv = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&u[static_cast<std::size_t>(r) * k + i]));
            const float y = (gv / (1.0f + std::exp(-gv))) * uv;
            const std::uint16_t yb = __bfloat16_as_ushort(__float2bfloat16(y));
            ref_out[static_cast<std::size_t>(r) * k + i] = yb;
            row[i] = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&yb));
        }
        check(quant::mxfp4::quantize_row(row.data(), k, kp,
                                         ref_canon.data() + static_cast<std::size_t>(r) * (kp / 2u),
                                         ref_scales.data() + static_cast<std::size_t>(r) * (kp / 32u)).ok(),
              "quantize_row (swiglu)");
    }

    shelf::bf16_t* dg = nullptr;
    shelf::bf16_t* du = nullptr;
    shelf::bf16_t* dout = nullptr;
    std::uint8_t* dc = nullptr;
    std::uint8_t* ds = nullptr;
    const std::size_t n = static_cast<std::size_t>(rows) * k;
    check_hip(hipMalloc(&dg, sizeof(std::uint16_t) * n), "sw g");
    check_hip(hipMalloc(&du, sizeof(std::uint16_t) * n), "sw u");
    check_hip(hipMalloc(&dout, sizeof(std::uint16_t) * n), "sw out");
    check_hip(hipMalloc(&dc, quant::mxfp4::tiled_bytes(rows, kp)), "sw codes");
    check_hip(hipMalloc(&ds, quant::mxfp4::scale_bytes(rows, kp)), "sw scales");
    check_hip(hipMemset(dc, 0, quant::mxfp4::tiled_bytes(rows, kp)), "sw zero codes");
    check_hip(hipMemset(ds, 0, quant::mxfp4::scale_bytes(rows, kp)), "sw zero scales");
    check_hip(hipMemcpy(dg, g.data(), sizeof(std::uint16_t) * n, hipMemcpyHostToDevice), "sw cg");
    check_hip(hipMemcpy(du, u.data(), sizeof(std::uint16_t) * n, hipMemcpyHostToDevice), "sw cu");
    check_hip(launch_swiglu_quantize_mxfp4(dg, du, dout, dc, ds, rows, k, kp, k, k, k, nullptr),
              "launch_swiglu_quantize_mxfp4");
    std::vector<std::uint8_t> got_codes(quant::mxfp4::tiled_bytes(rows, kp), 0u);   // パディング行は 0 のまま（参照も 0）
    std::vector<std::uint8_t> got_scales(quant::mxfp4::scale_bytes(rows, kp), 0u);
    std::vector<std::uint16_t> got_out(n, 0xAAAAu);
    check_hip(hipMemcpy(got_codes.data(), dc, got_codes.size(), hipMemcpyDeviceToHost), "sw cc");
    check_hip(hipMemcpy(got_scales.data(), ds, got_scales.size(), hipMemcpyDeviceToHost), "sw cs");
    check_hip(hipMemcpy(got_out.data(), dout, sizeof(std::uint16_t) * n, hipMemcpyDeviceToHost), "sw co");
    hipFree(dg); hipFree(du); hipFree(dout); hipFree(dc); hipFree(ds);

    std::vector<std::uint8_t> ref_codes(quant::mxfp4::tiled_bytes(rows, kp));
    check(quant::mxfp4::preshuffle_codes(ref_canon.data(), rows, k, kp, ref_codes.data()).ok(),
          "preshuffle (swiglu)");
    std::size_t cmis = 0, smis = 0, omis = 0;
    for (std::size_t i = 0; i < ref_codes.size(); ++i) if (ref_codes[i] != got_codes[i]) ++cmis;
    for (std::size_t i = 0; i < ref_scales.size(); ++i) if (ref_scales[i] != got_scales[i]) ++smis;
    for (std::size_t i = 0; i < n; ++i) if (ref_out[i] != got_out[i]) ++omis;
    std::printf("    codes %zu 不一致 / bf16 out %zu 不一致 / scales %zu 不一致\n", cmis, omis, smis);
    check(cmis == 0, "swiglu: コードが一致");
    check(smis == 0, "swiglu: スケールが一致");
    check(omis == 0, "swiglu: bf16 出力が一致");
}

}  // namespace

int main() {
    check(activation_quantize_mxfp4_supported(256u, 256u), "supported(256,256)");
    check(!activation_quantize_mxfp4_supported(256u, 272u), "k_padded が 32 の倍数でないと不受理");

    run_case(8u, 256u, 101u);
    run_case(16u, 512u, 102u);
    run_case(5u, 96u, 103u);
    run_case(5u, 128u, 104u);
    run_swiglu_case(8u, 256u, 201u);
    if (failures == 0) {
        std::printf("unit.hip_activation_mxfp4: OK\n");
        return 0;
    }
    std::printf("unit.hip_activation_mxfp4: %d 件失敗\n", failures);
    return 1;
}
