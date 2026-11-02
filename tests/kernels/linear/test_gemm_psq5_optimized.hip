#include <shelf/kernels/optimized/linear/psq5.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/quantization/psq5/psq5.h>
#include <shelf/quantization/codec/e4m3.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#define HIP_ASSERT(x) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        std::printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(err)); \
        return false; \
    } \
} while (0)

using ps::kernel::Psq5GemmConfig;
using ps::kernel::Psq5GemmConfigId;

namespace {

int g_pass = 0;
int g_fail = 0;
double g_worst_rel = 0.0;

ps::bf16_t f32_to_bf16(float f) {
    uint32_t ui = 0;
    std::memcpy(&ui, &f, sizeof(float));
    return ps::bf16_t{static_cast<uint16_t>((ui + 0x8000u) >> 16)};
}

float bf16_to_f32(ps::bf16_t b) {
    const uint32_t ui = static_cast<uint32_t>(b.data) << 16;
    float f = 0.0f;
    std::memcpy(&f, &ui, sizeof(float));
    return f;
}

Psq5GemmConfig config_for(uint32_t rows) {
    if (rows >= 8u * ps::kernel::kPsq5GemmRowTile) return {Psq5GemmConfigId::RowBlock8};
    if (rows >= 4u * ps::kernel::kPsq5GemmRowTile) return {Psq5GemmConfigId::RowBlock4};
    if (rows > ps::kernel::kPsq5GemmRowTile) return {Psq5GemmConfigId::RowBlock2};
    return {Psq5GemmConfigId::RowBlock1};
}

bool quantize_weight_psq5(const std::vector<float>& w, uint32_t n, uint32_t k,
                          std::vector<uint8_t>& mag, std::vector<uint8_t>& sg,
                          std::vector<uint8_t>& sc) {
    const uint32_t kp = (k + 31u) & ~31u;
    const uint32_t nb = kp / 32u;
    mag.assign(static_cast<std::size_t>(n) * nb * 16u, 0u);
    sg.assign(static_cast<std::size_t>(n) * nb * 4u, 0u);
    sc.assign(static_cast<std::size_t>(n) * nb, 0u);
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t ib = 0; ib < nb; ++ib) {
            float peak = 0.0f;
            for (uint32_t e = 0; e < 32u; ++e) {
                const uint32_t i = ib * 32u + e;
                const float v = i < k ? w[static_cast<std::size_t>(r) * k + i] : 0.0f;
                peak = std::max(peak, std::fabs(v));
            }
            const float target = peak / 28.0f * 512.0f;
            uint8_t best = 0u;
            float best_err = 1e30f;
            for (uint32_t b = 1u; b <= 0xEFu; ++b) {
                const float decoded = ps::quantization::psq5::ue4m4_value(static_cast<uint8_t>(b)) * 512.0f;
                const float err = std::fabs(decoded - target);
                if (err < best_err) { best_err = err; best = static_cast<uint8_t>(b); }
            }
            if (target <= 0.0f) best = 0u;
            const float restored = ps::quantization::psq5::ue4m4_value(best);
            sc[static_cast<std::size_t>(r) * nb + ib] = best;
            for (uint32_t e = 0; e < 32u; ++e) {
                const uint32_t i = ib * 32u + e;
                const float v = i < k ? w[static_cast<std::size_t>(r) * k + i] : 0.0f;
                uint8_t idx = 0u;
                if (restored > 0.0f) {
                    const float norm = v / restored;
                    const float av = std::fabs(norm);
                    const float table[16] = {0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28};
                    float bd = 1e30f;
                    for (uint8_t t = 0; t < 16u; ++t) {
                        const float d = std::fabs(av - table[t]);
                        if (d < bd) { bd = d; idx = t; }
                    }
                }
                const std::size_t mi = (static_cast<std::size_t>(r) * nb + ib) * 16u + (e >> 1);
                if ((e & 1u) == 0u) mag[mi] |= static_cast<uint8_t>(idx & 0x0Fu);
                else mag[mi] |= static_cast<uint8_t>((idx & 0x0Fu) << 4);
                if (v < 0.0f) {
                    const std::size_t si = (static_cast<std::size_t>(r) * nb + ib) * 4u + (e >> 3);
                    sg[si] |= static_cast<uint8_t>(1u << (e & 7u));
                }
            }
        }
    }
    return true;
}

bool run_case(ps::gpu::GpuArena& arena, uint32_t rows, uint32_t n, uint32_t k,
              const char* label, const Psq5GemmConfig* two_d = nullptr,
              const Psq5GemmConfig* wide = nullptr) {
    std::mt19937 rng(20261024u + rows * 31u + n * 7u + k);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-2.0f, 2.0f);

    std::vector<float> w(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(rows) * k);
    for (auto& v : w) v = wdist(rng);
    for (auto& v : x) v = xdist(rng);

    const uint32_t kp = (k + 31u) & ~31u;
    const uint32_t nb = kp / 32u;
    std::vector<uint8_t> mag, sg, sc;
    quantize_weight_psq5(w, n, k, mag, sg, sc);

    ps::quantization::psq5::Psq5CanonicalView cv;
    cv.n = n; cv.logical_k = k; cv.padded_k = kp;
    cv.magnitude = mag.data(); cv.magnitude_bytes = mag.size();
    cv.sign = sg.data(); cv.sign_bytes = sg.size();
    cv.scale = sc.data(); cv.scale_bytes = sc.size();
    ps::quantization::psq5::Psq5NativeHost native;
    if (!ps::quantization::psq5::preshuffle_psq5_native(cv, native)) {
        std::printf("FAIL: %s preshuffle\n", label);
        return false;
    }

    const uint32_t code_stride = kp;
    const uint32_t as_floats = (kp + 127u) / 128u;
    const uint32_t as_bytes = as_floats * 4u;

    std::vector<ps::bf16_t> hx(x.size());
    std::vector<float> xr(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        hx[i] = f32_to_bf16(x[i]);
        xr[i] = bf16_to_f32(hx[i]);
    }

    std::vector<float> wdec(static_cast<std::size_t>(n) * kp, 0.0f);
    ps::quantization::psq5::dequantize_psq5_canonical(cv, wdec.data());
    std::vector<float> adec(static_cast<std::size_t>(rows) * kp, 0.0f);
    const uint32_t act_groups = (kp + 127u) / 128u;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t g = 0; g < act_groups; ++g) {
            const uint32_t i0 = g * 128u;
            const uint32_t k_end = std::min(i0 + 128u, k);
            float peak = 0.0f;
            for (uint32_t i = i0; i < k_end; ++i)
                peak = std::max(peak, std::fabs(xr[static_cast<std::size_t>(m) * k + i]));
            const float scale = peak > 0.0f ? peak / 448.0f : 0.0f;
            const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            const uint32_t kp_end = std::min(i0 + 128u, kp);
            for (uint32_t i = i0; i < kp_end; ++i) {
                const float v = (i < k) ? xr[static_cast<std::size_t>(m) * k + i] : 0.0f;
                const uint8_t c = scale > 0.0f ? ps::quantization::codec::e4m3_encode_u8(v * inv) : 0u;
                adec[static_cast<std::size_t>(m) * kp + i] =
                    ps::quantization::codec::e4m3_decode_f32(c) * scale;
            }
        }
    }

    std::vector<float> expect(static_cast<std::size_t>(rows) * n, 0.0f);
    {
        const unsigned workers =
            std::min(16u, std::max(1u, std::thread::hardware_concurrency()));
        const uint32_t chunk = (rows + workers - 1u) / workers;
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (unsigned w = 0; w < workers; ++w) {
            const uint32_t m0 = w * chunk;
            const uint32_t m1 = std::min(rows, m0 + chunk);
            if (m0 >= m1)
                break;
            pool.emplace_back([&, m0, m1]() {
                for (uint32_t m = m0; m < m1; ++m) {
                    for (uint32_t o = 0; o < n; ++o) {
                        double acc = 0.0;
                        for (uint32_t i = 0; i < kp; ++i) {
                            acc += static_cast<double>(adec[static_cast<std::size_t>(m) * kp + i]) *
                                   static_cast<double>(wdec[static_cast<std::size_t>(o) * kp + i]);
                        }
                        expect[static_cast<std::size_t>(m) * n + o] = static_cast<float>(acc);
                    }
                }
            });
        }
        for (auto& t : pool)
            t.join();
    }

    const std::size_t act_codes_bytes =
        ((rows + 15u) / 16u) * static_cast<std::size_t>(code_stride) * 16u;
    const std::size_t act_scales_bytes = static_cast<std::size_t>(rows) * as_bytes;
    const std::size_t out_bytes = static_cast<std::size_t>(rows) * n * 4u;

    auto wmag_a = arena.allocate_aligned(native.magnitude_bytes, 16);
    auto wsg_a = arena.allocate_aligned(native.sign_bytes, 16);
    auto wsc_a = arena.allocate_aligned(native.scale_bytes, 16);
    auto xa = arena.allocate_aligned(hx.size() * sizeof(ps::bf16_t), 16);
    auto ac = arena.allocate_aligned(act_codes_bytes, 16);
    auto as = arena.allocate_aligned(act_scales_bytes, 16);
    auto ya = arena.allocate_aligned(out_bytes, 16);
    if (!wmag_a.ok() || !wsg_a.ok() || !wsc_a.ok() || !xa.ok() || !ac.ok() || !as.ok() || !ya.ok()) {
        std::printf("FAIL: %s alloc\n", label);
        return false;
    }
    auto* d_mag = static_cast<uint8_t*>(wmag_a.release().data());
    auto* d_sg = static_cast<uint8_t*>(wsg_a.release().data());
    auto* d_sc = static_cast<uint8_t*>(wsc_a.release().data());
    auto* d_x = static_cast<ps::bf16_t*>(xa.release().data());
    auto* d_ac = static_cast<uint8_t*>(ac.release().data());
    auto* d_as = static_cast<float*>(as.release().data());
    auto* d_y = static_cast<float*>(ya.release().data());

    HIP_ASSERT(hipMemset(d_ac, 0, act_codes_bytes));
    HIP_ASSERT(hipMemset(d_as, 0, act_scales_bytes));
    HIP_ASSERT(hipMemcpy(d_mag, native.magnitude.get(), native.magnitude_bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_sg, native.sign.get(), native.sign_bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_sc, native.scale.get(), native.scale_bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_x, hx.data(), hx.size() * sizeof(ps::bf16_t), hipMemcpyHostToDevice));

    hipError_t aq = ps::kernel::launch_activation_quantize_e4m3(
        d_x, d_ac, d_as, rows, k, kp, k, code_stride, as_bytes, 0);
    if (aq != hipSuccess) {
        std::printf("FAIL: %s activation quantize: %s\n", label, hipGetErrorString(aq));
        return false;
    }

    const uint32_t wss_bytes = nb * ps::quantization::psq5::kTileRows;
    std::vector<float> got(static_cast<std::size_t>(rows) * n);

    auto run_and_check = [&](const Psq5GemmConfig& config, const char* lbl) -> bool {
        hipError_t hk = ps::kernel::launch_gemm_psq5_w8a8_wmma(
            config, d_mag, d_sg, d_sc, reinterpret_cast<const int8_t*>(d_ac), d_as, d_y,
            ps::kernel::Psq5GemmOutputDType::F32, rows, n, kp,
            wss_bytes, code_stride, as_bytes, n, 0);
        if (hk != hipSuccess) {
            std::printf("FAIL: %s launch: %s\n", lbl, hipGetErrorString(hk));
            ++g_fail;
            return false;
        }
        HIP_ASSERT(hipStreamSynchronize(0));
        HIP_ASSERT(hipMemcpy(got.data(), d_y, out_bytes, hipMemcpyDeviceToHost));

        double worst = 0.0;
        double max_abs = 0.0;
        float max_mag = 0.0f;
        std::size_t worst_idx = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(expect[i]));
            const double e = static_cast<double>(expect[i]);
            if (diff > max_abs) { max_abs = diff; }
            if (std::fabs(static_cast<double>(expect[i])) > std::fabs(max_mag))
                max_mag = expect[i];
            if (std::fabs(e) > 1e-3) {
                const double rel = diff / std::fabs(e);
                if (rel > worst) { worst = rel; worst_idx = i; }
            }
        }

        const float atol = std::max(1.0f, max_mag) * 1e-4f;
        if (max_abs > atol) {
            std::printf("FAIL: %s rows=%u n=%u k=%u max_abs=%.6g max_rel=%.6g atol=%.6g at [%zu] got=%.6g expect=%.6g\n",
                        lbl, rows, n, k, max_abs, worst, atol, worst_idx,
                        static_cast<double>(got[worst_idx]), static_cast<double>(expect[worst_idx]));
            ++g_fail;
            return false;
        }
        g_worst_rel = std::max(g_worst_rel, worst);
        ++g_pass;
        std::printf("PASS: %-28s rows=%3u n=%4u k=%4u max_abs=%.3e max_rel=%.3e\n",
                    lbl, rows, n, k, max_abs, worst);
        return true;
    };

    if (!run_and_check(config_for(rows), label))
        return false;

    if (rows == 1u && (n & 15u) == 0u) {
        for (uint32_t unroll : {0u, 4u, 16u}) {
            hipError_t dk = ps::kernel::launch_gemm_psq5_w8a8_wmma_decode1(
                d_mag, d_sg, d_sc, reinterpret_cast<const int8_t*>(d_ac), d_as, d_y,
                ps::kernel::Psq5GemmOutputDType::F32, rows, n, kp,
                wss_bytes, code_stride, as_bytes, n, unroll, 0);
            if (dk != hipSuccess) {
                std::printf("FAIL: %s decode1 u=%u: %s\n", label, unroll, hipGetErrorString(dk));
                ++g_fail;
                return false;
            }
            HIP_ASSERT(hipStreamSynchronize(0));
            std::vector<float> d1(static_cast<std::size_t>(rows) * n);
            HIP_ASSERT(hipMemcpy(d1.data(), d_y, out_bytes, hipMemcpyDeviceToHost));
            if (std::memcmp(d1.data(), got.data(), out_bytes) != 0) {
                std::printf("FAIL: %s decode1 u=%u not bit-exact\n", label, unroll);
                ++g_fail;
                return false;
            }
        }
    }

    if (two_d != nullptr) {
        char label2d[64];
        std::snprintf(label2d, sizeof(label2d), "%s_2d", label);
        run_and_check(*two_d, label2d);
    }
    if (wide != nullptr) {
        char labelw[64];
        std::snprintf(labelw, sizeof(labelw), "%s_wide", label);
        run_and_check(*wide, labelw);
    }
    return true;
}

}

int main() {
    auto arena_result = ps::gpu::GpuArena::create(0, 6144ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: no GPU arena\n");
        return 0;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    struct Case {
        uint32_t rows, n, k;
        const char* label;
        const Psq5GemmConfig* two_d;
        const Psq5GemmConfig* wide;
    };
    const Psq5GemmConfig two_d_config{Psq5GemmConfigId::Prefill2D};
    const Psq5GemmConfig wide_config{Psq5GemmConfigId::Prefill2DWide};
    const Psq5GemmConfig wstage_config{Psq5GemmConfigId::Prefill2DWStage};
    const Case cases[] = {
        {1, 16, 128, "decode1_small", nullptr, nullptr},
        {1, 64, 128, "decode1_n64", nullptr, nullptr},
        {1, 5120, 5120, "decode1_27b_qproj", nullptr, nullptr},
        {1, 17408, 5120, "decode1_27b_gate", nullptr, nullptr},
        {1, 5120, 17408, "decode1_27b_down", nullptr, nullptr},
        {1, 10240, 5120, "decode1_gdn_qkv", nullptr, nullptr},
        {2, 64, 128, "mb1_rows2", nullptr, nullptr},
        {8, 128, 128, "mb1_rows8", nullptr, nullptr},
        {16, 256, 256, "mb2_rows16", nullptr, nullptr},
        {17, 128, 256, "mb2_rows17", nullptr, nullptr},
        {32, 256, 512, "mb4_rows32", nullptr, nullptr},
        {64, 512, 512, "mb8_rows64", nullptr, nullptr},
        {128, 1024, 1024, "mb8_rows128", nullptr, nullptr},
        {3, 96, 384, "mb1_odd", nullptr, nullptr},
        {5, 160, 640, "mb1_uneven", nullptr, nullptr},
        {1, 12288, 5120, "27b_q", nullptr, nullptr},
        {1, 6144, 5120, "27b_z", nullptr, nullptr},
        {1, 1024, 5120, "27b_kv", nullptr, nullptr},
        {1, 5120, 6144, "27b_o", nullptr, nullptr},
        {2048, 5120, 17408, "prefill_down", &two_d_config, nullptr},
        {2048, 17408, 5120, "prefill_gate", &two_d_config, nullptr},
        {2048, 12288, 5120, "prefill_q", &two_d_config, nullptr},
        {2048, 1024, 5120, "prefill_kv", &two_d_config, nullptr},
        {512, 5120, 6144, "prefill_o", &two_d_config, nullptr},
        {512, 5120, 17408, "prefill_down_512", &two_d_config, nullptr},
        {512, 17408, 5120, "prefill_gate_512", &two_d_config, nullptr},
        {2048, 5120, 6144, "prefill_o_2048", &two_d_config, nullptr},
        {2048, 5120, 17408, "wide_down", nullptr, &wide_config},
        {2048, 17408, 5120, "wide_gate", nullptr, &wide_config},
        {2048, 12288, 5120, "wide_q", nullptr, &wide_config},
        {2048, 1024, 5120, "wide_kv", nullptr, &wide_config},
        {512, 5120, 6144, "wide_o_512", nullptr, &wide_config},
        {512, 17408, 5120, "wide_gate_512", nullptr, &wide_config},
        {2048, 6144, 5120, "wide_z", nullptr, &wide_config},
        {1024, 1024, 1024, "wide_pow2", nullptr, &wide_config},
        {2048, 5120, 17408, "wstage_down", nullptr, &wstage_config},
        {2048, 17408, 5120, "wstage_gate", nullptr, &wstage_config},
        {2048, 12288, 5120, "wstage_q", nullptr, &wstage_config},
        {2048, 1024, 5120, "wstage_kv", nullptr, &wstage_config},
        {512, 5120, 6144, "wstage_o_512", nullptr, &wstage_config},
        {512, 17408, 5120, "wstage_gate_512", nullptr, &wstage_config},
        {1024, 1024, 1024, "wstage_pow2", nullptr, &wstage_config},
    };
    for (const Case& c : cases)
        run_case(arena, c.rows, c.n, c.k, c.label, c.two_d, c.wide);

    std::printf("\n=== PSQ5 GEMM: %d passed, %d failed, worst_rel=%.3e ===\n",
                g_pass, g_fail, g_worst_rel);
    return g_fail != 0 ? 1 : 0;
}
