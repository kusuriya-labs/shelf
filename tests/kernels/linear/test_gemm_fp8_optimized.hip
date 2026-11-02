#include <phaseshift/kernels/optimized/linear/fp8.h>
#include <phaseshift/kernels/optimized/activation_quantize.h>
#include <phaseshift/kernels/correctness/standalone/gemm_block_scaled_correctness.h>
#include <phaseshift/quantization/fp8/fp8.h>
#include <phaseshift/quantization/fp8/fp8_native.h>
#include <phaseshift/quantization/codec/e4m3.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_ASSERT(x) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        std::printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(err)); \
        return false; \
    } \
} while (0)

using ps::kernel::Fp8GemmConfig;
using ps::kernel::Fp8GemmConfigId;

namespace {

int g_pass = 0;
int g_fail = 0;
double g_worst_rel = 0.0;
float g_worst_abs = 0.0f;
float g_worst_mag = 0.0f;

ps::bf16_t f32_to_bf16(float f) {
    uint32_t ui = 0;
    std::memcpy(&ui, &f, sizeof(float));
    const uint16_t bf = static_cast<uint16_t>((ui + 0x8000u) >> 16);
    return ps::bf16_t{bf};
}

float bf16_to_f32(ps::bf16_t b) {
    const uint32_t ui = static_cast<uint32_t>(b.data) << 16;
    float f = 0.0f;
    std::memcpy(&f, &ui, sizeof(float));
    return f;
}

Fp8GemmConfig config_for(uint32_t rows) {
    if (rows >= 8u * ps::kernel::kFp8GemmRowTile) return {Fp8GemmConfigId::RowBlock8};
    if (rows >= 4u * ps::kernel::kFp8GemmRowTile) return {Fp8GemmConfigId::RowBlock4};
    if (rows > ps::kernel::kFp8GemmRowTile) return {Fp8GemmConfigId::RowBlock2};
    return {Fp8GemmConfigId::RowBlock1};
}

const char* config_name(Fp8GemmConfigId id) {
    switch (id) {
        case Fp8GemmConfigId::RowBlock1: return "MB1";
        case Fp8GemmConfigId::RowBlock2: return "MB2";
        case Fp8GemmConfigId::RowBlock4: return "MB4";
        case Fp8GemmConfigId::RowBlock8: return "MB8";
        case Fp8GemmConfigId::Prefill2D: return "2D";
    }
    return "?";
}

bool run_case(ps::gpu::GpuArena& arena, Fp8GemmConfig config, uint32_t rows,
              uint32_t n, uint32_t k, const char* label,
              uint32_t weight_scale_group = 128) {
    std::mt19937 rng(20260921u + rows * 31u + n * 7u + k);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-2.0f, 2.0f);

    std::vector<float> w(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(rows) * k);
    for (auto& v : w) v = wdist(rng);
    for (auto& v : x) v = xdist(rng);

    ps::quantization::fp8::Fp8BlockCanonicalStore store;
    const uint32_t kp = static_cast<uint32_t>((k + 127u) & ~127u);
    store.scale_block_k = weight_scale_group;
    store.init(1, n, k, kp);
    store.quantize_matrix(0, w.data(), k);

    ps::quantization::fp8::Fp8NativeHost native;
    if (!ps::quantization::fp8::preshuffle_fp8_native(store.view(), native)) {
        std::printf("FAIL: %s preshuffle\n", label);
        return false;
    }

    const uint32_t scale_k = kp / weight_scale_group;
    const uint32_t weight_scale_stride_bytes = scale_k * 4u;
    const uint32_t code_stride = kp;
    const uint32_t as_floats = (kp + 127u) / 128u;
    const uint32_t as_bytes = as_floats * 4u;

    std::vector<ps::bf16_t> hx(static_cast<std::size_t>(rows) * k);
    std::vector<float> xr(x.size());
    for (std::size_t i = 0; i < hx.size(); ++i) {
        hx[i] = f32_to_bf16(x[i]);
        xr[i] = bf16_to_f32(hx[i]);
    }

    std::vector<float> wdec(static_cast<std::size_t>(n) * kp, 0.0f);
    ps::quantization::fp8::dequantize_fp8_canonical_matrix(
        store.view(), 0, wdec.data(), kp);
    std::vector<float> adec(static_cast<std::size_t>(rows) * kp, 0.0f);
    const uint32_t act_groups = (kp + 127u) / 128u;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t g = 0; g < act_groups; ++g) {
            const uint32_t i0 = g * 128u;
            const uint32_t k_end = i0 + 128u < k ? i0 + 128u : k;
            float peak = 0.0f;
            for (uint32_t i = i0; i < k_end; ++i)
                peak = std::max(peak, std::fabs(xr[static_cast<std::size_t>(m) * k + i]));
            const float scale = peak > 0.0f ? peak / 448.0f : 0.0f;
            const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            const uint32_t kp_end = i0 + 128u < kp ? i0 + 128u : kp;
            for (uint32_t i = i0; i < kp_end; ++i) {
                const float v = (i < k) ? xr[static_cast<std::size_t>(m) * k + i] : 0.0f;
                const uint8_t c = scale > 0.0f ? ps::quantization::codec::e4m3_encode_u8(v * inv) : 0u;
                adec[static_cast<std::size_t>(m) * kp + i] =
                    ps::quantization::codec::e4m3_decode_f32(c) * scale;
            }
        }
    }

    const std::size_t act_codes_bytes = ((rows + 15u) / 16u) * static_cast<std::size_t>(code_stride) * 16u;
    const std::size_t act_scales_bytes = static_cast<std::size_t>(rows) * as_bytes;
    const std::size_t out_f32_bytes = static_cast<std::size_t>(rows) * n * 4u;

    auto wc = arena.allocate_aligned(native.codes_bytes, 16);
    auto ws = arena.allocate_aligned(native.scale_count * sizeof(float), 16);
    auto xa = arena.allocate_aligned(hx.size() * sizeof(ps::bf16_t), 16);
    auto ac = arena.allocate_aligned(act_codes_bytes, 16);
    auto as = arena.allocate_aligned(act_scales_bytes, 16);
    auto ya = arena.allocate_aligned(out_f32_bytes, 16);
    if (!wc.ok() || !ws.ok() || !xa.ok() || !ac.ok() || !as.ok() || !ya.ok()) {
        std::printf("FAIL: %s alloc\n", label);
        return false;
    }
    auto* d_wc = static_cast<uint8_t*>(wc.release().data());
    auto* d_ws = static_cast<float*>(ws.release().data());
    auto* d_x = static_cast<ps::bf16_t*>(xa.release().data());
    auto* d_ac = static_cast<uint8_t*>(ac.release().data());
    auto* d_as = static_cast<float*>(as.release().data());
    auto* d_y = static_cast<float*>(ya.release().data());

    HIP_ASSERT(hipMemset(d_ac, 0, act_codes_bytes));
    HIP_ASSERT(hipMemset(d_as, 0, act_scales_bytes));
    HIP_ASSERT(hipMemcpy(d_wc, native.codes.get(), native.codes_bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_ws, native.scales.get(), native.scale_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_x, hx.data(), hx.size() * sizeof(ps::bf16_t), hipMemcpyHostToDevice));

    hipError_t aq = ps::kernel::launch_activation_quantize_e4m3(
        d_x, d_ac, d_as, rows, k, kp, k, code_stride, as_bytes, 0);
    if (aq != hipSuccess) {
        std::printf("FAIL: %s activation quantize: %s\n", label, hipGetErrorString(aq));
        return false;
    }
    hipError_t hk = ps::kernel::launch_gemm_fp8_w8a8_wmma(
        config, d_wc, d_ws, reinterpret_cast<const int8_t*>(d_ac), d_as, d_y,
        ps::kernel::Fp8GemmOutputDType::F32, rows, n, kp,
        weight_scale_stride_bytes, code_stride, as_bytes, n, 0, weight_scale_group);
    if (hk != hipSuccess) {
        std::printf("FAIL: %s launch: %s\n", label, hipGetErrorString(hk));
        return false;
    }
    HIP_ASSERT(hipStreamSynchronize(0));

    std::vector<float> got(static_cast<std::size_t>(rows) * n);
    HIP_ASSERT(hipMemcpy(got.data(), d_y, out_f32_bytes, hipMemcpyDeviceToHost));

    if (rows == 1u) {
        const bool gate = (n & 15u) == 0u;
        if (!gate) {
            hipError_t dk = ps::kernel::launch_gemm_fp8_w8a8_wmma_decode1(
                d_wc, d_ws, reinterpret_cast<const int8_t*>(d_ac), d_as, d_y,
                ps::kernel::Fp8GemmOutputDType::F32, rows, n, kp,
                weight_scale_stride_bytes, code_stride, as_bytes, n, 0u, 0, weight_scale_group);
            if (dk != hipErrorNotSupported) {
                std::printf("FAIL: %s decode1 gate n=%u kp=%u: %s\n", label, n, kp,
                            hipGetErrorString(dk));
                return false;
            }
        } else {
            for (uint32_t unroll : {0u, 2u, 4u, 8u, 16u}) {
                hipError_t dk = ps::kernel::launch_gemm_fp8_w8a8_wmma_decode1(
                    d_wc, d_ws, reinterpret_cast<const int8_t*>(d_ac), d_as, d_y,
                    ps::kernel::Fp8GemmOutputDType::F32, rows, n, kp,
                    weight_scale_stride_bytes, code_stride, as_bytes, n, unroll, 0, weight_scale_group);
                if (dk != hipSuccess) {
                    std::printf("FAIL: %s decode1 F32 u=%u launch: %s\n", label, unroll,
                                hipGetErrorString(dk));
                    return false;
                }
                HIP_ASSERT(hipStreamSynchronize(0));
                std::vector<float> d1(static_cast<std::size_t>(rows) * n);
                HIP_ASSERT(hipMemcpy(d1.data(), d_y, out_f32_bytes, hipMemcpyDeviceToHost));
                if (std::memcmp(d1.data(), got.data(), out_f32_bytes) != 0) {
                    std::printf("FAIL: %s decode1 F32 u=%u not bit-exact n=%u k=%u kp=%u\n",
                                label, unroll, n, k, kp);
                    return false;
                }
            }
            hipError_t ha = ps::kernel::launch_gemm_fp8_w8a8_wmma_auto(
                config, d_wc, d_ws, reinterpret_cast<const int8_t*>(d_ac), d_as, d_y,
                ps::kernel::Fp8GemmOutputDType::F32, rows, n, kp,
                weight_scale_stride_bytes, code_stride, as_bytes, n, 0, weight_scale_group);
            if (ha != hipSuccess) {
                std::printf("FAIL: %s auto F32 launch: %s\n", label, hipGetErrorString(ha));
                return false;
            }
            HIP_ASSERT(hipStreamSynchronize(0));
            std::vector<float> a1(static_cast<std::size_t>(rows) * n);
            HIP_ASSERT(hipMemcpy(a1.data(), d_y, out_f32_bytes, hipMemcpyDeviceToHost));
            if (std::memcmp(a1.data(), got.data(), out_f32_bytes) != 0) {
                std::printf("FAIL: %s auto F32 not bit-exact vs RowBlock1 n=%u k=%u\n",
                            label, n, k);
                return false;
            }

            const std::size_t out_bf16_bytes = static_cast<std::size_t>(rows) * n * 2u;
            std::vector<uint8_t> gbf(out_bf16_bytes);
            hipError_t hg = ps::kernel::launch_gemm_fp8_w8a8_wmma(
                config, d_wc, d_ws, reinterpret_cast<const int8_t*>(d_ac), d_as, d_y,
                ps::kernel::Fp8GemmOutputDType::BF16, rows, n, kp,
                weight_scale_stride_bytes, code_stride, as_bytes, n, 0, weight_scale_group);
            if (hg != hipSuccess) {
                std::printf("FAIL: %s generic BF16 launch: %s\n", label, hipGetErrorString(hg));
                return false;
            }
            HIP_ASSERT(hipStreamSynchronize(0));
            HIP_ASSERT(hipMemcpy(gbf.data(), d_y, out_bf16_bytes, hipMemcpyDeviceToHost));
            for (uint32_t unroll : {0u, 8u, 16u}) {
                hipError_t dk = ps::kernel::launch_gemm_fp8_w8a8_wmma_decode1(
                    d_wc, d_ws, reinterpret_cast<const int8_t*>(d_ac), d_as, d_y,
                    ps::kernel::Fp8GemmOutputDType::BF16, rows, n, kp,
                    weight_scale_stride_bytes, code_stride, as_bytes, n, unroll, 0, weight_scale_group);
                if (dk != hipSuccess) {
                    std::printf("FAIL: %s decode1 BF16 u=%u launch: %s\n", label, unroll,
                                hipGetErrorString(dk));
                    return false;
                }
                HIP_ASSERT(hipStreamSynchronize(0));
                std::vector<uint8_t> dbf(out_bf16_bytes);
                HIP_ASSERT(hipMemcpy(dbf.data(), d_y, out_bf16_bytes, hipMemcpyDeviceToHost));
                if (std::memcmp(dbf.data(), gbf.data(), out_bf16_bytes) != 0) {
                    std::printf("FAIL: %s decode1 BF16 u=%u not bit-exact n=%u k=%u kp=%u\n",
                                label, unroll, n, k, kp);
                    return false;
                }
            }
        }
    }

    float max_abs = 0.0f;
    double max_rel = 0.0;
    float max_mag = 0.0f;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t o = 0; o < n; ++o) {
            double expected = 0.0;
            for (uint32_t kk = 0; kk < kp; ++kk) {
                expected += static_cast<double>(wdec[static_cast<std::size_t>(o) * kp + kk]) *
                            static_cast<double>(adec[static_cast<std::size_t>(m) * kp + kk]);
            }
            const float e = static_cast<float>(expected);
            const float g = got[static_cast<std::size_t>(m) * n + o];
            max_mag = std::max(max_mag, std::fabs(e));
            const float diff = std::fabs(g - e);
            max_abs = std::max(max_abs, diff);
            if (std::fabs(e) > 1e-3f)
                max_rel = std::max(max_rel, static_cast<double>(diff) / std::fabs(e));
        }
    }
    if (max_rel > g_worst_rel) { g_worst_rel = max_rel; g_worst_abs = max_abs; g_worst_mag = max_mag; }
    const float atol = std::max(1.0f, max_mag) * 1e-4f;
    const bool ok = max_abs <= atol;
    if (!ok)
        std::printf("FAIL: %s rows=%u N=%u K=%u Kp=%u %s max_abs=%.6g max_rel=%.4f atol=%.4f\n",
                    label, rows, n, k, kp, config_name(config.id), max_abs, max_rel, atol);
    return ok;
}

}  // namespace

int main() {
    std::printf("=== FP8 K128N128 W8A8 WMMA GEMM Tests ===\n");

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    auto arena_result = ps::gpu::GpuArena::create(0, 1024ull * 1024 * 1024);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    const uint32_t rows_list[] = {1, 2, 4, 8, 16, 17, 33, 64, 128};
    const uint32_t n_list[] = {16, 17, 128, 129, 130, 256};
    const uint32_t k_list[] = {128, 130, 256, 512, 2560};

    char label[128];
    for (uint32_t rows : rows_list) {
        for (uint32_t n : n_list) {
            for (uint32_t k : k_list) {
                if (static_cast<std::size_t>(rows) * n * k > 4000000u) continue;
                if (run_case(arena, config_for(rows), rows, n, k, "fp8b128")) ++g_pass; else ++g_fail;
            }
        }
    }

    for (uint32_t rows : {16u, 17u, 64u, 128u, 129u}) {
        for (Fp8GemmConfigId id : {Fp8GemmConfigId::RowBlock1,
                                           Fp8GemmConfigId::RowBlock2,
                                           Fp8GemmConfigId::RowBlock4,
                                           Fp8GemmConfigId::RowBlock8}) {
            std::snprintf(label, sizeof(label), "fp8b128-force-%s", config_name(id));
            if (run_case(arena, Fp8GemmConfig{id}, rows, 130u, 256u, label)) ++g_pass; else ++g_fail;
        }
    }

    for (uint32_t rows : {512u, 2048u}) {
        if (run_case(arena, config_for(rows), rows, 64u, 256u, "fp8b128-large")) ++g_pass; else ++g_fail;
        if (run_case(arena, Fp8GemmConfig{Fp8GemmConfigId::RowBlock1},
                     rows, 64u, 256u, "fp8b128-large-mb1")) ++g_pass; else ++g_fail;
    }

    const uint32_t prod_shapes[][2] = {
        {17408u, 5120u}, {5120u, 17408u}, {5120u, 6144u}, {10240u, 5120u},
        {6144u, 5120u}, {12288u, 5120u}, {1024u, 5120u}, {48u, 5120u},
        {8704u, 5120u}, {5120u, 8704u}, {3072u, 5120u},
        {512u, 5120u}, {5120u, 3072u}, {5120u, 5120u}, {24u, 5120u},
    };
    auto prod_arena_result = ps::gpu::GpuArena::create(0, 768ull * 1024 * 1024);
    if (!prod_arena_result.ok()) {
        std::printf("FAIL: prod arena create: %s\n",
                    prod_arena_result.status().message().c_str());
        return 1;
    }
    auto prod_arena = prod_arena_result.release();
    for (const auto& s : prod_shapes) {
        if (run_case(prod_arena, Fp8GemmConfig{Fp8GemmConfigId::RowBlock1},
                     1u, s[0], s[1], "fp8b128-decode1-prod"))
            ++g_pass;
        else
            ++g_fail;
    }

    for (uint32_t rows : {256u, 512u}) {
        for (uint32_t n : {64u, 128u}) {
            if (run_case(arena, Fp8GemmConfig{Fp8GemmConfigId::Prefill2D},
                         rows, n, 256u, "fp8b128-2d")) ++g_pass; else ++g_fail;
        }
    }

    for (uint32_t rows : {1u, 16u, 64u, 128u}) {
        if (run_case(arena, config_for(rows), rows, 128u, 256u, "fp8g64", 64u)) ++g_pass;
        else ++g_fail;
    }
    for (uint32_t rows : {1u, 17u, 64u}) {
        for (Fp8GemmConfigId id : {Fp8GemmConfigId::RowBlock1,
                                   Fp8GemmConfigId::RowBlock4}) {
            std::snprintf(label, sizeof(label), "fp8g64-force-%s", config_name(id));
            if (run_case(arena, Fp8GemmConfig{id}, rows, 130u, 256u, label, 64u)) ++g_pass;
            else ++g_fail;
        }
    }
    for (uint32_t rows : {256u, 512u}) {
        for (uint32_t n : {64u, 128u}) {
            if (run_case(arena, Fp8GemmConfig{Fp8GemmConfigId::Prefill2D},
                         rows, n, 256u, "fp8g64-2d", 64u)) ++g_pass;
            else ++g_fail;
        }
    }

    std::printf("\nWorst relative error vs exact host reference: %.3e (abs %.3e at |y|=%.3e)\n",
                g_worst_rel, static_cast<double>(g_worst_abs), static_cast<double>(g_worst_mag));
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
