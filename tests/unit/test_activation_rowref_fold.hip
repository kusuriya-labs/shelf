#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/quantization/codec/e4m3.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kRows = 32u;
constexpr uint32_t kK = 512u;
constexpr uint32_t kGroup = 128u;
constexpr uint32_t kGroups = kK / kGroup;
constexpr uint32_t kScaleStride = kGroups * 4u;

int g_fail = 0;

void fail(const char* msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg);
}

bool hip_ok(hipError_t e, const char* what) {
    if (e == hipSuccess)
        return true;
    std::printf("FAIL %s: %s\n", what, hipGetErrorString(e));
    return false;
}

// Same rule as the device kernel: the smallest power-of-two scale whose 448 ceiling
// still covers the group peak.
int32_t group_exponent(float amax) {
    if (!(amax > 0.0f) || !std::isfinite(amax))
        return -127;
    int32_t e = static_cast<int32_t>(std::ceil(std::log2(static_cast<double>(amax) / 448.0)));
    while (std::ldexp(448.0, e) < static_cast<double>(amax))
        ++e;
    while (e > -127 && std::ldexp(448.0, e - 1) >= static_cast<double>(amax))
        --e;
    return e;
}

}  // namespace

int main() {
    if (!hip_ok(hipSetDevice(0), "hipSetDevice"))
        return 1;
    hipStream_t stream = nullptr;
    if (!hip_ok(hipStreamCreate(&stream), "stream create"))
        return 1;

    // Group peaks are spread over ~20 octaves so the fold crosses the E4M3 normal
    // boundary, the subnormal band, and the collapse-to-zero tail.
    std::vector<__hip_bfloat16> h_in(static_cast<size_t>(kRows) * kK);
    std::mt19937 rng(20261103u);
    std::uniform_real_distribution<float> frac(0.0f, 1.0f);
    std::vector<int32_t> exp_of(static_cast<size_t>(kRows) * kGroups, 0);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t g = 0; g < kGroups; ++g) {
            const int32_t e = -6 + 5 * static_cast<int32_t>(g) - static_cast<int32_t>(r % 3);
            const float peak = static_cast<float>(std::ldexp(1.0, e));
            for (uint32_t j = 0; j < kGroup; ++j) {
                float v = frac(rng) * peak;
                if ((j % 17u) == 0u)
                    v = -v;
                if (j == 0u)
                    v = peak;
                h_in[static_cast<size_t>(r) * kK + static_cast<size_t>(g) * kGroup + j] =
                    __float2bfloat16(v);
            }
        }
    }

    // Derive the exponents from the BF16 values that actually reach the kernel.
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t g = 0; g < kGroups; ++g) {
            float peak = 0.0f;
            for (uint32_t j = 0; j < kGroup; ++j)
                peak = std::max(peak, std::fabs(__bfloat162float(
                    h_in[static_cast<size_t>(r) * kK + static_cast<size_t>(g) * kGroup + j])));
            exp_of[static_cast<size_t>(r) * kGroups + g] = group_exponent(peak);
        }
    }
    std::printf("rowref fold: exponents span %d .. %d octaves\n",
                *std::min_element(exp_of.begin(), exp_of.end()),
                *std::max_element(exp_of.begin(), exp_of.end()));

    const size_t codes_bytes = static_cast<size_t>(kRows) * kK;
    const size_t scales_bytes = static_cast<size_t>(kRows) * kScaleStride;
    __hip_bfloat16* d_in = nullptr;
    uint8_t* d_codes = nullptr;
    float* d_scales = nullptr;
    if (!hip_ok(hipMalloc((void**)&d_in, h_in.size() * sizeof(__hip_bfloat16)), "malloc in") ||
        !hip_ok(hipMalloc((void**)&d_codes, codes_bytes), "malloc codes") ||
        !hip_ok(hipMalloc((void**)&d_scales, scales_bytes), "malloc scales")) {
        return 1;
    }
    hipMemset(d_codes, 0xCD, codes_bytes);
    hipMemset(d_scales, 0, scales_bytes);
    if (!hip_ok(hipMemcpy(d_in, h_in.data(), h_in.size() * sizeof(__hip_bfloat16),
                          hipMemcpyHostToDevice), "h2d in"))
        return 1;

    const hipError_t herr = shelf::kernel::launch_activation_quantize_mxfp4_rowref(
        reinterpret_cast<const shelf::bf16_t*>(d_in), d_codes, d_scales, kRows, kK, kK,
        kK, kK, kScaleStride, stream);
    if (!hip_ok(herr, "rowref quantize launch"))
        return 1;
    if (!hip_ok(hipStreamSynchronize(stream), "rowref quantize sync"))
        return 1;

    std::vector<uint8_t> got_codes(codes_bytes, 0u);
    std::vector<float> got_scales(kRows * kGroups, 0.0f);
    if (!hip_ok(hipMemcpy(got_codes.data(), d_codes, codes_bytes, hipMemcpyDeviceToHost),
                "d2h codes") ||
        !hip_ok(hipMemcpy(got_scales.data(), d_scales, got_scales.size() * sizeof(float),
                          hipMemcpyDeviceToHost), "d2h scales"))
        return 1;

    std::size_t mismatch = 0;
    std::size_t folded_inexact = 0;
    std::size_t folded_zero = 0;
    std::size_t compared = 0;
    for (uint32_t r = 0; r < kRows; ++r) {
        int32_t ref = -127;
        for (uint32_t g = 0; g < kGroups; ++g)
            ref = std::max(ref, exp_of[static_cast<size_t>(r) * kGroups + g]);

        const double want_ref = std::ldexp(1.0, ref);
        const double got_ref = got_scales[static_cast<size_t>(r) * kGroups];
        if (got_ref != want_ref) {
            ++mismatch;
            std::printf("row %u ref got %.9g want %.9g\n", r, got_ref, want_ref);
        }
        for (uint32_t g = 1u; g < kGroups; ++g)
            if (got_scales[static_cast<size_t>(r) * kGroups + g] != 0.0f)
                ++mismatch;

        const uint8_t* tile = got_codes.data() +
            static_cast<size_t>(r >> 4u) * (static_cast<size_t>(kK) * 16u);
        for (uint32_t g = 0; g < kGroups; ++g) {
            const int32_t e = exp_of[static_cast<size_t>(r) * kGroups + g];
            const int32_t d = ref - e;
            for (uint32_t j = 0; j < kGroup; ++j) {
                const uint32_t i = static_cast<size_t>(g) * kGroup + j;
                const float x = __bfloat162float(h_in[static_cast<size_t>(r) * kK + i]);
                const uint8_t c = shelf::quantization::codec::e4m3_encode_u8(
                    x * static_cast<float>(std::ldexp(1.0, -e)));
                const float mid = shelf::quantization::codec::e4m3_decode_f32(c);
                const uint8_t cf = shelf::quantization::codec::e4m3_encode_u8(
                    mid * static_cast<float>(std::ldexp(1.0, -d)));
                const size_t off = static_cast<size_t>(i >> 6u) *
                                       shelf::quantization::fp8::kFragBlockBytes +
                                   shelf::quantization::fp8::fp8_frag_offset(r & 15u, i & 63u);
                if (tile[off] != cf) {
                    if (mismatch < 8u)
                        std::printf("row %u g %u j %u code got %02X want %02X (e=%d d=%d)\n",
                                    r, g, j, tile[off], cf, e, d);
                    ++mismatch;
                }
                ++compared;
                if (shelf::quantization::codec::e4m3_decode_f32(cf) !=
                    mid * std::ldexp(1.0, -d))
                    ++folded_inexact;
                if (mid != 0.0f && cf == 0u)
                    ++folded_zero;
            }
        }
    }

    std::printf("rowref fold: compared=%zu mismatch=%zu fold_inexact=%zu fold_to_zero=%zu\n",
                compared, mismatch, folded_inexact, folded_zero);
    if (mismatch != 0u)
        fail("device rowref quantize disagrees with the host model");
    if (folded_inexact == 0u)
        fail("the fixture never crossed the exact fold band (test is not exercising the fold)");

    (void)hipFree(d_in);
    (void)hipFree(d_codes);
    (void)hipFree(d_scales);
    (void)hipStreamDestroy(stream);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
