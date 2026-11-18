#include <shelf/quantization/codec/e4m3.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include <shelf/quantization/mxfp4/mxfp4_cpu_expert.h>

#include <cstddef>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SHELF_MXFP4_CPU_X86 1
#endif

namespace shelf::quantization::mxfp4 {
namespace {

constexpr std::uint32_t kBlockK = 32;
constexpr std::uint32_t kBlockBytes = 256;

const float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

alignas(64) const float kE4M3Pow[16] = {
    1.953125e-03f, 1.953125e-03f, 3.90625e-03f, 7.8125e-03f,
    1.5625e-02f,   3.125e-02f,    6.25e-02f,    1.25e-01f,
    2.5e-01f,      5.0e-01f,      1.0f,         2.0f,
    4.0f,          8.0f,          16.0f,        32.0f};

inline float e8m0_to_float(std::uint8_t exponent) {
    if (exponent == 0u) return 5.877471754111438e-39f;
    const std::uint32_t bits = static_cast<std::uint32_t>(exponent) << 23;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void deswizzle_lane(const std::uint8_t* tile, std::uint32_t lane, std::uint32_t k_padded,
                    std::uint8_t* dst) {
    for (std::uint32_t i = 0; i < k_padded; ++i) {
        dst[i] = tile[static_cast<std::size_t>(i >> 6u) * fp8::kFragBlockBytes +
                      fp8::fp8_frag_offset(lane, i & 63u)];
    }
}

void gemv_rows_scalar(const CpuExpertPlane& w, std::uint32_t slot,
                      const CpuActivationTile& a, std::uint32_t lane, float* out,
                      std::uint32_t out_stride, std::uint32_t begin, std::uint32_t end) {
    const std::uint8_t* codes = w.codes + static_cast<std::size_t>(slot) * w.codes_batch_stride;
    const std::uint8_t* scales =
        w.scales + static_cast<std::size_t>(slot) * w.scales_batch_stride;
    std::vector<std::uint8_t> linear(w.k_padded);
    deswizzle_lane(a.codes, lane, w.k_padded, linear.data());
    const std::uint8_t* acodes = linear.data();
    const float row_scale = *a.row_scale;
    for (std::uint32_t o = begin; o < end; ++o) {
        const std::uint8_t* row_codes = codes +
            static_cast<std::size_t>(o >> 4u) * w.blocks * kBlockBytes +
            static_cast<std::size_t>(o & 15u) * 16u;
        const std::uint8_t* row_scales =
            scales + static_cast<std::size_t>(o) * w.scale_row_stride;
        float acc = 0.0f;
        for (std::uint32_t ib = 0; ib < w.blocks; ++ib) {
            const std::uint8_t* packed = row_codes + static_cast<std::size_t>(ib) * kBlockBytes;
            const float ws = e8m0_to_float(row_scales[ib]);
            const std::uint8_t* x = acodes + static_cast<std::size_t>(ib) * kBlockK;
            float partial = 0.0f;
            for (std::uint32_t b = 0; b < 16u; ++b) {
                const float lo = kE2M1[packed[b] & 0xFu];
                const float hi = kE2M1[packed[b] >> 4u];
                partial += lo * codec::e4m3_decode_f32(x[b]);
                partial += hi * codec::e4m3_decode_f32(x[16u + b]);
            }
            acc += partial * ws;
        }
        out[static_cast<std::size_t>(o) * out_stride] = acc * row_scale;
    }
}

#ifdef SHELF_MXFP4_CPU_X86
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline __m512 e4m3_decode_vector(__m512i code, __m512i mant_mask, __m512i expo_mask,
                                 __m512i sign_mask, __m512i eight, __m512 pow_table) {
    const __m512i mant = _mm512_and_si512(code, mant_mask);
    const __m512i expo = _mm512_and_si512(code, expo_mask);
    const __mmask16 subnormal = _mm512_cmpeq_epi32_mask(expo, _mm512_setzero_si512());
    const __m512i ms = _mm512_mask_add_epi32(mant, ~subnormal, mant, eight);
    const __m512 shift = _mm512_permutexvar_ps(_mm512_srli_epi32(expo, 3), pow_table);
    const __m512 mag = _mm512_mul_ps(_mm512_cvtepi32_ps(ms), shift);
    const __mmask16 negative = _mm512_test_epi32_mask(code, sign_mask);
    return _mm512_mask_sub_ps(mag, negative, _mm512_setzero_ps(), mag);
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
void gemv_rows_avx512(const CpuExpertPlane& w, std::uint32_t slot,
                      const CpuActivationTile& a, std::uint32_t lane, float* out,
                      std::uint32_t out_stride, std::uint32_t begin, std::uint32_t end) {
    const std::uint8_t* codes = w.codes + static_cast<std::size_t>(slot) * w.codes_batch_stride;
    const std::uint8_t* scales =
        w.scales + static_cast<std::size_t>(slot) * w.scales_batch_stride;
    std::vector<std::uint8_t> linear(w.k_padded);
    deswizzle_lane(a.codes, lane, w.k_padded, linear.data());
    const std::uint8_t* acodes = linear.data();
    const float row_scale = *a.row_scale;
    const __m512 lut = _mm512_loadu_ps(kE2M1);
    const __m512 pow_table = _mm512_loadu_ps(kE4M3Pow);
    const __m512i low_mask = _mm512_set1_epi32(0xF);
    const __m512i mant_mask = _mm512_set1_epi32(7);
    const __m512i expo_mask = _mm512_set1_epi32(0x78);
    const __m512i sign_mask = _mm512_set1_epi32(0x80);
    const __m512i eight = _mm512_set1_epi32(8);
    for (std::uint32_t o = begin; o < end; ++o) {
        const std::uint8_t* row_codes = codes +
            static_cast<std::size_t>(o >> 4u) * w.blocks * kBlockBytes +
            static_cast<std::size_t>(o & 15u) * 16u;
        const std::uint8_t* row_scales =
            scales + static_cast<std::size_t>(o) * w.scale_row_stride;
        __m512 acc = _mm512_setzero_ps();
        for (std::uint32_t ib = 0; ib < w.blocks; ++ib) {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                row_codes + static_cast<std::size_t>(ib) * kBlockBytes));
            const __m512i v = _mm512_cvtepu8_epi32(packed);
            const __m512 ws = _mm512_set1_ps(e8m0_to_float(row_scales[ib]));
            const __m512 lo =
                _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(v, low_mask), lut), ws);
            const __m512 hi = _mm512_mul_ps(
                _mm512_permutexvar_ps(_mm512_srli_epi32(v, 4), lut), ws);
            const std::uint8_t* x = acodes + static_cast<std::size_t>(ib) * kBlockK;
            const __m512i c0 = _mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x)));
            const __m512i c1 = _mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + 16u)));
            const __m512 a0 =
                e4m3_decode_vector(c0, mant_mask, expo_mask, sign_mask, eight, pow_table);
            const __m512 a1 =
                e4m3_decode_vector(c1, mant_mask, expo_mask, sign_mask, eight, pow_table);
            acc = _mm512_fmadd_ps(lo, a0, acc);
            acc = _mm512_fmadd_ps(hi, a1, acc);
        }
        out[static_cast<std::size_t>(o) * out_stride] =
            _mm512_reduce_add_ps(acc) * row_scale;
    }
}
#endif

bool avx512_available() {
#ifdef SHELF_MXFP4_CPU_X86
    static const bool available = __builtin_cpu_supports("avx512f") &&
                                  __builtin_cpu_supports("avx512bw") &&
                                  __builtin_cpu_supports("avx512vl");
    return available;
#else
    return false;
#endif
}

}  // namespace

void gemv_rows(const CpuExpertPlane& weights, std::uint32_t slot,
               const CpuActivationTile& activation, std::uint32_t lane, float* out,
               std::uint32_t out_stride, std::uint32_t begin, std::uint32_t end) {
    if (begin >= end) return;
    const std::uint32_t stop = end < weights.rows ? end : weights.rows;
    if (begin >= stop) return;
#ifdef SHELF_MXFP4_CPU_X86
    if (avx512_available()) {
        gemv_rows_avx512(weights, slot, activation, lane, out, out_stride, begin, stop);
        return;
    }
#endif
    gemv_rows_scalar(weights, slot, activation, lane, out, out_stride, begin, stop);
}

void gemv_row(const CpuExpertPlane& weights, std::uint32_t slot,
              const CpuActivationTile& activation, std::uint32_t lane, float* out,
              std::uint32_t out_stride) {
    gemv_rows(weights, slot, activation, lane, out, out_stride, 0u, weights.rows);
}

}  // namespace shelf::quantization::mxfp4
