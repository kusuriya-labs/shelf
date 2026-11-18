#include <shelf/quantization/mxfp4/mxfp4_cpu_expert.h>

#include <cstddef>
#include <cstring>

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

inline float e8m0_to_float(std::uint8_t exponent) {
    const std::uint32_t bits = static_cast<std::uint32_t>(exponent) << 23;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void gemv_rows_scalar(const CpuExpertPlane& w, std::uint32_t slot,
                      const CpuActivationPlane& a, std::uint32_t row, float* out,
                      std::uint32_t out_stride, std::uint32_t begin, std::uint32_t end) {
    const std::uint8_t* codes = w.codes + static_cast<std::size_t>(slot) * w.codes_batch_stride;
    const std::uint8_t* scales =
        w.scales + static_cast<std::size_t>(slot) * w.scales_batch_stride;
    const std::int8_t* acodes = a.codes + static_cast<std::size_t>(row) * a.code_stride;
    const std::uint8_t* ascale_bytes =
        reinterpret_cast<const std::uint8_t*>(a.scales) +
        static_cast<std::size_t>(row) * a.scale_stride;
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
            const float as = *reinterpret_cast<const float*>(ascale_bytes + 4u * ib);
            const std::int8_t* x = acodes + static_cast<std::size_t>(ib) * kBlockK;
            float partial = 0.0f;
            for (std::uint32_t b = 0; b < 16u; ++b) {
                const float lo = kE2M1[packed[b] & 0xFu];
                const float hi = kE2M1[packed[b] >> 4u];
                partial += lo * static_cast<float>(x[b]);
                partial += hi * static_cast<float>(x[16u + b]);
            }
            acc += partial * ws * as;
        }
        out[static_cast<std::size_t>(o) * out_stride] = acc;
    }
}

#ifdef SHELF_MXFP4_CPU_X86
__attribute__((target("avx512f,avx512bw,avx512vl")))
void gemv_rows_avx512(const CpuExpertPlane& w, std::uint32_t slot,
                      const CpuActivationPlane& a, std::uint32_t row, float* out,
                      std::uint32_t out_stride, std::uint32_t begin, std::uint32_t end) {
    const std::uint8_t* codes = w.codes + static_cast<std::size_t>(slot) * w.codes_batch_stride;
    const std::uint8_t* scales =
        w.scales + static_cast<std::size_t>(slot) * w.scales_batch_stride;
    const std::int8_t* acodes = a.codes + static_cast<std::size_t>(row) * a.code_stride;
    const std::uint8_t* ascale_bytes =
        reinterpret_cast<const std::uint8_t*>(a.scales) +
        static_cast<std::size_t>(row) * a.scale_stride;
    const __m512 lut = _mm512_loadu_ps(kE2M1);
    const __m512i low_mask = _mm512_set1_epi32(0xF);
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
            const std::int8_t* x = acodes + static_cast<std::size_t>(ib) * kBlockK;
            const __m512 as = _mm512_set1_ps(
                *reinterpret_cast<const float*>(ascale_bytes + 4u * ib));
            const __m512 xl = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(x)))), as);
            const __m512 xh = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(x + 16u)))), as);
            acc = _mm512_fmadd_ps(lo, xl, acc);
            acc = _mm512_fmadd_ps(hi, xh, acc);
        }
        out[static_cast<std::size_t>(o) * out_stride] = _mm512_reduce_add_ps(acc);
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
               const CpuActivationPlane& activation, std::uint32_t row, float* out,
               std::uint32_t out_stride, std::uint32_t begin, std::uint32_t end) {
    if (begin >= end) return;
    const std::uint32_t stop = end < weights.rows ? end : weights.rows;
    if (begin >= stop) return;
#ifdef SHELF_MXFP4_CPU_X86
    if (avx512_available()) {
        gemv_rows_avx512(weights, slot, activation, row, out, out_stride, begin, stop);
        return;
    }
#endif
    gemv_rows_scalar(weights, slot, activation, row, out, out_stride, begin, stop);
}

void gemv_row(const CpuExpertPlane& weights, std::uint32_t slot,
              const CpuActivationPlane& activation, std::uint32_t row, float* out,
              std::uint32_t out_stride) {
    gemv_rows(weights, slot, activation, row, out, out_stride, 0u, weights.rows);
}

}  // namespace shelf::quantization::mxfp4
