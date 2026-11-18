#include <shelf/quantization/int6/int6_w32.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace shelf {
namespace quantization {
namespace int6 {

namespace {

inline float bf16_bits_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

inline shelf::fp16_t load_scale(const std::uint8_t* bytes) noexcept {
    shelf::fp16_t scale{};
    std::memcpy(&scale, bytes, sizeof(scale));
    return scale;
}

inline void store_scale(std::uint8_t* bytes, shelf::fp16_t scale) noexcept {
    std::memcpy(bytes, &scale, sizeof(scale));
}

void quantize_bf16_row_scalar(const bf16_t* src, std::uint32_t groups,
                              std::uint8_t* dst) noexcept {
    for (std::uint32_t group = 0; group < groups; ++group) {
        const bf16_t* source = src + static_cast<std::size_t>(group) * kGroupSize;
        float values[kGroupSize];
        float positive = 0.0f;
        float negative = 0.0f;
        for (std::uint32_t i = 0; i < kGroupSize; ++i) {
            values[i] = bf16_bits_to_f32(source[i].data);
            if (values[i] > positive) positive = values[i];
            if (values[i] < negative) negative = values[i];
        }

        const float raw = std::max(positive / kMaxPositive,
                                   -negative / kMaxNegativeMagnitude);
        const shelf::fp16_t stored =
            raw > 0.0f ? shelf::f32_to_fp16(std::max(raw, kMinScale))
                       : shelf::f32_to_fp16(1.0f);
        const float inv = 1.0f / shelf::fp16_to_f32(stored);

        std::uint8_t* codes = dst + static_cast<std::size_t>(group) * kGroupBytes;
        std::uint64_t bits = 0;
        std::uint32_t available = 0;
        std::uint32_t out_bytes = 0;
        for (std::uint32_t i = 0; i < kGroupSize; ++i) {
            const float scaled = std::nearbyintf(values[i] * inv);
            const float clamped =
                std::min(std::max(scaled, -kMaxNegativeMagnitude), kMaxPositive);
            const std::uint32_t code = std::isfinite(clamped)
                ? static_cast<std::uint32_t>(static_cast<std::int32_t>(clamped)) & 0x3Fu
                : 0u;
            bits |= static_cast<std::uint64_t>(code) << available;
            available += 6u;
            while (available >= 8u) {
                codes[out_bytes++] = static_cast<std::uint8_t>(bits & 0xFFu);
                bits >>= 8u;
                available -= 8u;
            }
        }
        store_scale(codes + kGroupCodeBytes, stored);
    }
}

#if defined(__x86_64__) || defined(__i386__)
bool cpu_has_avx512_bmi2() noexcept {
    static const bool available =
        __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("bmi2");
    return available;
}

__attribute__((target("avx512f,avx512bw,avx512vl,bmi2")))
void quantize_bf16_row_avx512(const bf16_t* src, std::uint32_t groups,
                              std::uint8_t* dst) noexcept {
    const __m512i sign_shift = _mm512_set1_epi32(16);
    const __m512 zero = _mm512_setzero_ps();
    const __m512i low = _mm512_set1_epi32(-static_cast<int>(kMaxNegativeMagnitude));
    const __m512i high = _mm512_set1_epi32(static_cast<int>(kMaxPositive));
    const __m512i code_mask = _mm512_set1_epi32(0x3F);
    constexpr std::uint64_t kByteMask = 0x3F3F3F3F3F3F3F3Full;
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint16_t* source =
            reinterpret_cast<const std::uint16_t*>(src) + group * kGroupSize;
        const __m256i raw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));
        const __m256i raw1 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + 16));
        const __m512 f0 =
            _mm512_castsi512_ps(_mm512_sllv_epi32(_mm512_cvtepu16_epi32(raw0), sign_shift));
        const __m512 f1 =
            _mm512_castsi512_ps(_mm512_sllv_epi32(_mm512_cvtepu16_epi32(raw1), sign_shift));

        const __m512 positive = _mm512_max_ps(zero, _mm512_max_ps(f0, f1));
        const __m512 negative = _mm512_min_ps(zero, _mm512_min_ps(f0, f1));
        const float pos = _mm512_reduce_max_ps(positive);
        const float neg = _mm512_reduce_min_ps(negative);
        const float raw = std::max(pos / kMaxPositive, -neg / kMaxNegativeMagnitude);
        const shelf::fp16_t stored = raw > 0.0f
            ? shelf::f32_to_fp16(std::max(raw, kMinScale))
            : shelf::f32_to_fp16(1.0f);
        const float inv = 1.0f / shelf::fp16_to_f32(stored);
        const __m512 scale = _mm512_set1_ps(inv);

        __m512i c0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(
            _mm512_mul_ps(f0, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m512i c1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(
            _mm512_mul_ps(f1, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c0 = _mm512_and_si512(_mm512_min_epi32(_mm512_max_epi32(c0, low), high), code_mask);
        c1 = _mm512_and_si512(_mm512_min_epi32(_mm512_max_epi32(c1, low), high), code_mask);

        alignas(32) std::uint8_t bytes[32];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(bytes),
                         _mm512_cvtepi32_epi8(c0));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(bytes + 16),
                         _mm512_cvtepi32_epi8(c1));

        std::uint8_t* codes = dst + static_cast<std::size_t>(group) * kGroupBytes;
        for (std::uint32_t i = 0; i < 4u; ++i) {
            std::uint64_t word = 0;
            std::memcpy(&word, bytes + i * 8u, 8u);
            const std::uint64_t packed = _pext_u64(word, kByteMask);
            std::memcpy(codes + i * 6u, &packed, 6u);
        }
        store_scale(codes + kGroupCodeBytes, stored);
    }
}
#endif

}  // namespace

void quantize_bf16_row(const bf16_t* src, std::uint32_t groups,
                       std::uint8_t* dst) noexcept {
#if defined(__x86_64__) || defined(__i386__)
    if (cpu_has_avx512_bmi2()) {
        quantize_bf16_row_avx512(src, groups, dst);
        return;
    }
#endif
    quantize_bf16_row_scalar(src, groups, dst);
}

void dequantize_row(const std::uint8_t* src, std::uint32_t groups,
                    float* dst) noexcept {
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint8_t* codes = src + static_cast<std::size_t>(group) * kGroupBytes;
        const float scale = shelf::fp16_to_f32(load_scale(codes + kGroupCodeBytes));

        std::uint64_t bits = 0;
        std::uint32_t available = 0;
        std::uint32_t in_bytes = 0;
        float* out = dst + static_cast<std::size_t>(group) * kGroupSize;
        for (std::uint32_t i = 0; i < kGroupSize; ++i) {
            while (available < 6u) {
                bits |= static_cast<std::uint64_t>(codes[in_bytes++]) << available;
                available += 8u;
            }
            const std::uint32_t code = static_cast<std::uint32_t>(bits & 0x3Fu);
            bits >>= 6u;
            available -= 6u;
            const std::int32_t value =
                (code & 0x20u) ? static_cast<std::int32_t>(code) - 64
                               : static_cast<std::int32_t>(code);
            out[i] = static_cast<float>(value) * scale;
        }
    }
}

}  // namespace int6
}  // namespace quantization
}  // namespace shelf
