#include <shelf/quantization/fp8/fp8_native.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace ps::quantization::fp8 {

namespace {

constexpr uint64_t kParallelThresholdBytes = 8ull * 1024ull * 1024ull;
constexpr unsigned kMaxThreads = 16u;

#if defined(__x86_64__) || defined(__i386__)
bool cpu_has_avx512() {
    static const bool value = __builtin_cpu_supports("avx512f") != 0;
    return value;
}

__attribute__((target("avx512f"))) void store_4rows_avx512(uint8_t* dst, const uint8_t* src,
                                                            uint64_t row_stride) {
    const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0u * row_stride));
    const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 1u * row_stride));
    const __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 2u * row_stride));
    const __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 3u * row_stride));
    const __m256i lo = _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
    const __m256i hi = _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r3, 1);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst),
                        _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
}
#endif

void store_4rows(uint8_t* dst, const uint8_t* src, uint64_t row_stride) {
#if defined(__x86_64__) || defined(__i386__)
    if (cpu_has_avx512()) {
        store_4rows_avx512(dst, src, row_stride);
        return;
    }
#endif
    for (uint32_t i = 0; i < 4u; ++i) {
        std::memcpy(dst + static_cast<std::size_t>(i) * 16u,
                    src + i * row_stride, 16u);
    }
}

void ensure_codes(Fp8NativeHost& out, uint64_t bytes) {
    if (out.codes == nullptr || out.codes_capacity < bytes) {
        out.codes.reset(new uint8_t[bytes]);
        out.codes_capacity = bytes;
    }
    out.codes_bytes = bytes;
}

void ensure_scales(Fp8NativeHost& out, uint64_t count) {
    if (out.scales == nullptr || out.scales_capacity < count) {
        out.scales.reset(new float[count]);
        out.scales_capacity = count;
    }
    out.scale_count = count;
}

void shuffle_range(const Fp8BlockCanonicalView& v, uint8_t* codes, uint64_t n, uint64_t kp,
                   uint64_t nb, uint64_t t_begin, uint64_t t_end) {
    for (uint64_t t = t_begin; t < t_end; ++t) {
        const uint8_t* const src_tile = v.codes + t * 16u * kp;
        for (uint64_t ib = 0; ib < nb; ++ib) {
            uint8_t* const block = codes + (t * nb + ib) * 512u;
            for (uint64_t half = 0; half < 2u; ++half) {
                uint8_t* const dst = block + half * 256u;
                const uint8_t* const src = src_tile + ib * 32u + half * 16u;
                for (uint64_t g = 0; g < 16u; g += 4u) {
                    const uint64_t o = t * 16u + g;
                    if (o + 4u <= n) {
                        store_4rows(dst + g * 16u, src + g * kp, kp);
                        continue;
                    }
                    for (uint64_t l = 0; l < 4u; ++l) {
                        uint8_t* d = dst + (g + l) * 16u;
                        if (o + l < n) {
                            std::memcpy(d, src + (g + l) * kp, 16u);
                        } else {
                            std::memset(d, 0, 16u);
                        }
                    }
                }
            }
        }
    }
}

void shuffle_all(const Fp8BlockCanonicalView& v, uint8_t* codes, uint64_t n, uint64_t kp,
                 uint64_t nb, uint64_t tiles) {
    const uint64_t total = tiles * nb * 512u;
    unsigned threads = std::thread::hardware_concurrency();
    if (threads == 0u) threads = 1u;
    if (threads > kMaxThreads) threads = kMaxThreads;
    if (total < kParallelThresholdBytes || threads <= 1u || tiles < 2u) {
        shuffle_range(v, codes, n, kp, nb, 0, tiles);
        return;
    }
    const uint64_t per = (tiles + threads - 1u) / threads;
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned i = 0; i < threads; ++i) {
        const uint64_t b = static_cast<uint64_t>(i) * per;
        if (b >= tiles) break;
        const uint64_t e = std::min(tiles, b + per);
        pool.emplace_back([&v, codes, n, kp, nb, b, e] {
            shuffle_range(v, codes, n, kp, nb, b, e);
        });
    }
    for (auto& th : pool) {
        th.join();
    }
}

}  // namespace

std::span<const uint8_t> Fp8NativeHost::codes_span() const noexcept {
    return std::span<const uint8_t>(codes.get(), static_cast<std::size_t>(codes_bytes));
}

std::span<const float> Fp8NativeHost::scales_span() const noexcept {
    return std::span<const float>(scales.get(), static_cast<std::size_t>(scale_count));
}

Fp8NativeView Fp8NativeHost::view() const {
    Fp8NativeView v;
    v.rows = rows;
    v.n = n;
    v.padded_k = padded_k;
    v.codes_row_stride_bytes = codes_row_stride_bytes;
    v.scale_n = scale_n;
    v.scale_k = scale_k;
    v.codes = codes.get();
    v.codes_bytes = codes_bytes;
    v.scales = scales.get();
    v.scales_bytes = scale_count * sizeof(float);
    return v;
}

bool preshuffle_fp8_native(const Fp8BlockCanonicalView& v,
                           Fp8NativeHost& out) {
    if (!v.validate()) return false;
    if (v.batch != 1) return false;
    const uint64_t n = v.n;
    const uint64_t kp = v.padded_k;
    if ((kp % 32u) != 0u) return false;
    const uint64_t nb = kp / 32u;
    const uint64_t tiles = (n + 15u) / 16u;

    out.n = n;
    out.padded_k = kp;
    out.rows = tiles * 16u;
    out.codes_row_stride_bytes = nb * 512u;
    out.scale_n = v.scale_n;
    out.scale_k = v.scale_k;
    ensure_codes(out, tiles * nb * 512u);
    ensure_scales(out, v.scale_n * v.scale_k);
    std::memcpy(out.scales.get(), v.scales, out.scale_count * sizeof(float));

    shuffle_all(v, out.codes.get(), n, kp, nb, tiles);
    return true;
}

}  // namespace ps::quantization::fp8
