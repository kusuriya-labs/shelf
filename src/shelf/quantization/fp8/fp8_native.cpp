#include <shelf/quantization/fp8/fp8_native.h>

#include <shelf/quantization/fp8/fp8_frag.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace shelf::quantization::fp8 {

namespace {

constexpr uint64_t kParallelThresholdBytes = 8ull * 1024ull * 1024ull;
constexpr unsigned kMaxThreads = 16u;

#if defined(__x86_64__) || defined(__i386__)
bool cpu_has_avx512() {
    static const bool value = __builtin_cpu_supports("avx512f") != 0;
    return value;
}

__attribute__((target("avx512f"))) void store_lane_pair_avx512(uint8_t* dst_kg0,
                                                               uint8_t* dst_kg1,
                                                               const uint8_t* src) {
    const __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(src));
    const __m512i even = _mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0);
    const __m512i odd = _mm512_set_epi64(0, 0, 0, 0, 7, 5, 3, 1);
    const __m256i lo = _mm512_castsi512_si256(_mm512_permutexvar_epi64(even, v));
    const __m256i hi = _mm512_castsi512_si256(_mm512_permutexvar_epi64(odd, v));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_kg0), lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_kg1), hi);
}
#endif

void store_lane_pair(uint8_t* dst_kg0, uint8_t* dst_kg1, const uint8_t* src) {
#if defined(__x86_64__) || defined(__i386__)
    if (cpu_has_avx512()) {
        store_lane_pair_avx512(dst_kg0, dst_kg1, src);
        return;
    }
#endif
    for (uint32_t f = 0u; f < 4u; ++f) {
        std::memcpy(dst_kg0 + f * 8u, src + f * 16u, 8u);
        std::memcpy(dst_kg1 + f * 8u, src + f * 16u + 8u, 8u);
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
                   uint64_t nblk, uint64_t t_begin, uint64_t t_end) {
    for (uint64_t t = t_begin; t < t_end; ++t) {
        const uint8_t* const src_tile = v.codes + t * 16u * kp;
        uint8_t* const dst_tile = codes + t * nblk * kFragBlockBytes;
        for (uint64_t r = 0; r < 16u; ++r) {
            const bool live = t * 16u + r < n;
            const uint8_t* const src = src_tile + r * kp;
            for (uint64_t b = 0; b < nblk; ++b) {
                uint8_t* const lane0 = dst_tile + b * kFragBlockBytes + r * kFragLaneBytes;
                uint8_t* const lane1 = lane0 + 16u * kFragLaneBytes;
                if (live) {
                    store_lane_pair(lane0, lane1, src + b * kFragBlockK);
                } else {
                    std::memset(lane0, 0, kFragLaneBytes);
                    std::memset(lane1, 0, kFragLaneBytes);
                }
            }
        }
    }
}

void shuffle_all(const Fp8BlockCanonicalView& v, uint8_t* codes, uint64_t n, uint64_t kp,
                 uint64_t nblk, uint64_t tiles) {
    const uint64_t total = tiles * nblk * kFragBlockBytes;
    unsigned threads = std::thread::hardware_concurrency();
    if (threads == 0u) threads = 1u;
    if (threads > kMaxThreads) threads = kMaxThreads;
    if (total < kParallelThresholdBytes || threads <= 1u || tiles < 2u) {
        shuffle_range(v, codes, n, kp, nblk, 0, tiles);
        return;
    }
    const uint64_t per = (tiles + threads - 1u) / threads;
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned i = 0; i < threads; ++i) {
        const uint64_t b = static_cast<uint64_t>(i) * per;
        if (b >= tiles) break;
        const uint64_t e = std::min(tiles, b + per);
        pool.emplace_back([&v, codes, n, kp, nblk, b, e] {
            shuffle_range(v, codes, n, kp, nblk, b, e);
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
    if ((kp % kFragBlockK) != 0u) return false;
    const uint64_t nblk = kp / kFragBlockK;
    const uint64_t tiles = (n + 15u) / 16u;

    out.n = n;
    out.padded_k = kp;
    out.rows = tiles * 16u;
    out.codes_row_stride_bytes = static_cast<uint32_t>(nblk * kFragBlockBytes);
    out.scale_n = v.scale_n;
    out.scale_k = v.scale_k;
    ensure_codes(out, fp8_frag_codes_bytes(static_cast<uint32_t>(tiles),
                                           static_cast<uint32_t>(kp)));
    ensure_scales(out, v.scale_n * v.scale_k);
    std::memcpy(out.scales.get(), v.scales, out.scale_count * sizeof(float));

    shuffle_all(v, out.codes.get(), n, kp, nblk, tiles);
    return true;
}

}  // namespace shelf::quantization::fp8
