#include <shelf/quantization/psq5/psq5.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace ps::quantization::psq5 {

namespace {

constexpr uint64_t kParallelThresholdBytes = 8ull * 1024ull * 1024ull;
constexpr unsigned kMaxThreads = 16u;

void ensure_magnitude(Psq5NativeHost& out, uint64_t bytes) {
    if (out.magnitude == nullptr || out.magnitude_capacity < bytes) {
        out.magnitude.reset(new uint8_t[bytes]);
        out.magnitude_capacity = bytes;
    }
    out.magnitude_bytes = bytes;
}

void ensure_sign(Psq5NativeHost& out, uint64_t bytes) {
    if (out.sign == nullptr || out.sign_capacity < bytes) {
        out.sign.reset(new uint8_t[bytes]);
        out.sign_capacity = bytes;
    }
    out.sign_bytes = bytes;
}

void ensure_scale(Psq5NativeHost& out, uint64_t bytes) {
    if (out.scale == nullptr || out.scale_capacity < bytes) {
        out.scale.reset(new uint8_t[bytes]);
        out.scale_capacity = bytes;
    }
    out.scale_bytes = bytes;
}

void shuffle_range(const Psq5CanonicalView& v, Psq5NativeHost& out, uint64_t n,
                   uint64_t nb, uint64_t t_begin, uint64_t t_end) {
    const uint64_t mag_row = nb * kMagnitudeBytesPerBlock;
    const uint64_t sign_row = nb * kSignBytesPerBlock;
    for (uint64_t t = t_begin; t < t_end; ++t) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            uint8_t* const mag_block =
                out.magnitude.get() + (t * nb + ib) * kTileRows * kMagnitudeBytesPerBlock;
            uint8_t* const sign_block =
                out.sign.get() + (t * nb + ib) * kTileRows * kSignBytesPerBlock;
            uint8_t* const scale_block =
                out.scale.get() + (t * nb + ib) * kTileRows;
            for (uint64_t r = 0; r < kTileRows; ++r) {
                const uint64_t row = t * kTileRows + r;
                uint8_t* const md = mag_block + r * kMagnitudeBytesPerBlock;
                uint8_t* const sd = sign_block + r * kSignBytesPerBlock;
                if (row < n) {
                    std::memcpy(md, v.magnitude + row * mag_row + ib * kMagnitudeBytesPerBlock,
                                kMagnitudeBytesPerBlock);
                    std::memcpy(sd, v.sign + row * sign_row + ib * kSignBytesPerBlock,
                                kSignBytesPerBlock);
                    scale_block[r] = v.scale[row * nb + ib];
                } else {
                    std::memset(md, 0, kMagnitudeBytesPerBlock);
                    std::memset(sd, 0, kSignBytesPerBlock);
                    scale_block[r] = 0;
                }
            }
        }
    }
}

void shuffle_all(const Psq5CanonicalView& v, Psq5NativeHost& out, uint64_t n,
                 uint64_t nb, uint64_t tiles) {
    const uint64_t total = tiles * nb * kBytesPerBlock * kTileRows;
    unsigned threads = std::thread::hardware_concurrency();
    if (threads == 0u) threads = 1u;
    if (threads > kMaxThreads) threads = kMaxThreads;
    if (total < kParallelThresholdBytes || threads <= 1u || tiles < 2u) {
        shuffle_range(v, out, n, nb, 0, tiles);
        return;
    }
    const uint64_t per = (tiles + threads - 1u) / threads;
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned i = 0; i < threads; ++i) {
        const uint64_t b = static_cast<uint64_t>(i) * per;
        if (b >= tiles) break;
        const uint64_t e = std::min(tiles, b + per);
        pool.emplace_back([&v, &out, n, nb, b, e] {
            shuffle_range(v, out, n, nb, b, e);
        });
    }
    for (auto& th : pool) {
        th.join();
    }
}

}

float ue4m4_value(uint8_t stored) noexcept {
    const uint32_t exponent = static_cast<uint32_t>(stored >> 4);
    const uint32_t mantissa = static_cast<uint32_t>(stored & 0x0Fu);
    if (exponent == 0u) return static_cast<float>(mantissa) / 1024.0f;
    if (exponent == 15u) return __builtin_nanf("");
    return (1.0f + static_cast<float>(mantissa) / 16.0f) *
           std::ldexp(1.0f, static_cast<int>(exponent) - 7);
}

std::span<const uint8_t> Psq5NativeHost::magnitude_span() const noexcept {
    return std::span<const uint8_t>(magnitude.get(), static_cast<std::size_t>(magnitude_bytes));
}

std::span<const uint8_t> Psq5NativeHost::sign_span() const noexcept {
    return std::span<const uint8_t>(sign.get(), static_cast<std::size_t>(sign_bytes));
}

std::span<const uint8_t> Psq5NativeHost::scale_span() const noexcept {
    return std::span<const uint8_t>(scale.get(), static_cast<std::size_t>(scale_bytes));
}

Psq5NativeView Psq5NativeHost::view() const {
    Psq5NativeView v;
    v.rows = rows;
    v.n = n;
    v.padded_k = padded_k;
    v.blocks_per_row = blocks_per_row;
    v.magnitude = magnitude.get();
    v.magnitude_bytes = magnitude_bytes;
    v.sign = sign.get();
    v.sign_bytes = sign_bytes;
    v.scale = scale.get();
    v.scale_bytes = scale_bytes;
    v.magnitude_row_stride_bytes = magnitude_row_stride_bytes;
    v.sign_row_stride_bytes = sign_row_stride_bytes;
    return v;
}

bool Psq5CanonicalView::validate() const noexcept {
    if (n == 0 || logical_k == 0) return false;
    if (padded_k < logical_k) return false;
    if ((padded_k % kBlockK) != 0u) return false;
    if (magnitude == nullptr || sign == nullptr || scale == nullptr) return false;
    const uint64_t nb = blocks_per_row();
    if (magnitude_bytes != n * nb * kMagnitudeBytesPerBlock) return false;
    if (sign_bytes != n * nb * kSignBytesPerBlock) return false;
    if (scale_bytes != n * nb * kScaleBytesPerBlock) return false;
    return true;
}

bool preshuffle_psq5_native(const Psq5CanonicalView& v, Psq5NativeHost& out) {
    if (!v.validate()) return false;
    const uint64_t n = v.n;
    const uint64_t kp = v.padded_k;
    const uint64_t nb = kp / kBlockK;
    const uint64_t tiles = (n + kTileRows - 1u) / kTileRows;

    out.n = n;
    out.padded_k = kp;
    out.rows = tiles * kTileRows;
    out.blocks_per_row = nb;
    out.magnitude_row_stride_bytes =
        static_cast<uint32_t>(kTileRows * kMagnitudeBytesPerBlock);
    out.sign_row_stride_bytes = static_cast<uint32_t>(kTileRows * kSignBytesPerBlock);
    ensure_magnitude(out, tiles * nb * kTileRows * kMagnitudeBytesPerBlock);
    ensure_sign(out, tiles * nb * kTileRows * kSignBytesPerBlock);
    ensure_scale(out, tiles * nb * kTileRows * kScaleBytesPerBlock);

    shuffle_all(v, out, n, nb, tiles);
    return true;
}

void dequantize_psq5_canonical(const Psq5CanonicalView& v, float* dst) {
    const uint64_t n = v.n;
    const uint64_t kp = v.padded_k;
    const uint64_t nb = kp / kBlockK;
    const uint64_t mag_row = nb * kMagnitudeBytesPerBlock;
    const uint64_t sign_row = nb * kSignBytesPerBlock;
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            const uint8_t* const mag = v.magnitude + row * mag_row + ib * kMagnitudeBytesPerBlock;
            const uint8_t* const sg = v.sign + row * sign_row + ib * kSignBytesPerBlock;
            const uint32_t sign_word =
                static_cast<uint32_t>(sg[0]) | (static_cast<uint32_t>(sg[1]) << 8) |
                (static_cast<uint32_t>(sg[2]) << 16) | (static_cast<uint32_t>(sg[3]) << 24);
            const float scale = ue4m4_value(v.scale[row * nb + ib]);
            for (uint32_t e = 0; e < kBlockK; ++e) {
                const uint8_t packed = mag[e >> 1];
                const uint8_t magnitude =
                    static_cast<uint8_t>((packed >> ((e & 1u) * 4u)) & 0x0Fu);
                const uint8_t sign_bit = static_cast<uint8_t>((sign_word >> e) & 1u);
                const uint8_t code = magnitude_sign_to_e4m3(magnitude, sign_bit);
                const uint32_t sign = (code >> 7) & 1u;
                const uint32_t exponent = (code >> 3) & 0x0Fu;
                const uint32_t mantissa = code & 0x07u;
                float value;
                if (exponent == 0u) {
                    value = std::ldexp(static_cast<float>(mantissa), -9);
                } else {
                    value = std::ldexp(static_cast<float>(mantissa | 8u),
                                       static_cast<int>(exponent) - 10);
                }
                if (sign != 0u) value = -value;
                dst[row * kp + ib * kBlockK + e] = value * scale;
            }
        }
    }
}

}
