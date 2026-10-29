// shelf quant — FP8 W8A8 の実装（E4M3 コーデックと断片順再配置）。
#include <cmath>
#include <cstring>
#include <limits>

#include <shelf/quant/fp8_w8a8.hpp>

namespace shelf::quant::fp8 {
namespace {

constexpr std::uint8_t kE4m3Nan = 0x7Fu;
constexpr std::uint8_t kE4m3MaxCode = 0x7Eu;   // +448

}  // namespace

float e4m3_decode(std::uint8_t code) noexcept {
    const std::uint32_t e = (code >> 3) & 0xFu;
    const std::uint32_t m = code & 7u;
    const float sign = (code & 0x80u) ? -1.0f : 1.0f;
    if (e == 15u && m == 7u) {
        // NaN の符号は保持する（code -> value -> code の往復で同一コードに戻る）。
        const float nan = std::numeric_limits<float>::quiet_NaN();
        return (code & 0x80u) ? -nan : nan;
    }
    if (e == 0u) return sign * static_cast<float>(m) * 0x1p-9f;   // 非正規: m * 2^-9
    const float frac = 1.0f + static_cast<float>(m) * 0.125f;
    return sign * std::ldexp(frac, static_cast<int>(e) - 7);
}

std::uint8_t e4m3_encode(float x) noexcept {
    std::uint32_t b = 0u;
    std::memcpy(&b, &x, sizeof(b));
    const std::uint8_t sign = static_cast<std::uint8_t>((b >> 24) & 0x80u);
    const std::uint32_t mag_bits = b & 0x7FFFFFFFu;
    if (mag_bits > 0x7F800000u) return static_cast<std::uint8_t>(sign | kE4m3Nan);
    if (mag_bits == 0x7F800000u) return static_cast<std::uint8_t>(sign | kE4m3MaxCode);  // inf -> 448
    if (mag_bits == 0u) return sign;   // +0 / -0

    const std::uint32_t e_field = mag_bits >> 23;
    const std::uint32_t mant = mag_bits & 0x7FFFFFu;
    if (e_field == 0u) return sign;    // f32 非正規（< 2^-126）は e4m3 では 0

    const int e = static_cast<int>(e_field) - 127;
    if (e < -6) {
        // e4m3 の非正規領域（量子は 2^-9）。scaled = 1.m を 2^23 倍した整数。
        const int sh = 14 - e;         // e <= -7 なので sh >= 21
        if (sh >= 25) return sign;     // 半量子未満は 0 へ
        const std::uint32_t scaled = (1u << 23) | mant;
        const std::uint32_t q = scaled >> sh;
        const std::uint32_t half = 1u << (sh - 1);
        const std::uint32_t rem = scaled & ((1u << sh) - 1u);
        const bool up = (rem > half) || (rem == half && (q & 1u) != 0u);
        const std::uint32_t r = q + (up ? 1u : 0u);
        if (r >= 8u) return static_cast<std::uint8_t>(sign | 0x08u);   // 2^-6 へ繰上げ
        return static_cast<std::uint8_t>(sign | r);
    }

    std::uint32_t q = mant >> 20;                 // 仮数 23 -> 3 ビット
    const std::uint32_t rem = mant & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (q & 1u) != 0u)) ++q;   // RNE
    int ee = e;
    if (q == 8u) { q = 0u; ++ee; }                // 仮数の繰上げが指数へ走る
    const int E = ee + 7;
    if (E > 15 || (E == 15 && q > 6u))
        return static_cast<std::uint8_t>(sign | kE4m3MaxCode);        // satfinite
    return static_cast<std::uint8_t>(sign | (static_cast<std::uint32_t>(E) << 3) | q);
}

Status validate_tiling(std::uint32_t rows, std::uint32_t cols, std::uint32_t k_padded) {
    if (rows == 0u || cols == 0u)
        return Status::invalid_argument("fp8 tiling: empty matrix", __FILE__, __LINE__);
    if ((k_padded % kTileK) != 0u)
        return Status::invalid_argument("fp8 tiling: k_padded must be a multiple of 32",
                                        __FILE__, __LINE__);
    if (k_padded < cols)
        return Status::invalid_argument("fp8 tiling: k_padded < cols", __FILE__, __LINE__);
    return Status::make_ok();
}

Status preshuffle_codes(const std::uint8_t* src, std::uint32_t rows, std::uint32_t cols,
                        std::uint32_t k_padded, std::uint8_t* dst) {
    if (src == nullptr || dst == nullptr)
        return Status::invalid_argument("fp8 preshuffle: null buffer", __FILE__, __LINE__);
    SHELF_RETURN_IF_ERROR(validate_tiling(rows, cols, k_padded));
    std::memset(dst, 0, tiled_bytes(rows, k_padded));   // 詰め領域は 0
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::uint8_t* srow = src + static_cast<std::size_t>(r) * cols;
        for (std::uint32_t c = 0; c < cols; ++c)
            dst[tiled_offset(r, c, k_padded)] = srow[c];
    }
    return Status::make_ok();
}

Status untile_codes(const std::uint8_t* tiled, std::uint32_t rows, std::uint32_t cols,
                    std::uint32_t k_padded, std::uint8_t* dst) {
    if (tiled == nullptr || dst == nullptr)
        return Status::invalid_argument("fp8 untile: null buffer", __FILE__, __LINE__);
    SHELF_RETURN_IF_ERROR(validate_tiling(rows, cols, k_padded));
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::uint8_t* drow = dst + static_cast<std::size_t>(r) * cols;
        for (std::uint32_t c = 0; c < cols; ++c) drow[c] = tiled[tiled_offset(r, c, k_padded)];
    }
    return Status::make_ok();
}

}  // namespace shelf::quant::fp8
