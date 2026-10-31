// shelf quant — MXFP4 のホスト側実装（契約は mxfp4.hpp）。
//
// 参照実装（phaseshift-mxfp4）の `quantize_block` / `dequantize_block` /
// `encode_scale_byte` / `e2m1_encode_nibble` / `preshuffle_mxfp4_native` と同じ
// 数値・同じビット配置にしてある。丸めは E2M1 の最近傍（同距離は偶数側）で、
// 飽和は最大絶対値 6 に clamp（MXFP4 に無限大は無い）。
#include <shelf/quant/mxfp4.hpp>

#include <cmath>
#include <cstring>

namespace shelf::quant::mxfp4 {
namespace {

// 2 の累乗の切り上げ指数。frexp の仮数が 0.5 のときは指数を 1 つ下げる
// （ちょうど 2 の累乗のときに次の段へ行かないため）。参照実装と同じ。
int ceil_log2_positive(double x) {
    int e = 0;
    const double m = std::frexp(x, &e);
    if (m == 0.5) return e - 1;
    return e;
}

// amax から E8M0 バイトを決める（amax / 6 を 2 の累乗で切り上げ）。
std::uint8_t encode_scale_byte(float amax) {
    if (!(amax > 0.0f) || !std::isfinite(amax)) return 0;
    const float raw = amax / kE2M1Magnitudes[7];
    int exponent = ceil_log2_positive(static_cast<double>(raw));
    if (exponent < -127) exponent = -127;
    if (exponent > 127) exponent = 127;
    return static_cast<std::uint8_t>(exponent + 127);
}

}  // namespace

float e2m1_decode_nibble(std::uint8_t nibble) noexcept {
    const float mag = kE2M1Magnitudes[nibble & 0x07u];
    return (nibble & 0x08u) ? -mag : mag;
}

std::uint8_t e2m1_encode_nibble(float x) noexcept {
    if (!std::isfinite(x)) return 0;
    const std::uint8_t sign = std::signbit(x) ? 0x08u : 0x00u;
    const float a = std::fabs(x);
    if (!(a > 0.0f)) return sign;
    if (a >= kE2M1Magnitudes[7]) return static_cast<std::uint8_t>(sign | 0x07u);
    int lo = 0;
    for (int i = 0; i < 8; ++i) {
        if (kE2M1Magnitudes[i] <= a) {
            lo = i;
        } else {
            break;
        }
    }
    if (lo >= 7) return static_cast<std::uint8_t>(sign | 0x07u);
    const float vlo = kE2M1Magnitudes[lo];
    const float vhi = kE2M1Magnitudes[lo + 1];
    const float dlo = a - vlo;
    const float dhi = vhi - a;
    int pick;
    if (dhi < dlo) {
        pick = lo + 1;
    } else if (dlo < dhi) {
        pick = lo;
    } else {
        // 同距離は偶数側（最近傍偶数丸めと同じ扱い）。
        pick = (((lo + 1) % 2) == 0) ? lo + 1 : lo;
    }
    return static_cast<std::uint8_t>(sign | static_cast<std::uint8_t>(pick));
}

float e8m0_decode_u8(std::uint8_t byte) noexcept {
    if (byte == 0xFFu) return std::nanf("");
    return std::ldexp(1.0f, static_cast<int>(byte) - 127);
}

std::uint8_t e8m0_encode_f32(float x) noexcept {
    if (!(x > 0.0f) || std::isnan(x)) return 0;
    if (std::isinf(x)) return 254;
    int exponent = ceil_log2_positive(static_cast<double>(x));
    if (exponent < -127) exponent = -127;
    if (exponent > 127) exponent = 127;
    return static_cast<std::uint8_t>(exponent + 127);
}

Status validate_tiling(std::uint32_t rows, std::uint32_t cols, std::uint32_t k_padded) {
    if (rows == 0u || cols == 0u) {
        return Status::invalid_state("mxfp4: rows/cols が 0", __FILE__, __LINE__);
    }
    if (k_padded == 0u || (k_padded % kBlockK) != 0u) {
        return Status::invalid_state("mxfp4: k_padded が 32 の倍数でない", __FILE__, __LINE__);
    }
    if (cols > k_padded) {
        return Status::invalid_state("mxfp4: cols が k_padded を超える", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status preshuffle_codes(const std::uint8_t* src, std::uint32_t rows, std::uint32_t cols,
                        std::uint32_t k_padded, std::uint8_t* dst) {
    if (src == nullptr || dst == nullptr) {
        return Status::invalid_state("mxfp4: preshuffle の引数が null", __FILE__, __LINE__);
    }
    const Status st = validate_tiling(rows, cols, k_padded);
    if (!st.ok()) return st;

    const std::size_t bytes = tiled_bytes(rows, k_padded);
    std::memset(dst, 0, bytes);   // 詰め領域（K と行の切り上げ分）は 0

    const std::uint32_t nb = k_padded / kBlockK;
    const std::size_t src_row_bytes = static_cast<std::size_t>(k_padded) / 2u;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::uint8_t* row = src + static_cast<std::size_t>(r) * src_row_bytes;
        for (std::uint32_t ib = 0; ib < nb; ++ib) {
            // 行優先の 16 B（= 32 要素）を、断片順の 16 B（lo 8 + hi 8 ではなく
            // ニブルで lo/hi を交互に持つ形）へ詰め替える。
            std::uint8_t* d = dst + tiled_offset(r, ib * kBlockK, k_padded);
            for (std::uint32_t b = 0; b < 16u; ++b) {
                const std::uint32_t lo_k = ib * kBlockK + b;
                const std::uint32_t hi_k = lo_k + 16u;
                const std::uint8_t lo = (lo_k < cols)
                    ? static_cast<std::uint8_t>((row[lo_k >> 1u] >> (4u * (lo_k & 1u))) & 0x0Fu)
                    : 0u;
                const std::uint8_t hi = (hi_k < cols)
                    ? static_cast<std::uint8_t>((row[hi_k >> 1u] >> (4u * (hi_k & 1u))) & 0x0Fu)
                    : 0u;
                d[b] = static_cast<std::uint8_t>(lo | (hi << 4u));
            }
        }
    }
    return Status::make_ok();
}

Status untile_codes(const std::uint8_t* tiled, std::uint32_t rows, std::uint32_t cols,
                    std::uint32_t k_padded, std::uint8_t* dst) {
    if (tiled == nullptr || dst == nullptr) {
        return Status::invalid_state("mxfp4: untile の引数が null", __FILE__, __LINE__);
    }
    const Status st = validate_tiling(rows, cols, k_padded);
    if (!st.ok()) return st;

    const std::size_t dst_row_bytes = static_cast<std::size_t>(k_padded) / 2u;
    std::memset(dst, 0, static_cast<std::size_t>(rows) * dst_row_bytes);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::uint8_t* row = dst + static_cast<std::size_t>(r) * dst_row_bytes;
        for (std::uint32_t c = 0; c < cols; ++c) {
            const std::uint8_t byte = tiled[tiled_offset(r, c, k_padded)];
            const std::uint8_t nib = static_cast<std::uint8_t>(
                (byte >> tiled_nibble_shift(c)) & 0x0Fu);
            row[c >> 1u] = static_cast<std::uint8_t>(
                row[c >> 1u] | (nib << (4u * (c & 1u))));
        }
    }
    return Status::make_ok();
}

Status quantize_row(const float* src, std::uint32_t logical_k, std::uint32_t k_padded,
                    std::uint8_t* codes, std::uint8_t* scales) {
    if (src == nullptr || codes == nullptr || scales == nullptr) {
        return Status::invalid_state("mxfp4: quantize の引数が null", __FILE__, __LINE__);
    }
    if (k_padded == 0u || (k_padded % kBlockK) != 0u) {
        return Status::invalid_state("mxfp4: k_padded が 32 の倍数でない", __FILE__, __LINE__);
    }
    if (logical_k > k_padded) {
        return Status::invalid_state("mxfp4: logical_k が k_padded を超える", __FILE__, __LINE__);
    }
    const std::uint32_t nb = k_padded / kBlockK;
    for (std::uint32_t ib = 0; ib < nb; ++ib) {
        float x[kBlockK];
        float amax = 0.0f;
        for (std::uint32_t i = 0; i < kBlockK; ++i) {
            const std::uint32_t idx = ib * kBlockK + i;
            const float v = (idx < logical_k) ? src[idx] : 0.0f;
            x[i] = v;
            const float a = std::fabs(v);
            if (std::isfinite(a) && a > amax) amax = a;
        }
        std::uint8_t* c = codes + static_cast<std::size_t>(ib) * kCodesBytesPerBlock;
        if (!(amax > 0.0f)) {
            std::memset(c, 0, kCodesBytesPerBlock);
            scales[ib] = 0u;
            continue;
        }
        const std::uint8_t sb = encode_scale_byte(amax);
        scales[ib] = sb;
        const float scale = std::ldexp(1.0f, static_cast<int>(sb) - 127);
        for (std::uint32_t j = 0; j < kCodesBytesPerBlock; ++j) {
            const std::uint8_t lo = e2m1_encode_nibble(x[2u * j] / scale);
            const std::uint8_t hi = e2m1_encode_nibble(x[2u * j + 1u] / scale);
            c[j] = static_cast<std::uint8_t>(lo | (hi << 4u));
        }
    }
    return Status::make_ok();
}

float read_element(const std::uint8_t* tiled_codes, const std::uint8_t* scales,
                   std::uint32_t row, std::uint32_t col, std::uint32_t k_padded) noexcept {
    const std::uint8_t byte = tiled_codes[tiled_offset(row, col, k_padded)];
    const std::uint8_t nib = static_cast<std::uint8_t>((byte >> tiled_nibble_shift(col)) & 0x0Fu);
    const float s = e8m0_decode_u8(scales[scale_offset(row, col, k_padded)]);
    return e2m1_decode_nibble(nib) * s;
}

Status dequantize(const std::uint8_t* tiled_codes, const std::uint8_t* scales,
                  std::uint32_t rows, std::uint32_t cols, std::uint32_t k_padded,
                  float* out) {
    if (tiled_codes == nullptr || scales == nullptr || out == nullptr) {
        return Status::invalid_state("mxfp4: dequantize の引数が null", __FILE__, __LINE__);
    }
    const Status st = validate_tiling(rows, cols, k_padded);
    if (!st.ok()) return st;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            out[static_cast<std::size_t>(r) * cols + c] =
                read_element(tiled_codes, scales, r, c, k_padded);
        }
    }
    return Status::make_ok();
}

}  // namespace shelf::quant::mxfp4
