// vendored from PhaseShift (own work); namespace adapted only
// (was ps::quantization::codec::device).
//
// SYCL port: the ROCm __hip_cvt_* FP8 intrinsics have no SYCL 2020
// equivalent, so E4M3 decode/encode are implemented in software. Decode is
// exact for all 256 codes (E4M3-FN: no inf, NaN = 0x7f, max = 448).
// Encode uses round-to-nearest-even with saturation of |x| >= 448 to 448
// and NaN -> 0x7f (|sign| preserved), matching __HIP_SATFINITE semantics
// used by the HIP backend.
#pragma once

#include <cstdint>
#include <cstring>

namespace shelf::syclk {

// Decode one E4M3 code to f32.
inline float e4m3_decode_f32(uint8_t x) {
    const uint32_t sign = (x & 0x80u) ? 0x80000000u : 0u;
    const uint32_t exp = (x >> 3u) & 0xFu;
    const uint32_t mant = x & 0x7u;
    uint32_t bits;
    if (exp == 0xFu && mant == 0x7u) {
        bits = sign | 0x7FC00000u;  // NaN
    } else if (exp == 0u) {
        // subnormal: value = mant * 2^-9 (mant==0 -> +/-0)
        const float v = static_cast<float>(mant) * (1.0f / 512.0f);
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        bits = sign | (u & 0x7FFFFFFFu);
    } else {
        // normal: value = (1 + mant/8) * 2^(exp-7)
        const uint32_t e = exp + (127u - 7u);
        bits = sign | (e << 23u) | (mant << 20u);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Round v (>0, double) to the nearest multiple of step, ties-to-even.
inline double round_to_step_rne(double v, double step) {
    const double t = v / step;
    const double fl = static_cast<double>(static_cast<int64_t>(t));  // t < 2^52
    double q = fl;
    const double frac = t - fl;
    if (frac > 0.5 || (frac == 0.5 && (static_cast<int64_t>(fl) & 1)))
        q = fl + 1.0;
    return q * step;
}

// Encode f32 -> E4M3 with satfinite + RNE.
inline uint8_t e4m3_encode_u8(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t sign = (u >> 31u) & 1u;
    const uint32_t abs_bits = u & 0x7FFFFFFFu;
    (void)abs_bits;
    const float a = x < 0.0f ? -x : x;

    // NaN -> 0x7f (+sign), Inf/overshoot saturate to 448 (0x7e).
    if (a != a) return static_cast<uint8_t>((sign << 7u) | 0x7Fu);
    if (!(a < 448.0f)) return static_cast<uint8_t>((sign << 7u) | 0x7Eu);

    // Round onto the E4M3 grid (exact in double throughout).
    double v = static_cast<double>(a);
    // Largest finite is 448 (0x7e). NaN code 0x7f stays unused.
    if (v >= 448.0) return static_cast<uint8_t>((sign << 7u) | 0x7Eu);

    uint8_t code;
    if (v < 0.001953125) {  // 2^-9: subnormal band (step 2^-9); <half of
                            // smallest step handled by RNE below
        code = static_cast<uint8_t>(
            static_cast<int>(round_to_step_rne(v, 1.0 / 512.0)));  // 0..7
        if (code > 7u) {  // rounded up into the smallest normal (2^-6)
            code = 0x08u;
        }
    } else {
        // normal band: find unbiased exponent e in [-6, 8]; step = 2^(e-3).
        int e = 0;
        {
            float t = a;
            while (t >= 2.0f) { t *= 0.5f; ++e; }     // a in [1,2) * 2^e
            while (t < 1.0f) { t *= 2.0f; --e; }
        }
        double step = 1.0;
        for (int k = 0; k < 3 - e; ++k) step *= 2.0;   // step = 2^(e-3)
        for (int k = 0; k < e - 3; ++k) step *= 0.5;
        const double rv = round_to_step_rne(v, step);
        if (rv >= 448.0) return static_cast<uint8_t>((sign << 7u) | 0x7Eu);
        // Re-encode rv (exact normal) as e4m3 bits.
        float f = static_cast<float>(rv);
        uint32_t b;
        std::memcpy(&b, &f, sizeof(b));
        const uint32_t fe = (b >> 23u) & 0xFFu;
        const uint32_t fm = b & 0x7FFFFFu;
        code = static_cast<uint8_t>(((fe - 127u + 7u) << 3u) | (fm >> 20u));
    }
    return static_cast<uint8_t>((sign << 7u) | code);
}

} // namespace shelf::syclk
