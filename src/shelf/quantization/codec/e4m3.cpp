#include <shelf/quantization/codec/e4m3.h>

#include <cmath>
#include <cstdint>

namespace shelf::quantization::codec {

namespace {

constexpr int kGridCount = 126;

struct GridEntry {
    double value;
    uint8_t code;
    uint8_t mant;
};

const GridEntry* e4m3_grid() {
    static GridEntry table[kGridCount];
    static bool init = false;
    if (!init) {
        int n = 0;
        for (uint32_t m = 1; m <= 7; ++m) {
            table[n].value = std::ldexp(static_cast<double>(m), -9);
            table[n].code = static_cast<uint8_t>(m);
            table[n].mant = static_cast<uint8_t>(m);
            ++n;
        }
        for (uint32_t e = 1; e <= 15; ++e) {
            const uint32_t mmax = (e == 15u) ? 6u : 7u;
            for (uint32_t m = 0; m <= mmax; ++m) {
                table[n].value = std::ldexp(static_cast<double>(8u + m), static_cast<int>(e) - 10);
                table[n].code = static_cast<uint8_t>((e << 3) | m);
                table[n].mant = static_cast<uint8_t>(m);
                ++n;
            }
        }
        init = true;
    }
    return table;
}

}  // namespace

float e4m3_decode_f32(uint8_t x) {
    const uint32_t sign = (static_cast<uint32_t>(x) >> 7) & 1u;
    const uint32_t e = (static_cast<uint32_t>(x) >> 3) & 0xFu;
    const uint32_t m = static_cast<uint32_t>(x) & 7u;
    if (x == 0x7Fu || x == 0xFFu) {
        return std::nanf("");
    }
    const float mag = (e == 0u)
        ? static_cast<float>(std::ldexp(static_cast<double>(m), -9))
        : static_cast<float>(std::ldexp(static_cast<double>(8u + m), static_cast<int>(e) - 10));
    return sign ? -mag : mag;
}

uint8_t e4m3_encode_u8(float x) {
    if (std::isnan(x)) {
        return 0x7Fu;
    }
    const uint8_t sign = std::signbit(x) ? 0x80u : 0x00u;
    if (x == 0.0f) {
        return sign;
    }
    if (std::isinf(x) || x >= 448.0f || x <= -448.0f) {
        return static_cast<uint8_t>(sign | 0x7Eu);
    }
    const double v = static_cast<double>(std::fabs(x));
    const GridEntry* g = e4m3_grid();
    int lo = -1;
    int hi = kGridCount;
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (g[mid].value >= v) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    const double vlo = (lo >= 0) ? g[lo].value : 0.0;
    const double vhi = g[hi].value;
    const double twice = 2.0 * v;
    const double mid2 = vlo + vhi;
    const GridEntry* picked = nullptr;
    if (twice < mid2) {
        picked = (lo >= 0) ? &g[lo] : nullptr;
    } else if (twice > mid2) {
        picked = &g[hi];
    } else {
        if ((g[hi].mant & 1u) == 0u) {
            picked = &g[hi];
        } else if (lo >= 0) {
            picked = &g[lo];
        }
    }
    const uint8_t code = (picked != nullptr) ? picked->code : 0x00u;
    return static_cast<uint8_t>(code | sign);
}

}  // namespace shelf::quantization::codec
