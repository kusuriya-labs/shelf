#include <shelf/quantization/codec/e4m3.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <set>

using shelf::quantization::codec::e4m3_decode_f32;
using shelf::quantization::codec::e4m3_encode_u8;

namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        ++g_passed;
    } else {
        ++g_failed;
        std::printf("FAIL: %s\n", msg);
    }
}

double expected_decode(uint8_t x) {
    const double sign = ((x >> 7) & 1u) ? -1.0 : 1.0;
    const uint32_t e = (x >> 3) & 0xFu;
    const uint32_t m = x & 7u;
    if (x == 0x7Fu || x == 0xFFu) return std::nan("");
    const double mag = (e == 0u)
        ? std::ldexp(static_cast<double>(m), -9)
        : std::ldexp(static_cast<double>(8u + m), static_cast<int>(e) - 10);
    return sign * mag;
}

void check_decode_bitexact(uint8_t x) {
    const float got = e4m3_decode_f32(x);
    const double want = expected_decode(x);
    if (std::isnan(want)) {
        check(std::isnan(got), "decode NaN");
        return;
    }
    const float wf = static_cast<float>(want);
    uint32_t a, b;
    std::memcpy(&a, &got, 4);
    std::memcpy(&b, &wf, 4);
    check(a == b, "decode bit-exact");
}

int run() {
    for (uint32_t x = 0; x < 256; ++x) {
        check_decode_bitexact(static_cast<uint8_t>(x));
    }
    check(e4m3_decode_f32(0x38) == 1.0f, "0x38 == 1.0");
    check(e4m3_decode_f32(0x08) == 0x1p-6f, "0x08 == 2^-6");
    check(e4m3_decode_f32(0x10) == 0x1p-5f, "0x10 == 2^-5");
    check(e4m3_decode_f32(0x7E) == 448.0f, "0x7E == 448");
    check(e4m3_decode_f32(0x00) == 0.0f && !std::signbit(e4m3_decode_f32(0x00)), "0x00 == +0");
    check(e4m3_decode_f32(0x80) == 0.0f && std::signbit(e4m3_decode_f32(0x80)), "0x80 == -0");

    {
        std::set<uint8_t> finite;
        for (uint32_t x = 0; x < 256; ++x) {
            if (x != 0x7Fu && x != 0xFFu) finite.insert(static_cast<uint8_t>(x));
        }
        for (uint8_t c : finite) {
            check(e4m3_encode_u8(e4m3_decode_f32(c)) == c, "roundtrip finite code");
        }
    }

    check(e4m3_encode_u8(0.0f) == 0x00u, "encode +0");
    check(e4m3_encode_u8(-0.0f) == 0x80u, "encode -0");
    check(e4m3_encode_u8(448.0f) == 0x7Eu, "encode +448");
    check(e4m3_encode_u8(-448.0f) == 0xFEu, "encode -448");
    check(e4m3_encode_u8(std::nextafterf(448.0f, 1e30f)) == 0x7Eu, "encode >448 sat");
    check(e4m3_encode_u8(-std::nextafterf(448.0f, 1e30f)) == 0xFEu, "encode <-448 sat");
    check(e4m3_encode_u8(std::numeric_limits<float>::infinity()) == 0x7Eu, "encode +inf");
    check(e4m3_encode_u8(-std::numeric_limits<float>::infinity()) == 0xFEu, "encode -inf");
    check(e4m3_encode_u8(std::nanf("")) == 0x7Fu, "encode NaN");
    check(e4m3_encode_u8(0x1p-9f) == 0x01u, "encode 2^-9");
    check(e4m3_encode_u8(7.0f * 0x1p-9f) == 0x07u, "encode 7*2^-9");
    check(e4m3_encode_u8(0x1p-11f) == 0x00u, "encode below subnormal half flush");
    check(e4m3_encode_u8(0x1p-10f) == 0x00u, "encode subnormal tie to even zero");
    check(e4m3_encode_u8(1.5f * 0x1p-9f) == 0x02u, "encode subnormal tie even");
    check(e4m3_encode_u8(1e-30f) == 0x00u, "encode tiny flush");
    check(e4m3_encode_u8(1.0625f) == 0x38u, "encode tie 1.0/1.125");
    check(e4m3_encode_u8(1.1875f) == 0x3Au, "encode tie 1.125/1.25");
    check(e4m3_encode_u8(0.0302734375f) == 0x10u, "encode tie exp boundary");

    std::mt19937 rng(20260826);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (int i = 0; i < 200000; ++i) {
        const float x = dist(rng);
        const uint8_t c = e4m3_encode_u8(x);
        const float d = e4m3_decode_f32(c);
        check(e4m3_encode_u8(d) == c, "random idempotent encode");
        if (std::fabs(x) > 1e-7f && d != 0.0f) {
            check(std::signbit(d) == std::signbit(x), "random sign preserved");
        }
    }
    return 0;
}

}  // namespace

int main() {
    run();
    std::printf("=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
