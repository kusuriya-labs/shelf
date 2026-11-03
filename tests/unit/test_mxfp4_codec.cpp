#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/quantization/quant_format.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace shelf::quantization;
using namespace shelf::quantization::mxfp4;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static const float* grid() {
    static const float kGrid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return kGrid;
}

static void test_byte_counts() {
    for (uint64_t n : {1u, 15u, 16u, 129u, 512u, 2048u}) {
        for (uint64_t k : {32u, 64u, 512u, 2048u, 17408u}) {
            const uint64_t kp = quant_padded_k(k, QuantFormatId::Mxfp4Finegrained);
            check(kp == k, "mxfp4 padded_k keeps a group-aligned K k=" + std::to_string(k));
            check(quant_codes_bytes(n, kp, kMxfp4FinegrainedFormatDesc) == n * (kp / 2u),
                  "mxfp4 codes bytes n=" + std::to_string(n) + " k=" + std::to_string(k));
            check(quant_metadata1_bytes(n, kp, kMxfp4FinegrainedFormatDesc) == n * (kp / 32u),
                  "mxfp4 scale bytes n=" + std::to_string(n) + " k=" + std::to_string(k));
            check(quant_metadata_bytes(3, n, kp, kMxfp4FinegrainedFormatDesc, 0) ==
                      3u * n * (kp / 32u),
                  "mxfp4 batched scale bytes n=" + std::to_string(n));
            check(quant_metadata_bytes(1, n, kp, kMxfp4FinegrainedFormatDesc, 1) == 0u,
                  "mxfp4 has no metadata2 stream");
        }
    }
    const uint64_t odd = quant_padded_k(33u, QuantFormatId::Mxfp4Finegrained);
    check(odd == 64u, "mxfp4 padded_k aligns up to 32");
}

// Sizes measured in models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 (routed experts) and
// models/Qwen3.8-27B-Quark-AWQ-MXFP4 (dense FFN).
static void test_measured_artifact_sizes() {
    struct Case {
        const char* tag;
        uint64_t n;
        uint64_t k;
        uint64_t codes;
        uint64_t scales;
    };
    const Case cases[] = {
        {"experts gate_proj", 512u, 2048u, 524288u, 32768u},
        {"experts down_proj", 2048u, 512u, 524288u, 32768u},
        {"dense gate_proj", 17408u, 5120u, 44564480u, 2785280u},
        {"dense out_proj", 5120u, 6144u, 15728640u, 983040u},
    };
    for (const Case& c : cases) {
        check(quant_codes_bytes(c.n, c.k, kMxfp4FinegrainedFormatDesc) == c.codes,
              std::string("mxfp4 measured codes ") + c.tag);
        check(quant_metadata1_bytes(c.n, c.k, kMxfp4FinegrainedFormatDesc) == c.scales,
              std::string("mxfp4 measured scales ") + c.tag);
    }
}

static void test_decode_table() {
    const float expected[16] = {
        0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    for (uint8_t code = 0; code < 16u; ++code) {
        const float value = e2m1_decode_f32(code);
        check(value == expected[code], "e2m1 decode code=" + std::to_string(code));
        if (code != 8u) {
            check(std::signbit(value) == ((code & 8u) != 0u),
                  "e2m1 decode sign code=" + std::to_string(code));
        }
    }
}

static void test_encode_roundtrip_grid() {
    for (uint8_t code = 0; code < 16u; ++code) {
        const float value = e2m1_decode_f32(code);
        const uint8_t again = e2m1_encode_u8(value);
        check(again == code, "e2m1 grid round trip code=" + std::to_string(code));
    }
}

// The packer that produced the Quark artifacts rounds half away from zero, not
// to even; the exact midpoints below are pinned so a future rewrite that swaps
// in textbook RNE fails here instead of silently disagreeing with an artifact.
static void test_encode_ties_and_saturation() {
    struct Case {
        float value;
        uint8_t code;
    };
    const Case cases[] = {
        {0.25f, 1u},   {1.25f, 3u},  {2.5f, 5u},   {5.0f, 7u},
        {-0.25f, 9u},  {-1.25f, 11u}, {-2.5f, 13u}, {-5.0f, 15u},
        {0.2499f, 0u}, {0.2501f, 1u}, {6.0f, 7u},   {6.01f, 7u},
        {100.0f, 7u},  {-100.0f, 15u}, {0.0f, 0u},  {-0.0f, 8u},
    };
    for (const Case& c : cases) {
        const uint8_t code = e2m1_encode_u8(c.value);
        check(code == c.code,
              "e2m1 encode " + std::to_string(c.value) + " expected " +
                  std::to_string(c.code) + " got " + std::to_string(code));
    }
    check(e2m1_encode_u8(std::numeric_limits<float>::infinity()) == 7u, "e2m1 encodes +inf as max");
    check(e2m1_encode_u8(-std::numeric_limits<float>::infinity()) == 15u, "e2m1 encodes -inf as -max");
}

static uint8_t nearest_code(float value) {
    const float magnitude = std::fabs(value);
    const float* values = grid();
    uint8_t best = 0u;
    float best_distance = std::fabs(magnitude - values[0]);
    for (uint8_t i = 1; i < 8u; ++i) {
        const float distance = std::fabs(magnitude - values[i]);
        if (distance < best_distance) {
            best_distance = distance;
            best = i;
        }
    }
    return static_cast<uint8_t>(best | (value < 0.0f ? 8u : 0u));
}

static void test_encode_nearest_grid() {
    std::mt19937 rng(20261103u);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    const float* values = grid();
    int compared = 0;
    for (int i = 0; i < 200000; ++i) {
        const float value = dist(rng);
        bool tie = false;
        for (uint8_t c = 0; c < 7u; ++c) {
            const float mid = (values[c] + values[c + 1u]) * 0.5f;
            if (std::fabs(std::fabs(value) - mid) < 1e-6f) tie = true;
        }
        if (tie || std::fabs(value) > 6.0f) continue;
        ++compared;
        const uint8_t expected = nearest_code(value);
        const uint8_t actual = e2m1_encode_u8(value);
        check(actual == expected,
              "e2m1 nearest grid value=" + std::to_string(value) + " expected " +
                  std::to_string(expected) + " got " + std::to_string(actual));
    }
    check(compared > 100000, "e2m1 nearest grid coverage");
}

static void test_e8m0() {
    check(e8m0_decode_f32(127u) == 1.0f, "e8m0 127 is 1.0");
    check(e8m0_decode_f32(125u) == 0.25f, "e8m0 125 is 0.25");
    check(e8m0_decode_f32(129u) == 4.0f, "e8m0 129 is 4.0");
    check(e8m0_decode_f32(1u) == std::ldexp(1.0f, -126), "e8m0 1 is 2^-126");
    check(e8m0_decode_f32(0u) == std::ldexp(1.0f, -127), "e8m0 0 is 2^-127");
    check(std::isnan(e8m0_decode_f32(255u)), "e8m0 255 is NaN");
    check(e8m0_scale_byte(0.0f) == 1u, "e8m0 zero amax clamps to 2^-126");
    check(e8m0_scale_byte(-0.0f) == 1u, "e8m0 negative zero amax clamps to 2^-126");
    check(e8m0_scale_byte(6.0f) == 127u, "e8m0 amax 6 is scale 1.0");
    check(e8m0_scale_byte(3.0f) == 126u, "e8m0 amax 3 is scale 0.5");
    check(e8m0_scale_byte(0.0234375f) == 119u, "e8m0 measured gate_proj byte");
    check(e8m0_scale_byte(std::numeric_limits<float>::min()) == 0u, "e8m0 f32 min normal clamps");
    check(e8m0_scale_byte(1e-45f) == 0u, "e8m0 subnormal amax clamps");
    check(e8m0_scale_byte(2.0e38f) == 252u, "e8m0 large amax keeps the group-32 exponent");
    // At amax >= 1.75 * 2^127 the E2M1 significand rounding overflows f32 to +inf
    // and the clamp maps it to byte 254 (2^127). The Quark packer does the same.
    check(e8m0_scale_byte(3.0e38f) == 254u, "e8m0 amax overflow clamps to 2^127");
    check(e8m0_scale_byte(std::numeric_limits<float>::max()) == 254u, "e8m0 max clamps to 2^127");
    check(e8m0_scale_byte(std::numeric_limits<float>::quiet_NaN()) == kE8m0NanByte,
          "e8m0 NaN amax maps to the NaN byte");
    for (int e = -120; e <= 120; ++e) {
        for (float mantissa : {1.0f, 1.5f, 1.75f, 1.9f}) {
            const float amax = std::ldexp(mantissa, e);
            const uint8_t byte = e8m0_scale_byte(amax);
            const float scale = e8m0_decode_f32(byte);
            check(scale > 0.0f && std::isfinite(scale), "e8m0 finite scale");
            check(amax / 16.0f < scale && scale <= amax / 3.0f,
                  "e8m0 scale bounds e=" + std::to_string(e));
            check(e8m0_decode_f32(e8m0_scale_byte(scale * 6.0f)) == scale,
                  "e8m0 idempotent at the grid top");
        }
    }
}

static void test_quantize_dequantize() {
    const uint64_t n = 4u;
    const uint64_t k = 96u;
    std::vector<float> src(n * k);
    std::mt19937 rng(4242u);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (float& value : src) value = dist(rng);
    for (uint64_t col = 0; col < k; col += 32u) src[col] = 0.75f;

    std::vector<uint8_t> codes(n * (k / 2u), 0u);
    std::vector<uint8_t> scales(n * (k / 32u), 0u);
    quantize_mxfp4_matrix(src.data(), k, n, k, k, codes.data(), scales.data());

    std::vector<float> dst(n * k);
    dequantize_mxfp4_matrix(codes.data(), scales.data(), n, k, dst.data());
    double worst = 0.0;
    for (uint64_t i = 0; i < n * k; ++i) {
        const float value = src[i];
        const uint64_t row = i / k;
        const uint64_t col = i % k;
        const float scale = e8m0_decode_f32(scales[row * (k / 32u) + col / 32u]);
        const float limit = std::fabs(value) * 0.25f + scale * 0.25f;
        const float error = std::fabs(dst[i] - value);
        if (error > limit) {
            fail("mxfp4 round trip error at " + std::to_string(i) + " error=" +
                 std::to_string(error) + " limit=" + std::to_string(limit));
        }
        worst = std::max(worst, static_cast<double>(error));
        check(!std::isnan(dst[i]) && !std::isinf(dst[i]), "mxfp4 dequant is finite");
    }
    check(worst > 0.0, "mxfp4 quantization is lossy");

    // A well-scaled group uses the top of the E2M1 grid: the largest magnitude
    // code in every group is 4 or 6, matching the measured checkpoints.
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t group = 0; group < k / 32u; ++group) {
            float top = 0.0f;
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint64_t col = group * 32u + j;
                const uint8_t byte = codes[row * (k / 2u) + col / 2u];
                const uint8_t code = (col & 1u) == 0u ? (byte & 0x0Fu) : (byte >> 4);
                top = std::max(top, std::fabs(e2m1_decode_f32(code)));
            }
            check(top == 4.0f || top == 6.0f,
                  "mxfp4 group top code row=" + std::to_string(row) +
                      " group=" + std::to_string(group));
        }
    }
}

static void test_zero_and_padded_groups() {
    const uint64_t n = 2u;
    const uint64_t k = 33u;
    const uint64_t kp = 64u;
    std::vector<float> src(n * kp, 0.0f);
    for (uint64_t col = 0; col < 32u; ++col) src[col] = col < 16u ? -0.3f : 0.3f;
    std::vector<uint8_t> codes(n * (kp / 2u), 0xABu);
    std::vector<uint8_t> scales(n * (kp / 32u), 0xCDu);
    quantize_mxfp4_matrix(src.data(), kp, n, k, kp, codes.data(), scales.data());

    // All-zero groups keep the E8M0 floor byte and zero codes.
    for (uint64_t group = 1; group < kp / 32u; ++group) {
        check(scales[group] == 1u, "mxfp4 zero group scale byte");
        for (uint64_t byte_index = group * 16u; byte_index < (group + 1u) * 16u; ++byte_index)
            check(codes[byte_index] == 0u, "mxfp4 zero group codes");
        check(scales[n * (kp / 32u) - (group + 1u)] == 1u, "mxfp4 zero group scale byte row 1");
    }
    // Columns in [k, kp) are padding and quantize to zero.
    for (uint64_t col = k; col < kp; ++col) {
        const uint8_t byte = codes[col / 2u];
        const uint8_t code = (col & 1u) == 0u ? (byte & 0x0Fu) : (byte >> 4);
        check(code == 0u, "mxfp4 padded column is zero code");
    }
    std::vector<float> dst(n * kp, 1.0f);
    dequantize_mxfp4_matrix(codes.data(), scales.data(), n, kp, dst.data());
    for (uint64_t i = 0; i < n * kp; ++i)
        check(!std::isnan(dst[i]) && !std::isinf(dst[i]), "mxfp4 padded dequant is finite");
}

static void test_canonical_view() {
    const uint64_t n = 128u;
    const uint64_t kp = 2048u;
    std::vector<uint8_t> codes(n * kp / 2u, 0u);
    std::vector<uint8_t> scales(n * kp / 32u, 119u);
    Mxfp4CanonicalView view;
    view.batch = 1;
    view.n = n;
    view.padded_k = kp;
    view.codes = codes.data();
    view.codes_bytes = codes.size();
    view.scales = scales.data();
    view.scales_bytes = scales.size();
    check(view.validate(), "mxfp4 canonical view valid");

    view.codes_bytes -= 1u;
    check(!view.validate(), "mxfp4 canonical view rejects short codes");
    view.codes_bytes = codes.size();
    view.scales_bytes += 32u;
    check(!view.validate(), "mxfp4 canonical view rejects extra scales");
    view.scales_bytes = scales.size();
    view.padded_k = 2050u;
    check(!view.validate(), "mxfp4 canonical view rejects unaligned padded_k");
    view.padded_k = kp;
    view.scales = nullptr;
    check(!view.validate(), "mxfp4 canonical view requires scales");
    view.scales = scales.data();
    view.batch = 2;
    check(!view.validate(), "mxfp4 canonical view sizes per batch");

    Mxfp4CanonicalView batched;
    batched.batch = 2;
    batched.n = n;
    batched.padded_k = kp;
    std::vector<uint8_t> big_codes(2u * n * kp / 2u, 0u);
    std::vector<uint8_t> big_scales(2u * n * kp / 32u, 119u);
    batched.codes = big_codes.data();
    batched.codes_bytes = big_codes.size();
    batched.scales = big_scales.data();
    batched.scales_bytes = big_scales.size();
    check(batched.validate(), "mxfp4 batched canonical view valid");
}

int main() {
    test_byte_counts();
    test_measured_artifact_sizes();
    test_decode_table();
    test_encode_roundtrip_grid();
    test_encode_ties_and_saturation();
    test_encode_nearest_grid();
    test_e8m0();
    test_quantize_dequantize();
    test_zero_and_padded_groups();
    test_canonical_view();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
