#include <shelf/quantization/exl3/exl3.h>

#include "exl3_codec_golden.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace shelf::quantization::exl3;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        g_fail++;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t crc32_u16(const uint16_t* values, std::size_t count) {
    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < count; ++i) {
        const uint8_t bytes[2] = {
            static_cast<uint8_t>(values[i] & 0xFFu),
            static_cast<uint8_t>(values[i] >> 8)};
        for (uint8_t b : bytes) {
            crc ^= b;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc & 1u) ? ((crc >> 1) ^ 0xEDB88320u) : (crc >> 1);
            }
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

static void test_codebook_tables() {
    std::vector<uint16_t> bits(65536);
    for (uint32_t state = 0; state < 65536u; ++state) {
        bits[state] = codebook_f16_bits(static_cast<uint16_t>(state));
    }
    check(crc32_u16(bits.data(), bits.size()) == testdata::kCodebookMul1Crc,
          "mul1 codebook table");
}

static void test_golden_tiles(const testdata::GoldenTile& golden, const std::string& tag) {
    check(layout_valid(golden.ka, golden.half), tag + " layout");
    check(words_per_tile(golden.ka, golden.half) == golden.word_count, tag + " word count");
    float tile[kTileWeights];
    check(decode_tile(golden.words, golden.ka, golden.half, tile), tag + " decode");
    std::vector<uint16_t> bits(kTileWeights);
    for (uint32_t i = 0; i < kTileWeights; ++i) bits[i] = f16_bits_from_f32(tile[i]);
    check(crc32_u16(bits.data(), bits.size()) == golden.tile_crc, tag + " tile crc");
}

static void test_tiles() {
    test_golden_tiles(testdata::kGoldenTile0, "K2");
    test_golden_tiles(testdata::kGoldenTile1, "K3");
    test_golden_tiles(testdata::kGoldenTile2, "K4");
    test_golden_tiles(testdata::kGoldenTile3, "K5");
    test_golden_tiles(testdata::kGoldenTile4, "K6");
    test_golden_tiles(testdata::kGoldenTile5, "K7");
    test_golden_tiles(testdata::kGoldenTile6, "K8");
    test_golden_tiles(testdata::kGoldenTile7, "K1.5");
    test_golden_tiles(testdata::kGoldenTile8, "K2.5");
    test_golden_tiles(testdata::kGoldenTile9, "K3.5");
}

static void test_hadamard() {
    float values[kHadamardBlock];
    for (float& v : values) v = 1.0f;
    hadamard128_inplace(values);
    check(std::fabs(values[0] - 11.3137085f) < 1e-4f, "hadamard of ones dc");
    for (uint32_t i = 1; i < kHadamardBlock; ++i) {
        check(std::fabs(values[i]) < 1e-5f, "hadamard of ones ac");
    }
}

static void test_dequant() {
    constexpr uint32_t k = 128;
    constexpr uint32_t n = 128;
    std::vector<float> out(static_cast<std::size_t>(k) * n);
    check(dequant(testdata::kDequantTrellis, testdata::kDequantSuh, testdata::kDequantSvh, k, n,
                  4, false, out.data()),
          "dequant decode");
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t r = testdata::kDequantCoords[i][0];
        const uint32_t c = testdata::kDequantCoords[i][1];
        const float got = out[static_cast<std::size_t>(r) * n + c];
        const float want = testdata::kDequantSamples[i];
        const float tolerance = 1e-4f * (1.0f + std::fabs(want));
        check(std::fabs(got - want) <= tolerance,
              "dequant sample " + std::to_string(i) + " got " + std::to_string(got) +
                  " want " + std::to_string(want));
    }
    check(trellis_bytes(k, n, 4, false) == sizeof(testdata::kDequantTrellis),
          "trellis byte count");
}

int main() {
    test_codebook_tables();
    test_tiles();
    test_hadamard();
    test_dequant();
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
