#include <shelf/quantization/exl3/exl3_int8.h>

#include "exl3_mul24_codec_fixture.h"

#include <cstdio>
#include <vector>

namespace {

template <size_t N>
bool check(const char* label, const uint16_t (&trellis)[N], uint64_t kt, uint64_t nt, uint32_t bits,
           const int8_t* expected, unsigned threads) {
    std::vector<int8_t> out(nt * 16 * kt * 16, 99);
    if (!shelf::quantization::exl3::exl3_decode_int8(trellis, kt, nt, bits, out.data(), threads)) {
        std::printf("FAIL %s: decode rejected\n", label);
        return false;
    }
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] != expected[i]) {
            std::printf("FAIL %s: mismatch at %zu (%d vs %d)\n", label, i, out[i], expected[i]);
            return false;
        }
    }
    return true;
}

}

int main() {
    bool ok = true;
    for (unsigned threads : {1u, 4u}) {
        ok &= check("4bit", exl3_fixture::kTrellis0, exl3_fixture::kKt0, exl3_fixture::kNt0,
                    exl3_fixture::kBits0, exl3_fixture::kExpected0, threads);
        ok &= check("6bit", exl3_fixture::kTrellis1, exl3_fixture::kKt1, exl3_fixture::kNt1,
                    exl3_fixture::kBits1, exl3_fixture::kExpected1, threads);
    }
    int8_t dummy[1];
    ok &= !shelf::quantization::exl3::exl3_decode_int8(exl3_fixture::kTrellis0, 2, 2, 9, dummy, 1);
    ok &= !shelf::quantization::exl3::exl3_decode_int8(nullptr, 2, 2, 4, dummy, 1);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
