#include <shelf/quantization/codec/e4m3.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include <shelf/quantization/mxfp4/mxfp4_cpu_expert.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++failures;
    }
}

float e8m0(std::uint8_t exponent) {
    const std::uint32_t bits = static_cast<std::uint32_t>(exponent) << 23;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float e2m1(std::uint8_t nibble) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = magnitude[nibble & 0x7u];
    return (nibble & 0x8u) != 0u ? -value : value;
}

float e8m0_scale(std::uint8_t exponent) {
    if (exponent == 0u) return 5.877471754111438e-39f;
    const std::uint32_t bits = static_cast<std::uint32_t>(exponent) << 23;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float e4m3_at(const std::vector<std::uint8_t>& plane, std::uint32_t k_padded,
              std::uint32_t i) {
    const std::uint8_t code =
        plane[static_cast<std::size_t>(i >> 6u) * shelf::quantization::fp8::kFragBlockBytes +
              shelf::quantization::fp8::fp8_frag_offset(0u, i & 63u)];
    return shelf::quantization::codec::e4m3_decode_f32(code);
}

float reference_row(const std::vector<std::uint8_t>& codes,
                    const std::vector<std::uint8_t>& scales, std::uint32_t slot,
                    std::uint32_t out_row, std::uint32_t blocks, std::uint32_t rows,
                    const std::vector<std::uint8_t>& acodes,
                    const std::vector<float>& ascales, std::uint32_t row,
                    std::uint32_t k_padded, float row_scale) {
    const std::uint32_t row_stride = k_padded / 2u;
    const std::uint8_t* base_codes =
        codes.data() + static_cast<std::size_t>(slot) * rows * row_stride;
    const std::uint8_t* row_codes = base_codes +
        static_cast<std::size_t>(out_row / 16u) * (blocks * 256u) +
        static_cast<std::size_t>(out_row % 16u) * 16u;
    const std::uint8_t* row_scales =
        scales.data() + static_cast<std::size_t>(slot) * rows * blocks +
        static_cast<std::size_t>(out_row) * blocks;
    float acc = 0.0f;
    for (std::uint32_t ib = 0; ib < blocks; ++ib) {
        const std::uint8_t* packed = row_codes + static_cast<std::size_t>(ib) * 256u;
        const float ws = e8m0_scale(row_scales[ib]);
        float partial = 0.0f;
        for (std::uint32_t b = 0; b < 16u; ++b) {
            partial += e2m1(packed[b] & 0xFu) * e4m3_at(acodes, k_padded, ib * 32u + b);
            partial += e2m1(packed[b] >> 4u) *
                       e4m3_at(acodes, k_padded, ib * 32u + 16u + b);
        }
        acc += partial * ws;
    }
    return acc * row_scale;
}

}  // namespace

int main() {
    const std::uint32_t rows = 32;
    const std::uint32_t k_padded = 128;
    const std::uint32_t blocks = k_padded / 32u;
    const std::uint32_t slots = 3;
    const std::uint32_t row_stride = k_padded / 2u;

    std::vector<std::uint8_t> codes(static_cast<std::size_t>(slots) * rows * row_stride);
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(slots) * rows * blocks);
    std::uint32_t seed = 12345u;
    const auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 16;
    };
    for (auto& value : codes) value = static_cast<std::uint8_t>(next() & 0xFFu);
    for (auto& value : scales) value = static_cast<std::uint8_t>(120u + (next() % 12u));

    std::vector<std::uint8_t> acodes(shelf::quantization::fp8::fp8_frag_tile_bytes(k_padded));
    std::vector<float> ascales(blocks);
    for (std::uint32_t i = 0; i < k_padded; ++i) {
        acodes[static_cast<std::size_t>(i >> 6u) * shelf::quantization::fp8::kFragBlockBytes +
               shelf::quantization::fp8::fp8_frag_offset(0u, i & 63u)] =
            shelf::quantization::codec::e4m3_encode_u8(
                std::sin(0.37f * static_cast<float>(i) + 0.11f) * 3.0f);
    }
    for (auto& value : ascales) value = 0.001f * static_cast<float>(1u + (next() % 8u));
    const float row_scale = ascales[0];

    shelf::quantization::mxfp4::CpuActivationTile activation;
    activation.codes = acodes.data();
    activation.row_scale = &ascales[0];
    activation.k_padded = k_padded;

    float max_error = 0.0f;
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        shelf::quantization::mxfp4::CpuExpertPlane weights;
        weights.codes = codes.data();
        weights.scales = scales.data();
        weights.rows = rows;
        weights.k_padded = k_padded;
        weights.blocks = blocks;
        weights.codes_batch_stride = rows * row_stride;
        weights.scales_batch_stride = rows * blocks;
        weights.scale_row_stride = blocks;

        std::vector<float> out(rows, 0.0f);
        shelf::quantization::mxfp4::gemv_row(weights, slot, activation, 0u, out.data(), 1u);

        for (std::uint32_t o = 0; o < rows; ++o) {
            const float want =
                reference_row(codes, scales, slot, o, blocks, rows, acodes, ascales, 0u,
                              k_padded, row_scale);
            const float diff = std::fabs(want - out[o]);
            const float scale = std::fabs(want) > 1.0f ? std::fabs(want) : 1.0f;
            if (diff / scale > max_error) max_error = diff / scale;
        }
    }
    check(max_error < 2e-3f, "gemv_row matches the e4m3 scalar reference");

    shelf::quantization::mxfp4::CpuExpertPlane odd;
    odd.codes = codes.data();
    odd.scales = scales.data();
    odd.rows = rows;
    odd.k_padded = k_padded;
    odd.blocks = blocks;
    odd.codes_batch_stride = rows * row_stride;
    odd.scales_batch_stride = rows * blocks;
    odd.scale_row_stride = blocks;

    std::vector<float> rows_out(rows * 2u, 0.0f);
    shelf::quantization::mxfp4::gemv_rows(odd, 1u, activation, 0u, rows_out.data(), 2u, 4u, 12u);
    bool range_ok = true;
    for (std::uint32_t o = 0; o < rows; ++o) {
        const float want =
            reference_row(codes, scales, 1u, o, blocks, rows, acodes, ascales, 0u, k_padded,
                         row_scale);
        if (o >= 4u && o < 12u) {
            range_ok = range_ok && std::fabs(rows_out[static_cast<std::size_t>(o) * 2u] - want) <
                                       std::fabs(want) * 2e-3f + 1e-3f;
        } else {
            range_ok = range_ok && rows_out[static_cast<std::size_t>(o) * 2u] == 0.0f;
        }
    }
    check(range_ok, "gemv_rows writes only the requested rows at the requested stride");

    check(shelf::quantization::mxfp4::silu_mul(0.0f, 2.0f) == 0.0f, "silu_mul at zero");

    if (failures == 0) {
        std::printf("PASS: mxfp4 cpu expert\n");
        return 0;
    }
    std::printf("FAIL: %d checks\n", failures);
    return 1;
}
