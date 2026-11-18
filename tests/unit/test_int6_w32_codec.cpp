#include <shelf/quantization/int6/int6_w32.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::fprintf(stderr, "ok: %s\n", what);
    }
}

shelf::bf16_t to_bf16(float value) {
    return shelf::bf16_t{shelf::f32_to_bf16_rne(value)};
}

float from_bf16(shelf::bf16_t value) {
    const std::uint32_t wide = static_cast<std::uint32_t>(value.data) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &wide, sizeof(result));
    return result;
}

void write_scale(std::uint8_t* dst, float scale) {
    const shelf::fp16_t stored = shelf::f32_to_fp16(scale);
    std::memcpy(dst, &stored, sizeof(stored));
}

}  // namespace

int main() {
    using namespace shelf::quantization::int6;

    check(kGroupSize == 32u, "INT6 W32 group size is 32");
    check(kGroupCodeBytes == 24u, "INT6 W32 group carries 24 code bytes");
    check(kGroupBytes == 26u, "INT6 W32 group is 26 bytes with FP16 scale");

    {
        std::vector<std::uint8_t> packed(kGroupBytes, 0u);
        write_scale(packed.data() + kGroupCodeBytes, 0.5f);
        std::vector<float> values(kGroupSize, 0.0f);
        dequantize_row(packed.data(), 1u, values.data());
        bool ok = true;
        for (float value : values) ok = ok && value == 0.0f;
        check(ok, "all-zero codes decode to zero");
    }

    {
        std::vector<std::uint8_t> packed(kGroupBytes, 0xFFu);
        write_scale(packed.data() + kGroupCodeBytes, 0.5f);
        std::vector<float> values(kGroupSize, 0.0f);
        dequantize_row(packed.data(), 1u, values.data());
        bool ok = true;
        for (float value : values) ok = ok && value == -0.5f;
        check(ok, "code 63 decodes to -1 * scale");
    }

    {
        std::vector<std::uint8_t> packed(kGroupBytes, 0u);
        const std::uint8_t code_one[kGroupCodeBytes] = {
            0x41, 0x10, 0x04, 0x41, 0x10, 0x04, 0x41, 0x10, 0x04, 0x41, 0x10, 0x04,
            0x41, 0x10, 0x04, 0x41, 0x10, 0x04, 0x41, 0x10, 0x04, 0x41, 0x10, 0x04};
        std::memcpy(packed.data(), code_one, kGroupCodeBytes);
        write_scale(packed.data() + kGroupCodeBytes, 0.25f);
        std::vector<float> values(kGroupSize, 0.0f);
        dequantize_row(packed.data(), 1u, values.data());
        bool ok = true;
        for (float value : values) ok = ok && value == 0.25f;
        check(ok, "packed 6-bit lane 1 decodes to +1 * scale");
    }

    {
        constexpr std::uint32_t groups = 4u;
        constexpr std::uint32_t count = groups * kGroupSize;
        std::vector<shelf::bf16_t> source(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::int32_t numerator = static_cast<std::int32_t>((i * 37u) % 101u) - 50;
            source[i] = to_bf16(static_cast<float>(numerator) / 64.0f);
        }
        std::vector<std::uint8_t> packed(groups * kGroupBytes, 0u);
        quantize_bf16_row(source.data(), groups, packed.data());
        std::vector<float> decoded(count, 0.0f);
        dequantize_row(packed.data(), groups, decoded.data());

        bool finite = true;
        bool bounded = true;
        for (std::uint32_t group = 0; group < groups; ++group) {
            float amax = 0.0f;
            for (std::uint32_t j = 0; j < kGroupSize; ++j) {
                amax = std::max(amax, std::fabs(from_bf16(source[group * kGroupSize + j])));
            }
            for (std::uint32_t j = 0; j < kGroupSize; ++j) {
                const std::uint32_t index = group * kGroupSize + j;
                finite = finite && std::isfinite(decoded[index]);
                const float error = std::fabs(decoded[index] - from_bf16(source[index]));
                bounded = bounded && error <= amax / 62.0f + amax / 1000.0f + 1e-6f;
            }
        }
        check(finite, "quantized row decodes to finite values");
        check(bounded, "quantized row stays within the INT6 W32 error bound");
    }

    {
        std::vector<shelf::bf16_t> source(kGroupSize, to_bf16(0.0f));
        std::vector<std::uint8_t> packed(kGroupBytes, 0xAAu);
        quantize_bf16_row(source.data(), 1u, packed.data());
        bool zero_codes = true;
        for (std::uint32_t i = 0; i < kGroupCodeBytes; ++i) {
            zero_codes = zero_codes && packed[i] == 0u;
        }
        const shelf::fp16_t scale =
            *reinterpret_cast<const shelf::fp16_t*>(packed.data() + kGroupCodeBytes);
        check(zero_codes && shelf::fp16_to_f32(scale) == 1.0f,
              "zero group stores zero codes and scale 1");
    }

    {
        std::vector<shelf::bf16_t> source(kGroupSize, to_bf16(0.0f));
        source[0] = to_bf16(-1.0f);
        std::vector<std::uint8_t> packed(kGroupBytes, 0u);
        quantize_bf16_row(source.data(), 1u, packed.data());
        std::vector<float> decoded(kGroupSize, 0.0f);
        dequantize_row(packed.data(), 1u, decoded.data());
        check(decoded[0] == -1.0f, "negative outlier uses the full -32 code");
        bool zeros = true;
        for (std::uint32_t i = 1; i < kGroupSize; ++i) zeros = zeros && decoded[i] == 0.0f;
        check(zeros, "negative outlier does not disturb the remaining group");
    }

    {
        std::vector<shelf::bf16_t> source(kGroupSize, to_bf16(0.0f));
        source[0] = to_bf16(1.0f);
        std::vector<std::uint8_t> packed(kGroupBytes, 0u);
        quantize_bf16_row(source.data(), 1u, packed.data());
        std::vector<float> decoded(kGroupSize, 0.0f);
        dequantize_row(packed.data(), 1u, decoded.data());
        check(std::fabs(decoded[0] - 1.0f) <= 1e-3f,
              "positive outlier uses the full +31 code");
    }

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
