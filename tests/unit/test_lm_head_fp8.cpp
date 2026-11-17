#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/lm_head_fp8.h>
#include <shelf/quantization/codec/e4m3.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
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

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

float e4m3_to_float(std::uint8_t bits) {
    const std::uint32_t sign = (bits >> 7u) & 1u;
    const std::uint32_t exponent = (bits >> 3u) & 0xFu;
    const std::uint32_t mantissa = bits & 0x7u;
    if (exponent == 0xFu && mantissa == 0x7u) return NAN;
    float value = 0.0f;
    if (exponent == 0u) {
        value = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                           static_cast<int>(exponent) - 7);
    }
    return sign != 0u ? -value : value;
}

std::vector<shelf::bf16_t> random_bf16(std::size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> out(count);
    for (auto& value : out) value.data = shelf::f32_to_bf16_rne(dist(rng));
    return out;
}

constexpr std::uint32_t kVocab = 1024u;
constexpr std::uint32_t kK = 5120u;
constexpr std::uint32_t kRows = 4u;

}  // namespace

int main() {
    std::mt19937 rng(20261117u);
    const std::vector<shelf::bf16_t> weight = random_bf16(kVocab * kK, rng);
    const std::vector<shelf::bf16_t> hidden = random_bf16(kRows * kK, rng);

    shelf::bf16_t* d_weight = nullptr;
    shelf::bf16_t* d_hidden = nullptr;
    shelf::bf16_t* d_hidden_strided = nullptr;
    std::uint8_t* d_codes = nullptr;
    shelf::fp16_t* d_scales = nullptr;
    float* d_logits = nullptr;
    const std::uint32_t hidden_stride = kK + 64u;
    const std::uint32_t logits_stride = kVocab + 96u;

    check(hipMalloc(&d_weight, weight.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc weight");
    check(hipMalloc(&d_hidden, hidden.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc hidden");
    check(hipMalloc(&d_hidden_strided,
                    static_cast<std::size_t>(kRows) * hidden_stride * sizeof(shelf::bf16_t)) ==
              hipSuccess,
          "malloc hidden strided");
    check(hipMalloc(&d_codes, static_cast<std::size_t>(kVocab) * kK) == hipSuccess, "malloc codes");
    check(hipMalloc(&d_scales, static_cast<std::size_t>(kVocab) * (kK / 32u) *
                                   sizeof(shelf::fp16_t)) == hipSuccess,
          "malloc scales");
    check(hipMalloc(&d_logits, static_cast<std::size_t>(kRows) * logits_stride * sizeof(float)) ==
              hipSuccess,
          "malloc logits");
    if (failures != 0) return 1;

    check(hipMemcpy(d_weight, weight.data(), weight.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "copy weight");
    for (std::uint32_t r = 0; r < kRows; ++r) {
        check(hipMemset(d_hidden_strided + static_cast<std::size_t>(r) * hidden_stride, 0,
                        hidden_stride * sizeof(shelf::bf16_t)) == hipSuccess,
              "clear hidden row");
        check(hipMemcpy(d_hidden_strided + static_cast<std::size_t>(r) * hidden_stride,
                        hidden.data() + static_cast<std::size_t>(r) * kK,
                        kK * sizeof(shelf::bf16_t), hipMemcpyHostToDevice) == hipSuccess,
              "copy hidden row");
    }

    check(shelf::kernel::launch_quantize_bf16_to_fp8_w32(
              d_weight, kK, d_codes, d_scales, kVocab, kK, nullptr) == hipSuccess,
          "quantize");
    check(hipMemset(d_logits, 0, static_cast<std::size_t>(kRows) * logits_stride * sizeof(float)) ==
              hipSuccess,
          "clear logits");
    check(shelf::kernel::launch_lm_head_fp8_w32(
              d_hidden_strided, hidden_stride, d_codes, d_scales, d_logits, logits_stride,
              kRows, kVocab, kK, nullptr) == hipSuccess,
          "lm head");
    check(hipDeviceSynchronize() == hipSuccess, "sync");

    std::vector<std::uint8_t> codes(static_cast<std::size_t>(kVocab) * kK);
    std::vector<shelf::fp16_t> scales(static_cast<std::size_t>(kVocab) * (kK / 32u));
    std::vector<float> logits(static_cast<std::size_t>(kRows) * logits_stride);
    check(hipMemcpy(codes.data(), d_codes, codes.size(), hipMemcpyDeviceToHost) == hipSuccess,
          "copy back codes");
    check(hipMemcpy(scales.data(), d_scales, scales.size() * sizeof(shelf::fp16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "copy back scales");
    check(hipMemcpy(logits.data(), d_logits, logits.size() * sizeof(float),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "copy back logits");

    float worst = 0.0f;
    float worst_scale = 0.0f;
    float bf16_worst = 0.0f;
    float reference_magnitude = 0.0f;
    float bf16_best[kRows];
    float fp8_best[kRows];
    std::uint32_t bf16_argmax[kRows] = {};
    std::uint32_t fp8_argmax[kRows] = {};
    for (std::uint32_t r = 0; r < kRows; ++r) {
        bf16_best[r] = -INFINITY;
        fp8_best[r] = -INFINITY;
    }
    std::uint32_t argmax_matches = 0;
    for (std::uint32_t v = 0; v < kVocab; ++v) {
        const std::uint8_t* code_row = codes.data() + static_cast<std::size_t>(v) * kK;
        const shelf::fp16_t* scale_row =
            scales.data() + static_cast<std::size_t>(v) * (kK / 32u);
        for (std::uint32_t block = 0; block < kK / 32u; ++block) {
            const float scale = shelf::fp16_to_f32(scale_row[block]);
            float block_amax = 0.0f;
            for (std::uint32_t i = 0; i < 32u; ++i) {
                const std::uint32_t index = block * 32u + i;
                const float original =
                    bf16_to_float(weight[static_cast<std::size_t>(v) * kK + index].data);
                const float dequant = e4m3_to_float(code_row[index]) * scale;
                block_amax = std::fmax(block_amax, std::fabs(original));
                worst_scale = std::fmax(worst_scale, std::fabs(dequant - original));
                (void)dequant;
                (void)index;
            }
            if (block_amax > 0.0f && std::fabs(scale * 448.0f - block_amax) > block_amax * 1e-2f) {
                check(false, "block scale matches the block maximum");
                block_amax = -1.0f;
            }
            if (block_amax > 0.0f && worst_scale > block_amax * 0.0625f + 1.0e-6f) {
                check(false, "quantization error stays inside the E4M3 step");
                block_amax = -1.0f;
            }
            (void)block_amax;
        }

        for (std::uint32_t r = 0; r < kRows; ++r) {
            double reference = 0.0;
            for (std::uint32_t i = 0; i < kK; ++i) {
                const float scale = shelf::fp16_to_f32(scale_row[i / 32u]);
                const float dequant = e4m3_to_float(code_row[i]) * scale;
                reference += static_cast<double>(dequant) *
                             bf16_to_float(hidden[static_cast<std::size_t>(r) * kK + i].data);
            }
            const float got = logits[static_cast<std::size_t>(r) * logits_stride + v];
            worst = std::fmax(worst, static_cast<float>(std::fabs(got - reference)));

            double bf16_reference = 0.0;
            for (std::uint32_t i = 0; i < kK; ++i) {
                bf16_reference +=
                    static_cast<double>(bf16_to_float(
                        weight[static_cast<std::size_t>(v) * kK + i].data)) *
                    bf16_to_float(hidden[static_cast<std::size_t>(r) * kK + i].data);
            }
            bf16_worst = std::fmax(bf16_worst, static_cast<float>(std::fabs(got - bf16_reference)));
            reference_magnitude = std::fmax(reference_magnitude,
                                            static_cast<float>(std::fabs(bf16_reference)));
            const float comparison =
                static_cast<float>(bf16_reference);
            if (comparison > bf16_best[r] || v == 0u) {
                bf16_best[r] = comparison;
                bf16_argmax[r] = v;
            }
            if (got > fp8_best[r] || v == 0u) {
                fp8_best[r] = got;
                fp8_argmax[r] = v;
            }
        }
    }

    for (std::uint32_t r = 0; r < kRows; ++r)
        if (bf16_argmax[r] == fp8_argmax[r]) ++argmax_matches;

    check(worst < 2.0e-2f, "logits match the dequantized reference");
    check(argmax_matches == kRows, "fp8 keeps the bf16 argmax");
    check(bf16_worst < reference_magnitude * 0.05f,
          "fp8 logits stay within 5% of the bf16 logits");
    std::fprintf(stderr,
                 "worst logit diff %.6f, worst bf16 diff %.6f (max |logit| %.3f), "
                 "worst weight diff %.6f, argmax %u/%u\n",
                 worst, bf16_worst, reference_magnitude, worst_scale, argmax_matches, kRows);

    {
        constexpr std::uint32_t kTinyRows = 4u;
        constexpr std::uint32_t kTinyK = 32u;
        const float amplitudes[kTinyRows] = {0.0f, 1.0e-7f, 1.0e-3f, 1.0f};
        std::vector<shelf::bf16_t> tiny(kTinyRows * kTinyK);
        for (std::uint32_t r = 0; r < kTinyRows; ++r) {
            for (std::uint32_t i = 0; i < kTinyK; ++i) {
                const float value =
                    amplitudes[r] * (i % 2u == 0u ? 1.0f : -1.0f) * (1.0f + 0.01f * float(i));
                tiny[static_cast<std::size_t>(r) * kTinyK + i].data = shelf::f32_to_bf16_rne(value);
            }
        }
        shelf::bf16_t* d_tiny = nullptr;
        std::uint8_t* d_tiny_codes = nullptr;
        shelf::fp16_t* d_tiny_scales = nullptr;
        check(hipMalloc(&d_tiny, tiny.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc tiny");
        check(hipMalloc(&d_tiny_codes, tiny.size()) == hipSuccess, "malloc tiny codes");
        check(hipMalloc(&d_tiny_scales, kTinyRows * sizeof(shelf::fp16_t)) == hipSuccess,
              "malloc tiny scales");
        check(hipMemcpy(d_tiny, tiny.data(), tiny.size() * sizeof(shelf::bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess, "copy tiny");
        check(shelf::kernel::launch_quantize_bf16_to_fp8_w32(
                  d_tiny, kTinyK, d_tiny_codes, d_tiny_scales, kTinyRows, kTinyK,
                  nullptr) == hipSuccess,
              "quantize tiny");
        std::vector<shelf::fp16_t> tiny_scales(kTinyRows);
        std::vector<std::uint8_t> tiny_codes(tiny.size());
        check(hipMemcpy(tiny_scales.data(), d_tiny_scales,
                        kTinyRows * sizeof(shelf::fp16_t), hipMemcpyDeviceToHost) == hipSuccess,
              "copy tiny scales");
        check(hipMemcpy(tiny_codes.data(), d_tiny_codes, tiny.size(),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "copy tiny codes");
        bool scales_ok = true;
        for (std::uint32_t r = 0; r < kTinyRows; ++r) {
            const float scale = shelf::fp16_to_f32(tiny_scales[r]);
            if (!(scale > 0.0f) || !std::isfinite(scale)) scales_ok = false;
            float block_amax = 0.0f;
            for (std::uint32_t i = 0; i < kTinyK; ++i)
                block_amax = std::fmax(
                    block_amax,
                    std::fabs(bf16_to_float(
                        tiny[static_cast<std::size_t>(r) * kTinyK + i].data)));
            const float bound = block_amax * 0.0625f + 1.0e-9f;
            for (std::uint32_t i = 0; i < kTinyK; ++i) {
                const float original = bf16_to_float(tiny[static_cast<std::size_t>(r) * kTinyK + i].data);
                const float dequant = e4m3_to_float(tiny_codes[static_cast<std::size_t>(r) * kTinyK + i]) * scale;
                if (std::fabs(dequant - original) > bound && original != 0.0f) scales_ok = false;
            }
        }
        check(scales_ok, "subnormal scales stay positive and representable");
        shelf::gpu::discard_cleanup_result(hipFree(d_tiny));
        shelf::gpu::discard_cleanup_result(hipFree(d_tiny_codes));
        shelf::gpu::discard_cleanup_result(hipFree(d_tiny_scales));
    }

    {
        // E4M3FN の特殊値。amax を 448 にして scale を 1.0 に固定し、
        // v_cvt_pk_fp8_f32 のコードを OCP の期待値と直接比較する。
        constexpr std::uint32_t kSpecials = 32u;
        const float values[kSpecials] = {
            0.0f, -0.0f, 448.0f, -448.0f, 449.0f, 1.0f, 2.0f, 0.5f,
            -0.5f, 0x1p-9f, 0x1p-13f, 0x1p-10f, 256.0f, 288.0f, 320.0f, 384.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const std::uint8_t expected[kSpecials] = {
            0x00u, 0x80u, 0x7Eu, 0xFEu, 0x7Eu, 0x38u, 0x40u, 0x30u,
            0xB0u, 0x01u, 0x00u, 0x00u, 0x78u, 0x79u, 0x7Au, 0x7Cu,
            0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u,
            0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u};
        std::vector<shelf::bf16_t> special(kSpecials);
        for (std::uint32_t i = 0; i < kSpecials; ++i)
            special[i].data = shelf::f32_to_bf16_rne(values[i]);
        shelf::bf16_t* d_special = nullptr;
        std::uint8_t* d_special_codes = nullptr;
        shelf::fp16_t* d_special_scales = nullptr;
        check(hipMalloc(&d_special, kSpecials * sizeof(shelf::bf16_t)) == hipSuccess,
              "malloc special");
        check(hipMalloc(&d_special_codes, kSpecials) == hipSuccess, "malloc special codes");
        check(hipMalloc(&d_special_scales, sizeof(shelf::fp16_t)) == hipSuccess,
              "malloc special scale");
        check(hipMemcpy(d_special, special.data(), kSpecials * sizeof(shelf::bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess, "copy special");
        check(shelf::kernel::launch_quantize_bf16_to_fp8_w32(
                  d_special, kSpecials, d_special_codes, d_special_scales, 1u, kSpecials,
                  nullptr) == hipSuccess,
              "quantize special");
        std::uint8_t got_codes[kSpecials] = {};
        shelf::fp16_t got_scale{};
        check(hipMemcpy(got_codes, d_special_codes, kSpecials, hipMemcpyDeviceToHost) ==
                  hipSuccess,
              "copy special codes");
        check(hipMemcpy(&got_scale, d_special_scales, sizeof(got_scale), hipMemcpyDeviceToHost) ==
                  hipSuccess,
              "copy special scale");
        check(std::fabs(shelf::fp16_to_f32(got_scale) - 1.0f) < 1e-6f,
              "special block keeps unit scale");
        bool codes_ok = true;
        for (std::uint32_t i = 0; i < kSpecials; ++i) {
            if (got_codes[i] != expected[i]) {
                std::fprintf(stderr, "special %u: got %02x expected %02x (value %a)\n", i,
                             got_codes[i], expected[i], values[i]);
                codes_ok = false;
            }
        }
        check(codes_ok, "e4m3 codes match the OCP special values");
        shelf::gpu::discard_cleanup_result(hipFree(d_special));
        shelf::gpu::discard_cleanup_result(hipFree(d_special_codes));
        shelf::gpu::discard_cleanup_result(hipFree(d_special_scales));
    }

    {
        // ホスト側エンコーダと GPU の v_cvt_pk_fp8_f32 が同じコードを返すこと。
        // 各 block に 448 を入れて scale を 1.0 に固定する。
        constexpr std::uint32_t kBlocks = 8u;
        constexpr std::uint32_t kSpan = kBlocks * 32u;
        std::mt19937 erng(7u);
        std::uniform_real_distribution<float> edist(-448.0f, 448.0f);
        std::vector<shelf::bf16_t> span(kSpan);
        for (std::uint32_t b = 0; b < kBlocks; ++b) {
            span[b * 32u].data = shelf::f32_to_bf16_rne(448.0f);
            for (std::uint32_t i = 1; i < 32u; ++i)
                span[b * 32u + i].data = shelf::f32_to_bf16_rne(edist(erng));
        }
        shelf::bf16_t* d_span = nullptr;
        std::uint8_t* d_span_codes = nullptr;
        shelf::fp16_t* d_span_scales = nullptr;
        check(hipMalloc(&d_span, kSpan * sizeof(shelf::bf16_t)) == hipSuccess, "malloc span");
        check(hipMalloc(&d_span_codes, kSpan) == hipSuccess, "malloc span codes");
        check(hipMalloc(&d_span_scales, kBlocks * sizeof(shelf::fp16_t)) == hipSuccess,
              "malloc span scales");
        check(hipMemcpy(d_span, span.data(), kSpan * sizeof(shelf::bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess, "copy span");
        check(shelf::kernel::launch_quantize_bf16_to_fp8_w32(
                  d_span, kSpan, d_span_codes, d_span_scales, 1u, kSpan, nullptr) == hipSuccess,
              "quantize span");
        std::vector<std::uint8_t> span_codes(kSpan);
        check(hipMemcpy(span_codes.data(), d_span_codes, kSpan, hipMemcpyDeviceToHost) ==
                  hipSuccess,
              "copy span codes");
        bool host_matches = true;
        for (std::uint32_t i = 0; i < kSpan; ++i) {
            const float value = bf16_to_float(span[i].data);
            const std::uint8_t host = shelf::quantization::codec::e4m3_encode_u8(value);
            if (host != span_codes[i]) {
                std::fprintf(stderr, "enc %u: host %02x gpu %02x (value %a)\n", i, host,
                             span_codes[i], value);
                host_matches = false;
            }
        }
        check(host_matches, "host encoder matches the gpu encoder");
        shelf::gpu::discard_cleanup_result(hipFree(d_span));
        shelf::gpu::discard_cleanup_result(hipFree(d_span_codes));
        shelf::gpu::discard_cleanup_result(hipFree(d_span_scales));
    }

    shelf::gpu::discard_cleanup_result(hipFree(d_weight));
    shelf::gpu::discard_cleanup_result(hipFree(d_hidden));
    shelf::gpu::discard_cleanup_result(hipFree(d_hidden_strided));
    shelf::gpu::discard_cleanup_result(hipFree(d_codes));
    shelf::gpu::discard_cleanup_result(hipFree(d_scales));
    shelf::gpu::discard_cleanup_result(hipFree(d_logits));

    if (failures != 0) {
        std::fprintf(stderr, "lm head fp8: %d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "lm head fp8: 0 failures\n");
    return 0;
}
