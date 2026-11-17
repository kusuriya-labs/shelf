#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/lm_head_int8.h>

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

float int8_to_float(std::uint8_t bits) {
    return static_cast<float>(static_cast<std::int8_t>(bits));
}

std::uint8_t int8_code(float value) {
    const float rounded = std::nearbyintf(value);
    const float clamped = std::fmin(std::fmax(rounded, -127.0f), 127.0f);
    return static_cast<std::uint8_t>(
        static_cast<std::int8_t>(static_cast<std::int32_t>(clamped)));
}

std::vector<shelf::bf16_t> random_bf16(std::size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> out(count);
    for (auto& value : out) value.data = shelf::f32_to_bf16_rne(dist(rng));
    return out;
}

constexpr std::uint32_t kVocab = 1024u;
constexpr std::uint32_t kK = 5120u;
constexpr std::uint32_t kRows = 16u;
constexpr std::uint32_t kCheckedRows[] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 12u, 15u, 16u};

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

    check(shelf::kernel::launch_quantize_bf16_to_int8_w32(
              d_weight, kK, d_codes, d_scales, kVocab, kK, nullptr) == hipSuccess,
          "quantize");

    std::vector<std::uint8_t> codes(static_cast<std::size_t>(kVocab) * kK);
    std::vector<shelf::fp16_t> scales(static_cast<std::size_t>(kVocab) * (kK / 32u));
    check(hipMemcpy(codes.data(), d_codes, codes.size(), hipMemcpyDeviceToHost) == hipSuccess,
          "copy back codes");
    check(hipMemcpy(scales.data(), d_scales, scales.size() * sizeof(shelf::fp16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "copy back scales");

    std::vector<double> reference(static_cast<std::size_t>(kVocab) * kRows);
    std::vector<double> bf16_reference(static_cast<std::size_t>(kVocab) * kRows);
    std::vector<float> dequant_row(kK);
    std::vector<float> weight_row(kK);
    float worst_scale = 0.0f;
    float reference_magnitude = 0.0f;
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
                const float dequant = int8_to_float(code_row[index]) * scale;
                block_amax = std::fmax(block_amax, std::fabs(original));
                worst_scale = std::fmax(worst_scale, std::fabs(dequant - original));
                dequant_row[index] = dequant;
                weight_row[index] = original;
            }
            if (block_amax > 0.0f &&
                std::fabs(scale * 127.0f - block_amax) > block_amax * 1e-2f) {
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
            const shelf::bf16_t* hidden_row = hidden.data() + static_cast<std::size_t>(r) * kK;
            double fp8_sum = 0.0;
            double bf16_sum = 0.0;
            for (std::uint32_t i = 0; i < kK; ++i) {
                const double h = bf16_to_float(hidden_row[i].data);
                fp8_sum += static_cast<double>(dequant_row[i]) * h;
                bf16_sum += static_cast<double>(weight_row[i]) * h;
            }
            reference[static_cast<std::size_t>(v) * kRows + r] = fp8_sum;
            bf16_reference[static_cast<std::size_t>(v) * kRows + r] = bf16_sum;
            reference_magnitude =
                std::fmax(reference_magnitude, static_cast<float>(std::fabs(bf16_sum)));
        }
    }

    std::vector<float> logits(static_cast<std::size_t>(kRows) * logits_stride);
    auto run_and_compare = [&](std::uint32_t rows, std::uint32_t vocab, const char* tag) {
        check(hipMemset(d_logits, 0,
                        static_cast<std::size_t>(kRows) * logits_stride * sizeof(float)) ==
                  hipSuccess,
              "clear logits");
        check(shelf::kernel::launch_lm_head_int8_w32(
                  d_hidden_strided, hidden_stride, d_codes, d_scales, d_logits, logits_stride,
                  rows, vocab, kK, nullptr) == hipSuccess,
              "lm head");
        check(hipDeviceSynchronize() == hipSuccess, "sync");
        check(hipMemcpy(logits.data(), d_logits, logits.size() * sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "copy back logits");

        float worst = 0.0f;
        float bf16_worst = 0.0f;
        std::uint32_t argmax_matches = 0;
        std::uint32_t argmax_ties = 0;
        for (std::uint32_t r = 0; r < rows; ++r) {
            float bf16_best = -INFINITY;
            float int8_best = -INFINITY;
            std::uint32_t bf16_argmax = 0;
            std::uint32_t int8_argmax = 0;
            for (std::uint32_t v = 0; v < vocab; ++v) {
                const std::size_t slot = static_cast<std::size_t>(v) * kRows + r;
                const float got = logits[static_cast<std::size_t>(r) * logits_stride + v];
                worst = std::fmax(
                    worst, static_cast<float>(std::fabs(got - reference[slot])));
                bf16_worst = std::fmax(
                    bf16_worst, static_cast<float>(std::fabs(got - bf16_reference[slot])));
                const float comparison = static_cast<float>(bf16_reference[slot]);
                if (comparison > bf16_best || v == 0u) {
                    bf16_best = comparison;
                    bf16_argmax = v;
                }
                if (got > int8_best || v == 0u) {
                    int8_best = got;
                    int8_argmax = v;
                }
            }
            if (bf16_argmax == int8_argmax) {
                ++argmax_matches;
            } else if (std::fabs(static_cast<float>(
                           bf16_reference[static_cast<std::size_t>(int8_argmax) * kRows + r]) -
                               bf16_best) <=
                       std::fabs(bf16_best) * 0.05f) {
                ++argmax_ties;
            }
        }
        check(worst < 2.0e-2f, (std::string(tag) + ": logits match the dequantized reference").c_str());
        check(argmax_matches + argmax_ties == rows,
              (std::string(tag) + ": int8 keeps the bf16 argmax inside the error").c_str());
        check(bf16_worst < reference_magnitude * 0.05f,
              (std::string(tag) + ": int8 logits stay within 5% of bf16").c_str());
        if (vocab < kVocab) {
            bool tail_ok = true;
            for (std::uint32_t r = 0; r < rows; ++r)
                for (std::uint32_t v = vocab; v < kVocab; ++v)
                    if (logits[static_cast<std::size_t>(r) * logits_stride + v] != 0.0f)
                        tail_ok = false;
            check(tail_ok,
                  (std::string(tag) + ": rows past the vocabulary are untouched").c_str());
        }
        std::fprintf(stderr,
                     "%-14s rows %2u: worst logit diff %.6f, worst bf16 diff %.6f "
                     "(max |logit| %.3f), argmax %u/%u (%u ties)\n",
                     tag, rows, worst, bf16_worst, reference_magnitude, argmax_matches, rows,
                     argmax_ties);
    };

    for (const std::uint32_t rows : kCheckedRows)
        run_and_compare(rows, kVocab, "full-vocab");
    for (const std::uint32_t vocab : {1000u, 1001u, 1007u, 127u, 129u})
        run_and_compare(8u, vocab, "vocab-tail");
    run_and_compare(16u, 1007u, "vocab-tail");
    run_and_compare(6u, 1000u, "vocab-tail");

    {
        // GEMV（rows<=4）と WMMA（rows>=5）が同じ行で同じ値になること。
        // 端数 rows でも先頭行が変わらないことを、既存の重みで確かめる。
        std::vector<float> gemv_logits(static_cast<std::size_t>(kRows) * logits_stride);
        std::vector<float> wmma_logits(static_cast<std::size_t>(kRows) * logits_stride);
        check(shelf::kernel::launch_lm_head_int8_w32(
                  d_hidden_strided, hidden_stride, d_codes, d_scales, d_logits, logits_stride,
                  4u, kVocab, kK, nullptr) == hipSuccess,
              "boundary gemv");
        check(hipMemcpy(gemv_logits.data(), d_logits, gemv_logits.size() * sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "boundary gemv copy");
        check(shelf::kernel::launch_lm_head_int8_w32(
                  d_hidden_strided, hidden_stride, d_codes, d_scales, d_logits, logits_stride,
                  5u, kVocab, kK, nullptr) == hipSuccess,
              "boundary wmma");
        check(hipMemcpy(wmma_logits.data(), d_logits, wmma_logits.size() * sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "boundary wmma copy");
        float boundary_worst = 0.0f;
        for (std::uint32_t r = 0; r < 4u; ++r)
            for (std::uint32_t v = 0; v < kVocab; ++v) {
                const std::size_t at = static_cast<std::size_t>(r) * logits_stride + v;
                boundary_worst = std::fmax(
                    boundary_worst, std::fabs(gemv_logits[at] - wmma_logits[at]));
            }
        check(boundary_worst < reference_magnitude * 1e-3f,
              "gemv and wmma agree on the shared rows");
        std::fprintf(stderr, "gemv/wmma boundary worst diff %.6f\n", boundary_worst);

        check(shelf::kernel::launch_lm_head_int8_w32(
                  d_hidden_strided, hidden_stride, d_codes, d_scales, d_logits, logits_stride,
                  17u, kVocab, kK, nullptr) == hipErrorInvalidValue,
              "rows above the tile are rejected");
    }

    {
        // k が WMMA のステージ幅で割り切れないときは GEMV へ戻る。そちらも
        // 逆量子化リファレンスと一致すること。
        constexpr std::uint32_t kTailK = 5248u;
        constexpr std::uint32_t kTailVocab = 256u;
        constexpr std::uint32_t kTailRows = 8u;
        std::mt19937 trng(23u);
        std::uniform_real_distribution<float> tdist(-1.0f, 1.0f);
        std::vector<shelf::bf16_t> tw(static_cast<std::size_t>(kTailVocab) * kTailK);
        for (auto& w : tw) w.data = shelf::f32_to_bf16_rne(tdist(trng));
        std::vector<shelf::bf16_t> th(static_cast<std::size_t>(kTailRows) * kTailK);
        for (auto& h : th) h.data = shelf::f32_to_bf16_rne(tdist(trng));
        shelf::bf16_t* d_tw = nullptr;
        shelf::bf16_t* d_th = nullptr;
        std::uint8_t* d_tc = nullptr;
        shelf::fp16_t* d_ts = nullptr;
        float* d_tl = nullptr;
        check(hipMalloc(&d_tw, tw.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc tail w");
        check(hipMalloc(&d_th, th.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc tail h");
        check(hipMalloc(&d_tc, tw.size()) == hipSuccess, "malloc tail codes");
        check(hipMalloc(&d_ts, kTailVocab * (kTailK / 32u) * sizeof(shelf::fp16_t)) == hipSuccess,
              "malloc tail scales");
        check(hipMalloc(&d_tl, static_cast<std::size_t>(kTailRows) * kTailVocab * sizeof(float)) ==
                  hipSuccess,
              "malloc tail logits");
        check(hipMemcpy(d_tw, tw.data(), tw.size() * sizeof(shelf::bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess, "copy tail w");
        check(hipMemcpy(d_th, th.data(), th.size() * sizeof(shelf::bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess, "copy tail h");
        check(shelf::kernel::launch_quantize_bf16_to_int8_w32(
                  d_tw, kTailK, d_tc, d_ts, kTailVocab, kTailK, nullptr) == hipSuccess,
              "quantize tail");
        std::vector<std::uint8_t> tc(tw.size());
        std::vector<shelf::fp16_t> ts(kTailVocab * (kTailK / 32u));
        check(hipMemcpy(tc.data(), d_tc, tc.size(), hipMemcpyDeviceToHost) == hipSuccess,
              "copy tail codes");
        check(hipMemcpy(ts.data(), d_ts, ts.size() * sizeof(shelf::fp16_t),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "copy tail scales");
        std::vector<float> tl(static_cast<std::size_t>(kTailRows) * kTailVocab);
        if (shelf::kernel::launch_lm_head_int8_w32(
                d_th, kTailK, d_tc, d_ts, d_tl, kTailVocab, kTailRows, kTailVocab, kTailK,
                nullptr) != hipSuccess) {
            check(false, "tail lm head launch");
        } else if (hipMemcpy(tl.data(), d_tl, tl.size() * sizeof(float),
                             hipMemcpyDeviceToHost) != hipSuccess) {
            check(false, "tail logits copy");
        } else {
            float tail_worst = 0.0f;
            for (std::uint32_t r = 0; r < kTailRows; ++r)
                for (std::uint32_t v = 0; v < kTailVocab; ++v) {
                    double want = 0.0;
                    for (std::uint32_t i = 0; i < kTailK; ++i) {
                        const std::uint8_t code =
                            tc[static_cast<std::size_t>(v) * kTailK + i];
                        const float scale = shelf::fp16_to_f32(
                            ts[static_cast<std::size_t>(v) * (kTailK / 32u) + i / 32u]);
                        want += static_cast<double>(int8_to_float(code) * scale) *
                                bf16_to_float(
                                    th[static_cast<std::size_t>(r) * kTailK + i].data);
                    }
                    tail_worst = std::fmax(
                        tail_worst,
                        static_cast<float>(
                            std::fabs(static_cast<double>(
                                tl[static_cast<std::size_t>(r) * kTailVocab + v]) - want)));
                }
            check(tail_worst < 5.0e-2f, "gemv fallback matches the dequantized reference");
            std::fprintf(stderr, "gemv fallback worst diff %.6f\n", tail_worst);
        }
        shelf::gpu::discard_cleanup_result(hipFree(d_tw));
        shelf::gpu::discard_cleanup_result(hipFree(d_th));
        shelf::gpu::discard_cleanup_result(hipFree(d_tc));
        shelf::gpu::discard_cleanup_result(hipFree(d_ts));
        shelf::gpu::discard_cleanup_result(hipFree(d_tl));
    }

    std::fprintf(stderr, "worst weight diff %.6f\n", worst_scale);

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
        check(shelf::kernel::launch_quantize_bf16_to_int8_w32(
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
            const float bound = scale * 0.5f + block_amax * 1.0e-3f + 1.0e-9f;
            for (std::uint32_t i = 0; i < kTinyK; ++i) {
                const float original = bf16_to_float(tiny[static_cast<std::size_t>(r) * kTinyK + i].data);
                const float dequant = int8_to_float(tiny_codes[static_cast<std::size_t>(r) * kTinyK + i]) * scale;
                if (std::fabs(dequant - original) > bound && original != 0.0f) scales_ok = false;
            }
        }
        check(scales_ok, "subnormal scales stay positive and representable");
        shelf::gpu::discard_cleanup_result(hipFree(d_tiny));
        shelf::gpu::discard_cleanup_result(hipFree(d_tiny_codes));
        shelf::gpu::discard_cleanup_result(hipFree(d_tiny_scales));
    }

    {
        // INT8 の丸め。scale が 2^-7 に固定されるよう amax を 127 * 2^-7 にして、
        // 偶数丸めと符号を直接比較する。
        constexpr std::uint32_t kSpecials = 32u;
        const float unit = 0x1p-8f;
        const float values[kSpecials] = {
            0.0f, -0.0f, 127.0f * unit, -127.0f * unit,
            unit, -unit, 0.5f * unit, 1.5f * unit,
            2.5f * unit, 3.5f * unit, 126.0f * unit, -126.0f * unit,
            0.49f * unit, 0.51f * unit, 1.49f * unit, 1.51f * unit,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const std::uint8_t expected[kSpecials] = {
            0x00u, 0x00u, 0x7Fu, 0x81u, 0x01u, 0xFFu, 0x00u, 0x02u,
            0x02u, 0x04u, 0x7Eu, 0x82u, 0x00u, 0x01u, 0x01u, 0x02u,
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
        check(shelf::kernel::launch_quantize_bf16_to_int8_w32(
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
        check(std::fabs(shelf::fp16_to_f32(got_scale) - unit) < 1e-9f,
              "special block stores the power of two scale");
        bool codes_ok = true;
        for (std::uint32_t i = 0; i < kSpecials; ++i) {
            if (got_codes[i] != expected[i]) {
                std::fprintf(stderr, "special %u: got %02x expected %02x (value %a)\n", i,
                             got_codes[i], expected[i], values[i]);
                codes_ok = false;
            }
        }
        check(codes_ok, "int8 codes round half to even");
        shelf::gpu::discard_cleanup_result(hipFree(d_special));
        shelf::gpu::discard_cleanup_result(hipFree(d_special_codes));
        shelf::gpu::discard_cleanup_result(hipFree(d_special_scales));
    }

    shelf::gpu::discard_cleanup_result(hipFree(d_weight));
    shelf::gpu::discard_cleanup_result(hipFree(d_hidden));
    shelf::gpu::discard_cleanup_result(hipFree(d_hidden_strided));
    shelf::gpu::discard_cleanup_result(hipFree(d_codes));
    shelf::gpu::discard_cleanup_result(hipFree(d_scales));
    shelf::gpu::discard_cleanup_result(hipFree(d_logits));

    if (failures != 0) {
        std::fprintf(stderr, "lm head int8: %d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "lm head int8: 0 failures\n");
    return 0;
}
