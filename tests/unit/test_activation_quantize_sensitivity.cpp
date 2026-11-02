#include <shelf/quantization/codec/e4m3.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using ps::quantization::codec::e4m3_decode_f32;
using ps::quantization::codec::e4m3_encode_u8;

namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        g_passed++;
        std::printf("PASS: %s\n", msg);
    } else {
        g_failed++;
        std::printf("FAIL: %s\n", msg);
    }
}

uint16_t f32_to_bf16_bits(float f) {
    uint32_t ui;
    std::memcpy(&ui, &f, 4);
    return static_cast<uint16_t>((ui + 0x8000u) >> 16);
}

float bf16_bits_to_f32(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

struct QuantResult {
    std::vector<uint8_t> codes;
    std::vector<float> scales;
};

QuantResult quantize_rows(const std::vector<uint16_t>& in, uint32_t rows, uint32_t k) {
    QuantResult r;
    const uint32_t groups = (k + 127u) / 128u;
    r.codes.assign(static_cast<size_t>(rows) * k, 0u);
    r.scales.assign(static_cast<size_t>(rows) * groups, 0.f);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t i0 = g * 128u;
            const uint32_t i1 = (i0 + 128u < k) ? i0 + 128u : k;
            float peak = 0.f;
            for (uint32_t i = i0; i < i1; ++i) {
                const float v = bf16_bits_to_f32(in[static_cast<size_t>(m) * k + i]);
                if (std::isfinite(v)) {
                    const float a = std::fabs(v);
                    if (a > peak) peak = a;
                }
            }
            const float scale = peak > 0.f ? peak / 448.0f : 0.f;
            const float inv = scale > 0.f ? 1.0f / scale : 0.f;
            r.scales[static_cast<size_t>(m) * groups + g] = scale;
            for (uint32_t i = i0; i < i1; ++i) {
                const float x = bf16_bits_to_f32(in[static_cast<size_t>(m) * k + i]);
                const float v = std::isfinite(x) ? x : 0.f;
                r.codes[static_cast<size_t>(m) * k + i] =
                    scale > 0.f ? e4m3_encode_u8(v * inv) : 0u;
            }
        }
    }
    return r;
}

std::vector<float> dequantize_rows(const QuantResult& q, uint32_t rows, uint32_t k) {
    const uint32_t groups = (k + 127u) / 128u;
    std::vector<float> out(static_cast<size_t>(rows) * k, 0.f);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t i0 = g * 128u;
            const uint32_t i1 = (i0 + 128u < k) ? i0 + 128u : k;
            const float s = q.scales[static_cast<size_t>(m) * groups + g];
            for (uint32_t i = i0; i < i1; ++i) {
                out[static_cast<size_t>(m) * k + i] =
                    e4m3_decode_f32(q.codes[static_cast<size_t>(m) * k + i]) * s;
            }
        }
    }
    return out;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

struct Sensitivity {
    double input_rel = 0.0;
    double dequant_rel = 0.0;
    double scale_changed_ratio = 0.0;
    double codes_changed_ratio = 0.0;
};

Sensitivity measure(uint32_t rows, uint32_t k, float eps, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);

    std::vector<uint16_t> x1(static_cast<size_t>(rows) * k);
    std::vector<uint16_t> x2(x1.size());
    std::vector<float> raw(x1.size());
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = dist(rng);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const float v1 = raw[i];
        const float v2 = raw[i] * (1.0f + eps * unit(rng));
        x1[i] = f32_to_bf16_bits(v1);
        x2[i] = f32_to_bf16_bits(v2);
    }

    std::vector<float> in1(x1.size());
    std::vector<float> in2(x1.size());
    for (std::size_t i = 0; i < x1.size(); ++i) {
        in1[i] = bf16_bits_to_f32(x1[i]);
        in2[i] = bf16_bits_to_f32(x2[i]);
    }

    const QuantResult q1 = quantize_rows(x1, rows, k);
    const QuantResult q2 = quantize_rows(x2, rows, k);
    const std::vector<float> d1 = dequantize_rows(q1, rows, k);
    const std::vector<float> d2 = dequantize_rows(q2, rows, k);

    Sensitivity s;
    s.input_rel = rel_l2(in1, in2);
    s.dequant_rel = rel_l2(d1, d2);
    std::size_t scale_diff = 0;
    for (std::size_t i = 0; i < q1.scales.size(); ++i) {
        if (std::memcmp(&q1.scales[i], &q2.scales[i], 4) != 0) ++scale_diff;
    }
    s.scale_changed_ratio = q1.scales.empty()
        ? 0.0
        : static_cast<double>(scale_diff) / static_cast<double>(q1.scales.size());
    std::size_t code_diff = 0;
    for (std::size_t i = 0; i < q1.codes.size(); ++i) {
        if (q1.codes[i] != q2.codes[i]) ++code_diff;
    }
    s.codes_changed_ratio = q1.codes.empty()
        ? 0.0
        : static_cast<double>(code_diff) / static_cast<double>(q1.codes.size());
    return s;
}

}  // namespace

int main() {
    std::printf("=== FP8 Activation Quantize Sensitivity (perturbed input) ===\n");
    const uint32_t rows = 16;
    const float eps_list[] = {0.003f, 0.01f, 0.03f, 0.07f, 0.10f};
    const uint32_t k_list[] = {5120u, 17408u};

    double worst_amp = 0.0;
    double first_amp = 0.0;
    double last_amp = 0.0;
    bool amp_above_one = true;
    bool amp_monotone = true;
    double prev_amp = 0.0;
    bool first_seen = false;
    for (uint32_t k : k_list) {
        prev_amp = 0.0;
        first_seen = false;
        for (float eps : eps_list) {
            const Sensitivity s = measure(rows, k, eps, 1234u + k + static_cast<uint32_t>(eps * 1000.0f));
            const double amp = s.input_rel > 0.0 ? s.dequant_rel / s.input_rel : 0.0;
            if (k == k_list[0]) {
                if (!first_seen) first_amp = amp;
                last_amp = amp;
                first_seen = true;
            }
            if (amp > worst_amp) worst_amp = amp;
            if (amp <= 1.0) amp_above_one = false;
            if (prev_amp > 0.0 && amp > prev_amp * 1.05) amp_monotone = false;
            prev_amp = amp;
            std::printf(
                "k=%-5u eps=%.3f input_rel=%.4e dequant_rel=%.4e amp=%.3f "
                "scale_changed=%.4f codes_changed=%.4f\n",
                k, static_cast<double>(eps), s.input_rel, s.dequant_rel, amp,
                s.scale_changed_ratio, s.codes_changed_ratio);
        }
    }

    char msg[128];
    std::snprintf(msg, sizeof(msg),
                  "small perturbation amplified above input (worst %.2fx)", worst_amp);
    check(amp_above_one, msg);
    std::snprintf(msg, sizeof(msg),
                  "amplification decreases as perturbation grows (%.2fx -> %.2fx)",
                  first_amp, last_amp);
    check(amp_monotone && first_amp > last_amp, msg);

    const Sensitivity tiny = measure(16u, 5120u, 0.0005f, 99u);
    std::printf("tiny eps=0.0005: input_rel=%.4e dequant_rel=%.4e codes_changed=%.4f\n",
                tiny.input_rel, tiny.dequant_rel, tiny.codes_changed_ratio);
    check(tiny.dequant_rel < 0.05, "tiny perturbation stays within e4m3 step scale");

    std::printf("Results: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
