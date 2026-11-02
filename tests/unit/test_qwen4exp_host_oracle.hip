#include <phaseshift/models/qwen4exp/model/qwen4exp_config.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_weights.h>
#include <phaseshift/quantization/bundle/psq5_manifest.h>
#include <phaseshift/quantization/bundle/quantized_model_reader.h>
#include <phaseshift/quantization/psq5/psq5.h>
#include <phaseshift/weights/weight_loader.h>

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

namespace q4e = ps::qwen4exp;
namespace pq = ps::quantization::psq5;
namespace qb = ps::quantization::bundle;

int g_fail = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

bool to_f32(const void* src, std::size_t bytes, std::vector<float>& out) {
    if (bytes % sizeof(std::uint16_t) != 0) return false;
    const std::size_t n = bytes / sizeof(std::uint16_t);
    const auto* h = static_cast<const std::uint16_t*>(src);
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t sign =
            (static_cast<std::uint32_t>(h[i]) & 0x8000u) << 16u;
        const std::uint32_t exp = (h[i] >> 10u) & 0x1Fu;
        const std::uint32_t man = h[i] & 0x3FFu;
        std::uint32_t bits = 0;
        if (exp == 0u) {
            if (man == 0u) {
                bits = sign;
            } else {
                int e = -124;
                std::uint32_t m = man;
                while ((m & 0x400u) == 0u) {
                    m <<= 1;
                    ++e;
                }
                m &= 0x3FFu;
                bits = sign | (static_cast<std::uint32_t>(e + 127) << 23u) | (m << 13u);
            }
        } else if (exp == 31u) {
            bits = sign | 0x7F800000u | (man << 13u);
        } else {
            bits = sign | ((exp + 112u) << 23u) | (man << 13u);
        }
        std::memcpy(&out[i], &bits, sizeof(float));
    }
    return true;
}

struct Stats {
    double mean_abs = 0.0;
    double max_abs = 0.0;
    std::size_t zeros = 0;
    std::size_t n = 0;
};

Stats summarize(const std::vector<float>& v) {
    Stats st;
    st.n = v.size();
    double sum = 0.0;
    for (float x : v) {
        const double a = std::fabs(static_cast<double>(x));
        sum += a;
        if (a > st.max_abs) st.max_abs = a;
        if (x == 0.0f) ++st.zeros;
    }
    st.mean_abs = st.n > 0 ? sum / static_cast<double>(st.n) : 0.0;
    return st;
}

}

int main(int argc, char** argv) {
    const char* model_dir = argc > 1
        ? argv[1]
        : "/opt/zen/work/models/Qwen3.8-Flash-Next-Quark-AWQ-PSQ";

    std::printf("test_qwen4exp_host_oracle\n");
    std::printf("model: %s\n", model_dir);

    auto cfg_result = q4e::load_qwen4exp_config(model_dir);
    if (!cfg_result.ok()) {
        std::printf("FAIL: config: %s\n", cfg_result.status().message().c_str());
        return 1;
    }
    const auto& cfg = cfg_result.value();

    const std::vector<std::string> names = {
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
        "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
        "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.3.self_attn.q_proj.weight",
    };

    auto source = qb::synthesize_psq5_quantized_source(model_dir, names, false, "qwen4exp");
    check(source.ok(), "synthesize mixed source for host oracle");
    if (!source.ok()) {
        std::printf("  reason: %s\n", source.status().message().c_str());
        return 1;
    }
    auto reader = qb::QuantizedModelReader::open_hf_source(source.release(), false);
    check(reader.ok(), "open quantized reader for host oracle");
    if (!reader.ok()) {
        std::printf("  reason: %s\n", reader.status().message().c_str());
        return 1;
    }
    qb::QuantizedModelReader rd = reader.release();

    for (const std::string& name : names) {
        auto view = rd.resolve(name);
        if (!view.ok()) {
            std::printf("FAIL: resolve %s: %s\n", name.c_str(),
                        view.status().message().c_str());
            ++g_fail;
            continue;
        }
        const auto& v = view.value();

        if (v.encoding == ps::quantization::manifest::QuantizedEncoding::Psq5) {
            const auto rows = static_cast<std::uint64_t>(v.logical_shape[0]);
            const auto cols = static_cast<std::uint64_t>(v.logical_shape[1]);
            const auto kp = static_cast<std::uint64_t>(v.k_padded);
            const std::uint64_t nb = kp / pq::kBlockK;

            std::size_t expect_mag = 0;
            std::size_t expect_sign = 0;
            std::size_t expect_scale = 0;
            const bool sizes_ok =
                v.codes.size == rows * nb * pq::kMagnitudeBytesPerBlock &&
                (expect_mag = v.codes.size, true) &&
                v.metadata1.size == rows * nb * pq::kSignBytesPerBlock &&
                (expect_sign = v.metadata1.size, true) &&
                v.metadata2.size == rows * nb * pq::kScaleBytesPerBlock &&
                (expect_scale = v.metadata2.size, true);
            check(sizes_ok, "psq5 stream sizes match canonical layout");

            pq::Psq5CanonicalView cv;
            cv.n = rows;
            cv.logical_k = cols;
            cv.padded_k = kp;
            cv.magnitude = static_cast<const std::uint8_t*>(v.codes.data);
            cv.magnitude_bytes = v.codes.size;
            cv.sign = static_cast<const std::uint8_t*>(v.metadata1.data);
            cv.sign_bytes = v.metadata1.size;
            cv.scale = static_cast<const std::uint8_t*>(v.metadata2.data);
            cv.scale_bytes = v.metadata2.size;
            check(cv.validate(), "psq5 canonical view validates");

            const std::size_t elems = static_cast<std::size_t>(rows * cols);
            std::vector<float> w(elems);
            pq::dequantize_psq5_canonical(cv, w.data());
            const Stats st = summarize(w);
            std::printf(
                "  PSQ5 %-52s rows=%llu cols=%llu mean_abs=%.6f max_abs=%.6f zero%%=%.2f\n",
                name.substr(name.rfind('.') == std::string::npos ? 0
                                                                  : name.rfind('/') + 1)
                    .c_str(),
                static_cast<unsigned long long>(rows),
                static_cast<unsigned long long>(cols), st.mean_abs, st.max_abs,
                100.0 * static_cast<double>(st.zeros) / static_cast<double>(st.n));
            check(st.n == elems, "psq5 dequant covers the full matrix");
            check(st.max_abs > 0.0, "psq5 dequant is non-trivial");
            check(std::isfinite(st.max_abs), "psq5 dequant is finite");
        } else if (v.encoding ==
                   ps::quantization::manifest::QuantizedEncoding::Fp8) {
            const auto rows = static_cast<std::uint64_t>(v.logical_shape[0]);
            const auto cols = static_cast<std::uint64_t>(v.logical_shape[1]);
            std::vector<float> codes;
            const bool ok = to_f32(v.codes.data, v.codes.size, codes);
            check(ok, "fp8 codes readable as bytes");
            std::vector<float> scale(v.metadata1.size / sizeof(float));
            std::memcpy(scale.data(), v.metadata1.data, v.metadata1.size);
            const Stats sc = summarize(scale);
            std::printf("  FP8  %-52s rows=%llu cols=%llu scales=%zu mean_abs=%.6f\n",
                        name.c_str(), static_cast<unsigned long long>(rows),
                        static_cast<unsigned long long>(cols), scale.size(),
                        sc.mean_abs);
            check(!scale.empty(), "fp8 scale staged as f32");
            check(sc.max_abs > 0.0, "fp8 scale non-trivial");
        } else {
            std::vector<float> w;
            const bool ok = to_f32(v.data.data, v.data.size, w);
            check(ok, "bf16 tensor converts to f32");
            const Stats st = summarize(w);
            std::printf("  BF16 %-52s n=%zu mean_abs=%.6f max_abs=%.6f\n", name.c_str(),
                        st.n, st.mean_abs, st.max_abs);
            check(!w.empty(), "bf16 host tensor non-empty");
        }
    }

    check(cfg.num_hidden_layers == 48, "48 layers in config");

    if (g_fail != 0) {
        std::printf("=== test_qwen4exp_host_oracle: %d FAIL ===\n", g_fail);
        return 1;
    }
    std::printf("=== test_qwen4exp_host_oracle: PASS ===\n");
    return 0;
}
