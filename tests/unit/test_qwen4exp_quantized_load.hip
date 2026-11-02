#include <phaseshift/models/qwen4exp/model/qwen4exp_config.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_weights.h>
#include <phaseshift/quantization/bundle/psq5_manifest.h>
#include <phaseshift/quantization/bundle/quantized_model_reader.h>
#include <phaseshift/weights/weight_loader.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_fail = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

std::uint64_t device_used_bytes() {
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return 0;
    return static_cast<std::uint64_t>(total_b - free_b);
}

}

int main(int argc, char** argv) {
    const char* model_dir = argc > 1
        ? argv[1]
        : "/opt/zen/work/models/Qwen3.8-Flash-Next-Quark-AWQ-PSQ";

    std::printf("test_qwen4exp_quantized_load\n");
    std::printf("model: %s\n", model_dir);

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    auto config_result = ps::qwen4exp::load_qwen4exp_config(model_dir);
    if (!config_result.ok()) {
        std::printf("FAIL: load config: %s\n", config_result.status().message().c_str());
        return 1;
    }
    const auto& config = config_result.value();
    std::printf("layers=%zu experts=%zu ple_layer=%zu is_qsa[3]=%d\n",
                config.num_hidden_layers, config.num_experts, config.ple_layer_index,
                static_cast<int>(config.layer_is_qsa(3)));

    check(config.num_hidden_layers == 48, "48 layers");
    check(config.num_experts == 512, "512 experts");

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::uint64_t baseline = device_used_bytes();
    std::printf("device used before load: %.3f GiB\n",
                static_cast<double>(baseline) / (1ull << 30));

    // VRAM 実測のため expert は除外して全層の非 expert 重みのみ読み込む。
    const std::uint64_t before = device_used_bytes();
    {
        auto arena_result = ps::gpu::GpuArena::create(0, 14ull << 30);
        if (!arena_result.ok()) {
            std::printf("FAIL: arena: %s\n", arena_result.status().message().c_str());
            return 1;
        }
        ps::gpu::GpuArena arena = arena_result.release();

        ps::qwen4exp::Qwen4ExpLoadOptions options;
        options.load_expert_weights = false;
        auto loaded = ps::qwen4exp::load_qwen4exp_weights_quantized(model_dir, arena, stream,
                                                                    options);
        if (!loaded.ok()) {
            std::printf("FAIL: load_qwen4exp_weights_quantized: %s\n",
                        loaded.status().message().c_str());
            return 1;
        }
        ps::qwen4exp::Qwen4ExpModelWeights w = loaded.release();

        check(w.layers.size() == config.num_hidden_layers, "all layers loaded");
        check(w.embed_tokens.rows > 0, "embed_tokens loaded");
        check(w.lm_head.rows > 0, "lm_head loaded");

        const auto& l0 = w.layers[0];
        check(l0.gdn.qkv.rows > 0, "layer0 GDN qkv loaded");
        check(l0.gdn.qkv.encoding == ps::weights::MatrixEncoding::Fp8,
              "layer0 GDN qkv is Fp8");
        check(l0.gdn.qkv.weight_scale_group == 128u, "layer0 GDN qkv scale group 128");
        check(l0.gdn.qkv.k_padded % 128u == 0u, "layer0 GDN qkv k_padded % 128");
        check(l0.gdn.out.rows > 0, "layer0 GDN out_proj loaded");
        check(l0.gdn.out.encoding == ps::weights::MatrixEncoding::Fp8,
              "layer0 GDN out_proj is Fp8");
        check(l0.gdn.b.encoding == ps::weights::MatrixEncoding::Bf16,
              "layer0 GDN in_proj_b is BF16");
        check(l0.moe.router.rows > 0, "layer0 router loaded");
        check(l0.moe.shared.gate.rows > 0, "layer0 shared expert gate loaded");
        check(l0.moe.expert_count > 0, "layer0 expert_count set");
        check(l0.moe.expert_gate_up.empty(), "layer0 experts skipped when disabled");

        const auto& l3 = w.layers[3];
        check(!l3.is_gdn, "layer3 is QSA");
        check(l3.qsa.q.rows > 0, "layer3 QSA q loaded");
        check(l3.qsa.q.encoding == ps::weights::MatrixEncoding::Bf16, "layer3 QSA q is BF16");
        check(l3.qsa.indexer.present, "layer3 indexer present");
        check(l3.qsa.indexer.index_qk.rows > 0, "layer3 indexer qk loaded");

        const auto& lp = w.layers[config.ple_layer_index];
        check(lp.ple.present, "PLE present");
        check(lp.ple.key_proj.rows > 0, "PLE key_proj loaded");
        check(!lp.ple.offsets.empty(), "PLE ngram offsets loaded (I64 passthrough)");

        std::printf("arena used (weights)   : %10.3f GiB\n",
                    static_cast<double>(arena.used()) / (1ull << 30));
        std::printf("arena committed        : %10.3f GiB\n",
                    static_cast<double>(arena.committed()) / (1ull << 30));
        std::printf("non-expert / layer avg : %10.3f GiB\n",
                    static_cast<double>(arena.used()) /
                        (static_cast<double>(config.num_hidden_layers) + 1.0) /
                        static_cast<double>(1ull << 30));
        std::uint64_t after = device_used_bytes();
        std::printf("device used after load : %10.3f GiB (delta %.3f GiB)\n",
                    static_cast<double>(after) / (1ull << 30),
                    static_cast<double>(after - before) / (1ull << 30));
    }

    // expert PSQ5 axis-slice の直接検証（reader のみ、非expertは再ロードしない）
    {
        auto arena_result = ps::gpu::GpuArena::create(0, 512ull << 20);
        if (!arena_result.ok()) {
            std::printf("FAIL: slice arena: %s\n",
                        arena_result.status().message().c_str());
            ++g_fail;
        } else {
            ps::gpu::GpuArena arena = arena_result.release();
            const std::string gu = "model.language_model.layers.0.mlp.experts.gate_up_proj";
            const std::uint64_t per_expert = 655360ull / config.num_experts;
            auto src = ps::quantization::bundle::synthesize_psq5_quantized_source(
                model_dir, {gu}, false, "qwen4exp");
            if (!src.ok()) {
                std::printf("FAIL: slice source: %s\n", src.status().message().c_str());
                ++g_fail;
            } else {
                auto rd = ps::quantization::bundle::QuantizedModelReader::open_hf_source(
                    src.release(), false);
                if (!rd.ok()) {
                    std::printf("FAIL: slice reader: %s\n", rd.status().message().c_str());
                    ++g_fail;
                } else {
                    auto slice = ps::weights::load_quantized_psq5_matrix_axis_slice(
                        rd.value(), gu, per_expert, per_expert, arena, stream, {});
                    if (!slice.ok()) {
                        std::printf("FAIL: expert slice: %s\n",
                                    slice.status().message().c_str());
                        ++g_fail;
                    } else {
                        const auto& sw = slice.value();
                        check(sw.encoding == ps::weights::MatrixEncoding::Psq5,
                              "expert slice encoding is Psq5");
                        check(sw.rows == per_expert, "expert slice rows");
                        check(sw.cols == 2560, "expert slice cols");
                        check(sw.k_padded == 2560, "expert slice k_padded");
                        check(sw.preshuffled, "expert slice preshuffled");
                        check(sw.codes.ndim() != 0 && sw.sign.ndim() != 0 &&
                                  sw.scales.ndim() != 0,
                              "expert slice carries magnitude/sign/scale on device");
                        std::printf("expert slice rows=%u cols=%u ok\n", sw.rows, sw.cols);
                    }
                }
            }
        }
    }

    if (g_fail != 0) {
        std::printf("=== test_qwen4exp_quantized_load: %d FAIL ===\n", g_fail);
        return 1;
    }
    std::printf("=== test_qwen4exp_quantized_load: PASS ===\n");
    return 0;
}
