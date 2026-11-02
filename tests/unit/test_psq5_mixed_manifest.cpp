#include <phaseshift/quantization/bundle/psq5_manifest.h>
#include <phaseshift/quantization/bundle/quantized_model_reader.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

}

int main(int argc, char** argv) {
    const char* model_dir = argc > 1
        ? argv[1]
        : "/opt/zen/work/models/Qwen3.8-Flash-Next-Quark-AWQ-PSQ";
    const char* cb28_dir = argc > 2
        ? argv[2]
        : "/opt/zen/work/models/Qwen3.8-27B-Quark-AWQ-PSQ";

    std::printf("test_psq5_mixed_manifest\n");
    std::printf("mixed: %s\n", model_dir);
    std::printf("cb28 : %s\n", cb28_dir);

    check(ps::quantization::bundle::is_psq5_model_dir(model_dir),
          "mixed manifest accepted by is_psq5_model_dir");
    check(ps::quantization::bundle::is_psq5_model_dir(cb28_dir),
          "PSQ5-CB28 manifest still accepted");

    const std::vector<std::string> names = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.embed_tokens.weight",
    };

    auto source_result = ps::quantization::bundle::synthesize_psq5_quantized_source(
        model_dir, names, false, "qwen4exp");
    if (!source_result.ok()) {
        std::printf("FAIL: synthesize mixed: %s\n",
                    source_result.status().message().c_str());
        ++g_fail;
    } else {
        auto& source = source_result.value();
        check(source.manifest.architecture == "qwen4exp", "architecture preserved");
        check(source.manifest.tensors.size() >= 6,
              "all requested mixed tensors synthesized");

        const auto expect_psq5 = source.manifest.tensors.find(names[0]);
        check(expect_psq5 != source.manifest.tensors.end(), "expert gate_up present");
        if (expect_psq5 != source.manifest.tensors.end()) {
            const auto& m = expect_psq5->second;
            check(m.encoding == ps::quantization::manifest::QuantizedEncoding::Psq5,
                  "expert gate_up encoding is Psq5");
            check(m.logical_shape.size() == 2, "expert gate_up is rank-2 logical");
            if (m.logical_shape.size() == 2) {
                check(m.logical_shape[0] == 655360 && m.logical_shape[1] == 2560,
                      "expert gate_up rows/columns from rows/columns fields");
            }
            check(m.codes.has_value() && m.metadata1.has_value() && m.metadata2.has_value(),
                  "expert gate_up carries magnitude/sign/scale streams");
            check(m.k_padded % 32u == 0, "expert gate_up k_padded aligned");
        }

        const auto expect_down = source.manifest.tensors.find(names[1]);
        check(expect_down != source.manifest.tensors.end(), "expert down present");
        if (expect_down != source.manifest.tensors.end()) {
            check(expect_down->second.logical_shape.size() == 2, "expert down rank-2");
            if (expect_down->second.logical_shape.size() == 2) {
                check(expect_down->second.logical_shape[0] == 1310720 &&
                          expect_down->second.logical_shape[1] == 640,
                      "expert down rows/columns from rows/columns fields");
            }
        }

        const auto expect_f8 = source.manifest.tensors.find(names[2]);
        check(expect_f8 != source.manifest.tensors.end(), "gdn qkv present");
        if (expect_f8 != source.manifest.tensors.end()) {
            const auto& m = expect_f8->second;
            check(m.encoding == ps::quantization::manifest::QuantizedEncoding::Fp8,
                  "gdn qkv encoding is Fp8");
            check(m.scale_block_n == 128u && m.scale_block_k == 128u,
                  "gdn qkv scale block 128x128");
            check(m.logical_shape.size() == 2 && m.logical_shape[1] % 128 == 0,
                  "gdn qkv K multiple of 128");
            check(m.metadata1.has_value() && m.metadata1->dtype == "F32",
                  "gdn qkv scale staged as F32");
            check(m.metadata1->tensor == names[2] + "_scale_inv",
                  "gdn qkv scale ref targets weight_scale_inv");
            check(source.fp32_scale_staging.find(m.metadata1->tensor) !=
                      source.fp32_scale_staging.end(),
                  "gdn qkv bf16->f32 staging materialized");
        }

        const auto expect_bf16 = source.manifest.tensors.find(names[5]);
        check(expect_bf16 != source.manifest.tensors.end(), "embed_tokens passthrough present");
        if (expect_bf16 != source.manifest.tensors.end()) {
            check(expect_bf16->second.encoding ==
                      ps::quantization::manifest::QuantizedEncoding::Bf16,
                  "embed_tokens encoding is Bf16");
        }
    }

    if (source_result.ok()) {
        auto reader_result = ps::quantization::bundle::QuantizedModelReader::open_hf_source(
            std::move(source_result).release(), false);
        if (!reader_result.ok()) {
            std::printf("FAIL: open_hf_source: %s\n",
                        reader_result.status().message().c_str());
            ++g_fail;
        } else {
            ps::quantization::bundle::QuantizedModelReader reader = reader_result.release();
            check(reader.manifest().tensors.size() >= 6, "reader holds synthesized tensors");

            for (const std::string& n : {names[0], names[1], names[2], names[5]}) {
                auto v = reader.resolve(n);
                if (!v.ok()) {
                    std::printf("FAIL: resolve %s: %s\n", n.c_str(),
                                v.status().message().c_str());
                    ++g_fail;
                    continue;
                }
                const auto& view = v.value();
                if (view.encoding == ps::quantization::manifest::QuantizedEncoding::Psq5) {
                    check(view.codes.data != nullptr && view.metadata1.data != nullptr &&
                              view.metadata2.data != nullptr,
                          "psq5 view exposes magnitude/sign/scale pointers");
                    const auto n_rows = static_cast<std::uint64_t>(view.logical_shape[0]);
                    const auto kk = static_cast<std::uint64_t>(view.logical_shape[1]);
                    check(view.codes.size == n_rows * (kk / 32u) * 16u,
                          "psq5 magnitude byte count");
                    check(view.metadata1.size == n_rows * (kk / 32u) * 4u,
                          "psq5 sign byte count");
                    check(view.metadata2.size == n_rows * (kk / 32u),
                          "psq5 scale byte count");
                } else if (view.encoding ==
                           ps::quantization::manifest::QuantizedEncoding::Fp8) {
                    check(view.codes.data != nullptr && view.metadata1.data != nullptr,
                          "fp8 view exposes codes and scales");
                    check(view.scale_block_n == 128u && view.scale_block_k == 128u,
                          "fp8 view scale block 128x128");
                    check(view.metadata1.size % sizeof(float) == 0,
                          "fp8 scale staged in f32");
                } else {
                    check(view.data.data != nullptr, "bf16 view exposes data pointer");
                }
            }
        }
    }

    auto cb28_source = ps::quantization::bundle::synthesize_psq5_quantized_source(
        cb28_dir, {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight"}, false);
    if (!cb28_source.ok()) {
        std::printf("FAIL: synthesize cb28: %s\n",
                    cb28_source.status().message().c_str());
        ++g_fail;
    } else {
        check(cb28_source.value().manifest.architecture == "qwen35_dense",
              "cb28 default architecture unchanged");
        const auto it = cb28_source.value().manifest.tensors.find(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
        check(it != cb28_source.value().manifest.tensors.end(), "cb28 tensor present");
        if (it != cb28_source.value().manifest.tensors.end()) {
            check(it->second.encoding == ps::quantization::manifest::QuantizedEncoding::Psq5,
                  "cb28 tensor stays Psq5");
        }
    }

    if (g_fail == 0) {
        std::printf("=== test_psq5_mixed_manifest: PASS ===\n");
        return 0;
    }
    std::printf("=== test_psq5_mixed_manifest: %d FAIL ===\n", g_fail);
    return 1;
}
