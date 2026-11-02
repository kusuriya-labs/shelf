#include <shelf/io/safetensors_reader.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/core/memory/types.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/quantization/manifest/runtime_resolution.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/weights/weight_loader.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace shelf;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const std::string& msg) {
    if (cond) {
        ++passed;
        std::printf("PASS: %s\n", msg.c_str());
    } else {
        ++failed;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

static void check_eq_u64(uint64_t got, uint64_t want, const std::string& msg) {
    check(got == want, msg + " got=" + std::to_string(got) + " want=" + std::to_string(want));
}

static void write_text(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << text;
}

static std::vector<uint8_t> pattern_bytes(uint64_t n, uint32_t seed) {
    std::vector<uint8_t> out(n);
    for (uint64_t i = 0; i < n; ++i) {
        out[i] = static_cast<uint8_t>((seed + i * 7u) & 0xFFu);
    }
    return out;
}

static std::vector<uint8_t> bf16_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> out(values.size() * 2u);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const uint16_t h = shelf::f32_to_bf16_rne(values[i]);
        std::memcpy(out.data() + i * 2, &h, 2);
    }
    return out;
}

struct Planned {
    std::string name;
    io::SType dtype;
    std::vector<std::size_t> shape;
    std::vector<uint8_t> bytes;
};

static bool write_shard(const std::string& path, const std::vector<Planned>& tensors) {
    auto w = io::SafetensorsWriter::create(path);
    if (!w.ok()) {
        check(false, "writer create " + path);
        return false;
    }
    io::SafetensorsWriter writer = w.release();
    for (const auto& t : tensors) {
        if (!writer.plan_tensor(t.name, t.dtype, t.shape).ok()) {
            check(false, "plan " + t.name);
            return false;
        }
    }
    if (!writer.write_header().ok()) {
        check(false, "write_header " + path);
        return false;
    }
    for (const auto& t : tensors) {
        if (!writer.write_tensor(t.name, t.bytes.data(), t.bytes.size()).ok()) {
            check(false, "write " + t.name);
            return false;
        }
    }
    if (!writer.finish().ok()) {
        check(false, "finish " + path);
        return false;
    }
    return true;
}

static void write_index(const std::string& dir,
                        const std::vector<std::pair<std::string, std::string>>& weight_map) {
    std::string j = "{\"weight_map\":{";
    bool first = true;
    for (const auto& kv : weight_map) {
        if (!first) j += ",";
        first = false;
        j += "\"" + kv.first + "\":\"" + kv.second + "\"";
    }
    j += "},\"metadata\":{\"total_size\":0}}";
    write_text(dir + "/model.safetensors.index.json", j);
}

static std::string fp8_config_json() {
    return std::string("{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,")
         + "\"text_config\":{\"model_type\":\"qwen3_5_text\",\"num_hidden_layers\":2},"
         + "\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\","
         + "\"activation_scheme\":\"dynamic\",\"weight_block_size\":[128,128]}}";
}

static std::string plain_config_json() {
    return std::string("{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,")
         + "\"text_config\":{\"model_type\":\"qwen3_5_text\",\"num_hidden_layers\":2}}";
}

static fs::path g_root;

static fs::path make_case_dir(const std::string& name) {
    const fs::path dir = g_root / name;
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir;
}

static const std::string kGate = "layers.0.mlp.gate_proj.weight";
static const std::string kUp = "layers.1.mlp.up_proj.weight";
static const std::string kLayernorm = "layers.0.input_layernorm.weight";
static const std::string kScaleInvGate = kGate + "_scale_inv";
static const std::string kScaleInvUp = kUp + "_scale_inv";
static const std::string kBareLmHead = "lm_head.weight";
static const std::string kPrefixedLmHead = "model.language_model.lm_head.weight";

static std::vector<uint8_t> g_gate_codes;
static std::vector<uint8_t> g_up_codes;
static std::vector<float> g_gate_scales = {1.0f, 2.0f, 3.0f, 4.0f};
static std::vector<float> g_up_scales = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};

static bool build_fp8_dir(const fs::path& dir, bool pad_unaligned) {
    const uint64_t gate_k = pad_unaligned ? 200u : 256u;
    g_gate_codes = pattern_bytes(130u * gate_k, 11u);
    g_up_codes = pattern_bytes(256u * 512u, 23u);
    const std::vector<uint8_t> gate_scales = bf16_bytes(g_gate_scales);
    const std::vector<uint8_t> up_scales = bf16_bytes(g_up_scales);
    const std::vector<uint8_t> layernorm = bf16_bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const std::vector<uint8_t> lm_head = pattern_bytes(4u * 8u * 2u, 31u);

    const std::string shard_a = "model-00001-of-00002.safetensors";
    const std::string shard_b = "model-00002-of-00002.safetensors";

    std::vector<Planned> a;
    a.push_back({kGate, io::SType::F8_E4M3, {130, static_cast<std::size_t>(gate_k)}, g_gate_codes});
    a.push_back({kScaleInvGate, io::SType::BF16, {2, 2}, gate_scales});
    a.push_back({kLayernorm, io::SType::BF16, {8}, layernorm});
    a.push_back({kBareLmHead, io::SType::BF16, {4, 8}, lm_head});

    std::vector<Planned> b;
    b.push_back({kUp, io::SType::F8_E4M3, {256, 512}, g_up_codes});
    b.push_back({kScaleInvUp, io::SType::BF16, {2, 4}, up_scales});

    if (!write_shard((dir / shard_a).string(), a)) return false;
    if (!write_shard((dir / shard_b).string(), b)) return false;
    write_index(dir.string(), {
        {kGate, shard_a},
        {kScaleInvGate, shard_a},
        {kLayernorm, shard_a},
        {kBareLmHead, shard_a},
        {kUp, shard_b},
        {kScaleInvUp, shard_b},
    });
    write_text((dir / "config.json").string(), fp8_config_json());
    return true;
}

static void test_fp8_manifest() {
    const fs::path dir = make_case_dir("fp8");
    if (!build_fp8_dir(dir, false)) return;

    check(weights::is_quantized_model_dir(dir.string()), "fp8 dir is quantized");

    auto method = quantization::bundle::detect_hf_quant_method(dir.string());
    check(method.ok() && method.value() == quantization::bundle::HfQuantMethod::Fp8,
          "fp8 method detected");

    const std::vector<std::string> names = {
        kGate, kUp, kLayernorm, kPrefixedLmHead, "layers.9.missing.weight"};
    auto source_result = quantization::bundle::synthesize_hf_quantized_source(dir.string(), names, true);
    if (!source_result.ok()) {
        check(false, "fp8 synthesize: " + source_result.status().message());
        return;
    }
    quantization::bundle::HfQuantizedSource source = source_result.release();
    const quantization::manifest::QuantizedManifest& manifest = source.manifest;

    check(manifest.architecture == "qwen35_dense", "fp8 manifest architecture");
    check_eq_u64(manifest.tensors.size(), 4, "fp8 manifest tensor count");
    check(manifest.aliases.at(kPrefixedLmHead) == kBareLmHead, "lm_head alias registered");
    check(manifest.tensors.count(kBareLmHead) == 1u, "lm_head canonical tensor present");
    check(manifest.tensors.count("layers.9.missing.weight") == 0u, "absent name skipped");

    check(quantization::manifest::validate_quantized_manifest_contract(manifest).ok(),
          "fp8 manifest contract");

    const auto& gate = manifest.tensors.at(kGate);
    check(gate.encoding == quantization::manifest::QuantizedEncoding::Fp8, "gate encoding fp8");
    check(gate.logical_shape == std::vector<int64_t>({130, 256}), "gate logical shape");
    check_eq_u64(gate.k_padded, 256, "gate k_padded");
    check(gate.codes && gate.codes->dtype == "U8", "gate codes dtype U8");
    check(gate.metadata1 && gate.metadata1->dtype == "F32", "gate scale dtype F32");

    auto gate_spec = quantization::manifest::resolve_quant_spec(manifest, gate);
    check(gate_spec.ok() &&
              gate_spec.value().spec_id == quantization::QuantSpecId::FP8_FINEGRAINED_F32_V1,
          "gate resolve_quant_spec fp8");
    const auto& layernorm = manifest.tensors.at(kLayernorm);
    auto layernorm_spec = quantization::manifest::resolve_quant_spec(manifest, layernorm);
    check(layernorm_spec.ok() &&
              layernorm_spec.value().spec_id == quantization::QuantSpecId::BF16_V1,
          "layernorm resolve_quant_spec bf16");

    auto reader_result = quantization::bundle::QuantizedModelReader::open_hf_source(
        std::move(source), true);
    if (!reader_result.ok()) {
        check(false, "fp8 open_hf_source: " + reader_result.status().message());
        return;
    }
    quantization::bundle::QuantizedModelReader reader = reader_result.release();

    auto logical = reader.list_logical_tensors();
    check(logical.ok() && logical.value().size() == 4, "reader logical tensor list");

    auto gate_view_result = reader.resolve(kGate);
    if (!gate_view_result.ok()) {
        check(false, "resolve gate");
        return;
    }
    const auto& gate_view = gate_view_result.value();
    check_eq_u64(gate_view.codes.size, 130u * 256u, "gate codes bytes == m*k_padded");
    check_eq_u64(gate_view.metadata1.size, 2u * 2u * 4u,
                 "gate scale bytes == ((m+127)/128)*(k_padded/128)*4");
    check(std::memcmp(gate_view.codes.data, g_gate_codes.data(), g_gate_codes.size()) == 0,
          "gate codes bytes match file");
    const auto* staged = static_cast<const float*>(gate_view.metadata1.data);
    bool staged_ok = true;
    for (std::size_t i = 0; i < g_gate_scales.size(); ++i) {
        staged_ok = staged_ok && staged[i] == g_gate_scales[i];
    }
    check(staged_ok, "fp8 scale staging widens BF16 to F32");

    auto up_view_result = reader.resolve(kUp);
    if (!up_view_result.ok()) {
        check(false, "resolve up");
        return;
    }
    check_eq_u64(up_view_result.value().codes.size, 256u * 512u, "up codes bytes");
    check_eq_u64(up_view_result.value().metadata1.size, 2u * 4u * 4u, "up scale bytes");

    auto alias_view_result = reader.resolve(kPrefixedLmHead);
    check(alias_view_result.ok() && alias_view_result.value().name == kBareLmHead,
          "lm_head alias resolve");
    if (alias_view_result.ok()) {
        check_eq_u64(alias_view_result.value().data.size, 4u * 8u * 2u, "lm_head data bytes");
    }

    auto norm_view_result = reader.resolve(kLayernorm);
    check(norm_view_result.ok() && norm_view_result.value().data.size == 8u * 2u,
          "layernorm data bytes");
}

static void test_fp8_unaligned_k() {
    const fs::path dir = make_case_dir("fp8_unaligned");
    if (!build_fp8_dir(dir, true)) return;

    auto source_result = quantization::bundle::synthesize_hf_quantized_source(
        dir.string(), {kGate}, false);
    check(!source_result.ok(), "unaligned fp8 K rejected");
    if (!source_result.ok()) {
        check(source_result.status().code() == Status::Code::unsupported,
              "unaligned fp8 K rejected as unsupported");
    }
}

static void test_unsupported_configs() {
    const fs::path fp8_fmt_dir = make_case_dir("fp8_bad_fmt");
    write_text((fp8_fmt_dir / "config.json").string(),
               std::string("{\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e5m2\",")
                   + "\"weight_block_size\":[128,128]}}");
    auto fmt_method = quantization::bundle::detect_hf_quant_method(fp8_fmt_dir.string());
    check(!fmt_method.ok() && fmt_method.status().code() == Status::Code::unsupported,
          "fp8 fmt e5m2 unsupported");
    check(weights::is_quantized_model_dir(fp8_fmt_dir.string()),
          "unsupported fp8 fmt still counts as quantized dir");

    const fs::path fp8_dir = make_case_dir("fp8_bad_block");
    write_text((fp8_dir / "config.json").string(),
               std::string("{\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",")
                   + "\"weight_block_size\":[64,64]}}");
    auto fp8_method = quantization::bundle::detect_hf_quant_method(fp8_dir.string());
    check(!fp8_method.ok() && fp8_method.status().code() == Status::Code::unsupported,
          "fp8 block 64x64 unsupported");

    const fs::path bf16_dir = make_case_dir("bf16_only");
    write_text((bf16_dir / "config.json").string(), plain_config_json());
    check(!weights::is_quantized_model_dir(bf16_dir.string()), "bf16 dir is not quantized");
    check(!quantization::bundle::is_hf_quantized_model_dir(bf16_dir.string()),
          "bf16 dir is not an HF quantized dir");
    auto none = quantization::bundle::synthesize_hf_quantized_source(bf16_dir.string(), {kGate}, false);
    check(!none.ok(), "bf16 dir synthesis rejected");
}

int main() {
    std::error_code ec;
    g_root = fs::temp_directory_path(ec) /
        ("shelf_hf_manifest_" + std::to_string(::getpid()));
    fs::create_directories(g_root, ec);

    test_fp8_manifest();
    test_fp8_unaligned_k();
    test_unsupported_configs();

    fs::remove_all(g_root, ec);
    std::printf("test_hf_manifest: %s (passed=%d failed=%d)\n",
                failed == 0 ? "PASS" : "FAIL", passed, failed);
    return failed == 0 ? 0 : 1;
}
