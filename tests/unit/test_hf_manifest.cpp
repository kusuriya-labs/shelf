#include <shelf/io/safetensors_reader.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/core/memory/types.h>
#include <shelf/quantization/exl3/exl3.h>
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

static std::string quark_config_json(int64_t group_size = 32,
                                     const char* scale_format = "e8m0",
                                     const char* weight_dtype = "fp4",
                                     const char* qscheme = "per_group",
                                     const char* weight_format = "real_quantized") {
    std::string json = std::string("{\"model_type\":\"qwen3_5_moe\",\"tie_word_embeddings\":false,")
        + "\"text_config\":{\"model_type\":\"qwen3_5_moe_text\",\"num_hidden_layers\":2,"
        + "\"num_experts\":256,\"moe_intermediate_size\":512},"
        + "\"quantization_config\":{\"quant_method\":\"quark\",\"version\":\"0.13\","
        + "\"global_quant_config\":{\"weight\":{"
        + "\"dtype\":\"" + weight_dtype + "\",\"qscheme\":\"" + qscheme + "\","
        + "\"group_size\":" + std::to_string(group_size) + ",\"is_dynamic\":false,"
        + "\"scale_format\":\"" + scale_format + "\",\"ch_axis\":-1},"
        + "\"input_tensors\":{\"dtype\":\"fp4\",\"qscheme\":\"per_group\",\"group_size\":32,"
        + "\"is_dynamic\":true,\"scale_format\":\"e8m0\"}},"
        + "\"export\":{\"weight_format\":\"" + weight_format + "\",\"pack_method\":\"reorder\"}}}";
    return json;
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

static const std::string kMxGate = "layers.0.mlp.gate_proj.weight";
static const std::string kMxGateScale = kMxGate + "_scale";
static const std::string kMxUp = "layers.1.mlp.up_proj.weight";
static const std::string kMxUpScale = kMxUp + "_scale";

static std::vector<uint8_t> g_mx_gate_codes;
static std::vector<uint8_t> g_mx_gate_scales;
static std::vector<uint8_t> g_mx_up_codes;
static std::vector<uint8_t> g_mx_up_scales;

// Shapes mirror the measured Quark checkpoints: codes are [N, K/2] U8 and
// scales are [N, K/32] U8 E8M0, with the tensor name suffix `.weight_scale`.
static bool build_mxfp4_dir(const fs::path& dir,
                            std::size_t gate_packed_k = 128u,
                            std::size_t gate_scale_cols = 8u,
                            bool gate_codes_bf16 = false,
                            uint8_t scale_fill = 119u,
                            bool with_scale_tensor = true) {
    const uint64_t gate_n = 130u;
    const uint64_t up_n = 256u;
    g_mx_gate_codes = pattern_bytes(gate_n * gate_packed_k, 5u);
    g_mx_gate_scales = std::vector<uint8_t>(gate_n * gate_scale_cols, scale_fill);
    g_mx_up_codes = pattern_bytes(up_n * 256u, 13u);
    g_mx_up_scales = std::vector<uint8_t>(up_n * 16u, 123u);
    const std::vector<uint8_t> layernorm =
        bf16_bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const std::vector<uint8_t> lm_head = pattern_bytes(4u * 8u * 2u, 37u);

    std::vector<Planned> tensors;
    if (gate_codes_bf16) {
        // Same byte count, wrong dtype: the payload must be rejected because the
        // codes stream is not U8.
        tensors.push_back({kMxGate, io::SType::BF16, {gate_n, gate_packed_k / 2u}, g_mx_gate_codes});
    } else {
        tensors.push_back({kMxGate, io::SType::U8, {gate_n, gate_packed_k}, g_mx_gate_codes});
    }
    if (with_scale_tensor)
        tensors.push_back({kMxGateScale, io::SType::U8, {gate_n, gate_scale_cols}, g_mx_gate_scales});
    tensors.push_back({kMxUp, io::SType::U8, {up_n, 256u}, g_mx_up_codes});
    tensors.push_back({kMxUpScale, io::SType::U8, {up_n, 16u}, g_mx_up_scales});
    tensors.push_back({kLayernorm, io::SType::BF16, {8}, layernorm});
    tensors.push_back({kBareLmHead, io::SType::BF16, {4, 8}, lm_head});

    if (!write_shard((dir / "model.safetensors").string(), tensors)) return false;
    write_text((dir / "config.json").string(), quark_config_json());
    return true;
}

static void test_mxfp4_manifest() {
    const fs::path dir = make_case_dir("mxfp4");
    if (!build_mxfp4_dir(dir)) return;

    check(weights::is_quantized_model_dir(dir.string()), "mxfp4 dir is quantized");
    auto method = quantization::bundle::detect_hf_quant_method(dir.string());
    check(method.ok() &&
              method.value() == quantization::bundle::HfQuantMethod::QuarkMxfp4,
          "quark mxfp4 method detected");

    const std::vector<std::string> names = {
        kMxGate, kMxUp, kLayernorm, kPrefixedLmHead, "layers.9.missing.weight"};
    auto source_result = quantization::bundle::synthesize_hf_quantized_source(
        dir.string(), names, true);
    if (!source_result.ok()) {
        check(false, "mxfp4 synthesize: " + source_result.status().message());
        return;
    }
    quantization::bundle::HfQuantizedSource source = source_result.release();
    const quantization::manifest::QuantizedManifest& manifest = source.manifest;

    check_eq_u64(manifest.tensors.size(), 4, "mxfp4 manifest tensor count");
    check(quantization::manifest::validate_quantized_manifest_contract(manifest).ok(),
          "mxfp4 manifest contract");

    const auto& gate = manifest.tensors.at(kMxGate);
    check(gate.encoding == quantization::manifest::QuantizedEncoding::Mxfp4, "gate encoding mxfp4");
    check(gate.logical_shape == std::vector<int64_t>({130, 256}),
          "mxfp4 logical K is the unpacked width");
    check_eq_u64(gate.k_padded, 256, "mxfp4 k_padded");
    check_eq_u64(gate.scale_block_n, 1, "mxfp4 scale block n is per row");
    check_eq_u64(gate.scale_block_k, 32, "mxfp4 scale block k is 32");
    check(gate.codes && gate.codes->dtype == "U8", "mxfp4 codes dtype U8");
    check(gate.metadata1 && gate.metadata1->dtype == "E8M0", "mxfp4 scale dtype E8M0");
    check(!gate.data.has_value(), "mxfp4 tensor carries no data stream");

    auto gate_spec = quantization::manifest::resolve_quant_spec(manifest, gate);
    check(gate_spec.ok() &&
              gate_spec.value().spec_id == quantization::QuantSpecId::MXFP4_E8M0_G32_V1,
          "gate resolve_quant_spec mxfp4");
    if (gate_spec.ok()) {
        const auto& descriptor = gate_spec.value().descriptor;
        check_eq_u64(descriptor.group_size, 32, "mxfp4 descriptor group size");
        check(descriptor.scale_type == quantization::StorageDType::U8, "mxfp4 scale storage U8");
        check(descriptor.weight_layout == quantization::QuantLayout::MXFP4_E2M1_BLOCK32_V1,
              "mxfp4 weight layout");
    }

    const auto& up = manifest.tensors.at(kMxUp);
    check_eq_u64(up.k_padded, 512, "mxfp4 up k_padded is the unpacked width");

    auto reader_result = quantization::bundle::QuantizedModelReader::open_hf_source(
        std::move(source), true);
    if (!reader_result.ok()) {
        check(false, "mxfp4 open_hf_source: " + reader_result.status().message());
        return;
    }
    quantization::bundle::QuantizedModelReader reader = reader_result.release();

    auto gate_view_result = reader.resolve(kMxGate);
    if (!gate_view_result.ok()) {
        check(false, "resolve mxfp4 gate");
        return;
    }
    const auto& gate_view = gate_view_result.value();
    check_eq_u64(gate_view.codes.size, 130u * 128u, "mxfp4 codes bytes == m * K/2");
    check_eq_u64(gate_view.metadata1.size, 130u * 8u, "mxfp4 scale bytes == m * K/32");
    check_eq_u64(static_cast<uint64_t>(gate_view.k_padded), 256, "mxfp4 view k_padded");
    check_eq_u64(gate_view.scale_block_k, 32, "mxfp4 view scale block k");
    check(std::memcmp(gate_view.codes.data, g_mx_gate_codes.data(), g_mx_gate_codes.size()) == 0,
          "mxfp4 codes bytes match file");
    check(std::memcmp(gate_view.metadata1.data, g_mx_gate_scales.data(), g_mx_gate_scales.size()) == 0,
          "mxfp4 scale bytes match file without staging");
    check(reader.resolve(kMxUp).ok(), "mxfp4 up resolves");
}

static void test_mxfp4_rejections() {
    const std::vector<std::string> names = {kMxGate};

    const fs::path group_dir = make_case_dir("mxfp4_group128");
    write_text((group_dir / "config.json").string(), quark_config_json(128));
    auto group_method = quantization::bundle::detect_hf_quant_method(group_dir.string());
    check(!group_method.ok() && group_method.status().code() == Status::Code::unsupported,
          "quark group_size 128 unsupported");

    const fs::path scale_fmt_dir = make_case_dir("mxfp4_scale_e4m3");
    write_text((scale_fmt_dir / "config.json").string(), quark_config_json(32, "e4m3"));
    check(!quantization::bundle::detect_hf_quant_method(scale_fmt_dir.string()).ok(),
          "quark scale_format e4m3 unsupported");

    const fs::path dtype_dir = make_case_dir("mxfp4_dtype_fp8");
    write_text((dtype_dir / "config.json").string(), quark_config_json(32, "e8m0", "fp8"));
    check(!quantization::bundle::detect_hf_quant_method(dtype_dir.string()).ok(),
          "quark weight dtype fp8 unsupported");

    const fs::path qscheme_dir = make_case_dir("mxfp4_per_tensor");
    write_text((qscheme_dir / "config.json").string(),
               quark_config_json(32, "e8m0", "fp4", "per_tensor"));
    check(!quantization::bundle::detect_hf_quant_method(qscheme_dir.string()).ok(),
          "quark qscheme per_tensor unsupported");

    const fs::path format_dir = make_case_dir("mxfp4_fake_quant");
    write_text((format_dir / "config.json").string(),
               quark_config_json(32, "e8m0", "fp4", "per_group", "fake_quant"));
    check(!quantization::bundle::detect_hf_quant_method(format_dir.string()).ok(),
          "quark weight_format fake_quant unsupported");

    const fs::path no_global_dir = make_case_dir("mxfp4_no_global_config");
    write_text((no_global_dir / "config.json").string(),
               std::string("{\"quantization_config\":{\"quant_method\":\"quark\"}}"));
    check(!quantization::bundle::detect_hf_quant_method(no_global_dir.string()).ok(),
          "quark without global_quant_config unsupported");

    const fs::path unaligned_dir = make_case_dir("mxfp4_unaligned_k");
    if (build_mxfp4_dir(unaligned_dir, 100u, 8u)) {
        auto result = quantization::bundle::synthesize_hf_quantized_source(
            unaligned_dir.string(), names, false);
        check(!result.ok() && result.status().code() == Status::Code::unsupported,
              "mxfp4 K that is not a multiple of 32 rejected");
    }

    const fs::path scale_shape_dir = make_case_dir("mxfp4_scale_shape");
    if (build_mxfp4_dir(scale_shape_dir, 128u, 4u)) {
        auto result = quantization::bundle::synthesize_hf_quantized_source(
            scale_shape_dir.string(), names, false);
        check(!result.ok() && result.status().code() == Status::Code::unsupported,
              "mxfp4 scale grid mismatch rejected");
    }

    const fs::path codes_dtype_dir = make_case_dir("mxfp4_codes_bf16");
    if (build_mxfp4_dir(codes_dtype_dir, 128u, 8u, true)) {
        auto result = quantization::bundle::synthesize_hf_quantized_source(
            codes_dtype_dir.string(), names, false);
        check(!result.ok() && result.status().code() == Status::Code::unsupported,
              "mxfp4 codes stored as BF16 rejected");
    }

    const fs::path fp8_scale_dir = make_case_dir("fp8_with_weight_scale");
    {
        std::vector<Planned> tensors;
        std::vector<uint8_t> codes = pattern_bytes(128u * 256u, 3u);
        std::vector<uint8_t> scales(128u * 8u, 119u);
        tensors.push_back({kMxGate, io::SType::F8_E4M3, {128, 256}, codes});
        tensors.push_back({kMxGateScale, io::SType::U8, {128, 8}, scales});
        write_shard((fp8_scale_dir / "model.safetensors").string(), tensors);
        write_text((fp8_scale_dir / "config.json").string(), fp8_config_json());
    }
    auto fp8_scale_result = quantization::bundle::synthesize_hf_quantized_source(
        fp8_scale_dir.string(), names, false);
    check(!fp8_scale_result.ok() &&
              fp8_scale_result.status().code() == Status::Code::unsupported,
          "weight_scale in an fp8-method model rejected");
}

static std::string exl3_config_json() {
    return std::string("{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,")
         + "\"text_config\":{\"model_type\":\"qwen3_5_text\",\"num_hidden_layers\":2},"
         + "\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4.0,"
         + "\"head_bits\":6,\"codebook\":\"mul1\"}}";
}

static bool build_exl3_dir(const fs::path& dir) {
    const std::string shard = "model.safetensors";
    const std::string p0 = "layers.0.mlp.down_proj";
    const std::string p1 = "layers.1.mlp.down_proj";

    const std::vector<uint8_t> trellis0 = pattern_bytes(8u * 8u * 64u * 2u, 41u);
    const std::vector<uint8_t> trellis1 = pattern_bytes(8u * 8u * 56u * 2u, 43u);
    const std::vector<uint8_t> suh = pattern_bytes(128u * 2u, 53u);
    const std::vector<uint8_t> svh = pattern_bytes(128u * 2u, 59u);
    const std::vector<uint8_t> f16_matrix = pattern_bytes(128u * 128u * 2u, 61u);
    const std::vector<uint8_t> mul1_marker = {0x2Du, 0xD1u, 0xDCu, 0x83u};

    std::vector<Planned> tensors;
    tensors.push_back({p0 + ".trellis", io::SType::I16, {8, 8, 64}, trellis0});
    tensors.push_back({p0 + ".suh", io::SType::F16, {128}, suh});
    tensors.push_back({p0 + ".svh", io::SType::F16, {128}, svh});
    tensors.push_back({p0 + ".mul1", io::SType::I32, {}, mul1_marker});
    tensors.push_back({p1 + ".trellis", io::SType::I16, {8, 8, 56}, trellis1});
    tensors.push_back({p1 + ".suh", io::SType::F16, {128}, suh});
    tensors.push_back({p1 + ".svh", io::SType::F16, {128}, svh});
    tensors.push_back({p1 + ".mul1", io::SType::I32, {}, mul1_marker});
    tensors.push_back(
        {"layers.0.linear_attn.in_proj_a.weight", io::SType::F16, {128, 128}, f16_matrix});

    if (!write_shard((dir / shard).string(), tensors)) return false;
    std::vector<std::pair<std::string, std::string>> map;
    for (const Planned& t : tensors) map.push_back({t.name, shard});
    write_index(dir.string(), map);
    write_text((dir / "config.json").string(), exl3_config_json());
    return true;
}

static void test_exl3_manifest() {
    const fs::path dir = make_case_dir("exl3");
    if (!build_exl3_dir(dir)) return;

    check(weights::is_quantized_model_dir(dir.string()), "exl3 dir is quantized");
    auto method = quantization::bundle::detect_hf_quant_method(dir.string());
    check(method.ok() && method.value() == quantization::bundle::HfQuantMethod::Exl3,
          "exl3 method detected");

    const std::vector<std::string> names = {
        "layers.0.mlp.down_proj.weight", "layers.1.mlp.down_proj.weight",
        "layers.0.linear_attn.in_proj_a.weight", "layers.9.missing.weight"};
    auto source_result =
        quantization::bundle::synthesize_hf_quantized_source(dir.string(), names, true);
    if (!source_result.ok()) {
        check(false, "exl3 synthesize: " + source_result.status().message());
        return;
    }
    quantization::bundle::HfQuantizedSource source = source_result.release();
    const auto& manifest = source.manifest;
    check_eq_u64(manifest.tensors.size(), 3, "exl3 manifest tensor count");
    check(quantization::manifest::validate_quantized_manifest_contract(manifest).ok(),
          "exl3 manifest contract");

    const auto& t0 = manifest.tensors.at("layers.0.mlp.down_proj.weight");
    check(t0.encoding == quantization::manifest::QuantizedEncoding::Exl3, "exl3 encoding");
    check(t0.logical_shape == std::vector<int64_t>({128, 128}), "exl3 logical shape");
    check_eq_u64(t0.k_padded, 128, "exl3 k_padded");
    check_eq_u64(t0.exl3_ka, 4, "exl3 integer bit rate");
    check(!t0.exl3_half, "exl3 integer rate flag");
    check(t0.codes && t0.codes->dtype == "U8", "exl3 trellis ref");
    check(t0.suh && t0.suh->dtype == "FP16", "exl3 suh ref");
    check(t0.svh && t0.svh->dtype == "FP16", "exl3 svh ref");
    check(t0.marker && t0.marker->dtype == "I32", "exl3 marker ref");

    const auto& t1 = manifest.tensors.at("layers.1.mlp.down_proj.weight");
    check_eq_u64(t1.exl3_ka, 3, "exl3 half bit rate integer part");
    check(t1.exl3_half, "exl3 half bit rate flag");

    auto reader_result =
        quantization::bundle::QuantizedModelReader::open_hf_source(std::move(source), true);
    if (!reader_result.ok()) {
        check(false, "exl3 open_hf_source: " + reader_result.status().message());
        return;
    }
    quantization::bundle::QuantizedModelReader reader = reader_result.release();
    auto view = reader.resolve("layers.0.mlp.down_proj.weight");
    if (!view.ok()) {
        check(false, "exl3 resolve: " + view.status().message());
        return;
    }
    check_eq_u64(view.value().codes.size, 8u * 8u * 64u * 2u, "exl3 trellis bytes");
    check_eq_u64(view.value().suh.size, 256u, "exl3 suh bytes");
    check_eq_u64(view.value().svh.size, 256u, "exl3 svh bytes");
    check_eq_u64(view.value().marker.size, 4u, "exl3 marker bytes");
    const auto view1 = reader.resolve("layers.1.mlp.down_proj.weight");
    check(view1.ok() && view1.value().exl3_half && view1.value().exl3_ka == 3,
          "exl3 resolved half rate");

    const auto& loaded = reader.manifest();
    const auto& f16 = loaded.tensors.at("layers.0.linear_attn.in_proj_a.weight");
    check(f16.encoding == quantization::manifest::QuantizedEncoding::Bf16,
          "exl3 f16 tensor becomes a bf16 entry");
    check(f16.data && f16.data->dtype == "BF16", "exl3 f16 tensor data is bf16");
    auto f16_view = reader.resolve("layers.0.linear_attn.in_proj_a.weight");
    check(f16_view.ok(), "exl3 f16 tensor resolves");
    if (f16_view.ok()) {
        check_eq_u64(f16_view.value().data.size, 128u * 128u * 2u,
                     "exl3 f16 tensor staged byte count");
        const std::vector<uint8_t> source = pattern_bytes(128u * 128u * 2u, 61u);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(source.data());
        const uint16_t* got = static_cast<const uint16_t*>(f16_view.value().data.data);
        const uint16_t want = shelf::f32_to_bf16_rne(
            quantization::exl3::f32_from_f16_bits(src[0]));
        check(got[0] == want, "exl3 f16 tensor is widened to bf16");
    }
}

static void test_exl3_mcg_rejected() {
    const fs::path dir = make_case_dir("exl3_mcg");
    const std::vector<uint8_t> trellis = pattern_bytes(8u * 8u * 64u * 2u, 67u);
    const std::vector<uint8_t> suh = pattern_bytes(128u * 2u, 69u);
    const std::vector<uint8_t> svh = pattern_bytes(128u * 2u, 71u);
    const std::vector<uint8_t> mcg = {0xEDu, 0x1Fu, 0xACu, 0xCBu};
    std::vector<Planned> tensors = {
        {"layer.trellis", io::SType::I16, {8, 8, 64}, trellis},
        {"layer.suh", io::SType::F16, {128}, suh},
        {"layer.svh", io::SType::F16, {128}, svh},
        {"layer.mcg", io::SType::I32, {}, mcg}};
    if (!write_shard((dir / "model.safetensors").string(), tensors)) return;
    std::vector<std::pair<std::string, std::string>> map;
    for (const Planned& t : tensors) map.push_back({t.name, "model.safetensors"});
    write_index(dir.string(), map);
    write_text((dir / "config.json").string(), exl3_config_json());

    auto source = quantization::bundle::synthesize_hf_quantized_source(
        dir.string(), {"layer.weight"}, false);
    check(!source.ok() && source.status().code() == Status::Code::unsupported,
          "exl3 mcg checkpoint rejected");
    check(!source.ok() && source.status().message().find("mcg") != std::string::npos,
          "exl3 mcg rejection names the codebook");
}

int main() {
    std::error_code ec;
    g_root = fs::temp_directory_path(ec) /
        ("shelf_hf_manifest_" + std::to_string(::getpid()));
    fs::create_directories(g_root, ec);

    test_fp8_manifest();
    test_fp8_unaligned_k();
    test_mxfp4_manifest();
    test_mxfp4_rejections();
    test_exl3_manifest();
    test_exl3_mcg_rejected();
    test_unsupported_configs();

    fs::remove_all(g_root, ec);
    std::printf("test_hf_manifest: %s (passed=%d failed=%d)\n",
                failed == 0 ? "PASS" : "FAIL", passed, failed);
    return failed == 0 ? 0 : 1;
}
