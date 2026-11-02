#include <shelf/quantization/manifest/manifest_v4.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <cstdio>
#include <string>

using namespace shelf;
using namespace shelf::quantization::manifest;

namespace {
int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) { ++g_passed; } else { ++g_failed; std::printf("FAIL: %s\n", msg); }
}

QuantizedTensorMetadata make_bf16(int64_t n, int64_t k) {
    QuantizedTensorMetadata t;
    t.role = "weight";
    t.encoding = QuantizedEncoding::Bf16;
    t.logical_shape = {n, k};
    t.k_padded = static_cast<uint64_t>(k);
    t.data = QuantizedTensorRef{"w", "BF16", "0x00000000"};
    return t;
}

QuantizedTensorMetadata make_quantized(QuantizedEncoding enc, int64_t n, int64_t k) {
    QuantizedTensorMetadata t;
    t.role = "weight";
    t.encoding = enc;
    t.logical_shape = {n, k};
    t.k_padded = static_cast<uint64_t>(k);
    const std::string name = std::string("w_") + std::to_string(static_cast<int>(enc));
    t.codes = QuantizedTensorRef{name + ".__shelf_codes", "U8", "0x11111111"};
    t.metadata1 = QuantizedTensorRef{
        name + ".__shelf_metadata1",
        enc == QuantizedEncoding::Fp8 ? "F32" : "U8",
        "0x22222222"};
    return t;
}

int run() {
    QuantizedManifest m;
    m.architecture = "qwen35_dense";
    m.preset = "test";
    m.source_dtype = "BF16";
    m.logical_parameter_count = 5 * 256 * 256;
    m.aliases["alias.w"] = "w_bf16";
    m.tensors["w_bf16"] = make_bf16(256, 256);
    m.tensors["w_fp8"] = make_quantized(QuantizedEncoding::Fp8, 256, 256);

    Result<std::string> json = serialize_quantized_manifest(m);
    check(json.ok(), "v4 serialize succeeds");
    if (!json.ok()) return 0;
    check(json.value().find("\"format_version\": 4") != std::string::npos,
          "v4 serialize writes version 4");
    check(json.value().find("\"domain\"") != std::string::npos,
          "v4 serialize writes tensor domain");
    check(json.value().find("\"block_shape\"") != std::string::npos,
          "v4 serialize writes scale block_shape");
    check(json.value().find("\"role\": \"scale\"") != std::string::npos,
          "v4 serialize writes payload role");
    check(json.value().find("fp8_e4m3_block128_f32") == std::string::npos,
          "v4 serialize drops v3 encoding names");

    Result<QuantizedManifest> parsed = parse_quantized_manifest(json.value());
    check(parsed.ok(), "v4 roundtrip parse succeeds");
    if (!parsed.ok()) return 0;
    const QuantizedManifest& m2 = parsed.value();
    check(m2.format_version == kQuantizedSafetensorsFormatVersion, "roundtrip keeps version 4");
    check(m2.architecture == "qwen35_dense", "roundtrip keeps architecture");
    check(m2.aliases.at("alias.w") == "w_bf16", "roundtrip keeps aliases");
    check(m2.tensors.at("w_bf16").encoding == QuantizedEncoding::Bf16, "roundtrip bf16");
    check(m2.tensors.at("w_bf16").data->dtype == "BF16", "roundtrip bf16 data dtype");
    const QuantizedTensorMetadata& fp8 = m2.tensors.at("w_fp8");
    check(fp8.encoding == QuantizedEncoding::Fp8, "roundtrip fp8 encoding");
    check(fp8.k_padded == 256, "roundtrip fp8 k_padded");
    check(fp8.codes->tensor == "w_5.__shelf_codes", "roundtrip fp8 codes name");
    check(fp8.codes->dtype == "U8", "roundtrip fp8 codes dtype tag");
    check(fp8.metadata1->dtype == "F32", "roundtrip fp8 scale dtype tag");
    check(fp8.layout == "fp8_e4m3_row_major_block128_v1", "roundtrip fp8 canonical layout");

    const char* v3_json = R"({
  "format": "shelf-fpx-safetensors",
  "format_version": 3,
  "architecture": "qwen35_dense",
  "tensors": {
    "w": {
      "role": "weight",
      "encoding": "fp8_e4m3_block128_f32",
      "layout": "fp8_e4m3_row_major_block128_v1",
      "logical_shape": [256, 256],
      "k_padded": 256,
      "codes": {"tensor": "w.__shelf_codes", "dtype": "U8", "crc32": "0x00000000"},
      "metadata1": {"tensor": "w.__shelf_metadata1", "dtype": "F32", "crc32": "0x00000000"}
    }
  }
})";
    Result<QuantizedManifest> v3 = parse_quantized_manifest(v3_json);
    check(v3.ok(), "v3 manifest still parses");
    if (v3.ok()) {
        check(v3.value().format_version == 3, "v3 keeps version 3");
        check(v3.value().tensors.at("w").encoding == QuantizedEncoding::Fp8,
              "v3 encoding names still resolve");
    }

    const char* v2_json = R"({"format": "shelf-fpx-safetensors", "format_version": 2, "tensors": {}})";
    check(!parse_quantized_manifest(v2_json).ok(), "rejects version 2");
    const char* v5_json = R"({"format": "shelf-fpx-safetensors", "format_version": 5, "tensors": {}})";
    check(!parse_quantized_manifest(v5_json).ok(), "rejects version 5");

    const char* bad_block = R"({
  "format": "shelf-fpx-safetensors",
  "format_version": 4,
  "tensors": {
    "w": {
      "domain": "weight",
      "logical_shape": [256, 256],
      "quant": {
        "encoding": "fp8_e4m3",
        "scale": {"granularity": "matrix_block", "block_shape": {"n": 128, "k": 64}, "dtype": "F32", "application": "multiply"}
      },
      "layout": {"layout_id": "fp8_e4m3_row_major_block128_v1", "storage_order": "row_major", "physical_shape": [256, 256], "k_padded": 256},
      "payloads": [
        {"role": "codes", "tensor": "w.__shelf_codes", "dtype": "U8", "shape": [256, 256], "crc32": "0x00000000"},
        {"role": "scale", "tensor": "w.__shelf_scale", "dtype": "F32", "shape": [2, 4], "crc32": "0x00000000"}
      ]
    }
  }
})";
    Result<QuantizedManifest> group64 = parse_quantized_manifest(bad_block);
    check(group64.ok(), "v4 accepts the fp8 group-64 block at format level");
    if (group64.ok()) {
        const QuantizedTensorMetadata& t64 = group64.value().tensors.at("w");
        check(t64.scale_block_n == 128 && t64.scale_block_k == 64,
              "v4 keeps the declared fp8 block");
        check(t64.metadata1->dtype == "F32", "fp8 group-64 scale tag stays F32");
    }

    const char* invalid_block = R"({
  "format": "shelf-fpx-safetensors",
  "format_version": 4,
  "tensors": {
    "w": {
      "domain": "weight",
      "logical_shape": [256, 256],
      "quant": {
        "encoding": "fp8_e4m3",
        "scale": {"granularity": "matrix_block", "block_shape": {"n": 128, "k": 16}, "dtype": "F32", "application": "multiply"}
      },
      "layout": {"layout_id": "fp8_e4m3_row_major_block128_v1", "storage_order": "row_major", "physical_shape": [256, 256], "k_padded": 256},
      "payloads": [
        {"role": "codes", "tensor": "w.__shelf_codes", "dtype": "U8", "shape": [256, 256], "crc32": "0x00000000"},
        {"role": "scale", "tensor": "w.__shelf_scale", "dtype": "F32", "shape": [2, 16], "crc32": "0x00000000"}
      ]
    }
  }
})";
    check(!parse_quantized_manifest(invalid_block).ok(),
          "v4 rejects a scale block below 32");

    const char* missing_scale = R"({
  "format": "shelf-fpx-safetensors",
  "format_version": 4,
  "tensors": {
    "w": {
      "domain": "weight",
      "logical_shape": [256, 256],
      "quant": {"encoding": "fp8_e4m3", "scale": {"granularity": "matrix_block", "block_shape": {"n": 128, "k": 128}, "dtype": "F32", "application": "multiply"}},
      "layout": {"layout_id": "fp8_e4m3_row_major_block128_v1", "storage_order": "row_major", "physical_shape": [256, 256], "k_padded": 256},
      "payloads": [
        {"role": "codes", "tensor": "w.__shelf_codes", "dtype": "U8", "shape": [256, 256], "crc32": "0x00000000"}
      ]
    }
  }
})";
    check(!parse_quantized_manifest(missing_scale).ok(), "v4 rejects missing scale payload");

    const char* bad_scale_dtype = R"({
  "format": "shelf-fpx-safetensors",
  "format_version": 4,
  "tensors": {
    "w": {
      "domain": "weight",
      "logical_shape": [256, 256],
      "quant": {"encoding": "fp8_e4m3", "scale": {"granularity": "matrix_block", "block_shape": {"n": 128, "k": 128}, "dtype": "BF16", "application": "multiply"}},
      "layout": {"layout_id": "fp8_e4m3_row_major_block128_v1", "storage_order": "row_major", "physical_shape": [256, 256], "k_padded": 256},
      "payloads": [
        {"role": "codes", "tensor": "w.__shelf_codes", "dtype": "U8", "shape": [256, 256], "crc32": "0x00000000"},
        {"role": "scale", "tensor": "w.__shelf_scale", "dtype": "BF16", "shape": [2, 2], "crc32": "0x00000000"}
      ]
    }
  }
})";
    check(!parse_quantized_manifest(bad_scale_dtype).ok(), "v4 rejects fp8 scale dtype mismatch");

    const char* kv_domain = R"({
  "format": "shelf-fpx-safetensors",
  "format_version": 4,
  "tensors": {
    "w": {
      "domain": "kv_cache",
      "logical_shape": [256, 256],
      "quant": {"encoding": "bf16"},
      "layout": {"layout_id": "", "storage_order": "row_major", "physical_shape": [256, 256], "k_padded": 256},
      "payloads": [
        {"role": "data", "tensor": "w", "dtype": "BF16", "shape": [256, 256], "crc32": "0x00000000"}
      ]
    }
  }
})";
    check(!parse_quantized_manifest(kv_domain).ok(), "v4 rejects non-weight domain tensors");
    return 0;
}
}  // namespace

int main() {
    run();
    std::printf("=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
