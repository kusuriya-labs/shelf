#pragma once
#include <shelf/core/status.h>
#include <shelf/quantization/quant_format.h>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace shelf::quantization::manifest {

constexpr const char* kQuantizedSafetensorsFormat = "shelf-fpx-safetensors";
constexpr uint32_t kQuantizedSafetensorsMinFormatVersion = 3;
constexpr uint32_t kQuantizedSafetensorsFormatVersion = 4;
constexpr const char* kQuantizedMetadataFile = "shelf_quantization.json";
constexpr const char* kFp8EncodingName = "fp8_e4m3_block128_f32";
constexpr const char* kFp8LayoutName = "fp8_e4m3_row_major_block128_v1";
constexpr const char* kMxfp4EncodingName = "mxfp4_e2m1_e8m0_g32";
constexpr const char* kMxfp4LayoutName = "mxfp4_e2m1_row_major_block32_v1";
constexpr const char* kCodesSuffix = ".__shelf_codes";
constexpr const char* kMetadata1Suffix = ".__shelf_metadata1";
constexpr const char* kMetadata2Suffix = ".__shelf_metadata2";
constexpr const char* kMetadata3Suffix = ".__shelf_metadata3";
constexpr const char* kMetadata4Suffix = ".__shelf_metadata4";

enum class QuantizedEncoding : uint8_t {
    Bf16 = 0,
    Fp8 = 5,
    Mxfp4 = 7,
};

struct QuantizedTensorRef {
    std::string tensor;
    std::string dtype;
    std::string crc32;
};

struct QuantizedTensorMetadata {
    std::string role;
    QuantizedEncoding encoding = QuantizedEncoding::Bf16;
    std::string codebook;
    std::string layout;
    std::vector<int64_t> logical_shape;
    uint64_t k_padded = 0;
    uint32_t scale_block_n = 0;
    uint32_t scale_block_k = 0;
    std::optional<QuantizedTensorRef> codes;
    std::optional<QuantizedTensorRef> metadata1;
    std::optional<QuantizedTensorRef> metadata2;
    std::optional<QuantizedTensorRef> metadata3;
    std::optional<QuantizedTensorRef> metadata4;
    std::optional<QuantizedTensorRef> data;
};

struct QuantizedImatrixMetadata {
    bool enabled = false;
    std::string policy;
    std::string sha256;
    std::string model_fingerprint;
    std::string corpus_sha256;
    uint64_t required = 0;
    uint64_t matched = 0;
};

struct QuantizedManifest {
    std::string format = kQuantizedSafetensorsFormat;
    uint32_t format_version = kQuantizedSafetensorsFormatVersion;
    std::string architecture;
    std::string preset;
    std::string scope = "full";
    std::string source_dtype;
    std::string source_model_fingerprint;
    uint64_t logical_parameter_count = 0;
    uint64_t payload_bytes = 0;
    uint64_t padding_bytes = 0;
    double effective_payload_bpw = 0.0;
    std::map<std::string, std::string> aliases;
    std::optional<QuantizedImatrixMetadata> imatrix;
    std::map<std::string, QuantizedTensorMetadata> tensors;
};

std::string quantized_encoding_name(QuantizedEncoding enc);
Result<QuantizedEncoding> parse_quantized_encoding(const std::string& name);
shelf::quantization::QuantFormatId quant_format_id(QuantizedEncoding enc);

Result<QuantizedManifest> parse_quantized_manifest(const std::string& json);
Result<std::string> serialize_quantized_manifest(const QuantizedManifest& manifest);

}
