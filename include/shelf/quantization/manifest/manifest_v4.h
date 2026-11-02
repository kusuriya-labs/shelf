#pragma once

#include <shelf/core/status.h>
#include <shelf/quantization/manifest/quantized_manifest.h>

#include <string>

namespace ps::quantization::manifest {

constexpr const char* kFp8V4EncodingName = "fp8_e4m3";
constexpr const char* kPsq5V4EncodingName = "psq5_cb28";

const char* v4_encoding_name(QuantizedEncoding enc);
Result<QuantizedEncoding> parse_v4_encoding(const std::string& name);

Result<QuantizedManifest> parse_quantized_manifest_v4(const std::string& json);
Result<std::string> serialize_quantized_manifest_v4(const QuantizedManifest& manifest);

}
