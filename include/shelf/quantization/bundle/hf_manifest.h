#pragma once
#include <shelf/core/status.h>
#include <shelf/io/safetensors_reader.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ps::quantization::bundle {

using ps::quantization::manifest::QuantizedManifest;

enum class HfQuantMethod : uint8_t {
    None = 0,
    Fp8 = 1,
};

struct HfQuantizedSource {
    std::string model_dir;
    QuantizedManifest manifest;
    std::map<std::string, std::string> shard_for_physical;
    std::map<std::string, std::vector<float>> fp32_scale_staging;
    std::map<std::string, std::unique_ptr<ps::io::SafetensorsReader>> shard_readers;
};

bool is_hf_quantized_model_dir(const std::string& model_dir);

Result<HfQuantMethod> detect_hf_quant_method(const std::string& model_dir);

Result<HfQuantizedSource> synthesize_hf_quantized_source(
    const std::string& model_dir,
    const std::vector<std::string>& logical_names,
    bool with_crc);

}
