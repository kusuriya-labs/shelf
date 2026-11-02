#pragma once

#include <shelf/core/status.h>
#include <shelf/quantization/bundle/hf_manifest.h>

#include <string>
#include <vector>

namespace ps::quantization::bundle {

bool is_psq5_model_dir(const std::string& model_dir);

Result<HfQuantizedSource> synthesize_psq5_quantized_source(
    const std::string& model_dir,
    const std::vector<std::string>& logical_names,
    bool with_crc,
    const std::string& architecture = "qwen35_dense");

}
