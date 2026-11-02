#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/bundle_util.h>
#include <shelf/quantization/manifest/crc32.h>
#include <shelf/quantization/quant_format.h>
#include <shelf/quantization/manifest/runtime_resolution.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <nlohmann/json.hpp>

namespace ps::quantization::bundle {

namespace fs = std::filesystem;

using ps::quantization::manifest::QuantizedEncoding;
using ps::quantization::manifest::QuantizedTensorMetadata;
using ps::quantization::manifest::QuantizedTensorRef;
using ps::quantization::manifest::crc32_iso_hdlc;
using ps::quantization::manifest::validate_quantized_manifest_contract;

namespace {

Result<nlohmann::json> read_config_json(const std::string& model_dir) {
    const fs::path config_path = fs::path(model_dir) / "config.json";
    if (!fs::exists(config_path)) {
        return Status::invalid_argument("config.json not found", __FILE__, __LINE__);
    }
    std::string text;
    Status st = read_file_text(config_path.string(), text);
    if (!st.ok()) return st;
    try {
        nlohmann::json config = nlohmann::json::parse(text);
        if (!config.is_object()) {
            return Status::invalid_argument("config.json is not an object", __FILE__, __LINE__);
        }
        return config;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("bad config.json: " + std::string(e.what())).c_str(), __FILE__, __LINE__);
    }
}

const nlohmann::json* find_quantization_config(const nlohmann::json& config) {
    auto root = config.find("quantization_config");
    if (root != config.end() && root->is_object()) return &*root;
    auto text_config = config.find("text_config");
    if (text_config != config.end() && text_config->is_object()) {
        auto nested = text_config->find("quantization_config");
        if (nested != text_config->end() && nested->is_object()) return &*nested;
    }
    return nullptr;
}

bool supported_hf_quant_method_name(const nlohmann::json* qc) {
    if (qc == nullptr) return false;
    auto it = qc->find("quant_method");
    if (it == qc->end() || !it->is_string()) return false;
    const std::string method = it->get<std::string>();
    return method == "fp8";
}

bool json_int_equals(const nlohmann::json& value, int64_t expected) {
    if (value.is_number_integer() || value.is_number_unsigned()) {
        return value.get<int64_t>() == expected;
    }
    return false;
}

bool block_size_is(const nlohmann::json& value, int64_t rows, int64_t cols) {
    return value.is_array() && value.size() == 2 &&
           json_int_equals(value[0], rows) && json_int_equals(value[1], cols);
}

uint64_t element_count(const std::vector<std::size_t>& shape) {
    uint64_t n = 1;
    for (const std::size_t d : shape) n *= static_cast<uint64_t>(d);
    return n;
}

Status spec_bytes(const ps::io::StTensorSpec& spec, std::size_t elem_size, uint64_t& out) {
    uint64_t bytes = element_count(spec.shape);
    if (elem_size != 0 && bytes > UINT64_MAX / elem_size) {
        return Status::overflow("hf tensor byte count overflow", __FILE__, __LINE__);
    }
    bytes *= elem_size;
    if (bytes != spec.data_end - spec.data_begin) {
        return Status::invalid_argument("hf tensor byte count mismatch", __FILE__, __LINE__);
    }
    out = bytes;
    return Status::make_ok();
}

std::string crc_hex(const void* data, std::size_t bytes) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08x", crc32_iso_hdlc(data, bytes));
    return buf;
}

float bf16_bits_to_f32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct HfShardSet {
    std::string model_dir;
    std::map<std::string, std::string> shard_for_tensor;
    std::map<std::string, std::unique_ptr<ps::io::SafetensorsReader>> readers;

    Result<const ps::io::SafetensorsReader*> reader_for_shard(const std::string& shard) {
        auto it = readers.find(shard);
        if (it != readers.end()) return it->second.get();
        Status path_status = validate_shard_path(shard);
        if (!path_status.ok()) return path_status;
        auto opened = ps::io::SafetensorsReader::open((fs::path(model_dir) / shard).string());
        if (!opened.ok()) return opened.status();
        auto inserted = readers.emplace(
            shard, std::make_unique<ps::io::SafetensorsReader>(opened.release()));
        return inserted.first->second.get();
    }
};

Result<std::unique_ptr<HfShardSet>> open_shard_set(const std::string& model_dir) {
    auto set = std::make_unique<HfShardSet>();
    set->model_dir = model_dir;

    std::vector<std::string> shards;
    const fs::path index_path = fs::path(model_dir) / "model.safetensors.index.json";
    if (fs::exists(index_path)) {
        std::string text;
        Status st = read_file_text(index_path.string(), text);
        if (!st.ok()) return st;
        nlohmann::json index;
        try {
            index = nlohmann::json::parse(text);
        } catch (const nlohmann::json::exception& e) {
            return Status::invalid_argument(
                ("bad safetensors index: " + std::string(e.what())).c_str(), __FILE__, __LINE__);
        }
        if (!index.is_object() || !index.contains("weight_map") || !index["weight_map"].is_object()) {
            return Status::invalid_argument("safetensors index missing weight_map", __FILE__, __LINE__);
        }
        std::set<std::string> unique_shards;
        for (auto it = index["weight_map"].begin(); it != index["weight_map"].end(); ++it) {
            if (!it.value().is_string()) {
                return Status::invalid_argument("safetensors index shard path is not a string", __FILE__, __LINE__);
            }
            const std::string shard = it.value().get<std::string>();
            Status path_status = validate_shard_path(shard);
            if (!path_status.ok()) return path_status;
            unique_shards.insert(shard);
        }
        shards.assign(unique_shards.begin(), unique_shards.end());
    } else {
        shards.push_back("model.safetensors");
    }

    for (const std::string& shard : shards) {
        auto reader_result = set->reader_for_shard(shard);
        if (!reader_result.ok()) return reader_result.status();
        auto names = reader_result.value()->list_tensors();
        if (!names.ok()) return names.status();
        for (const std::string& name : names.value()) {
            auto inserted = set->shard_for_tensor.emplace(name, shard);
            if (!inserted.second && inserted.first->second != shard) {
                return Status::invalid_argument(
                    ("hf tensor appears in multiple shards: " + name).c_str(), __FILE__, __LINE__);
            }
        }
    }
    return set;
}

QuantizedTensorRef make_ref(const std::string& physical, const std::string& dtype,
                            const void* data, std::size_t bytes, bool with_crc) {
    QuantizedTensorRef ref;
    ref.tensor = physical;
    ref.dtype = dtype;
    if (with_crc) ref.crc32 = crc_hex(data, bytes);
    return ref;
}

Status synthesize_fp8_tensor(
    const ps::io::SafetensorsReader* reader,
    HfQuantizedSource& source,
    const std::string& canonical,
    const std::string& scale_inv_name,
    const std::string& codes_shard,
    const std::string& scale_inv_shard,
    QuantizedTensorMetadata& out,
    bool with_crc) {
    auto spec_result = reader->tensor_spec(canonical);
    if (!spec_result.ok()) return spec_result.status();
    const ps::io::StTensorSpec& codes_spec = spec_result.value();
    if ((codes_spec.dtype != ps::io::SType::F8_E4M3 && codes_spec.dtype != ps::io::SType::U8) ||
        codes_spec.shape.size() != 2) {
        return Status::unsupported(
            ("hf fp8 weight payload must be a rank-2 F8_E4M3 matrix: " + canonical).c_str(),
            __FILE__, __LINE__);
    }
    uint64_t codes_bytes = 0;
    Status bytes_status = spec_bytes(codes_spec, 1, codes_bytes);
    if (!bytes_status.ok()) return bytes_status;
    const uint64_t n = codes_spec.shape[0];
    const uint64_t k = codes_spec.shape[1];
    if ((k % quantization::kFp8ScaleBlockK) != 0u) {
        return Status::unsupported(
            ("hf fp8 weight K is not a multiple of 128: " + canonical).c_str(),
            __FILE__, __LINE__);
    }

    auto scale_spec_result = reader->tensor_spec(scale_inv_name);
    if (!scale_spec_result.ok()) return scale_spec_result.status();
    const ps::io::StTensorSpec& scale_spec = scale_spec_result.value();
    if (scale_spec.dtype != ps::io::SType::BF16 || scale_spec.shape.size() != 2 ||
        scale_spec.shape[0] !=
            (n + quantization::kFp8ScaleBlockN - 1u) / quantization::kFp8ScaleBlockN ||
        scale_spec.shape[1] != k / quantization::kFp8ScaleBlockK) {
        return Status::unsupported(
            ("hf fp8 scale tensor shape or dtype mismatch: " + scale_inv_name).c_str(),
            __FILE__, __LINE__);
    }
    std::size_t scale_src_bytes = 0;
    auto scale_data_result = reader->tensor_data(scale_inv_name, scale_src_bytes);
    if (!scale_data_result.ok()) return scale_data_result.status();
    if (scale_src_bytes != scale_spec.shape[0] * scale_spec.shape[1] * 2u) {
        return Status::invalid_argument(
            ("hf fp8 scale tensor byte count mismatch: " + scale_inv_name).c_str(),
            __FILE__, __LINE__);
    }
    const auto* scale_src = static_cast<const uint16_t*>(scale_data_result.value());
    const std::size_t scale_count = scale_spec.shape[0] * scale_spec.shape[1];
    std::vector<float>& staging = source.fp32_scale_staging[scale_inv_name];
    staging.resize(scale_count);
    for (std::size_t i = 0; i < scale_count; ++i) {
        staging[i] = bf16_bits_to_f32(scale_src[i]);
    }

    std::size_t codes_src_bytes = 0;
    auto codes_data_result = reader->tensor_data(canonical, codes_src_bytes);
    if (!codes_data_result.ok()) return codes_data_result.status();
    if (codes_src_bytes != codes_bytes) {
        return Status::invalid_argument(
            ("hf fp8 weight byte count mismatch: " + canonical).c_str(), __FILE__, __LINE__);
    }

    out.role = "weight";
    out.encoding = QuantizedEncoding::Fp8;
    out.logical_shape = {static_cast<int64_t>(n), static_cast<int64_t>(k)};
    out.k_padded = quant_padded_k(k, quantization::QuantFormatId::Fp8Finegrained);
    out.codes = make_ref(canonical, "U8", codes_data_result.value(), codes_src_bytes, with_crc);
    out.metadata1 =
        make_ref(scale_inv_name, "F32", staging.data(), staging.size() * sizeof(float), with_crc);
    source.shard_for_physical[canonical] = codes_shard;
    source.shard_for_physical[scale_inv_name] = scale_inv_shard;
    return Status::make_ok();
}

Status synthesize_bf16_tensor(
    const ps::io::SafetensorsReader* reader,
    HfQuantizedSource& source,
    const std::string& canonical,
    const std::string& codes_shard,
    QuantizedTensorMetadata& out,
    bool with_crc) {
    auto spec_result = reader->tensor_spec(canonical);
    if (!spec_result.ok()) return spec_result.status();
    const ps::io::StTensorSpec& spec = spec_result.value();
    if (spec.dtype != ps::io::SType::BF16) {
        return Status::unsupported(
            ("hf bf16 tensor has a non-BF16 dtype: " + canonical).c_str(),
            __FILE__, __LINE__);
    }
    if (spec.shape.empty()) {
        return Status::invalid_argument(
            ("hf bf16 tensor has an empty shape: " + canonical).c_str(),
            __FILE__, __LINE__);
    }
    uint64_t data_bytes = 0;
    Status bytes_status = spec_bytes(spec, 2, data_bytes);
    if (!bytes_status.ok()) return bytes_status;
    std::size_t src_bytes = 0;
    auto data_result = reader->tensor_data(canonical, src_bytes);
    if (!data_result.ok()) return data_result.status();

    out.encoding = QuantizedEncoding::Bf16;
    for (const std::size_t d : spec.shape) {
        out.logical_shape.push_back(static_cast<int64_t>(d));
    }
    out.k_padded = quant_padded_k(spec.shape.back(), quantization::QuantFormatId::Bf16);
    out.data = make_ref(canonical, "BF16", data_result.value(), src_bytes, with_crc);
    out.role = spec.shape.size() >= 2 ? "weight" : "norm";
    source.shard_for_physical[canonical] = codes_shard;
    return Status::make_ok();
}

}

bool is_hf_quantized_model_dir(const std::string& model_dir) {
    auto config = read_config_json(model_dir);
    if (!config.ok()) return false;
    return supported_hf_quant_method_name(find_quantization_config(config.value()));
}

Result<HfQuantMethod> detect_hf_quant_method(const std::string& model_dir) {
    auto config_result = read_config_json(model_dir);
    if (!config_result.ok()) return config_result.status();
    const nlohmann::json* qc = find_quantization_config(config_result.value());
    if (qc == nullptr) {
        return Status::unsupported("no HF quantization_config", __FILE__, __LINE__);
    }
    auto method_it = qc->find("quant_method");
    if (method_it == qc->end() || !method_it->is_string()) {
        return Status::unsupported("HF quantization_config missing quant_method", __FILE__, __LINE__);
    }
    const std::string method = method_it->get<std::string>();

    if (method == "fp8") {
        auto fmt = qc->find("fmt");
        if (fmt == qc->end() || !fmt->is_string() || fmt->get<std::string>() != "e4m3") {
            return Status::unsupported("HF fp8 quantization requires fmt e4m3", __FILE__, __LINE__);
        }
        auto block = qc->find("weight_block_size");
        if (block == qc->end() || !block_size_is(*block, 128, 128)) {
            return Status::unsupported("HF fp8 quantization requires weight_block_size [128,128]", __FILE__, __LINE__);
        }
        return HfQuantMethod::Fp8;
    }

    return Status::unsupported("unsupported HF quant_method", __FILE__, __LINE__);
}

Result<HfQuantizedSource> synthesize_hf_quantized_source(
    const std::string& model_dir,
    const std::vector<std::string>& logical_names,
    bool with_crc)
{
    auto method_result = detect_hf_quant_method(model_dir);
    if (!method_result.ok()) return method_result.status();
    const HfQuantMethod method = method_result.value();

    auto shard_set_result = open_shard_set(model_dir);
    if (!shard_set_result.ok()) return shard_set_result.status();
    std::unique_ptr<HfShardSet> shard_set = shard_set_result.release();

    HfQuantizedSource source;
    source.model_dir = model_dir;
    source.manifest.architecture = "qwen35_dense";
    source.manifest.source_dtype = "BF16";

    const auto find_shard = [&](const std::string& tensor) -> const std::string* {
        auto it = shard_set->shard_for_tensor.find(tensor);
        return it == shard_set->shard_for_tensor.end() ? nullptr : &it->second;
    };

    std::set<std::string> processed;
    for (const std::string& logical : logical_names) {
        if (!processed.insert(logical).second) continue;

        std::string canonical = logical;
        const bool has_direct = find_shard(canonical) != nullptr;
        if (!has_direct && canonical == "model.language_model.lm_head.weight" &&
            find_shard("lm_head.weight") != nullptr) {
            canonical = "lm_head.weight";
            source.manifest.aliases[logical] = canonical;
        }

        const std::string* codes_shard = find_shard(canonical);
        const std::string scale_inv_name = canonical + "_scale_inv";
        const std::string scale_name = canonical + "_scale";
        const std::string* scale_inv_shard = find_shard(scale_inv_name);
        const std::string* scale_shard = find_shard(scale_name);

        if (codes_shard == nullptr && scale_inv_shard == nullptr && scale_shard == nullptr) {
            continue;
        }
        if (codes_shard == nullptr) {
            return Status::invalid_argument(
                ("hf quantized tensor missing weight payload: " + canonical).c_str(),
                __FILE__, __LINE__);
        }

        auto reader_result = shard_set->reader_for_shard(*codes_shard);
        if (!reader_result.ok()) return reader_result.status();
        const ps::io::SafetensorsReader* reader = reader_result.value();

        if (scale_inv_shard != nullptr && scale_shard != nullptr) {
            return Status::invalid_argument(
                ("hf tensor carries both weight_scale_inv and weight_scale: " + canonical).c_str(),
                __FILE__, __LINE__);
        }

        QuantizedTensorMetadata meta;
        if (scale_inv_shard != nullptr) {
            if (method != HfQuantMethod::Fp8) {
                return Status::unsupported(
                    ("hf tensor carries weight_scale_inv in a non-fp8 model: " + canonical).c_str(),
                    __FILE__, __LINE__);
            }
            Status adapter = synthesize_fp8_tensor(
                reader, source, canonical, scale_inv_name, *codes_shard, *scale_inv_shard, meta,
                with_crc);
            if (!adapter.ok()) return adapter;
        } else if (scale_shard != nullptr) {
            return Status::unsupported(
                ("hf tensor carries an unsupported weight_scale payload: " + canonical).c_str(),
                __FILE__, __LINE__);
        } else {
            Status adapter =
                synthesize_bf16_tensor(reader, source, canonical, *codes_shard, meta, with_crc);
            if (!adapter.ok()) return adapter;
        }

        if (meta.logical_shape.size() < 2 && meta.encoding != QuantizedEncoding::Bf16) {
            return Status::unsupported(
                ("hf quantized tensor must be a matrix: " + canonical).c_str(),
                __FILE__, __LINE__);
        }

        uint64_t tensor_bytes = 0;
        const QuantizedTensorRef* refs[5] = {
            meta.codes ? &*meta.codes : nullptr,
            meta.metadata1 ? &*meta.metadata1 : nullptr,
            meta.metadata2 ? &*meta.metadata2 : nullptr,
            meta.metadata3 ? &*meta.metadata3 : nullptr,
            meta.data ? &*meta.data : nullptr};
        for (const QuantizedTensorRef* ref : refs) {
            if (ref == nullptr) continue;
            auto ref_spec = shard_set->reader_for_shard(source.shard_for_physical.at(ref->tensor));
            if (!ref_spec.ok()) return ref_spec.status();
            auto spec_result = ref_spec.value()->tensor_spec(ref->tensor);
            if (!spec_result.ok()) return spec_result.status();
            tensor_bytes += spec_result.value().data_end - spec_result.value().data_begin;
        }
        source.manifest.payload_bytes += tensor_bytes;
        source.manifest.logical_parameter_count += quant_element_count(meta.logical_shape);
        source.manifest.tensors[canonical] = std::move(meta);
    }

    if (source.manifest.tensors.empty()) {
        return Status::invalid_argument(
            "no requested HF tensors were found in the model dir", __FILE__, __LINE__);
    }

    Status contract = validate_quantized_manifest_contract(source.manifest);
    if (!contract.ok()) return contract;
    return source;
}

}
