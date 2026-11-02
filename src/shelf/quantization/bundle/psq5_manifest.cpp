#include <shelf/quantization/bundle/psq5_manifest.h>
#include <shelf/quantization/bundle/bundle_util.h>
#include <shelf/quantization/manifest/crc32.h>
#include <shelf/quantization/quant_format.h>
#include <shelf/quantization/manifest/runtime_resolution.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <nlohmann/json.hpp>

namespace ps::quantization::bundle {

namespace fs = std::filesystem;

using ps::quantization::manifest::QuantizedEncoding;
using ps::quantization::manifest::QuantizedTensorMetadata;
using ps::quantization::manifest::QuantizedTensorRef;
using ps::quantization::manifest::crc32_iso_hdlc;

namespace {

constexpr const char* kPsq5ManifestFile = "model.safetensors.json";
constexpr const char* kPsq5FormatName = "PSQ5-CB28";

struct Psq5ShardSet {
    std::string model_dir;
    std::map<std::string, std::string> shard_for_tensor;
    std::set<std::string> shard_files;
    std::map<std::string, std::unique_ptr<ps::io::SafetensorsReader>> readers;

    Result<const ps::io::SafetensorsReader*> reader_for_shard(const std::string& shard) {
        auto it = readers.find(shard);
        if (it != readers.end()) return it->second.get();
        const fs::path full = fs::path(model_dir) / shard;
        auto reader_result = ps::io::SafetensorsReader::open(full.string());
        if (!reader_result.ok()) return reader_result.status();
        auto inserted = readers.emplace(shard, std::make_unique<ps::io::SafetensorsReader>(reader_result.release()));
        return inserted.first->second.get();
    }
};

Result<std::unique_ptr<Psq5ShardSet>> open_psq5_shard_set(const std::string& model_dir) {
    auto set = std::make_unique<Psq5ShardSet>();
    set->model_dir = model_dir;
    const fs::path index_path = fs::path(model_dir) / "model.safetensors.index.json";
    if (!fs::exists(index_path)) {
        return Status::invalid_argument("model.safetensors.index.json not found", __FILE__, __LINE__);
    }
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
    if (!index.contains("weight_map") || !index["weight_map"].is_object()) {
        return Status::invalid_argument("safetensors index missing weight_map", __FILE__, __LINE__);
    }
    for (auto it = index["weight_map"].begin(); it != index["weight_map"].end(); ++it) {
        if (!it.value().is_string()) {
            return Status::invalid_argument("safetensors index shard path is not a string", __FILE__, __LINE__);
        }
        const std::string shard = it.value().get<std::string>();
        Status path_status = validate_shard_path(shard);
        if (!path_status.ok()) return path_status;
        set->shard_files.insert(shard);
    }
    for (const std::string& shard : set->shard_files) {
        auto reader_result = set->reader_for_shard(shard);
        if (!reader_result.ok()) return reader_result.status();
        auto names = reader_result.value()->list_tensors();
        if (!names.ok()) return names.status();
        for (const std::string& name : names.value()) {
            auto inserted = set->shard_for_tensor.emplace(name, shard);
            if (!inserted.second && inserted.first->second != shard) {
                return Status::invalid_argument(
                    ("psq5 tensor appears in multiple shards: " + name).c_str(), __FILE__, __LINE__);
            }
        }
    }
    return set;
}

Result<nlohmann::json> read_psq5_manifest(const std::string& model_dir) {
    const fs::path path = fs::path(model_dir) / kPsq5ManifestFile;
    if (!fs::exists(path)) {
        return Status::invalid_argument("model.safetensors.json not found", __FILE__, __LINE__);
    }
    std::string text;
    Status st = read_file_text(path.string(), text);
    if (!st.ok()) return st;
    try {
        nlohmann::json manifest = nlohmann::json::parse(text);
        if (!manifest.is_object()) {
            return Status::invalid_argument("model.safetensors.json is not an object", __FILE__, __LINE__);
        }
        return manifest;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("bad model.safetensors.json: " + std::string(e.what())).c_str(), __FILE__, __LINE__);
    }
}

std::string crc_hex(const void* data, std::size_t bytes) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08x", crc32_iso_hdlc(data, bytes));
    return buf;
}

float bits_to_f32(std::uint32_t bits) {
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float bf16_bits_to_f32_local(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16u;
    const std::uint32_t exp = (h >> 10u) & 0x1Fu;
    const std::uint32_t man = h & 0x3FFu;
    if (exp == 0u) {
        if (man == 0u) return bits_to_f32(sign);
        int e = -124;
        std::uint32_t m = man;
        while ((m & 0x400u) == 0u) { m <<= 1; ++e; }
        m &= 0x3FFu;
        return bits_to_f32(sign | (static_cast<std::uint32_t>(e + 127) << 23u) | (m << 13u));
    }
    if (exp == 31u) {
        return bits_to_f32(sign | 0x7F800000u | (man << 13u));
    }
    return bits_to_f32(sign | ((exp + 112u) << 23u) | (man << 13u));
}

QuantizedTensorRef make_ref(const std::string& tensor, const std::string& dtype,
                             const void* data, std::size_t bytes, bool with_crc) {
    QuantizedTensorRef ref;
    ref.tensor = tensor;
    ref.dtype = dtype;
    if (with_crc) ref.crc32 = crc_hex(data, bytes);
    return ref;
}

Status synthesize_fp8_block_entry(
    Psq5ShardSet& shard_set,
    const std::string& canonical,
    const nlohmann::json& entry,
    HfQuantizedSource& source,
    QuantizedTensorMetadata& out,
    bool with_crc) {
    auto codes_key = entry.find("codes_key");
    auto scale_key = entry.find("scale_key");
    auto shard = entry.find("shard");
    auto shape = entry.find("weight_shape");
    auto block = entry.find("block_size");
    if (codes_key == entry.end() || !codes_key->is_string() ||
        scale_key == entry.end() || !scale_key->is_string() ||
        shard == entry.end() || !shard->is_string() ||
        shape == entry.end() || !shape->is_array() || shape->size() != 2 ||
        block == entry.end() || !block->is_array() || block->size() != 2) {
        return Status::invalid_argument(
            ("mixed manifest fp8_block entry malformed: " + canonical).c_str(),
            __FILE__, __LINE__);
    }
    const std::string codes_name = codes_key->get<std::string>();
    const std::string scale_name = scale_key->get<std::string>();
    const std::string entry_shard = shard->get<std::string>();
    const std::uint64_t n = (*shape)[0].get<std::uint64_t>();
    const std::uint64_t k = (*shape)[1].get<std::uint64_t>();
    const std::uint64_t bn = (*block)[0].get<std::uint64_t>();
    const std::uint64_t bk = (*block)[1].get<std::uint64_t>();
    if (bk != 128u || bn != 128u) {
        return Status::unsupported(
            ("mixed manifest fp8_block only supports 128x128: " + canonical).c_str(),
            __FILE__, __LINE__);
    }

    auto reader_result = shard_set.reader_for_shard(entry_shard);
    if (!reader_result.ok()) return reader_result.status();
    const ps::io::SafetensorsReader* reader = reader_result.value();

    auto codes_spec = reader->tensor_spec(codes_name);
    if (!codes_spec.ok()) return codes_spec.status();
    const auto& cs = codes_spec.value();
    if ((cs.dtype != ps::io::SType::F8_E4M3 && cs.dtype != ps::io::SType::U8) ||
        cs.shape.size() != 2 || cs.shape[0] != n || cs.shape[1] != k) {
        return Status::unsupported(
            ("mixed manifest fp8_block codes shape/dtype mismatch: " + canonical).c_str(),
            __FILE__, __LINE__);
    }

    auto scale_spec = reader->tensor_spec(scale_name);
    if (!scale_spec.ok()) return scale_spec.status();
    const auto& ss = scale_spec.value();
    const std::uint64_t expect_sn = (n + bn - 1u) / bn;
    const std::uint64_t expect_sk = k / bk;
    if (ss.dtype != ps::io::SType::BF16 || ss.shape.size() != 2 ||
        ss.shape[0] != expect_sn || ss.shape[1] != expect_sk) {
        return Status::unsupported(
            ("mixed manifest fp8_block scale shape/dtype mismatch: " + canonical).c_str(),
            __FILE__, __LINE__);
    }
    if ((k % 128u) != 0u) {
        return Status::unsupported(
            ("mixed manifest fp8_block K not multiple of 128: " + canonical).c_str(),
            __FILE__, __LINE__);
    }

    std::size_t codes_bytes = 0;
    auto codes_data = reader->tensor_data(codes_name, codes_bytes);
    if (!codes_data.ok()) return codes_data.status();
    const std::uint64_t expect_codes = n * k;
    if (codes_bytes != expect_codes) {
        return Status::invalid_argument(
            ("mixed manifest fp8_block codes byte count mismatch: " + canonical).c_str(),
            __FILE__, __LINE__);
    }

    std::size_t scale_bytes = 0;
    auto scale_data = reader->tensor_data(scale_name, scale_bytes);
    if (!scale_data.ok()) return scale_data.status();
    const std::uint64_t expect_scale = expect_sn * expect_sk * 2u;
    if (scale_bytes != expect_scale) {
        return Status::invalid_argument(
            ("mixed manifest fp8_block scale byte count mismatch: " + canonical).c_str(),
            __FILE__, __LINE__);
    }

    const auto* scale_src = static_cast<const std::uint16_t*>(scale_data.value());
    const std::size_t scale_count = expect_sn * expect_sk;
    std::vector<float>& staging = source.fp32_scale_staging[scale_name];
    staging.resize(scale_count);
    for (std::size_t i = 0; i < scale_count; ++i) {
        staging[i] = bf16_bits_to_f32_local(scale_src[i]);
    }

    out.role = "weight";
    out.encoding = QuantizedEncoding::Fp8;
    out.logical_shape = {static_cast<int64_t>(n), static_cast<int64_t>(k)};
    out.k_padded = quant_padded_k(k, quantization::QuantFormatId::Fp8);
    out.scale_block_n = 128u;
    out.scale_block_k = 128u;
    out.codes = make_ref(codes_name, "U8", codes_data.value(), codes_bytes, with_crc);
    out.metadata1 = make_ref(scale_name, "F32", staging.data(),
                            staging.size() * sizeof(float), with_crc);
    source.shard_for_physical[codes_name] = entry_shard;
    source.shard_for_physical[scale_name] = entry_shard;
    return Status::make_ok();
}

Status synthesize_bf16_passthrough(
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
            ("psq5 passthrough tensor has a non-BF16 dtype: " + canonical).c_str(),
            __FILE__, __LINE__);
    }
    if (spec.shape.empty()) {
        return Status::invalid_argument(
            ("psq5 passthrough tensor has an empty shape: " + canonical).c_str(),
            __FILE__, __LINE__);
    }
    uint64_t data_bytes = 0;
    uint64_t elements = 1;
    for (const std::size_t d : spec.shape) {
        if (d != 0 && elements > (std::numeric_limits<uint64_t>::max() / d)) {
            return Status::overflow("psq5 passthrough tensor element count overflow", __FILE__, __LINE__);
        }
        elements *= d;
    }
    if (elements > std::numeric_limits<uint64_t>::max() / 2u) {
        return Status::overflow("psq5 passthrough tensor byte count overflow", __FILE__, __LINE__);
    }
    data_bytes = elements * 2u;
    std::size_t src_bytes = 0;
    auto data_result = reader->tensor_data(canonical, src_bytes);
    if (!data_result.ok()) return data_result.status();
    if (src_bytes != data_bytes) {
        return Status::invalid_argument(
            ("psq5 passthrough tensor byte count mismatch: " + canonical).c_str(),
            __FILE__, __LINE__);
    }
    out.encoding = QuantizedEncoding::Bf16;
    for (const std::size_t d : spec.shape) {
        out.logical_shape.push_back(static_cast<int64_t>(d));
    }
    const int64_t logical_k = out.logical_shape.back();
    out.k_padded = static_cast<uint64_t>(logical_k);
    out.data = make_ref(canonical, "BF16", data_result.value(), src_bytes, with_crc);
    source.shard_for_physical[canonical] = codes_shard;
    return Status::make_ok();
}

}

bool is_psq5_model_dir(const std::string& model_dir) {
    const fs::path path = fs::path(model_dir) / kPsq5ManifestFile;
    if (!fs::exists(path)) return false;
    auto manifest_result = read_psq5_manifest(model_dir);
    if (!manifest_result.ok()) return false;
    const nlohmann::json& manifest = manifest_result.value();
    auto format = manifest.find("format");
    if (format == manifest.end() || !format->is_string()) return false;
    if (format->get<std::string>() != kPsq5FormatName) return false;
    auto block_k = manifest.find("block_k");
    if (block_k == manifest.end() || !block_k->is_number_unsigned() ||
        block_k->get<std::uint64_t>() != 32u) {
        return false;
    }
    auto scale_format = manifest.find("scale_format");
    if (scale_format == manifest.end() || !scale_format->is_string() ||
        scale_format->get<std::string>() != "ue4m4") {
        return false;
    }
    return true;
}

Result<HfQuantizedSource> synthesize_psq5_quantized_source(
    const std::string& model_dir,
    const std::vector<std::string>& logical_names,
    bool with_crc,
    const std::string& architecture)
{
    if (!is_psq5_model_dir(model_dir)) {
        return Status::unsupported("not a PSQ5-CB28 ue4m4 model dir", __FILE__, __LINE__);
    }
    auto manifest_result = read_psq5_manifest(model_dir);
    if (!manifest_result.ok()) return manifest_result.status();
    const nlohmann::json& pq = manifest_result.value();
    auto tensors_it = pq.find("tensors");
    if (tensors_it == pq.end() || !tensors_it->is_object()) {
        return Status::invalid_argument("psq5 manifest missing tensors", __FILE__, __LINE__);
    }
    const nlohmann::json& pq_tensors = *tensors_it;

    auto shard_set_result = open_psq5_shard_set(model_dir);
    if (!shard_set_result.ok()) return shard_set_result.status();
    std::unique_ptr<Psq5ShardSet> shard_set = shard_set_result.release();

    HfQuantizedSource source;
    source.model_dir = model_dir;
    source.manifest.architecture = architecture;
    source.manifest.source_dtype = "BF16";

    const auto find_shard = [&](const std::string& tensor) -> const std::string* {
        auto it = shard_set->shard_for_tensor.find(tensor);
        return it == shard_set->shard_for_tensor.end() ? nullptr : &it->second;
    };

    std::set<std::string> processed;
    for (const std::string& logical : logical_names) {
        if (!processed.insert(logical).second) continue;

        const auto fp8b_it = pq.find("fp8_block");
        const nlohmann::json* fp8b_tensors = nullptr;
        if (fp8b_it != pq.end() && fp8b_it->is_object()) {
            auto t = fp8b_it->find("tensors");
            if (t != fp8b_it->end() && t->is_object()) fp8b_tensors = &(*t);
        }
        std::string canonical = logical;
        if (fp8b_tensors != nullptr) {
            auto f8 = fp8b_tensors->find(canonical);
            if (f8 != fp8b_tensors->end() && f8->is_object()) {
                QuantizedTensorMetadata f8_meta;
                Status f8_st = synthesize_fp8_block_entry(
                    *shard_set, canonical, *f8, source, f8_meta, with_crc);
                if (!f8_st.ok()) return f8_st;
                source.manifest.tensors[canonical] = std::move(f8_meta);
                continue;
            }
        }
        auto pq_it = pq_tensors.find(canonical);
        if (pq_it == pq_tensors.end()) {
            if (canonical == "model.language_model.lm_head.weight" &&
                pq_tensors.find("lm_head.weight") != pq_tensors.end()) {
                canonical = "lm_head.weight";
                source.manifest.aliases[logical] = canonical;
                pq_it = pq_tensors.find(canonical);
            }
        }
        if (pq_it == pq_tensors.end()) {
            std::string bf16_canonical = canonical;
            const std::string* bf16_shard = find_shard(bf16_canonical);
            if (bf16_shard == nullptr &&
                bf16_canonical == "model.language_model.lm_head.weight") {
                bf16_canonical = "lm_head.weight";
                bf16_shard = find_shard(bf16_canonical);
                if (bf16_shard != nullptr) {
                    source.manifest.aliases[canonical] = bf16_canonical;
                }
            }
            if (bf16_shard == nullptr) continue;
            auto reader_result = shard_set->reader_for_shard(*bf16_shard);
            if (!reader_result.ok()) return reader_result.status();
            QuantizedTensorMetadata bf16_meta;
            Status bf16_status = synthesize_bf16_passthrough(
                reader_result.value(), source, bf16_canonical, *bf16_shard, bf16_meta, with_crc);
            if (!bf16_status.ok()) return bf16_status;
            source.manifest.tensors[bf16_canonical] = std::move(bf16_meta);
            continue;
        }

        const nlohmann::json& entry = *pq_it;
        auto streams = entry.find("streams");
        auto shape = entry.find("weight_shape");
        auto shard = entry.find("shard");
        if (streams == entry.end() || !streams->is_object() || shape == entry.end() ||
            !shape->is_array() || shard == entry.end() || !shard->is_string()) {
            return Status::invalid_argument(
                ("psq5 tensor entry malformed: " + canonical).c_str(), __FILE__, __LINE__);
        }
        std::uint64_t n = 0;
        std::uint64_t k = 0;
        auto rows_it = entry.find("rows");
        auto cols_it = entry.find("columns");
        if (rows_it != entry.end() && rows_it->is_number_unsigned() &&
            cols_it != entry.end() && cols_it->is_number_unsigned()) {
            n = rows_it->get<std::uint64_t>();
            k = cols_it->get<std::uint64_t>();
        } else if (shape->size() == 2) {
            n = (*shape)[0].get<std::uint64_t>();
            k = (*shape)[1].get<std::uint64_t>();
        } else {
            return Status::unsupported(
                ("psq5 tensor entry has neither rows/columns nor rank-2 weight_shape: " + canonical)
                    .c_str(), __FILE__, __LINE__);
        }
        auto mag = streams->find("magnitude");
        auto sign = streams->find("sign");
        auto scale = streams->find("scale");
        if (mag == streams->end() || !mag->is_string() || sign == streams->end() ||
            !sign->is_string() || scale == streams->end() || !scale->is_string()) {
            return Status::invalid_argument(
                ("psq5 tensor streams malformed: " + canonical).c_str(), __FILE__, __LINE__);
        }
        const std::string mag_name = mag->get<std::string>();
        const std::string sign_name = sign->get<std::string>();
        const std::string scale_name = scale->get<std::string>();
        const std::string entry_shard = shard->get<std::string>();
        const std::uint64_t kp = quant_padded_k(k, quantization::QuantFormatId::Psq5);
        const std::uint64_t nb = kp / 32u;

        auto reader_result = shard_set->reader_for_shard(entry_shard);
        if (!reader_result.ok()) return reader_result.status();
        const ps::io::SafetensorsReader* reader = reader_result.value();

        std::size_t mag_bytes = 0;
        auto mag_res = reader->tensor_data(mag_name, mag_bytes);
        if (!mag_res.ok()) return mag_res.status();
        std::size_t sign_bytes = 0;
        auto sign_res = reader->tensor_data(sign_name, sign_bytes);
        if (!sign_res.ok()) return sign_res.status();
        std::size_t scale_bytes = 0;
        auto scale_res = reader->tensor_data(scale_name, scale_bytes);
        if (!scale_res.ok()) return scale_res.status();

        QuantizedTensorMetadata meta;
        meta.role = "weight";
        meta.encoding = QuantizedEncoding::Psq5;
        meta.logical_shape = {static_cast<int64_t>(n), static_cast<int64_t>(k)};
        meta.k_padded = kp;
        meta.scale_block_n = 1;
        meta.scale_block_k = 32;
        meta.codes = make_ref(mag_name, "U8", mag_res.value(), mag_bytes, with_crc);
        meta.metadata1 = make_ref(sign_name, "U8", sign_res.value(), sign_bytes, with_crc);
        meta.metadata2 = make_ref(scale_name, "U8", scale_res.value(), scale_bytes, with_crc);
        source.shard_for_physical[mag_name] = entry_shard;
        source.shard_for_physical[sign_name] = entry_shard;
        source.shard_for_physical[scale_name] = entry_shard;
        source.manifest.tensors[canonical] = std::move(meta);
    }

    if (source.manifest.tensors.empty()) {
        return Status::invalid_argument("psq5 synthesis produced no tensors", __FILE__, __LINE__);
    }
    return source;
}

}
