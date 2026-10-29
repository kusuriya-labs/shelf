// shelf apps/common — 読取ヘルパの実装（shelf-run / shelf-bench 共有）。
#include "app_support.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include <shelf/quant/fp8_w8a8.hpp>
#ifdef SHELF_HAVE_HIP
#include <shelf/backend/hip/hip_api.hpp>
#include <shelf/runtime/boot/probe.hpp>
#endif


namespace shelf::app {

std::string read_file_str(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::size_t numel(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

Status open_shards(const std::string& model_dir, ShardSet* out) {
    if (out == nullptr)
        return Status::invalid_argument("open_shards: null out", __FILE__, __LINE__);
    const std::string index_path = model_dir + "/model.safetensors.index.json";
    const std::string text = read_file_str(index_path);
    if (text.empty())
        return Status::not_found("cannot read " + index_path, __FILE__, __LINE__);

    nlohmann::json index;
    try {
        index = nlohmann::json::parse(text);
    } catch (const std::exception& e) {
        return Status::invalid_argument(std::string("parse index.json: ") + e.what(),
                                        __FILE__, __LINE__);
    }
    if (!index.contains("weight_map") || !index.at("weight_map").is_object())
        return Status::invalid_argument("index.json: missing weight_map", __FILE__, __LINE__);

    std::vector<std::string> shard_files;
    for (const auto& f : index.at("weight_map").items()) {
        if (!f.value().is_string()) continue;
        const std::string file = f.value().get<std::string>();
        if (std::find(shard_files.begin(), shard_files.end(), file) == shard_files.end())
            shard_files.push_back(file);
    }
    if (shard_files.empty())
        return Status::invalid_argument("index.json: empty weight_map", __FILE__, __LINE__);

    out->readers.clear();
    out->where.clear();
    out->readers.reserve(shard_files.size());
    for (const auto& file : shard_files) {
        auto r = artifact::safetensors::Reader::open(model_dir + "/" + file);
        if (!r.ok())
            return Status::invalid_argument("open " + file + ": " + r.status().message(),
                                            __FILE__, __LINE__);
        out->readers.push_back(
            std::make_unique<artifact::safetensors::Reader>(r.release()));
    }
    // reader の実体は unique_ptr で固定（vector のムーブでポインタが死なない）。
    for (auto& rp : out->readers)
        for (const auto& name : rp->list_tensors())
            if (!out->where.contains(name)) out->where.emplace(name, rp.get());
    return Status::make_ok();
}

Result<runtime::load::TensorBlob> ShardSource::read(const std::string& name) const {
    auto it = where_.find(name);
    if (it == where_.end())
        return Status::not_found("tensor not in shards: " + name, __FILE__, __LINE__);
    const auto* reader = it->second;
    const auto* entry = reader->catalog().find(name);
    if (entry == nullptr)
        return Status::not_found("catalog miss: " + name, __FILE__, __LINE__);
    runtime::load::TensorBlob blob;
    blob.dtype = entry->dtype;
    blob.bytes.resize(static_cast<std::size_t>(entry->end - entry->begin));
    if (!blob.bytes.empty()) {
        SHELF_RETURN_IF_ERROR(
            reader->read_tensor(name, blob.bytes.data(), blob.bytes.size()));
    }
    return blob;
}

Result<std::vector<bf16_t>> read_bf16(
    const std::map<std::string, artifact::safetensors::Reader*>& where,
    const std::string& name) {
    return runtime::load::read_bf16(ShardSource(where), name);
}

Result<std::vector<float>> read_f32(
    const std::map<std::string, artifact::safetensors::Reader*>& where,
    const std::string& name) {
    return runtime::load::read_f32(ShardSource(where), name);
}

Result<std::vector<std::uint8_t>> read_u8(
    const std::map<std::string, artifact::safetensors::Reader*>& where,
    const std::string& name) {
    return runtime::load::read_u8(ShardSource(where), name);
}

WeightKind classify(const std::map<std::string, artifact::safetensors::Reader*>& where,
                    const std::string& name) {
    return runtime::load::classify(ShardSource(where), name);
}

Result<weights::TensorCatalog> build_catalog(const ShardSet& shards) {
    std::vector<artifact::RawTensorEntry> raw;
    for (const auto& r : shards.readers) {
        const auto& cat = r->catalog();
        raw.insert(raw.end(), cat.entries().begin(), cat.entries().end());
    }
    if (raw.empty())
        return Status::invalid_state("shard にテンソルが 1 つも無い", __FILE__, __LINE__);
    return weights::TensorCatalog::from_raw(raw);
}

namespace {

const char* dtype_label(DataType dt) {
    switch (dt) {
    case DataType::bf16: return "bf16";
    case DataType::f32: return "f32";
    case DataType::fp8_e4m3: return "F8_E4M3";
    case DataType::u8: return "u8";
    case DataType::i32: return "i32";
    default: return "?";
    }
}

std::size_t shape_elems(const TensorShape& s) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < s.rank(); ++i) n *= s.dim(i);
    return n;
}

std::size_t shape_elems_of(const std::vector<std::size_t>& s) {
    std::size_t n = 1;
    for (const auto d : s) n *= d;
    return n;
}

}  // namespace

Result<MaterializationPlan> plan_materialization(const weights::TensorCatalog& catalog,
                                                 const model::qwen35::StepGraphPlan& plan) {
    MaterializationPlan out;
    out.requests.reserve(plan.weights.size());
    for (const auto& w : plan.weights) {
        const auto* spec = catalog.find(w.checkpoint_name);
        if (spec == nullptr)
            return Status::not_found("checkpoint に無いテンソル: " + w.checkpoint_name,
                                     __FILE__, __LINE__);
        // 次数が同じなら次元ごとに一致を要求する。次数が違う場合は要素数だけを見る:
        // checkpoint はコンテナ側の軸を残すことがある（例: conv1d の [C, 1, K] に
        // 対して IR の論理形状は [C, K]）。同じペイロードを別の形で見ているだけ。
        if (spec->shape.rank() == w.shape.size()) {
            for (std::size_t i = 0; i < w.shape.size(); ++i) {
                if (spec->shape.dim(i) != w.shape[i])
                    return Status::invalid_state(
                        "形状が違う: " + w.checkpoint_name + "（ファイル " +
                            std::to_string(spec->shape.dim(i)) + " / IR " +
                            std::to_string(w.shape[i]) + "）",
                        __FILE__, __LINE__);
            }
        } else if (shape_elems(spec->shape) != shape_elems_of(w.shape)) {
            return Status::invalid_state(
                "要素数が違う: " + w.checkpoint_name + "（ファイル " +
                    std::to_string(shape_elems(spec->shape)) + " / IR " +
                    std::to_string(shape_elems_of(w.shape)) + "）",
                __FILE__, __LINE__);
        }
        switch (spec->logical_dtype) {
        case DataType::bf16: ++out.bf16; break;
        case DataType::f32: ++out.f32; break;
        case DataType::fp8_e4m3: ++out.fp8; break;
        default:
            return Status::unsupported(
                "実体化できない dtype: " + w.checkpoint_name + "（" +
                    std::string(dtype_label(spec->logical_dtype)) + "）",
                __FILE__, __LINE__);
        }
        out.bytes += shape_elems(spec->shape) * element_size(spec->logical_dtype);
        out.requests.push_back(runtime::load::WeightRequest{w.checkpoint_name, w.shape});
    }
    out.summary = "bf16 " + std::to_string(out.bf16) + " / f32 " + std::to_string(out.f32) +
                  " / fp8 " + std::to_string(out.fp8) + " / " +
                  std::to_string(out.bytes >> 20) + " MiB";
    return out;
}

#ifdef SHELF_HAVE_HIP
shelf::runtime::boot::Probe probe_devices(bool want_device_execution) {
    shelf::runtime::boot::Probe p;
    p.cpu_reference = true;
    p.backend = want_device_execution ? "hip" : "cpu";
    if (!want_device_execution) return p;
    auto devs = shelf::hip::enumerate_devices();
    if (devs.ok()) p.devices = devs.release();
    // 失敗は「デバイス 0 台」として扱う（evaluate が理由付きで拒否する）。
    return p;
}
#endif

Result<Fp8ModelFormat> read_fp8_format(const std::string& config_json_text) {
    return runtime::load::read_fp8_format(config_json_text);
}

Result<Fp8WeightPayload> build_fp8_weight(
    const std::map<std::string, artifact::safetensors::Reader*>& where, const std::string& name,
    const std::vector<std::size_t>& shape, const Fp8ModelFormat& fmt) {
    runtime::load::WeightRequest req{name, shape};
    return runtime::load::materialize(ShardSource(where), req, fmt);
}

}  // namespace shelf::app
