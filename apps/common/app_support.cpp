// shelf apps/common — 読取ヘルパの実装（shelf-run / shelf-bench 共有）。
#include "app_support.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

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

Result<std::vector<bf16_t>> read_bf16(
    const std::map<std::string, artifact::safetensors::Reader*>& where,
    const std::string& name) {
    auto it = where.find(name);
    if (it == where.end())
        return Status::not_found("tensor not in shards: " + name, __FILE__, __LINE__);
    const auto* reader = it->second;
    const auto* entry = reader->catalog().find(name);
    if (!entry)
        return Status::not_found("catalog miss: " + name, __FILE__, __LINE__);
    if (entry->dtype != DataType::bf16)
        return Status::unsupported("not bf16 on disk: " + name, __FILE__, __LINE__);
    const std::size_t bytes = entry->end - entry->begin;
    std::vector<bf16_t> out(bytes / 2);
    SHELF_RETURN_IF_ERROR(reader->read_tensor(name, out.data(), bytes));
    return out;
}

Result<std::vector<float>> read_f32(
    const std::map<std::string, artifact::safetensors::Reader*>& where,
    const std::string& name) {
    auto it = where.find(name);
    if (it == where.end())
        return Status::not_found("tensor not in shards: " + name, __FILE__, __LINE__);
    const auto* reader = it->second;
    const auto* entry = reader->catalog().find(name);
    if (!entry)
        return Status::not_found("catalog miss: " + name, __FILE__, __LINE__);
    const std::size_t bytes = entry->end - entry->begin;
    if (entry->dtype == DataType::f32) {
        std::vector<float> out(bytes / sizeof(float));
        SHELF_RETURN_IF_ERROR(reader->read_tensor(name, out.data(), bytes));
        return out;
    }
    if (entry->dtype == DataType::bf16) {
        std::vector<bf16_t> raw(bytes / 2);
        SHELF_RETURN_IF_ERROR(reader->read_tensor(name, raw.data(), bytes));
        std::vector<float> out(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) out[i] = bf16_to_f32(raw[i]);
        return out;
    }
    return Status::unsupported("read_f32: dtype for " + name, __FILE__, __LINE__);
}

}  // namespace shelf::app
