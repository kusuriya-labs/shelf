// shelf artifact/safetensors — implementation.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <utility>

#include <nlohmann/json.hpp>

#include <shelf/artifact/safetensors/reader.hpp>

namespace shelf::artifact::safetensors {
namespace {

bool parse_u64_le(const std::uint8_t* p, std::uint64_t& out) {
    out = 0;
    for (int i = 0; i < 8; ++i) out |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return true;
}

bool map_dtype(const std::string& s, DataType& out) {
    if (s == "F64")  { out = DataType::f64;  return true; }
    if (s == "F32")  { out = DataType::f32;  return true; }
    if (s == "F16")  { out = DataType::fp16; return true; }
    if (s == "BF16") { out = DataType::bf16; return true; }
    if (s == "I64")  { out = DataType::i64;  return true; }
    if (s == "I32")  { out = DataType::i32;  return true; }
    if (s == "U8")   { out = DataType::u8;   return true; }
    return false;
}

} // namespace

Result<Reader> Reader::open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return Status::not_found("cannot open artifact: " + path, __FILE__, __LINE__);

    std::uint8_t len_buf[8];
    f.read(reinterpret_cast<char*>(len_buf), 8);
    if (f.gcount() != 8) {
        return Status::invalid_argument("file shorter than header length field", __FILE__, __LINE__);
    }
    std::uint64_t header_len = 0;
    parse_u64_le(len_buf, header_len);

    f.seekg(0, std::ios::end);
    const std::uint64_t file_size = static_cast<std::uint64_t>(f.tellg());
    if (header_len == 0 || 8 + header_len > file_size || header_len > (1ull << 30)) {
        return Status::invalid_argument("implausible safetensors header length", __FILE__, __LINE__);
    }

    std::vector<char> header(header_len);
    f.seekg(8);
    f.read(header.data(), static_cast<std::streamsize>(header_len));
    if (static_cast<std::uint64_t>(f.gcount()) != header_len) {
        return Status::invalid_argument("truncated safetensors header", __FILE__, __LINE__);
    }

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(header.begin(), header.end());
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(std::string("header JSON error: ") + e.what(),
                                        __FILE__, __LINE__);
    }
    if (!root.is_object()) {
        return Status::invalid_argument("safetensors header is not an object", __FILE__, __LINE__);
    }

    Reader r;
    r.path_ = path;
    r.file_size_ = file_size;
    r.data_base_offset_ = 8 + header_len;
    const std::uint64_t data_size = file_size - r.data_base_offset_;

    std::vector<RawTensorEntry> parsed;
    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string& key = it.key();
        if (key == "__metadata__") {
            if (!it.value().is_object()) {
                return Status::invalid_argument("__metadata__ is not an object", __FILE__, __LINE__);
            }
            for (auto mit = it.value().begin(); mit != it.value().end(); ++mit) {
                if (!mit.value().is_string()) {
                    return Status::invalid_argument("__metadata__ value is not a string",
                                                    __FILE__, __LINE__);
                }
                r.metadata_[mit.key()] = mit.value().get<std::string>();
            }
            continue;
        }
        const nlohmann::json& meta = it.value();
        if (!meta.is_object()) {
            return Status::invalid_argument("tensor meta is not an object: " + key,
                                            __FILE__, __LINE__);
        }
        RawTensorEntry entry;
        entry.name = key;

        const std::string dt = meta.contains("dtype") && meta["dtype"].is_string()
                                   ? meta["dtype"].get<std::string>()
                                   : std::string();
        if (!map_dtype(dt, entry.dtype)) {
            return Status::unsupported("unsupported dtype '" + dt + "' for tensor " + key,
                                       __FILE__, __LINE__);
        }

        if (!meta.contains("shape") || !meta["shape"].is_array()) {
            return Status::invalid_argument("missing shape for tensor " + key, __FILE__, __LINE__);
        }
        std::vector<std::size_t> dims;
        for (const auto& d : meta["shape"]) {
            if (!d.is_number_unsigned()) {
                return Status::invalid_argument("shape dims must be unsigned: " + key,
                                                __FILE__, __LINE__);
            }
            dims.push_back(d.get<std::size_t>());
        }
        auto shape = TensorShape::create(dims.data(), dims.size());
        if (!shape.ok()) {
            return Status::invalid_argument("bad shape for tensor " + key + ": " +
                                                shape.status().message(), __FILE__, __LINE__);
        }
        entry.shape = shape.release();

        if (!meta.contains("data_offsets") || !meta["data_offsets"].is_array() ||
            meta["data_offsets"].size() != 2 || !meta["data_offsets"][0].is_number_unsigned() ||
            !meta["data_offsets"][1].is_number_unsigned()) {
            return Status::invalid_argument("bad data_offsets for tensor " + key, __FILE__, __LINE__);
        }
        entry.begin = meta["data_offsets"][0].get<std::uint64_t>();
        entry.end = meta["data_offsets"][1].get<std::uint64_t>();
        if (entry.end > data_size) {
            return Status::invalid_argument("payload range exceeds data region: " + key,
                                            __FILE__, __LINE__);
        }
        parsed.push_back(std::move(entry));
    }

    // Overlap check on sorted ranges.
    std::sort(parsed.begin(), parsed.end(),
              [](const RawTensorEntry& a, const RawTensorEntry& b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < parsed.size(); ++i) {
        if (parsed[i].begin < parsed[i - 1].end) {
            return Status::invalid_argument("overlapping payload ranges: " + parsed[i - 1].name +
                                                " / " + parsed[i].name, __FILE__, __LINE__);
        }
    }
    for (const auto& entry : parsed) {
        SHELF_RETURN_IF_ERROR(r.catalog_.add(entry));  // shape/size/dup validation
    }
    return r;
}

Reader::Reader(Reader&&) noexcept = default;
Reader& Reader::operator=(Reader&&) noexcept = default;
Reader::~Reader() = default;

std::vector<std::string> Reader::list_tensors() const {
    std::vector<std::string> names;
    names.reserve(catalog_.size());
    for (const auto& e : catalog_.entries()) names.push_back(e.name);
    return names;
}

Status Reader::read_tensor(const std::string& name, void* dst, std::size_t dst_bytes) const {
    const RawTensorEntry* entry = catalog_.find(name);
    if (entry == nullptr) {
        return Status::not_found("tensor not found: " + name, __FILE__, __LINE__);
    }
    if (dst == nullptr) {
        return Status::invalid_argument("null destination", __FILE__, __LINE__);
    }
    const std::size_t needed = static_cast<std::size_t>(entry->end - entry->begin);
    if (dst_bytes != needed) {
        return Status::insufficient_memory(
            "exact destination size required for " + name +
                " (got " + std::to_string(dst_bytes) + ", need " + std::to_string(needed) + ")",
            __FILE__, __LINE__);
    }
    std::ifstream f(path_, std::ios::binary);
    if (!f) return Status::not_found("cannot reopen artifact: " + path_, __FILE__, __LINE__);
    f.seekg(static_cast<std::streamoff>(data_base_offset_ + entry->begin));
    f.read(static_cast<char*>(dst), static_cast<std::streamsize>(needed));
    if (static_cast<std::size_t>(f.gcount()) != needed) {
        return Status::invalid_state("short read for tensor " + name, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status Reader::read_tensor_bf16(const std::string& name, std::vector<bf16_t>& out) const {
    const RawTensorEntry* entry = catalog_.find(name);
    if (entry == nullptr) {
        return Status::not_found("tensor not found: " + name, __FILE__, __LINE__);
    }
    if (entry->dtype != DataType::bf16) {
        return Status::invalid_state("read_tensor_bf16 called on non-bf16 tensor " + name,
                                     __FILE__, __LINE__);
    }
    auto n = entry->shape.num_elements();
    SHELF_RETURN_IF_ERROR(n.status());
    out.resize(n.value());
    return read_tensor(name, out.data(), out.size() * sizeof(bf16_t));
}

} // namespace shelf::artifact::safetensors
