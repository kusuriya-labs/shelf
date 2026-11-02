#include <shelf/io/safetensors_writer.h>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace ps {
namespace io {

namespace {

const char* dtype_name(SType dt) {
    return stype_name(dt);
}

std::size_t elem_size(SType dt) {
    const std::size_t esz = stype_elem_size(dt);
    return esz != 0 ? esz : 2;
}

}

struct PlannedTensor {
    std::string name;
    SType dtype = SType::BF16;
    std::vector<std::size_t> shape;
    std::uint64_t payload_begin = 0;
    std::uint64_t payload_end = 0;
};

struct SafetensorsWriter::Impl {
    std::string path;
    FILE* file = nullptr;
    std::vector<PlannedTensor> tensors;
    std::map<std::string, std::string> metadata;
    std::map<std::string, std::size_t> name_index;
    std::uint64_t cursor = 0;
    std::uint64_t payload_offset = 0;
    bool header_written = false;
    bool finished = false;
};

Result<SafetensorsWriter> SafetensorsWriter::create(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "w+b");
    if (!f) {
        return Status::invalid_argument("cannot open safetensors for writing", __FILE__, __LINE__);
    }
    auto pimpl = std::make_unique<Impl>();
    pimpl->path = path;
    pimpl->file = f;

    SafetensorsWriter w;
    w.pimpl_ = pimpl.release();
    return w;
}

SafetensorsWriter::~SafetensorsWriter() noexcept {
    if (pimpl_) {
        if (pimpl_->file) {
            std::fclose(pimpl_->file);
        }
        delete pimpl_;
    }
}

SafetensorsWriter::SafetensorsWriter(SafetensorsWriter&& other) noexcept : pimpl_(other.pimpl_) {
    other.pimpl_ = nullptr;
}

SafetensorsWriter& SafetensorsWriter::operator=(SafetensorsWriter&& other) noexcept {
    if (this != &other) {
        delete pimpl_;
        pimpl_ = other.pimpl_;
        other.pimpl_ = nullptr;
    }
    return *this;
}

Status SafetensorsWriter::plan_tensor(const std::string& name, SType dtype,
                                      const std::vector<std::size_t>& shape) {
    if (pimpl_->header_written) {
        return Status::invalid_argument("plan_tensor after write_header", __FILE__, __LINE__);
    }
    if (name == "__metadata__") {
        return Status::invalid_argument("reserved tensor name", __FILE__, __LINE__);
    }
    if (pimpl_->name_index.find(name) != pimpl_->name_index.end()) {
        return Status::invalid_argument("duplicate tensor name", __FILE__, __LINE__);
    }

    std::size_t elements = 1;
    for (const std::size_t d : shape) {
        if (d == 0) {
            return Status::invalid_argument("zero dimension", __FILE__, __LINE__);
        }
        if (elements > (std::numeric_limits<std::size_t>::max() / d)) {
            return Status::invalid_argument("shape overflow", __FILE__, __LINE__);
        }
        elements *= d;
    }
    const std::size_t esz = elem_size(dtype);
    if (elements > (std::numeric_limits<std::size_t>::max() / esz)) {
        return Status::invalid_argument("element count overflow", __FILE__, __LINE__);
    }
    const std::uint64_t bytes = static_cast<std::uint64_t>(elements) * esz;
    if (pimpl_->cursor > (std::numeric_limits<std::uint64_t>::max() - bytes)) {
        return Status::invalid_argument("payload size overflow", __FILE__, __LINE__);
    }

    PlannedTensor pt;
    pt.name = name;
    pt.dtype = dtype;
    pt.shape = shape;
    pt.payload_begin = pimpl_->cursor;
    pt.payload_end = pimpl_->cursor + bytes;
    pimpl_->cursor = pt.payload_end;

    pimpl_->name_index[name] = pimpl_->tensors.size();
    pimpl_->tensors.push_back(std::move(pt));
    return Status::make_ok();
}

Status SafetensorsWriter::set_metadata(const std::string& key, const std::string& value) {
    if (pimpl_->header_written) {
        return Status::invalid_argument("set_metadata after write_header", __FILE__, __LINE__);
    }
    pimpl_->metadata[key] = value;
    return Status::make_ok();
}

Result<std::uint64_t> SafetensorsWriter::write_header() {
    if (pimpl_->header_written) {
        return Status::invalid_argument("write_header called twice", __FILE__, __LINE__);
    }

    nlohmann::json header = nlohmann::json::object();
    if (!pimpl_->metadata.empty()) {
        header["__metadata__"] = pimpl_->metadata;
    }
    for (const auto& pt : pimpl_->tensors) {
        nlohmann::json entry;
        entry["dtype"] = dtype_name(pt.dtype);
        entry["shape"] = pt.shape;
        entry["data_offsets"] = {pt.payload_begin, pt.payload_end};
        header[pt.name] = std::move(entry);
    }

    std::string header_text = header.dump();
    const std::uint64_t header_len = static_cast<std::uint64_t>(header_text.size());

    if (std::fseek(pimpl_->file, 0, SEEK_SET) != 0) {
        return Status::invalid_argument("seek failed", __FILE__, __LINE__);
    }
    if (std::fwrite(&header_len, 1, 8, pimpl_->file) != 8) {
        return Status::invalid_argument("write header_len failed", __FILE__, __LINE__);
    }
    if (std::fwrite(header_text.data(), 1, header_text.size(), pimpl_->file) != header_text.size()) {
        return Status::invalid_argument("write header failed", __FILE__, __LINE__);
    }

    const std::uint64_t payload_offset = 8 + header_len;
    pimpl_->payload_offset = payload_offset;
    pimpl_->header_written = true;
    return payload_offset;
}

Status SafetensorsWriter::write_tensor(const std::string& name, const void* data, std::size_t bytes) {
    if (!pimpl_->header_written) {
        return Status::invalid_argument("write_tensor before write_header", __FILE__, __LINE__);
    }
    auto it = pimpl_->name_index.find(name);
    if (it == pimpl_->name_index.end()) {
        return Status::invalid_argument("tensor not planned", __FILE__, __LINE__);
    }
    const PlannedTensor& pt = pimpl_->tensors[it->second];
    const std::uint64_t expected = pt.payload_end - pt.payload_begin;
    if (static_cast<std::uint64_t>(bytes) != expected) {
        return Status::invalid_argument("tensor size mismatch", __FILE__, __LINE__);
    }

    const std::uint64_t abs_offset = pimpl_->payload_offset + pt.payload_begin;
    if (std::fseek(pimpl_->file, static_cast<long>(abs_offset), SEEK_SET) != 0) {
        return Status::invalid_argument("seek failed", __FILE__, __LINE__);
    }
    if (std::fwrite(data, 1, bytes, pimpl_->file) != bytes) {
        return Status::invalid_argument("write tensor data failed", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status SafetensorsWriter::write_tensor_chunk(const std::string& name, const void* data,
                                             std::size_t offset_in_tensor, std::size_t bytes) {
    if (!pimpl_->header_written) {
        return Status::invalid_argument("write_tensor_chunk before write_header", __FILE__, __LINE__);
    }
    auto it = pimpl_->name_index.find(name);
    if (it == pimpl_->name_index.end()) {
        return Status::invalid_argument("tensor not planned", __FILE__, __LINE__);
    }
    const PlannedTensor& pt = pimpl_->tensors[it->second];
    const std::uint64_t expected = pt.payload_end - pt.payload_begin;
    if (static_cast<std::uint64_t>(offset_in_tensor) + bytes > expected) {
        return Status::invalid_argument("chunk out of tensor bounds", __FILE__, __LINE__);
    }

    const std::uint64_t abs_offset = pimpl_->payload_offset + pt.payload_begin + offset_in_tensor;
    if (std::fseek(pimpl_->file, static_cast<long>(abs_offset), SEEK_SET) != 0) {
        return Status::invalid_argument("seek failed", __FILE__, __LINE__);
    }
    if (std::fwrite(data, 1, bytes, pimpl_->file) != bytes) {
        return Status::invalid_argument("write chunk failed", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Result<std::string> SafetensorsWriter::finish() {
    if (pimpl_->finished) {
        return Status::invalid_argument("finish called twice", __FILE__, __LINE__);
    }
    if (std::fflush(pimpl_->file) != 0) {
        return Status::invalid_argument("flush failed", __FILE__, __LINE__);
    }
    if (std::fclose(pimpl_->file) != 0) {
        pimpl_->file = nullptr;
        return Status::invalid_argument("close failed", __FILE__, __LINE__);
    }
    pimpl_->file = nullptr;
    pimpl_->finished = true;
    return std::string();
}

}
}
