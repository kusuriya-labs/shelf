// shelf artifact/safetensors — BF16-era safetensors reader (plan PR-004).
//
// Container interpretation ONLY: this reader never knows about model
// semantics, quantized codecs or devices. Layout facts follow the
// safetensors spec: <u64 header_len><JSON header><data region>.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <shelf/artifact/common/raw_catalog.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::artifact::safetensors {

class Reader {
public:
    static Result<Reader> open(const std::string& path);

    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    ~Reader();

    const RawTensorCatalog& catalog() const noexcept { return catalog_; }
    std::vector<std::string> list_tensors() const;

    // Copies the full tensor payload. dst_bytes must match exactly — an exact
    // match catches spec/size misunderstandings at the boundary (use
    // catalog().find() first when unsure).
    Status read_tensor(const std::string& name, void* dst, std::size_t dst_bytes) const;

    // Typed convenience: validates dtype is bf16 and sizes exactly.
    Status read_tensor_bf16(const std::string& name, std::vector<bf16_t>& out) const;

    // Parsed "__metadata__" strings (empty if absent).
    const std::map<std::string, std::string>& metadata() const noexcept { return metadata_; }

    // File offset where the data region begins (8 + header_len).
    std::uint64_t data_base_offset() const noexcept { return data_base_offset_; }
    std::uint64_t file_size() const noexcept { return file_size_; }

private:
    Reader() = default;
    std::string path_;
    RawTensorCatalog catalog_;
    std::map<std::string, std::string> metadata_;
    std::uint64_t data_base_offset_ = 0;
    std::uint64_t file_size_ = 0;
};

} // namespace shelf::artifact::safetensors
