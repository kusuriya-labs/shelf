#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/io/safetensors_reader.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ps {
namespace io {

class SafetensorsWriter {
 public:
    // Opens/creates the output file (truncated) for writing a safetensors file.
    static Result<SafetensorsWriter> create(const std::string& path);

    ~SafetensorsWriter() noexcept;
    SafetensorsWriter(SafetensorsWriter&& other) noexcept;
    SafetensorsWriter& operator=(SafetensorsWriter&& other) noexcept;
    SafetensorsWriter(const SafetensorsWriter&) = delete;
    SafetensorsWriter& operator=(const SafetensorsWriter&) = delete;

    // Registers a tensor's header entry and reserves its payload region.
    // All plan_tensor calls must happen before write_header().
    Status plan_tensor(const std::string& name, SType dtype,
                       const std::vector<std::size_t>& shape);

    // Sets a "__metadata__" key/value entry (values are stored as strings).
    Status set_metadata(const std::string& key, const std::string& value);

    // Serializes the header, writes [8B header_len][header] to the file start,
    // and returns the absolute payload base offset where tensor data begins.
    Result<std::uint64_t> write_header();

    // Writes tensor payload at its planned region. Must be called after
    // write_header(). Exactly data_bytes (the planned size) must be supplied.
    Status write_tensor(const std::string& name, const void* data, std::size_t bytes);

    // Writes a sub-range of a tensor's payload at offset_in_tensor. Enables
    // bounded-memory streaming without buffering a whole tensor. Must stay
    // within the tensor's planned byte range.
    Status write_tensor_chunk(const std::string& name, const void* data,
                              std::size_t offset_in_tensor, std::size_t bytes);

    // Finalizes the file (fsync + close). Returns the index JSON if a
    // weight_map is provided, otherwise an empty string.
    Result<std::string> finish();

 private:
    SafetensorsWriter() noexcept : pimpl_(nullptr) {}

    struct Impl;
    Impl* pimpl_;
};

}
}
