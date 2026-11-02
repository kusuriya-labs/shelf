#pragma once

#include <shelf/core/status.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace ps::io {

enum class SsdReadPath {
    none = 0,
    hipfile = 1,
    posix = 2,
};

const char* ssd_read_path_name(SsdReadPath path) noexcept;

struct SsdReadResult {
    SsdReadPath path = SsdReadPath::none;
    std::uint64_t bytes_read = 0;
    bool short_read = false;
};

bool hipfile_available() noexcept;

class HipFileReader {
 public:
    HipFileReader() noexcept = default;
    ~HipFileReader() noexcept;

    HipFileReader(HipFileReader&& other) noexcept;
    HipFileReader& operator=(HipFileReader&& other) noexcept;
    HipFileReader(const HipFileReader&) = delete;
    HipFileReader& operator=(const HipFileReader&) = delete;

    Status open(const std::string& path, bool odirect);
    Status register_buffer(const void* device_ptr, std::size_t bytes);
    void deregister_buffer() noexcept;
    Status read(void* device_ptr, std::size_t bytes, std::uint64_t file_offset,
                std::size_t buffer_offset, SsdReadResult& out);
    void shutdown() noexcept;
    bool opened() const noexcept;
    bool buffer_registered() const noexcept;

 private:
    void close_fd() noexcept;

    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace ps::io
