#pragma once
#include <phaseshift/core/status.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace ps::quantization::bundle {

inline Status read_file_text(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return Status::invalid_argument("cannot open file", __FILE__, __LINE__);
    }
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return Status::make_ok();
}

// A shard path must stay relative and must not escape the bundle root.
inline Status validate_shard_path(const std::string& shard) {
    const std::filesystem::path path(shard);
    if (shard.empty() || path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        return Status::invalid_argument("bundle shard path must be relative", __FILE__, __LINE__);
    }
    for (const std::filesystem::path& component : path) {
        if (component == "..") {
            return Status::invalid_argument("bundle shard path escapes bundle root", __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

}