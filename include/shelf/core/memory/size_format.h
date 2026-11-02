#pragma once

#include <cstddef>
#include <cstdio>
#include <string>

namespace ps {
namespace gpu {

inline std::string format_gib(std::size_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f GiB",
                  static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return std::string(buf);
}

}  // namespace gpu
}  // namespace ps
