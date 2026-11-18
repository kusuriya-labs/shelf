#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace shelf::io {
std::vector<int32_t> read_input_ids(const std::string &path);
}
