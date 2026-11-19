#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace shelf::hy4::snapshot {
using Digest = std::array<uint8_t, 32>;
Digest digest(std::span<const uint8_t> data);
void save(std::span<const uint8_t> data);
std::vector<uint8_t> load(size_t limit);
}
