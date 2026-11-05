#pragma once

#include <cstddef>

namespace shelf::qwen35::runtime {

inline constexpr std::size_t kExl3DenseBytes = 192u << 20;
inline constexpr std::size_t kExl3MaxOutFeatures = 17408u;

inline constexpr std::size_t exl3_dense_accum_bytes(std::size_t rows) noexcept {
    return rows * kExl3MaxOutFeatures * sizeof(float);
}

}  // namespace shelf::qwen35::runtime
