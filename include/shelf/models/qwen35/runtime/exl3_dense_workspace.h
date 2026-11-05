#pragma once

#include <cstddef>

namespace shelf::qwen35::runtime {

inline constexpr std::size_t kExl3DenseBytes = 192u << 20;
inline constexpr std::size_t kExl3XrotBytes = 80u << 20;
inline constexpr std::size_t kExl3MaxOutFeatures = 17408u;

inline constexpr std::size_t exl3_dense_accum_bytes(std::size_t rows) noexcept {
    return rows * kExl3MaxOutFeatures * sizeof(float);
}

// The rotated-domain trellis form keeps the rotated activations, a dense copy of one
// matrix, the split accumulators and the row sums. The program that runs such a linear
// needs them; a program that does not must not hold them, because they come out of the
// same arena the KV pool is sized from.
inline constexpr std::size_t exl3_workspace_bytes(std::size_t rows) noexcept {
    auto align_up = [](std::size_t bytes) noexcept { return (bytes + 255u) & ~std::size_t{255u}; };
    return align_up(kExl3XrotBytes) + align_up(kExl3DenseBytes) +
           align_up(exl3_dense_accum_bytes(rows)) + align_up(rows * sizeof(float));
}

}  // namespace shelf::qwen35::runtime
