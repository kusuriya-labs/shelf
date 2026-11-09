#pragma once

#include <shelf/kernels/optimized/linear/exl3_int8.h>
#include <shelf/weights/matrix_weight.h>

#include <cstddef>

namespace shelf::qwen35::runtime {

inline constexpr std::size_t kExl3DenseBytes = 192u << 20;
inline constexpr std::size_t kExl3XrotBytes = 80u << 20;
inline constexpr std::size_t kExl3MaxOutFeatures = 17408u;

struct Exl3WorkspaceBytes {
    std::size_t xrot = 0u;
    std::size_t dense = 0u;
    std::size_t accum = 0u;
    std::size_t rowsum = 0u;

    std::size_t total() const noexcept { return xrot + dense + accum + rowsum; }
};

// The EXL3-INT8 prefill path also keeps ::shelf::kernel::exl3_int8_flag_cap(rows) rows of F32 output after the int32 accumulators.
inline constexpr std::size_t exl3_dense_accum_bytes(std::size_t rows,
                                                    std::size_t max_out_features) noexcept {
    return (rows + ::shelf::kernel::exl3_int8_flag_cap(rows)) * max_out_features * sizeof(float);
}

// The rotated-domain trellis form keeps the rotated activations, a dense copy of one
// matrix, the split accumulators and the row sums. A model states the largest linear it
// runs; an empty shape keeps the model-independent upper bound.
inline Exl3WorkspaceBytes exl3_workspace_bytes(
    std::size_t rows, const shelf::weights::Exl3LinearShape& shape = {}) noexcept {
    auto align_up = [](std::size_t bytes) noexcept { return (bytes + 255u) & ~std::size_t{255u}; };
    Exl3WorkspaceBytes out;
    if (shape.present()) {
        out.xrot = align_up(rows * shape.max_k * sizeof(::shelf::bf16_t));
        out.dense = align_up(shape.max_dense_bytes);
        out.accum = align_up(exl3_dense_accum_bytes(rows, shape.max_out_features));
    } else {
        out.xrot = align_up(kExl3XrotBytes);
        out.dense = align_up(kExl3DenseBytes);
        out.accum = align_up(exl3_dense_accum_bytes(rows, kExl3MaxOutFeatures));
    }
    if (out.accum < shape.max_int8_decode_accum_bytes)
        out.accum = align_up(shape.max_int8_decode_accum_bytes);
    out.rowsum = align_up(rows * sizeof(float));
    return out;
}

}  // namespace shelf::qwen35::runtime
