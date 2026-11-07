#pragma once

#include <shelf/core/memory/tensor.h>
#include <shelf/weights/matrix_weight.h>

#include <cstddef>
#include <cstdint>

namespace shelf::weights {

// Resident bytes of one device tensor: the span the allocation must cover, or 0 when the
// tensor is not backed (a weight a stage or a rank does not hold).
inline std::uint64_t tensor_bytes(const gpu::Tensor& t) {
    if (t.ndim() == 0) return 0u;
    std::size_t bytes = 0u;
    if (!t.required_span_bytes(bytes).ok()) return 0u;
    return static_cast<std::uint64_t>(bytes);
}

// Resident bytes of one matrix weight. A bf16 matrix is accounted by its shape, because the
// memory plan needs the number before the weight is read; a quantized matrix is accounted by
// the planes it actually carries.
inline void add_matrix_bytes(std::uint64_t& total, const MatrixWeight& w) {
    if (w.encoding == MatrixEncoding::Bf16) {
        total += static_cast<std::uint64_t>(w.rows) * w.cols * 2u;
        return;
    }
    total += tensor_bytes(w.codes);
    total += tensor_bytes(w.scales);
    total += tensor_bytes(w.compute_codes);
    total += tensor_bytes(w.compute_scales_bf16);
    total += tensor_bytes(w.exl3_suh);
    total += tensor_bytes(w.exl3_svh);
    total += tensor_bytes(w.exl3_marker);
}

}  // namespace shelf::weights
