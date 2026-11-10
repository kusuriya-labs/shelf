#include <shelf/kernels/optimized/linear/detail/workspace.h>
#include <shelf/kernels/optimized/linear/exl3.h>

#include <algorithm>
#include <cstdint>

namespace shelf::kernel::exl3_detail {
namespace {

inline uint64_t align256(uint64_t x) noexcept {
    return (x + 255ull) & ~255ull;
}

}  // namespace

Exl3Mul24WorkspaceLayout make_mul24_workspace_layout(
    uint32_t rows, uint32_t n, uint32_t k,
    uint64_t xrot_capacity, uint64_t accum_capacity, uint64_t dense_capacity) {
    Exl3Mul24WorkspaceLayout l;
    const uint32_t groups = k / 128u;
    l.code_stride = k;
    l.scale_stride_bytes = (groups * 4u + 15u) & ~15u;
    const uint64_t per_row_in = static_cast<uint64_t>(l.code_stride) + l.scale_stride_bytes;
    const uint64_t rows_in = per_row_in == 0u ? 0u : xrot_capacity / per_row_in;
    const uint64_t rows_out = n == 0u ? 0u : accum_capacity / (static_cast<uint64_t>(n) * 4u);
    const uint64_t chunk64 = std::min<uint64_t>(std::min<uint64_t>(rows_in, rows_out), rows);
    l.chunk_rows = static_cast<uint32_t>(chunk64);
    l.scratch_rows = static_cast<uint32_t>(std::min<uint64_t>(rows_out, 0xFFFFFFFFull));
    l.dense_bytes = static_cast<uint64_t>(k) * n;
    if (l.chunk_rows == 0u) {
        l.fits = false;
        return l;
    }
    const uint32_t chunk = l.chunk_rows;
    l.flag_capacity = static_cast<uint32_t>(::shelf::kernel::exl3_mul24_flag_cap(chunk));
    l.codes_offset = 0u;
    l.scales_offset = align256(static_cast<uint64_t>(chunk) * l.code_stride);
    l.slots_offset = align256(l.scales_offset + static_cast<uint64_t>(chunk) * l.scale_stride_bytes);
    l.count_offset = l.slots_offset + static_cast<uint64_t>(chunk) * 4u;
    l.flagged_codes_offset = align256(l.count_offset + 4u);
    l.flagged_scales_offset =
        align256(l.flagged_codes_offset + static_cast<uint64_t>(l.flag_capacity) * l.code_stride);
    l.xrot_bytes =
        l.flagged_scales_offset + static_cast<uint64_t>(l.flag_capacity) * l.scale_stride_bytes;
    l.yprime_offset = 0u;
    l.flagged_yprime_offset = align256(static_cast<uint64_t>(chunk) * n * 4u);
    l.accum_bytes = l.flagged_yprime_offset + static_cast<uint64_t>(l.flag_capacity) * n * 4u;
    l.fits = l.xrot_bytes <= xrot_capacity && l.accum_bytes <= accum_capacity &&
             l.dense_bytes <= dense_capacity;
    return l;
}

}  // namespace shelf::kernel::exl3_detail
