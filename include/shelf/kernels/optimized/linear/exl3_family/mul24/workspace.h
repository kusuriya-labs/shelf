#pragma once

// mul24 workspace layout. Pure size/offset arithmetic: no allocation, no device access.
// The offsets are relative to two distinct arenas:
//   xrot  : codes, scales, slots + count, flagged codes/scales
//   accum : yprime (int32/f32 partials), flagged yprime2
// Every region starts on a 256-byte boundary, matching the previous hand calculation.

#include <cstdint>

namespace shelf::kernel::exl3_detail::mul24 {

struct WorkspaceLayout {
    // xrot arena
    uint64_t codes_offset = 0;
    uint64_t scales_offset = 0;
    uint64_t slots_offset = 0;
    uint64_t count_offset = 0;
    uint64_t flagged_codes_offset = 0;
    uint64_t flagged_scales_offset = 0;
    uint64_t xrot_bytes = 0;
    // accum arena
    uint64_t yprime_offset = 0;
    uint64_t flagged_yprime_offset = 0;
    uint64_t accum_bytes = 0;
    // dense trellis expansion (one int8 per weight)
    uint64_t dense_bytes = 0;
    // sizing
    uint32_t chunk_rows = 0;
    uint32_t scratch_rows = 0;
    uint32_t flag_capacity = 0;
    uint32_t code_stride = 0;
    uint32_t scale_stride_bytes = 0;
    bool fits = false;
};

// `xrot_capacity`, `accum_capacity` and `dense_capacity` are the arena sizes in bytes.
// `chunk_rows` is the largest row count whose codes + scales + flagged buffers fit the
// xrot arena and whose F32 output fits the accum arena, capped by `rows`.
WorkspaceLayout make_workspace_layout(
    uint32_t rows, uint32_t n, uint32_t k,
    uint64_t xrot_capacity, uint64_t accum_capacity, uint64_t dense_capacity);

}  // namespace shelf::kernel::exl3_detail::mul24
