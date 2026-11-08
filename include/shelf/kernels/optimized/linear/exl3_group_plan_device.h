#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/exl3.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel::exl3_detail {

// Routed-expert group plan: turns args into the view of the (fused) groups this block owns. Returns false when the
// block has no work. group_index = blockIdx.z; with expert_group_begin, only the first group of each kGroups stride
// of an expert does the work for the whole stride (its rows are the sum of the stride's groups).
template <uint32_t kGroups>
__device__ __forceinline__ bool resolve_group_plan(Exl3GemmArgs& args, uint32_t group_index,
                                                   uint32_t& r0, uint32_t& active_rows) {
    constexpr uint32_t kRows = kExl3W16A16RowTile * kGroups;
    if (group_index >= args.group_count[0]) return false;
    const uint32_t expert = args.group_expert[group_index];
    uint32_t rows_here = args.group_rows[group_index];
    if (args.expert_group_begin != nullptr) {
        const uint32_t begin = args.expert_group_begin[expert];
        const uint32_t end = args.expert_group_begin[expert + 1u];
        if (((group_index - begin) % kGroups) != 0u) return false;
        const uint32_t chunk_end = (group_index + kGroups) < end ? (group_index + kGroups) : end;
        rows_here = 0u;
        for (uint32_t gi = group_index; gi < chunk_end; ++gi) rows_here += args.group_rows[gi];
    }
    if (rows_here == 0u) return false;
    args.trellis = static_cast<const std::uint8_t*>(args.trellis) +
                   static_cast<std::uint64_t>(expert) * args.trellis_batch_stride;
    args.suh = static_cast<const std::uint8_t*>(args.suh) +
               static_cast<std::uint64_t>(expert) * args.suh_batch_stride;
    args.svh = static_cast<const std::uint8_t*>(args.svh) +
               static_cast<std::uint64_t>(expert) * args.svh_batch_stride;
    const std::uint64_t output_offset =
        static_cast<std::uint64_t>(group_index) * kExl3W16A16RowTile * args.output_row_stride;
    if (args.row_token == nullptr) {
        const std::uint64_t input_offset =
            static_cast<std::uint64_t>(group_index) * kExl3W16A16RowTile * args.input_row_stride;
        args.input = static_cast<const shelf::bf16_t*>(args.input) + input_offset;
    }
    args.output = args.output_dtype == 0u
                      ? static_cast<void*>(static_cast<shelf::bf16_t*>(args.output) + output_offset)
                      : static_cast<void*>(static_cast<float*>(args.output) + output_offset);
    args.rows = rows_here;
    r0 = 0u;
    active_rows = rows_here < kRows ? rows_here : kRows;
    return true;
}

}  // namespace shelf::kernel::exl3_detail
