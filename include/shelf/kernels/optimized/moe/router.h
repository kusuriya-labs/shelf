#pragma once

#include <shelf/kernels/optimized/linear/config.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// MoE routing: f32 softmax over experts, top-k selection, renormalization, and the
// expert-major slot plan consumed by the grouped linear.
//
// Decision order contract (MoE routing):
//   - probabilities are materialized in f32 as exp(logit - max(logit)) / sum,
//   - the k largest probabilities win; ties go to the smaller expert index (strict >),
//   - the selected weights are divided by their sum, always.
//
// A slot is one (token, selected expert) pair, so there are exactly rows * top_k slots.
// Slots are numbered expert index ascending, and inside one expert by ascending token.
// expert_row_begin[e] .. expert_row_begin[e+1] is expert e's slot range (the array has
// experts + 1 entries and the last one is the slot total).
//
// That slot space is cut into groups of kMxfp4GemmRowTile slots that never cross an
// expert, and the grouped linear works on one group per workgroup. Groups are numbered
// in the same expert-ascending order, so a group index is also monotone in expert index.
//   group_count[0]   number of groups
//   group_expert[g]  expert of group g
//   group_rows[g]    slots in group g (1 .. kMxfp4GemmRowTile)
//   expert_group_begin[e] .. expert_group_begin[e+1] is expert e's group range
//     (experts + 1 entries; group g covers slots
//      expert_row_begin[group_expert[g]] + (g - expert_group_begin[group_expert[g]]) *
//      kMxfp4GemmRowTile ..)
//
// The activation plane and every grouped output are addressed by **group row**, not by
// slot index: group g owns plane rows [g * kMxfp4GemmRowTile, (g + 1) *
// kMxfp4GemmRowTile), which keeps each group inside one fragment-order row tile.
// The plan therefore maps a token's j-th selection straight to its group row:
//   slot_row_of[t * top_k + j]  group row of token t's j-th selection
//   sel_weight[t * top_k + j]   renormalized weight of that selection
// and tells the activation producer which token owns each group row:
//   row_token[r]                token of group row r, -1 for a row no group claims
//     (the padding tail of a group and every row past group_count)
// Because group rows increase with expert index, a consumer that walks a token's
// selections in group row order accumulates expert ascending.
//
// moe_group_row_capacity(rows, experts, top_k) is the host-side bound on the number of
// group rows, and tile_counts (experts * moe_slot_tile_count(rows) entries) is
// accumulated by the router kernel, so the caller must zero it first.
constexpr uint32_t kMoeRouterMaxExperts = 512u;
constexpr uint32_t kMoeRouterMaxTopK = 16u;
constexpr uint32_t kMoeSlotTileTokens = 64u;

constexpr uint32_t moe_slot_tile_count(uint32_t rows) noexcept {
    return (rows + kMoeSlotTileTokens - 1u) / kMoeSlotTileTokens;
}

constexpr uint32_t moe_group_count_upper_bound(uint32_t rows, uint32_t experts,
                                               uint32_t top_k) noexcept {
    const uint64_t slots = static_cast<uint64_t>(rows) * top_k;
    const uint64_t bound = slots / kMxfp4GemmRowTile + experts;
    return bound < slots ? static_cast<uint32_t>(bound) : static_cast<uint32_t>(slots);
}

constexpr uint32_t moe_group_row_capacity(uint32_t rows, uint32_t experts,
                                          uint32_t top_k) noexcept {
    return moe_group_count_upper_bound(rows, experts, top_k) * kMxfp4GemmRowTile;
}

hipError_t launch_moe_router(
    const float* logits,
    uint32_t logits_row_stride,
    int32_t* sel_expert,
    float* sel_weight,
    uint32_t* tile_counts,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    hipStream_t stream);

hipError_t launch_moe_slot_scan(
    const uint32_t* tile_counts,
    uint32_t* expert_row_begin,
    uint32_t* tile_base,
    uint32_t* expert_group_begin,
    uint32_t* group_expert,
    uint32_t* group_rows,
    uint32_t* group_count,
    int32_t* row_token,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    hipStream_t stream);

hipError_t launch_moe_slot_scatter(
    const int32_t* sel_expert,
    const uint32_t* tile_base,
    const uint32_t* expert_row_begin,
    const uint32_t* expert_group_begin,
    int32_t* row_token,
    int32_t* slot_row_of,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    hipStream_t stream);

}  // namespace shelf::kernel
