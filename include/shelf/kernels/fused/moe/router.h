#pragma once

#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/runtime/graph/grouped_row_shape.h>

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
constexpr uint32_t kMoeSlotTileTokens = ::shelf::runtime::kGroupedSlotTileRows;

// A selection whose expert this device does not own. The routed-expert split is along the
// expert axis, so every rank selects the same global top-k but only materializes the slots
// it owns. sel_weight stays renormalized over the whole selection, which is what makes the
// ranks' partial sums equal the unsplit result.
constexpr int32_t kMoeSlotNotOwned = -1;

static_assert(kMxfp4GemmRowTile == ::shelf::runtime::kGroupedRowTile,
              "a group is one grouped GEMM row tile, so the plan and the GEMM tile must match");

// The plan capacities are the runtime's grouped row contract: the same numbers size the
// program workspace, so the launcher and the allocator cannot drift apart.
constexpr uint32_t moe_slot_tile_count(uint32_t rows) noexcept {
    return ::shelf::runtime::grouped_slot_tile_count(rows);
}

constexpr uint32_t moe_group_count_upper_bound(uint32_t rows, uint32_t experts,
                                               uint32_t top_k) noexcept {
    return ::shelf::runtime::grouped_row_tile_count(rows, {experts, top_k});
}

constexpr uint32_t moe_group_row_capacity(uint32_t rows, uint32_t experts,
                                          uint32_t top_k) noexcept {
    return ::shelf::runtime::grouped_row_capacity(rows, {experts, top_k});
}

// A row whose logits hold a value that is not finite (NaN, +INF or -INF) is a numeric
// anomaly: the Qwen3.5 MoE router puts no -INF mask on its logits, so -INF is an anomaly too.
// The kernel keeps the plan usable in that case (the selected experts stay in
// [0, experts), the weights stay finite, and a row with no finite logit takes the lowest
// expert indices with weight 0) and reports the anomaly into error_word with
// ProgramStatus::NUMERIC_ERROR. The fallback exists so a broken model cannot fault the
// context; it does not make the result usable, and the caller has to stop the run.
// error_word may be null, which skips the report.
hipError_t launch_moe_router(
    const float* logits,
    uint32_t logits_row_stride,
    int32_t* sel_expert,
    float* sel_weight,
    uint32_t* tile_counts,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    // The expert range this device owns. `experts` is always the logits width, because the
    // softmax and the top-k selection have to be the same on every rank. A selection inside
    // the owned range is written as its index in that range, and one outside it as
    // kMoeSlotNotOwned, so the plan arrays are indexed by the local expert count
    // `owned_experts` and tile_counts holds owned_experts * moe_slot_tile_count(rows) entries.
    uint32_t expert_begin,
    uint32_t owned_experts,
    uint32_t* error_word,
    hipStream_t stream);

hipError_t launch_moe_slot_scan(
    uint32_t* tile_counts,
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
