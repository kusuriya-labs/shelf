#pragma once

#include <cstdint>

namespace shelf::runtime {

// The row space of a grouped GEMM. A grouped consumer owns one tile of output rows, and a
// plan never lets a group cross a tile, so the rows of an activation plane and of a grouped
// output are counted in group rows, not in tokens. The count grows with the expert count
// because every expert can claim a partial tile.
constexpr uint32_t kGroupedRowTile = 16u;

// The router accumulates slot counts per (expert, token tile), so token tiles are the unit
// of the plan's count arrays.
constexpr uint32_t kGroupedSlotTileRows = 64u;

struct GroupedRowShape {
    uint32_t experts = 0u;
    uint32_t top_k = 0u;
};

constexpr uint64_t grouped_slot_count(uint32_t rows, GroupedRowShape shape) noexcept {
    return static_cast<uint64_t>(rows) * shape.top_k;
}

constexpr uint32_t grouped_slot_tile_count(uint32_t rows) noexcept {
    return (rows + kGroupedSlotTileRows - 1u) / kGroupedSlotTileRows;
}

// Upper bound on the number of groups a batch of `rows` tokens can produce: a group holds at
// most kGroupedRowTile slots and never crosses an expert. Summing ceil(c_e / tile) over
// experts, the slots wasted in a partial group are at most tile - 1 per expert, which gives
// (slots + experts * (tile - 1)) / tile. Two more constraints bind: a token never selects the
// same expert twice (c_e <= rows, so one expert holds at most ceil(rows / tile) groups), and a
// group is never empty (groups <= slots). This is the exact maximum over routings, not a slack
// bound; tests/unit/test_grouped_row_shape.cpp enumerates it.
constexpr uint32_t grouped_row_tile_count(uint32_t rows, GroupedRowShape shape) noexcept {
    const uint64_t slots = grouped_slot_count(rows, shape);
    const uint64_t waste_bound =
        (slots + static_cast<uint64_t>(shape.experts) * (kGroupedRowTile - 1u)) / kGroupedRowTile;
    const uint64_t per_expert_bound =
        static_cast<uint64_t>(shape.experts) *
        ((rows + kGroupedRowTile - 1u) / kGroupedRowTile);
    uint64_t bound = waste_bound < per_expert_bound ? waste_bound : per_expert_bound;
    if (slots < bound) bound = slots;
    return static_cast<uint32_t>(bound);
}

constexpr uint32_t grouped_row_capacity(uint32_t rows, GroupedRowShape shape) noexcept {
    return grouped_row_tile_count(rows, shape) * kGroupedRowTile;
}

}
