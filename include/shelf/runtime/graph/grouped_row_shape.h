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

// Upper bound on the number of groups a batch of `rows` tokens can produce: a group holds
// at most kGroupedRowTile slots and never crosses an expert, so the bound is the full
// groups over all slots plus one partial group per expert, clamped to the slot count.
constexpr uint32_t grouped_row_tile_count(uint32_t rows, GroupedRowShape shape) noexcept {
    const uint64_t slots = grouped_slot_count(rows, shape);
    const uint64_t bound = slots / kGroupedRowTile + shape.experts;
    return bound < slots ? static_cast<uint32_t>(bound) : static_cast<uint32_t>(slots);
}

constexpr uint32_t grouped_row_capacity(uint32_t rows, GroupedRowShape shape) noexcept {
    return grouped_row_tile_count(rows, shape) * kGroupedRowTile;
}

}
