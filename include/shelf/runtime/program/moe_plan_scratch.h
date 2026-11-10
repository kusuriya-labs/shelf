#pragma once

#include <shelf/runtime/graph/grouped_row_shape.h>

#include <cstdint>

namespace shelf::runtime {

// The device-side scratch one MoE routing round produces and consumes: the top-k selection
// and the group plan. It is not a value (it carries no features and its rows are slots and
// groups, not tokens), so the program owns it as one workspace range laid out here. Every
// consumer derives its array pointers from this same layout, so the plan cannot be read at
// a different offset than it was written.
//
// Offsets and sizes are in uint32 units from the start of the range. The arrays that the
// kernels read as int32 or float are the same width, and the base of a workspace range is
// aligned, so every array is aligned for its own element type.
struct MoePlanScratch {
    uint32_t tokens = 0;
    uint32_t experts = 0;
    uint32_t top_k = 0;

    uint32_t slots = 0;
    uint32_t slot_tiles = 0;
    uint32_t group_capacity = 0;
    uint32_t row_capacity = 0;

    uint32_t sel_expert_offset = 0;
    uint32_t sel_weight_offset = 0;
    uint32_t tile_counts_offset = 0;
    uint32_t expert_row_begin_offset = 0;
    uint32_t tile_base_offset = 0;
    uint32_t expert_group_begin_offset = 0;
    uint32_t group_expert_offset = 0;
    uint32_t group_rows_offset = 0;
    uint32_t group_count_offset = 0;
    uint32_t row_token_offset = 0;
    uint32_t slot_row_of_offset = 0;

    uint32_t capacity_units = 0;

    static MoePlanScratch make(uint32_t tokens, uint32_t experts, uint32_t top_k) noexcept {
        MoePlanScratch s;
        s.tokens = tokens;
        s.experts = experts;
        s.top_k = top_k;
        if (tokens == 0u || experts == 0u || top_k == 0u) return s;

        const GroupedRowShape shape{experts, top_k};
        s.slots = static_cast<uint32_t>(grouped_slot_count(tokens, shape));
        s.slot_tiles = grouped_slot_tile_count(tokens);
        s.group_capacity = grouped_row_tile_count(tokens, shape);
        s.row_capacity = grouped_row_capacity(tokens, shape);

        const uint32_t count_arrays = experts * s.slot_tiles;
        uint32_t at = 0;
        s.sel_expert_offset = at;
        at += s.slots;
        s.sel_weight_offset = at;
        at += s.slots;
        s.tile_counts_offset = at;
        at += count_arrays;
        s.expert_row_begin_offset = at;
        at += experts + 1u;
        s.tile_base_offset = at;
        at += count_arrays;
        s.expert_group_begin_offset = at;
        at += experts + 1u;
        s.group_expert_offset = at;
        at += s.group_capacity;
        s.group_rows_offset = at;
        at += s.group_capacity;
        s.group_count_offset = at;
        at += 1u;
        s.row_token_offset = at;
        at += s.row_capacity;
        s.slot_row_of_offset = at;
        at += s.slots;
        s.capacity_units = at;
        return s;
    }

    uint64_t bytes() const noexcept {
        return static_cast<uint64_t>(capacity_units) * sizeof(uint32_t);
    }

    uint64_t mul24_offset() const noexcept { return (bytes() + 255u) & ~uint64_t{255u}; }

    static uint64_t mul24_bytes(uint32_t rows, uint32_t k) noexcept {
        return static_cast<uint64_t>(rows) * (static_cast<uint64_t>(k) + (k / 128u) * sizeof(float));
    }

    bool valid() const noexcept {
        return tokens != 0u && experts != 0u && top_k != 0u && capacity_units != 0u;
    }
};

}
