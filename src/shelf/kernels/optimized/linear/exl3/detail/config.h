#pragma once

// Per-codebook kernel traits. mul1 and mul24 are separate types even where the values
// currently coincide, so one can change without silently changing the other. Lengths
// are in elements and threads, not bytes.

#include <cstdint>

namespace shelf::kernel::exl3_detail {

struct Exl3Mul1Traits {
    static constexpr uint32_t kRowTile = 16u;    // output rows per block tile
    static constexpr uint32_t kOutBlock = 128u;  // output columns per block
    static constexpr uint32_t kThreads = 256u;   // threads per block
    static constexpr uint32_t kTrellisTile = 16u;
    static constexpr uint32_t kKBlock = 128u;
};

struct Exl3Mul24Traits {
    static constexpr uint32_t kRowTile = 16u;
    static constexpr uint32_t kOutBlock = 128u;
    static constexpr uint32_t kThreads = 256u;
    static constexpr uint32_t kTrellisTile = 16u;
    static constexpr uint32_t kKBlock = 128u;
};

}  // namespace shelf::kernel::exl3_detail
