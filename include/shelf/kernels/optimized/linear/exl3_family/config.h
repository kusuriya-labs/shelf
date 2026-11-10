#pragma once

// EXL3 constants and per-codebook kernel traits. mul1 and mul24 are separate types even
// where the values currently coincide, so one can change without silently changing the
// other. Lengths are in elements and threads, not bytes.

#include <cstdint>

namespace shelf::kernel {

inline constexpr const char* kExl3W16A16Symbol = "shelf_gemm_exl3_w16a16";
inline constexpr uint32_t kExl3W16A16OutBlock = 128u;
inline constexpr uint32_t kExl3W16A16RowTile = 16u;
inline constexpr uint32_t kExl3W16A16Threads = 256u;
inline constexpr uint32_t kExl3GroupedFuse = 2u;

}  // namespace shelf::kernel

namespace shelf::kernel::exl3_detail::mul1 {

struct Traits {
    static constexpr uint32_t kRowTile = 16u;    // output rows per block tile
    static constexpr uint32_t kOutBlock = 128u;  // output columns per block
    static constexpr uint32_t kThreads = 256u;   // threads per block
    static constexpr uint32_t kTrellisTile = 16u;
    static constexpr uint32_t kKBlock = 128u;
};

}  // namespace shelf::kernel::exl3_detail::mul1

namespace shelf::kernel::exl3_detail::mul24 {

struct Traits {
    static constexpr uint32_t kRowTile = 16u;
    static constexpr uint32_t kOutBlock = 128u;
    static constexpr uint32_t kThreads = 256u;
    static constexpr uint32_t kTrellisTile = 16u;
    static constexpr uint32_t kKBlock = 128u;
};

}  // namespace shelf::kernel::exl3_detail::mul24
