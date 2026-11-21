#pragma once


#include <cstdint>

namespace shelf::kernel {

inline constexpr uint32_t kExl3W16A16OutBlock = 128u;
inline constexpr uint32_t kExl3W16A16RowTile = 16u;
inline constexpr uint32_t kExl3W16A16Threads = 256u;
inline constexpr uint32_t kExl3GroupedFuse = 2u;

}  // namespace shelf::kernel

namespace shelf::kernel::exl3_detail::mul24 {

struct Traits {
    static constexpr uint32_t kRowTile = 16u;
    static constexpr uint32_t kOutBlock = 128u;
    static constexpr uint32_t kThreads = 256u;
    static constexpr uint32_t kTrellisTile = 16u;
    static constexpr uint32_t kKBlock = 128u;
};

}  // namespace shelf::kernel::exl3_detail::mul24
