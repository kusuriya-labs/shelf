#pragma once

#include <cstdint>
#include <limits>

namespace ps {
namespace state {

using PageId = uint32_t;
using SequenceSlotId = uint32_t;

inline constexpr PageId kInvalidPageId =
    std::numeric_limits<PageId>::max();

inline constexpr SequenceSlotId kInvalidSlot =
    std::numeric_limits<SequenceSlotId>::max();

}
}
