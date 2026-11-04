#pragma once

#include <shelf/core/status.h>
#include <shelf/runtime/program/program.h>

#include <cstddef>

namespace shelf::execution {

// Host-side contract check for one WeightSlot, before it is copied into the device weight
// table. Every consumer reads the slot through the same fields, so this is where a slot
// that no kernel can consume is rejected.
Status validate_host_weight_slot(const ::shelf::runtime::WeightSlot& slot, std::size_t index);

}
