#pragma once

// mul1 (W16A16) decode constants. The shared trellis layout and FWHT variants live in
// trellis.h / fwht.h; the mul1 decode loops stay in exl3.hip.

#include "fwht.h"
#include "trellis.h"

namespace shelf::kernel::exl3_detail {

constexpr float kMul1Inv = 0.00676727294921875f;
constexpr float kMul1Bias = -10.3828125f;
constexpr float kMul1BiasOverInv = kMul1Bias / kMul1Inv;

}  // namespace shelf::kernel::exl3_detail
