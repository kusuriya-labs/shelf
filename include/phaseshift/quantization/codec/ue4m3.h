#pragma once
#include <cstdint>

namespace ps::quantization::codec {

// Finite unsigned E4M3 (UE4M3) scale helpers. Valid scale bytes are 0x00..0x7e.
bool ue4m3_is_valid(uint8_t e);
float ue4m3_to_fp32(uint8_t e);
// Nearest valid scale byte to `target` (ties keep the lower byte). Returns 0
// when target is not a positive finite number.
uint8_t ue4m3_nearest_scale(float target);

}  // namespace ps::quantization::codec
