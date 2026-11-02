#include <phaseshift/quantization/codec/ue4m3.h>
#include <cmath>

namespace ps::quantization::codec {

namespace {

constexpr float scale_value(uint8_t e) {
    const uint32_t expf = e >> 3;
    const uint32_t man = e & 7;
    if (expf == 0) {
        return static_cast<float>(man) / 1024.0f;
    }
    if (expf >= 11) {
        return static_cast<float>(8 + man) * static_cast<float>(1u << (expf - 11));
    }
    return static_cast<float>(8 + man) / static_cast<float>(1u << (11 - expf));
}

}  // namespace

bool ue4m3_is_valid(uint8_t e) {
    return e <= 0x7e;
}

float ue4m3_to_fp32(uint8_t e) {
    return ue4m3_is_valid(e) ? scale_value(e) : 0.0f;
}

uint8_t ue4m3_nearest_scale(float target) {
    if (!(target > 0.0f) || !std::isfinite(target)) {
        return 0;
    }
    int lo = 1;
    int hi = 126;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (ue4m3_to_fp32(static_cast<uint8_t>(mid)) < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo == 1) {
        return 1;
    }
    const float hi_scale = ue4m3_to_fp32(static_cast<uint8_t>(lo));
    const float lo_scale = ue4m3_to_fp32(static_cast<uint8_t>(lo - 1));
    return (target - lo_scale <= hi_scale - target)
               ? static_cast<uint8_t>(lo - 1)
               : static_cast<uint8_t>(lo);
}

}  // namespace ps::quantization::codec
