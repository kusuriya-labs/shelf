#pragma once
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {
struct Exl3SwigluProducer {
    const bf16_t* gate;
    const bf16_t* up;
    uint32_t gate_stride;
    uint32_t up_stride;
    bool intermediate;
};
hipError_t launch_exl3_swiglu_rotate_rows(const Exl3SwigluProducer&, const uint16_t*, int8_t*, float*,
                                           uint32_t, uint32_t, uint32_t, uint32_t, hipStream_t);
}
