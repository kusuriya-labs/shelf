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
struct Exl3AttnProducer {
    const float* input;
    const bf16_t* gate;
    uint32_t input_stride;
    uint32_t gate_stride;
};
struct Exl3GdnProducer {
    const float* input;
    const bf16_t* gate;
    const bf16_t* weight;
    uint32_t input_stride;
    uint32_t gate_stride;
    float eps;
};
hipError_t launch_exl3_attn_rotate_rows(const Exl3AttnProducer&, const uint16_t*, int8_t*, float*,
                                       uint32_t, uint32_t, uint32_t, uint32_t, hipStream_t);
hipError_t launch_exl3_gdn_rotate_rows(const Exl3GdnProducer&, const uint16_t*, int8_t*, float*,
                                      uint32_t, uint32_t, uint32_t, uint32_t, hipStream_t);
hipError_t launch_exl3_swiglu_rotate_rows(const Exl3SwigluProducer&, const uint16_t*, int8_t*, float*,
                                           uint32_t, uint32_t, uint32_t, uint32_t, hipStream_t);
}
