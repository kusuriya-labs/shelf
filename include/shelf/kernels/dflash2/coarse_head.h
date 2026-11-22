#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

inline constexpr uint32_t kDFlash2CoarseMaxRows = 16;
inline constexpr uint32_t kDFlash2CoarseKeep = 128;

hipError_t build_dflash2_coarse_head(const uint16_t* trellis, uint32_t* packed,
                                     uint32_t vocab, uint32_t hidden, hipStream_t stream);

struct DFlash2CoarseHeadArgs {
    const uint32_t* packed;
    const uint16_t* trellis;
    const uint16_t* svh;
    const int8_t* activations;
    const float* scales;
    float* accum;
    float* logits;
    float* maxima;
    int32_t* selected;
    uint32_t rows;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t code_stride;
    uint32_t scale_stride;
    uint32_t keep;
};

hipError_t launch_dflash2_coarse_head(const DFlash2CoarseHeadArgs& args, hipStream_t stream);

}
