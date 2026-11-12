#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

struct PlamoExl3Projection {
    const uint16_t *trellis = nullptr;
    const uint16_t *suh = nullptr;
    const uint16_t *svh = nullptr;
    int8_t *codes = nullptr;
    float *scales = nullptr;
    float *yprime = nullptr;
    bf16_t *out = nullptr;
    uint32_t n = 0;
    uint32_t splits = 0;
    uint32_t out_stride = 0;
};

struct PlamoExl3ProjectionGroup {
    const bf16_t *input = nullptr;
    PlamoExl3Projection p[3];
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t input_stride = 0;
    uint32_t scale_stride_bytes = 0;
    uint32_t count = 0;
};

hipError_t launch_plamo_exl3_projection(const PlamoExl3ProjectionGroup &args, uint32_t bits,
                                        hipStream_t stream);

}
