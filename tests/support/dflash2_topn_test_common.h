#pragma once

#include <shelf/kernels/optimized/radix_topn.h>

#include <cstdint>

namespace dflash2_topn_test {

struct Rng {
    uint32_t state = 0x1234567u;

    uint32_t next_u32() {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }

    float unit() {
        return static_cast<float>(next_u32()) / static_cast<float>(1u << 24);
    }

    float sym() {
        return unit() * 2.0f - 1.0f;
    }

    uint32_t below(uint32_t n) {
        return next_u32() % (n == 0u ? 1u : n);
    }
};

}  // namespace dflash2_topn_test
