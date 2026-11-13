#pragma once
#include <cstdint>

namespace shelf::execution {
struct BoundaryDeferred {
    const float* yprime = nullptr;
    const uint16_t* svh = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t output_stride = 0;
    uint32_t splits = 0;
};
}
