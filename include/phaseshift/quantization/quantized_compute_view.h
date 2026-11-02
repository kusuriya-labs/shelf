#pragma once
#include <cstdint>

namespace ps::quantization {

struct alignas(16) QuantizedComputeView {
    const void* codes = nullptr;
    const void* scales_bf16 = nullptr;
    uint32_t k_padded = 0;
    uint32_t codes_row_stride_bytes = 0;
    uint32_t scale_row_stride_bytes = 0;
    uint32_t weight_scale_group = 0;
    uint32_t rows = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};

static_assert(alignof(QuantizedComputeView) >= 16);
static_assert(sizeof(QuantizedComputeView) % 16 == 0);

}
