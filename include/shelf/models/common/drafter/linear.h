#pragma once

#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::drafter {

struct LinearWorkspace {
    gpu::Tensor codes;
    gpu::Tensor scales;
    gpu::Tensor accum;
};

Status quantized_linear(const weights::MatrixWeight& weight, const bf16_t* input,
                        uint32_t input_stride, bf16_t* output, uint32_t output_stride,
                        uint32_t rows, LinearWorkspace& workspace, hipStream_t stream);

}
