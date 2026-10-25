// shelf backend/hip — auxiliary bf16 glue ops (see kernels/aux_ops.hip)
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include <shelf/foundation/numeric.hpp>

namespace shelf::hipk {

hipError_t launch_bf16_row_slice(const shelf::bf16_t* in, shelf::bf16_t* out, uint32_t rows,
                                 uint32_t cols, uint32_t begin, uint32_t in_row_stride,
                                 hipStream_t stream);
hipError_t launch_bf16_sigmoid_gate(const shelf::bf16_t* x, const shelf::bf16_t* z,
                                    shelf::bf16_t* out, uint32_t n, hipStream_t stream);
hipError_t launch_bf16_add(const shelf::bf16_t* a, const shelf::bf16_t* b, shelf::bf16_t* out,
                           uint32_t n, hipStream_t stream);
hipError_t launch_f32_argmax(const float* logits, uint32_t n, uint32_t* out_tok,
                             hipStream_t stream);
hipError_t launch_bf16_to_f32(const shelf::bf16_t* in, float* out, uint32_t n,
                              hipStream_t stream);
hipError_t launch_f32_to_bf16(const float* in, shelf::bf16_t* out, uint32_t n,
                              hipStream_t stream);

} // namespace shelf::hipk
