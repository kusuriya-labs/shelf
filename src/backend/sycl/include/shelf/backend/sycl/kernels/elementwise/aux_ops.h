// shelf backend/sycl — auxiliary bf16 glue ops (see kernels/elementwise/aux_ops.cpp)
// (SYCL port of backend/hip kernels/elementwise/aux_ops.h: hipError_t ->
// shelf::Status, hipStream_t -> Stream.)
#pragma once

#include <cstdint>

#include <shelf/backend/sycl/sycl_api.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

::shelf::Status launch_bf16_row_slice(const shelf::bf16_t* in, shelf::bf16_t* out, uint32_t rows,
                                      uint32_t cols, uint32_t begin, uint32_t in_row_stride,
                                      const ::shelf::sycl::Stream& stream);
::shelf::Status launch_bf16_sigmoid_gate(const shelf::bf16_t* x, const shelf::bf16_t* z,
                                         shelf::bf16_t* out, uint32_t n,
                                         const ::shelf::sycl::Stream& stream);
::shelf::Status launch_bf16_add(const shelf::bf16_t* a, const shelf::bf16_t* b, shelf::bf16_t* out,
                                uint32_t n, const ::shelf::sycl::Stream& stream);
::shelf::Status launch_f32_argmax(const float* logits, uint32_t n, uint32_t* out_tok,
                                  const ::shelf::sycl::Stream& stream);
::shelf::Status launch_bf16_to_f32(const shelf::bf16_t* in, float* out, uint32_t n,
                                   const ::shelf::sycl::Stream& stream);
::shelf::Status launch_f32_to_bf16(const float* in, shelf::bf16_t* out, uint32_t n,
                                   const ::shelf::sycl::Stream& stream);

} // namespace shelf::syclk
