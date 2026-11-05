#pragma once

#include <shelf/kernels/optimized/linear/exl3.h>
#include <hip/hip_runtime.h>

namespace shelf::execution {

hipError_t exl3_dense_gemm(const shelf::kernel::Exl3GemmArgs& args, const void* dense,
                           void* accum, hipStream_t stream);

}  // namespace shelf::execution
