#pragma once

#include <shelf/kernels/optimized/linear/exl3_mul1.h>

#include <hip/hip_runtime.h>

namespace shelf::kernel {

hipError_t launch_epilogue_exl3_dense(const Exl3GemmArgs& args, const void* rowsum,
                                      const void* accum, hipStream_t stream);

}  // namespace shelf::kernel
