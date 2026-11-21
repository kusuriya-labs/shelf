#pragma once

#include <shelf/kernels/optimized/linear/exl3_mul24.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_exl3_mul24_grouped(const Exl3GemmArgs& args, uint32_t n_tiles,
                                     uint32_t grid_x, uint32_t grid_z, hipStream_t stream);

}  // namespace shelf::kernel
