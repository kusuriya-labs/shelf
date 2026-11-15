#pragma once

#include <shelf/kernels/optimized/linear/exl3_mul1.h>

#include <hip/hip_runtime.h>

namespace shelf::kernel {

hipError_t launch_gemm_exl3_w16a16(const Exl3GemmArgs& args, hipStream_t stream);
hipError_t launch_gemm_exl3_w16a16_pre(const Exl3GemmArgs& args, const void* xrot,
                                       hipStream_t stream);
hipError_t launch_gemm_exl3_w16a16_grouped(const Exl3GroupedArgs& args,
                                           hipStream_t stream);

}  // namespace shelf::kernel
