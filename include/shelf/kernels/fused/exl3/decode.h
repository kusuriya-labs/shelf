#pragma once

#include <shelf/kernels/optimized/linear/exl3_family/types.h>

namespace shelf::kernel {

hipError_t launch_gemm_exl3_int8_grouped(const Exl3GroupedArgs& args, hipStream_t stream);

}
