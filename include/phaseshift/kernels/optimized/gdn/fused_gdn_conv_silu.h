#pragma once

#include <phaseshift/kernels/optimized/gdn/conv1d.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

struct FusedGdnConvSiluArgs {
    GdnConv1dArgs conv{};
    bf16_t* act_out = nullptr;
    uint32_t act_row_stride = 0;
};

inline constexpr const char* kFusedGdnConvSiluSymbol =
    "phaseshift_fused_gdn_conv_silu";

hipError_t launch_fused_gdn_conv_silu(
    const FusedGdnConvSiluArgs& args,
    hipStream_t stream);

}  // namespace ps::kernel
