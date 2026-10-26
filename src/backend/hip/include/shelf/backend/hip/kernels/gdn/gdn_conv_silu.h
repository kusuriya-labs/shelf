// vendored from PhaseShift (own work); namespace/type adapted only.
//
// Fused causal depthwise conv step (kernel width 4) + SiLU. Emits BOTH the raw
// conv result (fp32) and silu(conv) (bf16). shelf's causal_conv_step IR output
// corresponds to the bf16 act_out stream; the f32 output stream is kept for
// parity/debug (PhaseShift contract).
#pragma once

#include <shelf/backend/hip/kernels/gdn/gdn_conv1d.h>
#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::hipk {

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

}  // namespace shelf::hipk
