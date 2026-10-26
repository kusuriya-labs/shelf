// vendored from PhaseShift (own work); namespace/type adapted only.
//
// Fused causal depthwise conv step (kernel width 4) + SiLU. Emits BOTH the raw
// conv result (fp32) and silu(conv) (bf16). shelf's causal_conv_step IR output
// corresponds to the bf16 act_out stream; the f32 output stream is kept for
// parity/debug (PhaseShift contract).
// (SYCL port of backend/hip kernels/gdn/gdn_conv_silu.h.)
#pragma once

#include <cstdint>

#include <shelf/backend/sycl/kernels/gdn/gdn_conv1d.h>
#include <shelf/backend/sycl/sycl_api.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

struct FusedGdnConvSiluArgs {
    GdnConv1dArgs conv{};
    bf16_t* act_out = nullptr;
    uint32_t act_row_stride = 0;
};

::shelf::Status launch_fused_gdn_conv_silu(
    const FusedGdnConvSiluArgs& args,
    const ::shelf::sycl::Stream& stream);

} // namespace shelf::syclk
