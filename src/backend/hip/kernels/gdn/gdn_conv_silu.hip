// vendored from PhaseShift (own work); namespace/type adapted only.
//
// shelf adaptations (see gdn_conv1d.h note): fp32 conv state in shelf [C,3]
// channel-major layout; shelf GdnRequestDescriptor; serving-only branches
// (row_parent trees, history capture, SPEC-VERIFY) not ported. Conv math and
// SiLU (x / (1 + expf(-x))) identical to PhaseShift.
#include <shelf/backend/hip/kernels/gdn/gdn_conv_silu.h>
#include <shelf/backend/hip/kernels/detail/gdn_prepare_device.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace shelf::hipk {

namespace {

using bf16 = __hip_bfloat16;

constexpr uint32_t kConvSiluThreads = 256;
static_assert(kConvSiluThreads == kGdnConv1dThreads);
constexpr uint32_t kConvSiluRowTile = 32;

uint32_t conv_silu_row_tile() {
    const char* e = std::getenv("PHASESHIFT_GDN_CONV1D_TILE");
    if (e == nullptr) return kConvSiluRowTile;
    const long v = std::strtol(e, nullptr, 10);
    if (v <= 0) return 0;
    return static_cast<uint32_t>(v);
}

}  // namespace

extern "C" __global__ __launch_bounds__(kConvSiluThreads)
void phaseshift_fused_gdn_conv_silu(
    const bf16* __restrict__ input,
    float* __restrict__ output,
    bf16* __restrict__ act_output,
    const bf16* __restrict__ conv_weight,
    float* __restrict__ conv_state,
    const GdnRequestDescriptor* __restrict__ requests,
    uint32_t actual_rows,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    uint32_t act_row_stride,
    uint32_t conv_dim,
    uint32_t state_index,
    uint32_t max_sequences,
    uint64_t conv_slot_stride,
    uint64_t conv_layer_stride,
    uint32_t row_tile) {
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;
    const GdnRequestDescriptor req = requests[blockIdx.y];
    const uint32_t slot = req.slot;
    if (slot >= max_sequences) return;

    const uint32_t t0 = blockIdx.z * row_tile;
    if (t0 >= req.row_count) return;
    const uint32_t remaining = req.row_count - t0;
    const uint32_t t_end = remaining < row_tile ? remaining : row_tile;
    const uint32_t row0 = req.row_begin + t0;

    // shelf conv state: [slot][layer][C][3] fp32, h=0 oldest.
    float* state_row = conv_state
                     + static_cast<size_t>(slot) * conv_slot_stride
                     + static_cast<size_t>(state_index) * conv_layer_stride
                     + static_cast<size_t>(ch) * 3u;
    const bf16* w = conv_weight + static_cast<size_t>(ch) * 4u;

    float s0 = 0.f;
    float s1 = 0.f;
    float s2 = 0.f;
    if (t0 == 0u) {
        s0 = state_row[0];
        s1 = state_row[1];
        s2 = state_row[2];
    } else {
        s0 = __bfloat162float(
            input[static_cast<size_t>(row0 - 3u) * input_row_stride + ch]);
        s1 = __bfloat162float(
            input[static_cast<size_t>(row0 - 2u) * input_row_stride + ch]);
        s2 = __bfloat162float(
            input[static_cast<size_t>(row0 - 1u) * input_row_stride + ch]);
    }
    const float w0 = __bfloat162float(w[0]);
    const float w1 = __bfloat162float(w[1]);
    const float w2 = __bfloat162float(w[2]);
    const float w3 = __bfloat162float(w[3]);

    for (uint32_t t = 0; t < t_end; ++t) {
        const uint32_t row = row0 + t;
        if (row >= actual_rows) continue;
        const float x = __bfloat162float(
            input[static_cast<size_t>(row) * input_row_stride + ch]);
        const float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
        s0 = s1;
        s1 = s2;
        s2 = x;
        output[static_cast<size_t>(row) * output_row_stride + ch] = y;
        act_output[static_cast<size_t>(row) * act_row_stride + ch] =
            __float2bfloat16(detail::gdn_silu_f32(y));
    }

    if (t0 + t_end >= req.row_count) {
        state_row[0] = s0;
        state_row[1] = s1;
        state_row[2] = s2;
    }
}

hipError_t launch_fused_gdn_conv_silu(
    const FusedGdnConvSiluArgs& args,
    hipStream_t stream) {
    const GdnConv1dArgs& c = args.conv;
    if (c.input == nullptr || c.output == nullptr ||
        c.conv_weight == nullptr || c.conv_state == nullptr ||
        c.requests == nullptr)
        return hipErrorInvalidValue;
    if (args.act_out == nullptr)
        return hipErrorInvalidValue;
    if (c.num_requests == 0u || c.actual_rows == 0u || c.conv_dim == 0u)
        return hipErrorInvalidValue;
    if (c.conv_history != 3u)
        return hipErrorNotSupported;
    if (c.input_row_stride < c.conv_dim || c.output_row_stride < c.conv_dim ||
        args.act_row_stride < c.conv_dim)
        return hipErrorInvalidValue;
    const uint32_t row_tile = conv_silu_row_tile();
    const uint32_t tiled_rows =
        c.max_request_rows != 0u ? c.max_request_rows : c.actual_rows;
    const uint32_t tiles = gdn_conv1d_tiles(row_tile, tiled_rows);
    const uint32_t blocks_x = gdn_conv1d_blocks_x(c.conv_dim);
    dim3 grid(blocks_x, c.num_requests, tiles);
    phaseshift_fused_gdn_conv_silu<<<grid, kConvSiluThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(c.input), c.output,
        reinterpret_cast<bf16*>(args.act_out),
        reinterpret_cast<const bf16*>(c.conv_weight), c.conv_state,
        c.requests,
        c.actual_rows, c.input_row_stride, c.output_row_stride,
        args.act_row_stride,
        c.conv_dim, c.state_index, c.max_sequences,
        c.conv_slot_stride, c.conv_layer_stride,
        gdn_conv1d_resolved_row_tile(row_tile, c.actual_rows));
    return hipGetLastError();
}

}  // namespace shelf::hipk
