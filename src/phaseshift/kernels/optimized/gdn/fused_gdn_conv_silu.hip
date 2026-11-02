#include <phaseshift/kernels/optimized/gdn/fused_gdn_conv_silu.h>
#include <phaseshift/kernels/optimized/gdn/detail/gdn_prepare_device.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace ps::kernel {

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
    bf16* __restrict__ conv_state,
    const ::ps::runtime::DeviceRequestDescriptor* __restrict__ requests,
    uint32_t actual_rows,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    uint32_t act_row_stride,
    uint32_t conv_dim,
    uint32_t state_index,
    uint32_t max_sequences,
    uint64_t conv_slot_stride,
    uint64_t conv_layer_stride,
    uint64_t conv_history_stride,
    uint32_t row_tile,
    bf16* __restrict__ conv_history_store,
    uint64_t conv_history_row_stride,
    uint32_t capture_rows,
    uint32_t capture_verify_only,
    const int32_t* __restrict__ row_parent) {
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;
    const ::ps::runtime::DeviceRequestDescriptor req = requests[blockIdx.y];
    const uint32_t slot = req.request_handle.slot;
    if (slot >= max_sequences) return;

    const uint32_t t0 = blockIdx.z * row_tile;
    if (t0 >= req.row_count) return;
    const uint32_t remaining = req.row_count - t0;
    const uint32_t t_end = remaining < row_tile ? remaining : row_tile;
    const uint32_t row0 = req.row_begin + t0;

    bf16* state_row = conv_state
                    + static_cast<size_t>(slot) * conv_slot_stride
                    + static_cast<size_t>(state_index) * conv_layer_stride;
    const bf16* w = conv_weight + static_cast<size_t>(ch) * 4u;

    float s0 = 0.f;
    float s1 = 0.f;
    float s2 = 0.f;
    if (row_parent == nullptr) {
        if (t0 == 0u) {
            s0 = __bfloat162float(state_row[ch]);
            s1 = __bfloat162float(state_row[conv_history_stride + ch]);
            s2 = __bfloat162float(state_row[static_cast<size_t>(2) * conv_history_stride + ch]);
        } else {
            s0 = __bfloat162float(
                input[static_cast<size_t>(row0 - 3u) * input_row_stride + ch]);
            s1 = __bfloat162float(
                input[static_cast<size_t>(row0 - 2u) * input_row_stride + ch]);
            s2 = __bfloat162float(
                input[static_cast<size_t>(row0 - 1u) * input_row_stride + ch]);
        }
    }
    const float w0 = __bfloat162float(w[0]);
    const float w1 = __bfloat162float(w[1]);
    const float w2 = __bfloat162float(w[2]);
    const float w3 = __bfloat162float(w[3]);

    for (uint32_t t = 0; t < t_end; ++t) {
        const uint32_t row = row0 + t;
        if (row >= actual_rows) continue;
        if (row_parent != nullptr) {
            const int32_t parent = row_parent[row];
            if (t > 0u && parent == static_cast<int32_t>(row) - 1) {
            } else if (parent < 0) {
                s0 = __bfloat162float(state_row[ch]);
                s1 = __bfloat162float(state_row[conv_history_stride + ch]);
                s2 = __bfloat162float(
                    state_row[static_cast<size_t>(2) * conv_history_stride + ch]);
            } else if (conv_history_store != nullptr &&
                       static_cast<uint32_t>(parent) < capture_rows) {
                const bf16* hp = conv_history_store
                               + static_cast<size_t>(parent) * conv_history_row_stride
                               + static_cast<size_t>(state_index) * conv_layer_stride;
                s0 = __bfloat162float(hp[ch]);
                s1 = __bfloat162float(hp[conv_history_stride + ch]);
                s2 = __bfloat162float(hp[static_cast<size_t>(2) * conv_history_stride + ch]);
            } else {
                s0 = 0.f;
                s1 = 0.f;
                s2 = 0.f;
            }
        }
        const float x = __bfloat162float(
            input[static_cast<size_t>(row) * input_row_stride + ch]);
        const float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
        s0 = s1;
        s1 = s2;
        s2 = x;
        output[static_cast<size_t>(row) * output_row_stride + ch] = y;
        act_output[static_cast<size_t>(row) * act_row_stride + ch] =
            __float2bfloat16(detail::gdn_silu_f32(y));
        const bool verify_only = capture_verify_only != 0u;
        const uint32_t hist_row = row_parent != nullptr
                                      ? row
                                      : (verify_only ? (row - req.row_begin) : row);
        if (conv_history_store != nullptr && hist_row < capture_rows &&
            (!verify_only ||
             req.execution_class ==
                 ::ps::runtime::ExecutionClass::SPEC_VERIFY)) {
            bf16* hp = conv_history_store
                     + static_cast<size_t>(hist_row) * conv_history_row_stride
                     + static_cast<size_t>(state_index) * conv_layer_stride;
            hp[ch] = __float2bfloat16(s0);
            hp[conv_history_stride + ch] = __float2bfloat16(s1);
            hp[static_cast<size_t>(2) * conv_history_stride + ch] = __float2bfloat16(s2);
        }
    }

    if (t0 + t_end >= req.row_count) {
        state_row[ch] = __float2bfloat16(s0);
        state_row[conv_history_stride + ch] = __float2bfloat16(s1);
        state_row[static_cast<size_t>(2) * conv_history_stride + ch] = __float2bfloat16(s2);
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
    const uint32_t row_tile =
        c.row_parent != nullptr ? 0u : conv_silu_row_tile();
    const uint32_t tiled_rows =
        c.max_request_rows != 0u ? c.max_request_rows : c.actual_rows;
    const uint32_t tiles = gdn_conv1d_tiles(row_tile, tiled_rows);
    const uint32_t blocks_x = gdn_conv1d_blocks_x(c.conv_dim);
    dim3 grid(blocks_x, c.num_requests, tiles);
    phaseshift_fused_gdn_conv_silu<<<grid, kConvSiluThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(c.input), c.output,
        reinterpret_cast<bf16*>(args.act_out),
        reinterpret_cast<const bf16*>(c.conv_weight),
        reinterpret_cast<bf16*>(c.conv_state), c.requests,
        c.actual_rows, c.input_row_stride, c.output_row_stride,
        args.act_row_stride,
        c.conv_dim, c.state_index, c.max_sequences,
        c.conv_slot_stride, c.conv_layer_stride, c.conv_history_stride,
        gdn_conv1d_resolved_row_tile(row_tile, c.actual_rows),
        reinterpret_cast<bf16*>(c.conv_history_store), c.conv_history_row_stride,
        c.capture_rows, c.capture_verify_only, c.row_parent);
    return hipGetLastError();
}

}  // namespace ps::kernel
