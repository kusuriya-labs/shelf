// vendored from PhaseShift (own work); namespace/type adapted only.
//
// shelf adaptations (see gdn_conv1d.h note): fp32 conv state in shelf [C,3]
// channel-major layout; shelf GdnRequestDescriptor; serving-only branches
// (row_parent trees, history capture, SPEC-VERIFY) not ported. Conv math and
// SiLU (x / (1 + expf(-x))) identical to PhaseShift.
// (SYCL port of backend/hip kernels/gdn/gdn_conv_silu.hip.)
#include <shelf/backend/sycl/kernels/gdn/gdn_conv_silu.h>
#include <shelf/backend/sycl/kernels/detail/gdn_prepare_device.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace shelf::syclk {

namespace {

using bf16 = shelf::bf16_t;

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

} // namespace

::shelf::Status launch_fused_gdn_conv_silu(
    const FusedGdnConvSiluArgs& args,
    const ::shelf::sycl::Stream& stream) {
    const GdnConv1dArgs& c = args.conv;
    if (c.input == nullptr || c.output == nullptr ||
        c.conv_weight == nullptr || c.conv_state == nullptr ||
        c.requests == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (args.act_out == nullptr)
        return ::shelf::Status::invalid_argument("null act_out", __FILE__, __LINE__);
    if (c.num_requests == 0u || c.actual_rows == 0u || c.conv_dim == 0u)
        return ::shelf::Status::invalid_argument("empty launch rejected", __FILE__, __LINE__);
    if (c.conv_history != 3u)
        return ::shelf::Status::unsupported("conv_history != 3", __FILE__, __LINE__);
    if (c.input_row_stride < c.conv_dim || c.output_row_stride < c.conv_dim ||
        args.act_row_stride < c.conv_dim)
        return ::shelf::Status::invalid_argument("row stride too small", __FILE__, __LINE__);
    const uint32_t row_tile = conv_silu_row_tile();
    const uint32_t tiled_rows =
        c.max_request_rows != 0u ? c.max_request_rows : c.actual_rows;
    const uint32_t tiles = gdn_conv1d_tiles(row_tile, tiled_rows);
    const uint32_t blocks_x = gdn_conv1d_blocks_x(c.conv_dim);
    const uint32_t rt = gdn_conv1d_resolved_row_tile(row_tile, c.actual_rows);

    const bf16* input = c.input;
    float* output = c.output;
    bf16* act_output = args.act_out;
    const bf16* conv_weight = c.conv_weight;
    float* conv_state = c.conv_state;
    const GdnRequestDescriptor* requests = c.requests;
    const uint32_t actual_rows = c.actual_rows;
    const uint32_t input_row_stride = c.input_row_stride;
    const uint32_t output_row_stride = c.output_row_stride;
    const uint32_t act_row_stride = args.act_row_stride;
    const uint32_t conv_dim = c.conv_dim;
    const uint32_t state_index = c.state_index;
    const uint32_t max_sequences = c.max_sequences;
    const uint64_t conv_slot_stride = c.conv_slot_stride;
    const uint64_t conv_layer_stride = c.conv_layer_stride;

    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<3>{
                ::sycl::range<3>{blocks_x * kConvSiluThreads, c.num_requests, tiles},
                ::sycl::range<3>{kConvSiluThreads, 1, 1}},
            [=](::sycl::nd_item<3> it) {
                const uint32_t ch = static_cast<uint32_t>(
                    it.get_group(0) * kConvSiluThreads + it.get_local_linear_id());
                if (ch >= conv_dim) return;
                const GdnRequestDescriptor req =
                    requests[static_cast<uint32_t>(it.get_group(1))];
                const uint32_t slot = req.slot;
                if (slot >= max_sequences) return;

                const uint32_t t0 = static_cast<uint32_t>(it.get_group(2)) * rt;
                if (t0 >= req.row_count) return;
                const uint32_t remaining = req.row_count - t0;
                const uint32_t t_end = remaining < rt ? remaining : rt;
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
                    s0 = ::shelf::bf16_to_f32(
                        input[static_cast<size_t>(row0 - 3u) * input_row_stride + ch]);
                    s1 = ::shelf::bf16_to_f32(
                        input[static_cast<size_t>(row0 - 2u) * input_row_stride + ch]);
                    s2 = ::shelf::bf16_to_f32(
                        input[static_cast<size_t>(row0 - 1u) * input_row_stride + ch]);
                }
                const float w0 = ::shelf::bf16_to_f32(w[0]);
                const float w1 = ::shelf::bf16_to_f32(w[1]);
                const float w2 = ::shelf::bf16_to_f32(w[2]);
                const float w3 = ::shelf::bf16_to_f32(w[3]);

                for (uint32_t t = 0; t < t_end; ++t) {
                    const uint32_t row = row0 + t;
                    if (row >= actual_rows) continue;
                    const float x = ::shelf::bf16_to_f32(
                        input[static_cast<size_t>(row) * input_row_stride + ch]);
                    const float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
                    s0 = s1;
                    s1 = s2;
                    s2 = x;
                    output[static_cast<size_t>(row) * output_row_stride + ch] = y;
                    act_output[static_cast<size_t>(row) * act_row_stride + ch] =
                        ::shelf::f32_to_bf16(detail::gdn_silu_f32(y));
                }

                if (t0 + t_end >= req.row_count) {
                    state_row[0] = s0;
                    state_row[1] = s1;
                    state_row[2] = s2;
                }
            });
    });
}

} // namespace shelf::syclk
