// vendored from PhaseShift (own work); namespace/type adapted only.
//
// shelf adaptations (see gdn_conv1d.h for the full note): conv state is fp32
// in shelf's [C,3] channel-major IR layout (PhaseShift: bf16 [3,C]
// history-major); requests are shelf GdnRequestDescriptor; row_parent/history
// capture/SPEC-VERIFY serving branches not ported. Math per element is
// identical: y = w0*s0 + w1*s1 + w2*s2 + w3*x with s0..s2 oldest-first, then
// the history shift s0=s1, s1=s2, s2=x.
// (SYCL port of backend/hip kernels/gdn/gdn_conv1d.hip.)
#include <shelf/backend/sycl/kernels/gdn/gdn_conv1d.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace shelf::syclk {

namespace {

using bf16 = shelf::bf16_t;

constexpr uint32_t kConvThreads = 256;
static_assert(kConvThreads == kGdnConv1dThreads);
constexpr uint32_t kConvRowTile = 32;

uint32_t conv_row_tile() {
    const char* e = std::getenv("PHASESHIFT_GDN_CONV1D_TILE");
    if (e == nullptr) return kConvRowTile;
    const long v = std::strtol(e, nullptr, 10);
    if (v <= 0) return 0;
    return static_cast<uint32_t>(v);
}

} // namespace

uint32_t gdn_conv1d_row_tile() { return conv_row_tile(); }

::shelf::Status launch_gdn_conv1d_bf16_f32(
    const GdnConv1dArgs& args,
    const ::shelf::sycl::Stream& stream) {
    if (args.input == nullptr || args.output == nullptr ||
        args.conv_weight == nullptr || args.conv_state == nullptr ||
        args.requests == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (args.num_requests == 0u || args.actual_rows == 0u || args.conv_dim == 0u)
        return ::shelf::Status::invalid_argument("empty launch rejected", __FILE__, __LINE__);
    if (args.conv_history != 3u)
        return ::shelf::Status::unsupported("conv_history != 3", __FILE__, __LINE__);
    if (args.input_row_stride < args.conv_dim ||
        args.output_row_stride < args.conv_dim)
        return ::shelf::Status::invalid_argument("row stride too small", __FILE__, __LINE__);
    const uint32_t row_tile = conv_row_tile();
    const uint32_t tiled_rows =
        args.max_request_rows != 0u ? args.max_request_rows : args.actual_rows;
    const uint32_t tiles = gdn_conv1d_tiles(row_tile, tiled_rows);
    const uint32_t blocks_x = gdn_conv1d_blocks_x(args.conv_dim);
    const uint32_t rt = gdn_conv1d_resolved_row_tile(row_tile, args.actual_rows);

    const bf16* input = args.input;
    float* output = args.output;
    const bf16* conv_weight = args.conv_weight;
    float* conv_state = args.conv_state;
    const GdnRequestDescriptor* requests = args.requests;
    const uint32_t actual_rows = args.actual_rows;
    const uint32_t input_row_stride = args.input_row_stride;
    const uint32_t output_row_stride = args.output_row_stride;
    const uint32_t conv_dim = args.conv_dim;
    const uint32_t state_index = args.state_index;
    const uint32_t max_sequences = args.max_sequences;
    const uint64_t conv_slot_stride = args.conv_slot_stride;
    const uint64_t conv_layer_stride = args.conv_layer_stride;

    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<3>{
                ::sycl::range<3>{blocks_x * kConvThreads, args.num_requests, tiles},
                ::sycl::range<3>{kConvThreads, 1, 1}},
            [=](::sycl::nd_item<3> it) {
                const uint32_t ch = static_cast<uint32_t>(
                    it.get_group(0) * kConvThreads + it.get_local_linear_id());
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
                    // tile continuation: history comes from this request's
                    // input rows (requires row_begin >= 3; decode launches
                    // one row, t0 always 0).
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
