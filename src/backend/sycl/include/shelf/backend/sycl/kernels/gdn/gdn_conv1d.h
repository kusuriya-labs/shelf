// vendored from PhaseShift (own work); namespace/type adapted only.
//
// Layout/type adaptations vs PhaseShift (math unchanged):
//   * conv state element type: PhaseShift stores bf16 history
//     ([slot][layer][hist=3][C], i.e. state_row[h * conv_history_stride + ch]);
//     shelf stores fp32 and uses the shelf IR StateKind::conv contract
//     [C,3] channel-major: state[c * 3 + h], h=0 oldest. The strided
//     slot/layer addressing below is unchanged.
//   * the serving-only machinery (row_parent trees, per-row history capture,
//     SPEC-VERIFY gating) is not ported; shelf decode drives one row per
//     request. GdnConv1dAqlArgs (PhaseShift async-queue launch blob) is also
//     not ported.
// (SYCL port of backend/hip kernels/gdn/gdn_conv1d.h.)
#pragma once

#include <cstdint>

#include <shelf/backend/sycl/kernels/gdn/gdn_types.h>
#include <shelf/backend/sycl/sycl_api.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

using shelf::bf16_t;

struct GdnConv1dArgs {
    const bf16_t* input = nullptr;
    float* output = nullptr;
    const bf16_t* conv_weight = nullptr;  // [C,4]
    float* conv_state = nullptr;          // [slot][layer][C][3] (shelf layout)

    const GdnRequestDescriptor* requests = nullptr;

    uint32_t num_requests = 0;
    uint32_t actual_rows = 0;
    uint32_t max_request_rows = 0;

    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;

    uint32_t conv_dim = 0;
    uint32_t conv_history = 3;
    uint32_t state_index = 0;   // GDN layer index within the state pool
    uint32_t max_sequences = 0;

    uint64_t conv_slot_stride = 0;    // elements per slot      (C * 3 * layers)
    uint64_t conv_layer_stride = 0;   // elements per layer     (C * 3)
};

inline constexpr uint32_t kGdnConv1dThreads = 256u;

constexpr uint32_t gdn_conv1d_blocks_x(uint32_t conv_dim) {
    return (conv_dim + kGdnConv1dThreads - 1u) / kGdnConv1dThreads;
}

constexpr uint32_t gdn_conv1d_resolved_row_tile(uint32_t row_tile,
                                                uint32_t actual_rows) {
    return row_tile == 0u ? actual_rows : row_tile;
}

constexpr uint32_t gdn_conv1d_tiles(uint32_t row_tile, uint32_t tiled_rows) {
    return row_tile == 0u ? 1u : (tiled_rows + row_tile - 1u) / row_tile;
}

uint32_t gdn_conv1d_row_tile();

::shelf::Status launch_gdn_conv1d_bf16_f32(
    const GdnConv1dArgs& args,
    const ::shelf::sycl::Stream& stream);

} // namespace shelf::syclk
