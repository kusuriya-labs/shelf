// vendored from PhaseShift (own work); namespace/type adapted only.
//
// Decode-path recurrent gated delta-rule update (single token per request).
// Ported from phaseshift recurrence.hip's portable `gdn_recurrence_f32_impl`
// kernel (f32 compute). Adaptations:
//   * recurrent state element type fp16 -> fp32 (shelf IR recurrent state is
//     fp32; PhaseShift's GPU state pool was fp16 — this is the only dtype
//     change; the algebra is identical to the shelf CPU reference).
//   * state layout UNCHANGED: [slot][layer][v_head][head_k][head_v],
//     i.e. per-head S[kk * head_v + vcol] row-major — matches the shelf IR
//     recurrent state [nvh, dk, dv].
//   * beta/g are computed INSIDE the kernel from RAW projections:
//     b (=in_proj_b out), a (=in_proj_a out), plus bf16 weights A_log and
//     dt_bias:  beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias);
//     softplus here = __logf(1 + __expf(x)) (PhaseShift formula; equals the
//     shelf reference's thresholded softplus within f32 rounding for x <= 20).
//   * q/k inputs are ALREADY normalized/scaled (feed launch_fused_gdn_qk_prepare
//     sq into q and nk into k); v stays bf16.
//   * serving-only machinery not ported: recurrent history capture, row_parent
//     trees, compact delta/k/a buffers, and the gfx12 WMMA chunk/decode1/
//     decode_rows/commit kernels (those are fp16-state, fixed-head-shape
//     fast paths).
#pragma once

#include <shelf/backend/hip/kernels/gdn/gdn_types.h>
#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::hipk {

using shelf::bf16_t;

struct GdnRecurrenceArgs {
    const float* q = nullptr;          // scaled+normalized q (sq_out)
    const float* k = nullptr;          // normalized k (nk_out)
    const bf16_t* v = nullptr;         // raw v projections
    const bf16_t* a = nullptr;         // raw in_proj_a projections
    const bf16_t* b = nullptr;         // raw in_proj_b projections

    const bf16_t* dt_bias = nullptr;   // [num_v_heads]
    const bf16_t* a_log = nullptr;     // [num_v_heads]

    float* recurrent_state = nullptr;  // [slot][layer][nvh][head_k][head_v] f32
    float* output = nullptr;           // [rows][nvh * head_v]

    const GdnRequestDescriptor* requests = nullptr;

    uint32_t num_requests = 0;
    uint32_t actual_rows = 0;

    uint32_t q_row_stride = 0;
    uint32_t k_row_stride = 0;
    uint32_t v_row_stride = 0;
    uint32_t a_row_stride = 0;
    uint32_t b_row_stride = 0;
    uint32_t output_row_stride = 0;

    uint32_t state_index = 0;
    uint32_t max_sequences = 0;

    uint32_t key_heads = 0;
    uint32_t num_v_heads = 0;

    uint32_t head_k = 0;
    uint32_t head_v = 0;

    uint64_t recurrent_slot_stride = 0;   // elements per slot  (nvh*dk*dv*layers)
    uint64_t recurrent_layer_stride = 0;  // elements per layer (nvh*dk*dv)
};

hipError_t launch_gdn_recurrence_f32(
    const GdnRecurrenceArgs& args,
    hipStream_t stream);

inline constexpr const char* kGdnRecurrenceF32Symbol =
    "gdn_recurrence_f32_impl";

constexpr uint32_t gdn_recurrence_f32_grid_x(uint32_t num_requests,
                                             uint32_t num_v_heads,
                                             uint32_t head_v) {
    if (num_requests == 0u || num_v_heads == 0u || head_v == 0u)
        return 0u;
    const uint32_t v_tiles = (head_v + 31u) >> 5;
    return num_requests * num_v_heads * v_tiles;
}

}  // namespace shelf::hipk
