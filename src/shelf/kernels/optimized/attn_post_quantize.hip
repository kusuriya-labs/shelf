#include <shelf/kernels/optimized/attn_post_quantize.h>
#include "detail/sigmoid_device.h"
#include "detail/vector_io.h"
#include "detail/activation_quantize_device.h"
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kThreads = 256u;
constexpr uint32_t kTileCols = kThreads * detail::kVec;
using detail::bf16;

template <uint32_t K>
__global__ __launch_bounds__(kThreads)
void fused_attn_post_quantize_kernel(
    const void* gate, uint32_t gate_rs,
    const void* ctx, uint32_t ctx_rs,
    void* gated, uint32_t gated_rs,
    uint8_t* codes, float* scales,
    uint32_t, uint32_t features,
    uint32_t code_rs, uint32_t scale_rs, uint32_t ctx_first) {
    const uint32_t row = blockIdx.x;
    const bf16* gate_row =
        static_cast<const bf16*>(gate) + static_cast<size_t>(row) * gate_rs;
    const float* ctx_row =
        static_cast<const float*>(ctx) + static_cast<size_t>(row) * ctx_rs;
    bf16* g_base = static_cast<bf16*>(gated);
    bf16* out_row = g_base + static_cast<size_t>(row) * gated_rs;
    for (uint32_t col0 = threadIdx.x * detail::kVec; col0 < features;
         col0 += kTileCols) {
        if (col0 + detail::kVec - 1u < features && detail::aligned8(gate_row + col0) &&
            detail::aligned16(ctx_row + col0) && detail::aligned8(out_row + col0)) {
            const detail::F4 g = detail::load_bf16x4(gate_row + col0);
            const detail::F4 c = detail::load_f32x4(ctx_row + col0);
            detail::F4 y;
#pragma unroll
            for (uint32_t e = 0u; e < detail::kVec; ++e) {
                const float s = detail::sigmoid_f32(g.v[e]);
                y.v[e] = ctx_first != 0u ? c.v[e] * s : s * c.v[e];
            }
            detail::store_bf16x4(out_row + col0, y);
        } else {
            for (uint32_t i = 0u; i < detail::kVec; ++i) {
                const uint32_t c = col0 + i;
                if (c >= features) break;
                const float s =
                    detail::sigmoid_f32(detail::bf16_to_f32(gate_row[c]));
                const float p =
                    ctx_first != 0u ? ctx_row[c] * s : s * ctx_row[c];
                out_row[c] = detail::f32_to_bf16(p);
            }
        }
    }
    __syncthreads();
    detail::activation_quantize_e4m3_vec_row_body<K, K>(
        reinterpret_cast<const ::ps::bf16_t*>(g_base), codes, scales, row,
        gated_rs, code_rs, scale_rs);
}

}

hipError_t launch_fused_attn_post_quantize(const FusedAttnPostQuantizeArgs& a,
                                           hipStream_t stream) {
    if (a.rows == 0u) return hipSuccess;
    if (a.gate == nullptr || a.ctx == nullptr || a.gated_out == nullptr ||
        a.codes == nullptr || a.scales == nullptr) {
        return hipErrorInvalidValue;
    }
    if (a.features != 6144u && a.features != 3072u) return hipErrorInvalidValue;
    if (a.gate_row_stride < a.features || a.ctx_row_stride < a.features ||
        a.gated_row_stride < a.features) {
        return hipErrorInvalidValue;
    }
    if (a.features == 6144u) {
        fused_attn_post_quantize_kernel<6144u><<<a.rows, kThreads, 0, stream>>>(
            a.gate, a.gate_row_stride, a.ctx, a.ctx_row_stride, a.gated_out,
            a.gated_row_stride, a.codes, a.scales, a.rows, a.features,
            a.code_row_stride, a.scale_row_stride, a.ctx_first);
        return hipGetLastError();
    }
    fused_attn_post_quantize_kernel<3072u><<<a.rows, kThreads, 0, stream>>>(
        a.gate, a.gate_row_stride, a.ctx, a.ctx_row_stride, a.gated_out,
        a.gated_row_stride, a.codes, a.scales, a.rows, a.features,
        a.code_row_stride, a.scale_row_stride, a.ctx_first);
    return hipGetLastError();
}

}
