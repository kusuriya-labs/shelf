// vendored from PhaseShift (own work); namespace/type adapted only.
#include <shelf/backend/hip/kernels/gdn/gdn_qk_prepare.h>
#include <shelf/backend/hip/kernels/detail/row_grid.h>
#include <shelf/backend/hip/kernels/detail/gdn_prepare_device.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <algorithm>
#include <cstdint>

namespace shelf::hipk {

namespace {

using bf16 = __hip_bfloat16;

constexpr uint32_t kQkThreads = detail::kGdnL2Threads;
static_assert(kQkThreads == 32u);
static_assert(kQkThreads == kFusedGdnQkPrepareThreads);

}  // namespace

extern "C" __global__ __launch_bounds__(32)
void phaseshift_fused_gdn_qk_prepare(FusedGdnQkPrepareArgs args) {
    const uint32_t row = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t groups = args.features / args.group_size;
    const uint32_t half = blockIdx.x;
    if (half < groups) {
        const size_t in_off = static_cast<size_t>(row) * args.q_row_stride +
                              static_cast<size_t>(half) * args.group_size;
        const size_t out_off = static_cast<size_t>(row) * args.nq_row_stride +
                               static_cast<size_t>(half) * args.group_size;
        const size_t scaled_off = static_cast<size_t>(row) * args.sq_row_stride +
                                  static_cast<size_t>(half) * args.group_size;
        const bool aligned = (in_off & 3u) == 0u;
        detail::gdn_l2_normalize_head_body<true>(
            reinterpret_cast<const bf16*>(args.q) + in_off,
            args.nq_out + out_off, args.sq_out + scaled_off, args.scale,
            lane, args.group_size, args.eps, aligned);
    } else {
        const uint32_t g = half - groups;
        const size_t in_off = static_cast<size_t>(row) * args.k_row_stride +
                              static_cast<size_t>(g) * args.group_size;
        const size_t out_off = static_cast<size_t>(row) * args.nk_row_stride +
                               static_cast<size_t>(g) * args.group_size;
        const bool aligned = (in_off & 3u) == 0u;
        detail::gdn_l2_normalize_head_body<false>(
            reinterpret_cast<const bf16*>(args.k) + in_off,
            args.nk_out + out_off, nullptr, 0.0f,
            lane, args.group_size, args.eps, aligned);
    }
}

hipError_t launch_fused_gdn_qk_prepare(
    const FusedGdnQkPrepareArgs& args,
    hipStream_t stream) {
    if (args.q == nullptr || args.k == nullptr || args.nq_out == nullptr ||
        args.nk_out == nullptr || args.sq_out == nullptr)
        return hipErrorInvalidValue;
    if (args.rows == 0u || args.features == 0u || args.group_size == 0u)
        return hipErrorInvalidValue;
    if (args.features % args.group_size != 0u)
        return hipErrorInvalidValue;
    if (args.q_row_stride < args.features || args.k_row_stride < args.features ||
        args.nq_row_stride < args.features || args.nk_row_stride < args.features ||
        args.sq_row_stride < args.features)
        return hipErrorInvalidValue;
    const uint32_t groups = args.features / args.group_size;
    // 行方向（token 行 × key head 数）が grid.y 上限（65535）を超える場合は
    // 基底ポインタを進めて分割発射する。
    for (uint32_t begin = 0u; begin < args.rows; begin += detail::kMaxGridY) {
        FusedGdnQkPrepareArgs chunk = args;
        const uint32_t n = std::min(args.rows - begin, detail::kMaxGridY);
        chunk.q = args.q + static_cast<size_t>(begin) * args.q_row_stride;
        chunk.k = args.k + static_cast<size_t>(begin) * args.k_row_stride;
        chunk.nq_out = args.nq_out + static_cast<size_t>(begin) * args.nq_row_stride;
        chunk.nk_out = args.nk_out + static_cast<size_t>(begin) * args.nk_row_stride;
        chunk.sq_out = args.sq_out + static_cast<size_t>(begin) * args.sq_row_stride;
        chunk.rows = n;
        phaseshift_fused_gdn_qk_prepare<<<dim3(2u * groups, n), kQkThreads, 0, stream>>>(chunk);
        const hipError_t e = hipGetLastError();
        if (e != hipSuccess) return e;
    }
    return hipSuccess;
}

}  // namespace shelf::hipk
