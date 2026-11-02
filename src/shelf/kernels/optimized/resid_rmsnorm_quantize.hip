#include <shelf/kernels/optimized/resid_rmsnorm_quantize.h>
#include "detail/residual_add_device.h"
#include "detail/rmsnorm_device.h"
#include "detail/activation_quantize_device.h"
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kThreads = 256u;
constexpr uint32_t kK = 5120u;
constexpr uint32_t kTileCols = kThreads * detail::kVec;
using detail::bf16;

template <uint32_t K>
__global__ __launch_bounds__(kThreads)
void fused_resid_rmsnorm_quantize_kernel(
    const void* resid_a, uint32_t resid_a_rs,
    const void* resid_b, uint32_t resid_b_rs,
    void* residual, uint32_t residual_rs,
    void* normed, uint32_t normed_rs,
    const void* weight, uint8_t* codes, float* scales,
    uint32_t, uint32_t features, uint32_t group_size,
    uint32_t code_rs, uint32_t scale_rs, float eps) {
    const uint32_t row = blockIdx.x;
    const bf16* a_row =
        static_cast<const bf16*>(resid_a) + static_cast<size_t>(row) * resid_a_rs;
    const bf16* b_row =
        static_cast<const bf16*>(resid_b) + static_cast<size_t>(row) * resid_b_rs;
    bf16* s_base = static_cast<bf16*>(residual);
    bf16* n_base = static_cast<bf16*>(normed);
    for (uint32_t col0 = threadIdx.x * detail::kVec; col0 < features;
         col0 += kTileCols) {
        detail::residual_add_bf16_body(a_row, b_row,
                                       s_base + static_cast<size_t>(row) * residual_rs,
                                       col0, features);
    }
    __syncthreads();
    detail::rmsnorm_row_body<RmsNormDataType::BF16, RmsNormDataType::BF16,
                             RmsNormDataType::BF16, RmsNormWeightLayout::PER_FEATURE,
                             RmsNormWeightMode::ONE_PLUS>(
        s_base, weight, n_base, row, 0u, group_size, residual_rs, normed_rs, eps);
    __syncthreads();
    detail::activation_quantize_e4m3_vec_row_body<K, K>(
        reinterpret_cast<const ::ps::bf16_t*>(n_base), codes, scales, row,
        normed_rs, code_rs, scale_rs);
}

}

hipError_t launch_fused_resid_rmsnorm_quantize(const FusedResidRmsNormQuantizeArgs& a,
                                               hipStream_t stream) {
    if (a.rows == 0u) return hipSuccess;
    if (a.resid_a == nullptr || a.resid_b == nullptr || a.residual == nullptr ||
        a.normed == nullptr || a.weight == nullptr || a.codes == nullptr ||
        a.scales == nullptr) {
        return hipErrorInvalidValue;
    }
    if (a.features != kK || a.group_size != a.features) return hipErrorInvalidValue;
    if (a.resid_a_row_stride < a.features || a.resid_b_row_stride < a.features ||
        a.residual_row_stride < a.features || a.normed_row_stride < a.features) {
        return hipErrorInvalidValue;
    }
    fused_resid_rmsnorm_quantize_kernel<kK><<<a.rows, kThreads, 0, stream>>>(
        reinterpret_cast<const ::ps::bf16_t*>(a.resid_a), a.resid_a_row_stride,
        reinterpret_cast<const ::ps::bf16_t*>(a.resid_b), a.resid_b_row_stride,
        reinterpret_cast<::ps::bf16_t*>(a.residual), a.residual_row_stride,
        reinterpret_cast<::ps::bf16_t*>(a.normed), a.normed_row_stride,
        a.weight, a.codes, a.scales, a.rows, a.features, a.group_size,
        a.code_row_stride, a.scale_row_stride, a.eps);
    return hipGetLastError();
}

}
