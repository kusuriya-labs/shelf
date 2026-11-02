#include <phaseshift/kernels/optimized/gdn/fused_gdn_post.h>
#include <phaseshift/kernels/optimized/detail/rmsnorm_device.h>
#include <phaseshift/kernels/optimized/detail/vector_io.h>
#include <phaseshift/kernels/optimized/gdn/detail/gdn_prepare_device.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

using bf16 = detail::bf16;

__global__ __launch_bounds__(32)
void fused_gdn_post_kernel(
    const float* __restrict__ rec, uint32_t rec_rs,
    const bf16* __restrict__ norm_weight, const bf16* __restrict__ z,
    uint32_t z_rs, float* __restrict__ rms_out, uint32_t rms_rs,
    float* __restrict__ gated_z_out, uint32_t gz_rs,
    bf16* __restrict__ gated_out, uint32_t g_rs, float eps,
    uint32_t group_size) {
    const uint32_t row = blockIdx.y;
    const uint32_t gi = blockIdx.x;

    detail::rmsnorm_row_body<RmsNormDataType::F32, RmsNormDataType::F32,
                             RmsNormDataType::BF16, RmsNormWeightLayout::PER_GROUP,
                             RmsNormWeightMode::DIRECT>(
        rec, norm_weight, rms_out, row, gi, group_size, rec_rs, rms_rs, eps);

    const uint32_t g0 = gi * group_size;
    const uint32_t lane = threadIdx.x;
    const bf16* z_row = z + static_cast<size_t>(row) * z_rs + g0;
    float* gz_row = gated_z_out + static_cast<size_t>(row) * gz_rs + g0;
    const float* a_row = rms_out + static_cast<size_t>(row) * rms_rs + g0;
    bf16* c_row = gated_out + static_cast<size_t>(row) * g_rs + g0;

    for (uint32_t base = lane * detail::kVec; base < group_size;
         base += 32u * detail::kVec) {
        if (base + detail::kVec - 1u < group_size && detail::aligned8(z_row + base) &&
            detail::aligned16(gz_row + base)) {
            const detail::F4 x = detail::load_bf16x4(z_row + base);
            detail::F4 y;
            y.v[0] = detail::gdn_silu_f32(x.v[0]);
            y.v[1] = detail::gdn_silu_f32(x.v[1]);
            y.v[2] = detail::gdn_silu_f32(x.v[2]);
            y.v[3] = detail::gdn_silu_f32(x.v[3]);
            detail::store_f32x4(gz_row + base, y);
        } else {
            for (uint32_t i = 0u; i < detail::kVec; ++i) {
                const uint32_t c = base + i;
                if (c >= group_size) break;
                gz_row[c] = detail::gdn_silu_f32(__bfloat162float(z_row[c]));
            }
        }
    }

    for (uint32_t base = lane * detail::kVec; base < group_size;
         base += 32u * detail::kVec) {
        if (base + detail::kVec - 1u < group_size && detail::aligned16(a_row + base) &&
            detail::aligned16(gz_row + base) && detail::aligned8(c_row + base)) {
            const detail::F4 xa = detail::load_f32x4(a_row + base);
            const detail::F4 xb = detail::load_f32x4(gz_row + base);
            detail::F4 y;
            y.v[0] = xa.v[0] * xb.v[0];
            y.v[1] = xa.v[1] * xb.v[1];
            y.v[2] = xa.v[2] * xb.v[2];
            y.v[3] = xa.v[3] * xb.v[3];
            detail::store_bf16x4(c_row + base, y);
        } else {
            for (uint32_t i = 0u; i < detail::kVec; ++i) {
                const uint32_t c = base + i;
                if (c >= group_size) break;
                c_row[c] = __float2bfloat16(a_row[c] * gz_row[c]);
            }
        }
    }
}

}

hipError_t launch_fused_gdn_post(const FusedGdnPostArgs& a, hipStream_t stream) {
    if (a.rows == 0u) return hipSuccess;
    if (a.rec == nullptr || a.norm_weight == nullptr || a.z == nullptr ||
        a.rms_out == nullptr || a.gated_z_out == nullptr || a.gated_out == nullptr) {
        return hipErrorInvalidValue;
    }
    if ((a.features != 6144u && a.features != 3072u) || a.group_size == 0u ||
        a.group_size > 128u || a.features % a.group_size != 0u ||
        a.group_size % 4u != 0u) {
        return hipErrorInvalidValue;
    }
    if (a.rec_row_stride < a.features || a.z_row_stride < a.features ||
        a.rms_row_stride < a.features || a.gated_z_row_stride < a.features ||
        a.gated_row_stride < a.features) {
        return hipErrorInvalidValue;
    }
    const uint32_t groups = a.features / a.group_size;
    dim3 grid(groups, a.rows);
    fused_gdn_post_kernel<<<grid, 32u, 0, stream>>>(
        a.rec, a.rec_row_stride, reinterpret_cast<const bf16*>(a.norm_weight),
        reinterpret_cast<const bf16*>(a.z), a.z_row_stride, a.rms_out,
        a.rms_row_stride, a.gated_z_out, a.gated_z_row_stride,
        reinterpret_cast<bf16*>(a.gated_out), a.gated_row_stride, a.eps, a.group_size);
    return hipGetLastError();
}

}
