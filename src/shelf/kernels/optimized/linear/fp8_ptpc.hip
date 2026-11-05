#include <shelf/kernels/optimized/linear/fp8_ptpc.h>

#include <shelf/core/memory/types.h>
#include <shelf/quantization/fp8/fp8_frag.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {
namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using float8_t = float __attribute__((ext_vector_type(8)));

using namespace shelf::quantization::fp8;

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kThreadsPerBlock = 32u;

#define FP8_PTPC_ZERO8 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}

#define FP8_PTPC_LOAD4(p, lo, hi)                                              \
    do {                                                                        \
        lo = *reinterpret_cast<const int32x4_t*>(p);                            \
        hi = *reinterpret_cast<const int32x4_t*>((p) + 16u);                    \
    } while (0)

#define FP8_PTPC_WMM4(c, al, ah, wl, wh)                                       \
    do {                                                                        \
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(               \
            int32x2_t{(al)[0], (al)[1]}, int32x2_t{(wl)[0], (wl)[1]}, c);       \
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(               \
            int32x2_t{(al)[2], (al)[3]}, int32x2_t{(wl)[2], (wl)[3]}, c);       \
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(               \
            int32x2_t{(ah)[0], (ah)[1]}, int32x2_t{(wh)[0], (wh)[1]}, c);       \
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(               \
            int32x2_t{(ah)[2], (ah)[3]}, int32x2_t{(wh)[2], (wh)[3]}, c);       \
    } while (0)

__global__ __launch_bounds__(kThreadsPerBlock) void gemm_fp8_ptpc_kernel(
    const uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k_padded,
    const uint32_t activation_code_stride,
    const uint32_t activation_scale_stride,
    const uint32_t output_row_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4;

    const uint32_t out_start = blockIdx.x * kTileOut;
    const uint32_t row_start = blockIdx.y * kTileOut;
    if (out_start >= out_features || row_start >= rows) return;

    const bool out_valid = out_start + m_lane < out_features;
    const uint32_t o_self = out_start + m_lane;

    const uint8_t* wrow =
        weight_codes +
        static_cast<std::size_t>(out_start >> 4) *
            static_cast<std::size_t>(k_padded) * 16u +
        static_cast<std::size_t>(tid) * kFragLaneBytes;
    const uint8_t* arow =
        activation_codes +
        static_cast<std::size_t>(row_start >> 4) *
            static_cast<std::size_t>(activation_code_stride) * 16u +
        static_cast<std::size_t>(tid) * kFragLaneBytes;

    float8_t acc = float8_t FP8_PTPC_ZERO8;
    const uint32_t blocks = k_padded / kFragBlockK;
    for (uint32_t blk = 0; blk < blocks; ++blk) {
        const std::size_t offset = static_cast<std::size_t>(blk) * kFragBlockBytes;
        int32x4_t wl = {0, 0, 0, 0};
        int32x4_t wh = {0, 0, 0, 0};
        if (out_valid) FP8_PTPC_LOAD4(wrow + offset, wl, wh);
        int32x4_t al = {0, 0, 0, 0};
        int32x4_t ah = {0, 0, 0, 0};
        FP8_PTPC_LOAD4(arow + offset, al, ah);
        float8_t c = float8_t FP8_PTPC_ZERO8;
        FP8_PTPC_WMM4(c, al, ah, wl, wh);
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] += c[j];
    }

    if (!out_valid) return;
    const float weight_scale = weight_scales[o_self];
    const uint32_t r_base = row_start + 8u * k_group;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t r = r_base + static_cast<uint32_t>(j);
        if (r >= rows) continue;
        const float value =
            acc[j] * weight_scale *
            activation_scales[static_cast<std::size_t>(r) *
                              (activation_scale_stride / sizeof(float))];
        const std::size_t out_idx =
            static_cast<std::size_t>(r) * output_row_stride + o_self;
        if (output_dtype ==
            static_cast<uint32_t>(Fp8PertokenGemmOutputDType::BF16)) {
            static_cast<::shelf::bf16_t*>(output)[out_idx] = ::shelf::bf16_t{
                __bfloat16_as_ushort(__float2bfloat16(value))};
        } else {
            static_cast<float*>(output)[out_idx] = value;
        }
    }
}

}  // namespace

hipError_t launch_gemm_fp8_ptpc(
    const uint8_t* weight_codes,
    const float* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8PertokenGemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (weight_codes == nullptr || weight_scales == nullptr ||
        activation_codes == nullptr || activation_scales == nullptr ||
        output == nullptr || rows == 0u || out_features == 0u || k_padded == 0u ||
        (k_padded % kFragBlockK) != 0u ||
        activation_code_stride_bytes < k_padded ||
        activation_scale_stride_bytes < sizeof(float) ||
        output_row_stride < out_features) {
        return hipErrorInvalidValue;
    }
    const dim3 grid((out_features + kTileOut - 1u) / kTileOut,
                    (rows + kTileOut - 1u) / kTileOut);
    gemm_fp8_ptpc_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        weight_codes, weight_scales, activation_codes, activation_scales, output,
        static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
        activation_code_stride_bytes, activation_scale_stride_bytes,
        output_row_stride);
    return hipGetLastError();
}

}  // namespace shelf::kernel
