#include <shelf/kernels/fused/exl3/rotate_out.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul24;

// Same as rotate_out but the input is the int32 hipBLASLt accumulator: yprime = acc * row scale * S_FIX.
// Rows with a slot take the F32 result of the fused per-group kernel (yprime2[slot]) instead.
template <bool BF16>
__global__ void __launch_bounds__(kBlock) exl3_mul24_rotate_out_i32_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ rowscale,
    const int32_t* __restrict__ slots,
    const float* __restrict__ yprime2,
    const uint16_t* __restrict__ svh,
    void* __restrict__ out,
    const uint32_t n,
    const uint32_t out_row_stride) {
    __shared__ float smem[kBlock];
    const uint32_t t = threadIdx.x;
    const uint32_t m = blockIdx.x;
    const uint32_t col = blockIdx.y * kBlock + t;
    const int32_t slot = slots[m];
    float v = slot >= 0 ? yprime2[static_cast<size_t>(slot) * n + col]
                        : static_cast<float>(acc[static_cast<size_t>(m) * n + col]) * rowscale[m] *
                              ::shelf::quantization::exl3::kExl3Int8GridStep;
    v = fwht128_smem(v, smem, t) * kInvSqrt128 * f16_bits_to_f32(svh[col]);
    if constexpr (BF16) {
        reinterpret_cast<::shelf::bf16_t*>(out)[static_cast<size_t>(m) * out_row_stride + col] =
            ::shelf::bf16_t{f32_to_bf16_rne_device(v)};
    } else {
        reinterpret_cast<float*>(out)[static_cast<size_t>(m) * out_row_stride + col] = v;
    }
}

// Same math, float4 form: one CTA per 8 rows, 8 waves, one wave per row. int4 activation load /
// float4 flagged-row load, H128 and svh inside one wave, one 64-bit output store.
// Each wave walks the whole output row, or with kTiled every (8 * gridDim.y)-th 128-column block
// of the single row the CTA owns.
template <bool BF16, bool kFlaggedOnly = false, bool kTiled = false>
__global__ void __launch_bounds__(256) exl3_mul24_rotate_out_i32_f4_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ rowscale,
    const int32_t* __restrict__ slots,
    const float* __restrict__ yprime2,
    const uint16_t* __restrict__ svh,
    void* __restrict__ out,
    const uint32_t n,
    const uint32_t out_row_stride,
    const uint32_t rows) {
    const uint32_t m = kTiled ? blockIdx.x : blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (m >= rows) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const int32_t slot = slots[m];
    if constexpr (kFlaggedOnly) {
        if (slot < 0) return;
    }
    const float rs = rowscale[m];
    const uint32_t col_blocks = n / kBlock;
    const uint32_t first = kTiled ? blockIdx.y * 8u + wave : 0u;
    const uint32_t stride = kTiled ? 8u * gridDim.y : 1u;
    for (uint32_t cb = first; cb < col_blocks; cb += stride) {
        const uint32_t col = cb * kBlock + lane * 4u;
        float v0, v1, v2, v3;
        if (slot >= 0) {
            const float4 y = *reinterpret_cast<const float4*>(yprime2 + static_cast<size_t>(slot) * n + col);
            v0 = y.x;
            v1 = y.y;
            v2 = y.z;
            v3 = y.w;
        } else {
            const int4 a = *reinterpret_cast<const int4*>(acc + static_cast<size_t>(m) * n + col);
            v0 = (static_cast<float>(a.x) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
            v1 = (static_cast<float>(a.y) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
            v2 = (static_cast<float>(a.z) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
            v3 = (static_cast<float>(a.w) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
        }
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
        v0 = (v0 * kInvSqrt128) * f16_bits_to_f32(svh[col + 0u]);
        v1 = (v1 * kInvSqrt128) * f16_bits_to_f32(svh[col + 1u]);
        v2 = (v2 * kInvSqrt128) * f16_bits_to_f32(svh[col + 2u]);
        v3 = (v3 * kInvSqrt128) * f16_bits_to_f32(svh[col + 3u]);
        const size_t index = static_cast<size_t>(m) * out_row_stride + col;
        if constexpr (BF16) {
            uint2 packed;
            packed.x = static_cast<uint32_t>(f32_to_bf16_rne_device(v0)) |
                       (static_cast<uint32_t>(f32_to_bf16_rne_device(v1)) << 16u);
            packed.y = static_cast<uint32_t>(f32_to_bf16_rne_device(v2)) |
                       (static_cast<uint32_t>(f32_to_bf16_rne_device(v3)) << 16u);
            *reinterpret_cast<uint2*>(reinterpret_cast<::shelf::bf16_t*>(out) + index) = packed;
        } else {
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(out) + index) = float4{v0, v1, v2, v3};
        }
    }
}
}  // namespace

template <bool kF4, bool kBF16, bool kFlaggedOnly = false>
hipError_t launch_rotate_out_i32_impl(const int32_t* acc, const float* rowscale, const int32_t* slots,
                                      const float* yprime2, const uint16_t* svh, void* out, uint32_t rows,
                                      uint32_t n, uint32_t out_row_stride, hipStream_t stream) {
    if constexpr (kF4) {
        const dim3 grid((rows + 7u) / 8u);
        exl3_mul24_rotate_out_i32_f4_kernel<kBF16, kFlaggedOnly, false>
            <<<grid, 256, 0, stream>>>(acc, rowscale, slots, yprime2, svh, out, n, out_row_stride, rows);
    } else {
        const dim3 grid(rows, n / kBlock);
        exl3_mul24_rotate_out_i32_kernel<kBF16>
            <<<grid, kBlock, 0, stream>>>(acc, rowscale, slots, yprime2, svh, out, n, out_row_stride);
    }
    return hipGetLastError();
}

hipError_t launch_exl3_mul24_rotate_out_i32_flagged(
    const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (n == 0u || (n % kBlock) != 0u || (out_row_stride % 4u) != 0u) return hipErrorInvalidValue;
    // One CTA per row and up to kColumnTiles column tiles: a wave walks every (8 * tiles)-th
    // 128-column block, so the column parallelism grows without letting the grid follow n.
    constexpr uint32_t kColumnTilesBudget = 16384u;
    const uint32_t tiles = min((n / kBlock + 7u) / 8u, max(1u, kColumnTilesBudget / rows));
    const dim3 grid(rows, tiles);
    exl3_mul24_rotate_out_i32_f4_kernel<true, true, true>
        <<<grid, 256, 0, stream>>>(nullptr, rowscale, slots, yprime2, svh, out, n, out_row_stride, rows);
    return hipGetLastError();
}

hipError_t launch_exl3_mul24_rotate_out_i32(
    const int32_t* acc, const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, Exl3OutputDType out_dtype, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (n == 0u || (n % kBlock) != 0u) return hipErrorInvalidValue;
    const bool f4 = rotate_f4_enabled() && rows >= kRotateOutF4MinRows && (out_row_stride % 4u) == 0u;
    const bool bf16 = out_dtype == Exl3OutputDType::BF16;
    if (f4)
        return bf16 ? launch_rotate_out_i32_impl<true, true>(acc, rowscale, slots, yprime2, svh, out, rows, n,
                                                             out_row_stride, stream)
                    : launch_rotate_out_i32_impl<true, false>(acc, rowscale, slots, yprime2, svh, out, rows, n,
                                                              out_row_stride, stream);
    return bf16 ? launch_rotate_out_i32_impl<false, true>(acc, rowscale, slots, yprime2, svh, out, rows, n,
                                                          out_row_stride, stream)
                : launch_rotate_out_i32_impl<false, false>(acc, rowscale, slots, yprime2, svh, out, rows, n,
                                                           out_row_stride, stream);
}


}  // namespace shelf::kernel
