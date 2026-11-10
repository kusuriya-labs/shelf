#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/quantization/exl3/exl3_int8.h>

#include <cstdlib>

namespace shelf::kernel {
namespace {

constexpr uint32_t kBlock = 128u;
constexpr float kInvSqrt128 = 0.08838834764831845f;

// SHELF_EXL3_INT8_ROTATE_F4=1 replaces the 5-shuffle / 2-smem hybrid below with the full float4 form
// (launch_exl3_mul24_rotate_out_i32), so all 7 H128 stages stay inside one wave.
bool rotate_f4_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHELF_EXL3_INT8_ROTATE_F4");
        return env != nullptr && std::strtoul(env, nullptr, 10) != 0ul;
    }();
    return enabled;
}

__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7F800000u) == 0x7F800000u)
        return (u & 0x007FFFFFu) != 0u ? 0x7FC0u : static_cast<uint16_t>(u >> 16);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

template <bool BF16>
__global__ void __launch_bounds__(kBlock) exl3_mul24_rotate_out_i32_prefill(
    const int32_t* __restrict__ acc, const float* __restrict__ rowscale, const int32_t* __restrict__ slots,
    const float* __restrict__ yprime2, const uint16_t* __restrict__ svh, void* __restrict__ out,
    uint32_t n, uint32_t out_row_stride) {
    __shared__ float smem[kBlock];
    const uint32_t t = threadIdx.x;
    const uint32_t m = blockIdx.x;
    const uint32_t col = blockIdx.y * kBlock + t;
    const int32_t slot = slots[m];
    float v = slot >= 0 ? yprime2[static_cast<size_t>(slot) * n + col]
                        : static_cast<float>(acc[static_cast<size_t>(m) * n + col]) * rowscale[m] *
                              ::shelf::quantization::exl3::kExl3Int8GridStep;
    #pragma unroll
    for (uint32_t h = 1u; h < 32u; h <<= 1u) {
        const float b = __shfl_xor(v, h);
        v = (t & h) ? (b - v) : (v + b);
    }
    smem[t] = v;
    __syncthreads();
    #pragma unroll
    for (uint32_t h = 32u; h < kBlock; h <<= 1u) {
        const float a = smem[t];
        const float b = smem[t ^ h];
        __syncthreads();
        smem[t] = (t & h) ? (b - a) : (a + b);
        __syncthreads();
    }
    v = smem[t] * kInvSqrt128 * static_cast<float>(__builtin_bit_cast(_Float16, svh[col]));
    if constexpr (BF16)
        reinterpret_cast<::shelf::bf16_t*>(out)[static_cast<size_t>(m) * out_row_stride + col] =
            ::shelf::bf16_t{f32_to_bf16_rne(v)};
    else
        reinterpret_cast<float*>(out)[static_cast<size_t>(m) * out_row_stride + col] = v;
}

}

hipError_t launch_exl3_mul24_prefill_rotate_out_i32(
    const int32_t* acc, const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, Exl3OutputDType out_dtype, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (n == 0u || (n % kBlock) != 0u) return hipErrorInvalidValue;
    if (rotate_f4_enabled()) {
        return launch_exl3_mul24_rotate_out_i32(acc, rowscale, slots, yprime2, svh, out, out_dtype, rows, n,
                                               out_row_stride, stream);
    }
    const dim3 grid(rows, n / kBlock);
    if (out_dtype == Exl3OutputDType::BF16)
        exl3_mul24_rotate_out_i32_prefill<true><<<grid, kBlock, 0, stream>>>(acc, rowscale, slots, yprime2, svh, out, n, out_row_stride);
    else
        exl3_mul24_rotate_out_i32_prefill<false><<<grid, kBlock, 0, stream>>>(acc, rowscale, slots, yprime2, svh, out, n, out_row_stride);
    return hipGetLastError();
}

}
