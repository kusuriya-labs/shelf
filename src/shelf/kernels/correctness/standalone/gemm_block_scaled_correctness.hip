#include <shelf/kernels/correctness/standalone/gemm_block_scaled_correctness.h>
#include <shelf/core/status.h>
#include <shelf/quantization/codec/e4m3_detail.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ps {
namespace kernel {
namespace detail {

namespace {

constexpr uint32_t kBlockScaledThreads = 256;

__device__ __forceinline__ float bs_bf16_to_f32(bf16_t x) {
    return __bfloat162float(__ushort_as_bfloat16(x.data));
}

__device__ __forceinline__ bf16_t bs_f32_to_bf16(float v) {
    return bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
}
template <BlockScaledKind Kind>
__global__ void gemm_block_scaled_reference_kernel(
    const uint8_t* __restrict__ wcodes,
    const void* __restrict__ wscales,
    uint32_t out_features,
    uint32_t k,
    uint32_t k_padded,
    const bf16_t* __restrict__ input,
    bf16_t* __restrict__ output,
    uint32_t rows)
{
    extern __shared__ uint8_t a_codes[];
    __shared__ float s_scale;
    __shared__ float s_warp[kBlockScaledThreads / 32u];

    const uint32_t m = blockIdx.x;
    if (m >= rows) return;
    const uint32_t t = threadIdx.x;
    const uint32_t nt = blockDim.x;
    const bf16_t* xrow = input + static_cast<std::size_t>(m) * k;

    float peak = 0.0f;
    for (uint32_t i = t; i < k; i += nt) {
        const float v = bs_bf16_to_f32(xrow[i]);
        if (isfinite(v)) {
            const float a = fabsf(v);
            if (a > peak) peak = a;
        }
    }
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffULL, peak, off));
    if ((t & 31u) == 0u) s_warp[t >> 5u] = peak;
    __syncthreads();
    if (t == 0u) {
        float v = 0.0f;
        for (uint32_t w = 0; w < kBlockScaledThreads / 32u; ++w) v = fmaxf(v, s_warp[w]);
        s_scale = v > 0.0f ? v / 448.0f : 0.0f;
    }
    __syncthreads();

    const float ascale = s_scale;
    for (uint32_t i = t; i < k_padded; i += nt) {
        float v = 0.0f;
        if (i < k) {
            const float x = bs_bf16_to_f32(xrow[i]);
            v = isfinite(x) ? x : 0.0f;
        }
        a_codes[i] = ascale > 0.0f
            ? ::ps::quantization::codec::device::e4m3_encode_u8(v / ascale)
            : 0u;
    }
    __syncthreads();

    for (uint32_t o = t; o < out_features; o += nt) {
        float acc = 0.0f;
        {
            const uint32_t sk = k_padded >> 7;
            const uint8_t* wrow = wcodes + static_cast<std::size_t>(o) * k_padded;
            const float* srow = static_cast<const float*>(wscales) +
                static_cast<std::size_t>(o >> 7u) * sk;
            for (uint32_t kb = 0; kb < sk; ++kb) {
                const uint32_t base = kb * 128u;
                float p = 0.0f;
                for (uint32_t j = 0; j < 128u; ++j) {
                    p = fmaf(::ps::quantization::codec::device::e4m3_decode_f32(wrow[base + j]),
                             ::ps::quantization::codec::device::e4m3_decode_f32(a_codes[base + j]),
                             p);
                }
                acc += p * srow[kb];
            }
        }
        acc *= ascale;
        output[static_cast<std::size_t>(m) * out_features + o] = bs_f32_to_bf16(acc);
    }
}

Status bs_validate(const BlockScaledWeightView& w, const bf16_t* input,
                   const bf16_t* output, uint32_t rows) {
    (void)input;
    (void)output;
    (void)rows;
    if (w.codes == nullptr || w.scales == nullptr)
        return Status::invalid_argument("block scaled weight is not bound", __FILE__, __LINE__);
    if (w.out_features == 0 || w.k_padded == 0 || w.k_padded < w.k)
        return Status::invalid_argument("block scaled weight geometry invalid", __FILE__, __LINE__);
    if (w.kind == BlockScaledKind::Fp8) {
        if ((w.k_padded % 128u) != 0u)
            return Status::invalid_argument("fp8 k_padded is not a multiple of 128", __FILE__, __LINE__);
    } else {
        if ((w.k_padded % 32u) != 0u)
            return Status::invalid_argument("block scaled k_padded is not a multiple of 32", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

hipError_t launch_gemm_block_scaled_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    uint32_t rows,
    hipStream_t stream) {
    Status status = bs_validate(weight, input, output, rows);
    if (!status.ok())
        return hipErrorInvalidValue;
    if (rows == 0)
        return hipSuccess;
    const std::size_t smem = weight.k_padded;
    switch (weight.kind) {
        case BlockScaledKind::Fp8:
            gemm_block_scaled_reference_kernel<BlockScaledKind::Fp8>
                <<<rows, kBlockScaledThreads, smem, stream>>>(
                    weight.codes, weight.scales, weight.out_features, weight.k,
                    weight.k_padded, input, output, rows);
            break;
    }
    return hipGetLastError();
}

hipError_t launch_gemv_block_scaled_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    hipStream_t stream) {
    return launch_gemm_block_scaled_reference(weight, input, output, 1u, stream);
}

hipError_t launch_gemm_fp8_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    uint32_t rows,
    hipStream_t stream) {
    BlockScaledWeightView w = weight;
    w.kind = BlockScaledKind::Fp8;
    return launch_gemm_block_scaled_reference(w, input, output, rows, stream);
}

hipError_t launch_gemv_fp8_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    hipStream_t stream) {
    BlockScaledWeightView w = weight;
    w.kind = BlockScaledKind::Fp8;
    return launch_gemm_block_scaled_reference(w, input, output, 1u, stream);
}
}
}
}
