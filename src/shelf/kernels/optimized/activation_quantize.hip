#include <shelf/kernels/optimized/activation_quantize.h>
#include "detail/activation_quantize_device.h"
#include <shelf/quantization/codec/e4m3_detail.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace shelf::kernel {

namespace {

constexpr uint32_t kThreadsPerRow = 256u;

__global__ void __launch_bounds__(kThreadsPerRow) activation_quantize_fp8_finegrained_kernel(
    const ::shelf::bf16_t* __restrict__ input,
    uint8_t* __restrict__ codes,
    float* __restrict__ scales,
    const uint32_t k,
    const uint32_t k_padded,
    const uint32_t input_row_stride,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride) {
    constexpr uint32_t kGroup = 128u;
    constexpr uint32_t kWaves = kThreadsPerRow / 32u;
    constexpr uint32_t kElemsPerLane = kGroup / 32u;
    const uint32_t m = blockIdx.x;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const __hip_bfloat16* xb = reinterpret_cast<const __hip_bfloat16*>(input) +
        static_cast<size_t>(m) * input_row_stride;
    uint8_t* crow = codes +
        static_cast<size_t>(m >> 4u) * (static_cast<size_t>(code_row_stride) * 16u);
    const uint32_t m_tile = m & 15u;
    float* row_scales = scales + static_cast<size_t>(m) * (scale_row_stride / 4u);
    const uint32_t groups = (k_padded + kGroup - 1u) / kGroup;

    for (uint32_t g = wave; g < groups; g += kWaves) {
        const uint32_t base = g * kGroup;
        const uint32_t i0 = base + lane * kElemsPerLane;
        const bool full4 = (i0 + kElemsPerLane <= k);
        uint64_t raw = 0u;
        if (full4)
            raw = *reinterpret_cast<const uint64_t*>(xb + i0);
        float peak = 0.0f;
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            float v;
            if (full4) {
                const uint32_t word =
                    static_cast<uint32_t>((raw >> (16u * l)) & 0xffffu);
                v = __uint_as_float(word << 16);
            } else {
                v = (i0 + l < k) ? __bfloat162float(xb[i0 + l]) : 0.0f;
            }
            if (isfinite(v)) {
                const float av = std::fabs(v);
                if (av > peak) peak = av;
            }
        }
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1)
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffULL, peak, off));
        const float scale = __shfl_sync(0xffffffffULL, peak, 0u);
        const float s = scale > 0.0f ? scale / 448.0f : 0.0f;
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        if (lane == 0u) row_scales[g] = s;
        uint32_t packed = 0u;
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            float v = 0.0f;
            if (i0 + l < k) {
                if (full4) {
                    const uint32_t word =
                        static_cast<uint32_t>((raw >> (16u * l)) & 0xffffu);
                    const float x = __uint_as_float(word << 16);
                    v = isfinite(x) ? x : 0.0f;
                } else {
                    const float x = __bfloat162float(xb[i0 + l]);
                    v = isfinite(x) ? x : 0.0f;
                }
            }
            const uint8_t c = s > 0.0f
                ? ::shelf::quantization::codec::device::e4m3_encode_u8(v * inv)
                : 0u;
            packed |= static_cast<uint32_t>(c) << (8u * l);
        }
        if (i0 + kElemsPerLane <= k_padded) {
            const uint32_t off = (i0 >> 6u) * ::shelf::quantization::fp8::kFragBlockBytes +
                                 ::shelf::quantization::fp8::fp8_frag_offset(m_tile, i0 & 63u);
            *reinterpret_cast<uint32_t*>(crow + off) = packed;
        }
    }
}

template <uint32_t K, uint32_t KP>
__global__ void __launch_bounds__(kThreadsPerRow)
activation_quantize_fp8_finegrained_vec_kernel(
    const ::shelf::bf16_t* __restrict__ input,
    uint8_t* __restrict__ codes,
    float* __restrict__ scales,
    const uint32_t input_row_stride,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride) {
    detail::activation_quantize_fp8_finegrained_vec_row_body<K, KP>(
        input, codes, scales, blockIdx.x, input_row_stride, code_row_stride,
        scale_row_stride);
}

}  // namespace

bool activation_quantize_fp8_finegrained_vec_supported(
    const ::shelf::bf16_t* input, const uint8_t* codes,
    uint32_t k, uint32_t k_padded, uint32_t input_row_stride) {
    if (k != k_padded || (k & 7u) != 0u)
        return false;
    if (k != 5120u && k != 6144u && k != 17408u && k != 8704u && k != 3072u)
        return false;
    if ((reinterpret_cast<uintptr_t>(input) & 15u) != 0u)
        return false;
    if ((reinterpret_cast<uintptr_t>(codes) & 7u) != 0u)
        return false;
    if ((input_row_stride & 7u) != 0u)
        return false;
    return true;
}

hipError_t launch_activation_quantize_fp8_finegrained_generic(
    const ::shelf::bf16_t* input,
    uint8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream) {
    if (rows == 0 || k_padded == 0)
        return hipSuccess;
    if ((k_padded % ::shelf::quantization::fp8::kFragBlockK) != 0u ||
        (code_row_stride_bytes % ::shelf::quantization::fp8::kFragBlockK) != 0u)
        return hipErrorNotSupported;
    activation_quantize_fp8_finegrained_kernel<<<rows, kThreadsPerRow, 0, stream>>>(
        input, codes, scales, k, k_padded,
        input_row_stride, code_row_stride_bytes, scale_row_stride_bytes);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void shelf_activation_quantize_fp8_finegrained_k5120(
    ActivationQuantizeFp8FinegrainedK5120Args args);

hipError_t launch_activation_quantize_fp8_finegrained(
    const ::shelf::bf16_t* input,
    uint8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream,
    bool* specialized) {
    if (specialized != nullptr) *specialized = false;
    if (rows == 0 || k_padded == 0)
        return hipSuccess;
    if ((k_padded % ::shelf::quantization::fp8::kFragBlockK) != 0u ||
        (code_row_stride_bytes % ::shelf::quantization::fp8::kFragBlockK) != 0u)
        return hipErrorNotSupported;
    if (activation_quantize_fp8_finegrained_vec_supported(input, codes, k, k_padded, input_row_stride)) {
        switch (k) {
            case 5120u: {
                ActivationQuantizeFp8FinegrainedK5120Args k5120{};
                k5120.input = input;
                k5120.codes = codes;
                k5120.scales = scales;
                k5120.input_row_stride = input_row_stride;
                k5120.code_row_stride = code_row_stride_bytes;
                k5120.scale_row_stride = scale_row_stride_bytes;
                shelf_activation_quantize_fp8_finegrained_k5120
                    <<<rows, 256, 0, stream>>>(k5120);
                if (specialized != nullptr) *specialized = true;
                return hipGetLastError();
            }
            case 6144u:
                activation_quantize_fp8_finegrained_vec_kernel<6144u, 6144u>
                    <<<rows, kThreadsPerRow, 0, stream>>>(
                        input, codes, scales, input_row_stride,
                        code_row_stride_bytes, scale_row_stride_bytes);
                if (specialized != nullptr) *specialized = true;
                return hipGetLastError();
            case 17408u:
                activation_quantize_fp8_finegrained_vec_kernel<17408u, 17408u>
                    <<<rows, kThreadsPerRow, 0, stream>>>(
                        input, codes, scales, input_row_stride,
                        code_row_stride_bytes, scale_row_stride_bytes);
                if (specialized != nullptr) *specialized = true;
                return hipGetLastError();
            case 8704u:
                activation_quantize_fp8_finegrained_vec_kernel<8704u, 8704u>
                    <<<rows, kThreadsPerRow, 0, stream>>>(
                        input, codes, scales, input_row_stride,
                        code_row_stride_bytes, scale_row_stride_bytes);
                if (specialized != nullptr) *specialized = true;
                return hipGetLastError();
            case 3072u:
                activation_quantize_fp8_finegrained_vec_kernel<3072u, 3072u>
                    <<<rows, kThreadsPerRow, 0, stream>>>(
                        input, codes, scales, input_row_stride,
                        code_row_stride_bytes, scale_row_stride_bytes);
                if (specialized != nullptr) *specialized = true;
                return hipGetLastError();
            default:
                break;
        }
    }
    return launch_activation_quantize_fp8_finegrained_generic(
        input, codes, scales, rows, k, k_padded,
        input_row_stride, code_row_stride_bytes, scale_row_stride_bytes, stream);
}

extern "C" __global__ __launch_bounds__(256)
void shelf_activation_quantize_fp8_finegrained_k5120(
    ActivationQuantizeFp8FinegrainedK5120Args args) {
    detail::activation_quantize_fp8_finegrained_vec_row_body<5120u, 5120u>(
        reinterpret_cast<const ::shelf::bf16_t*>(args.input),
        reinterpret_cast<uint8_t*>(args.codes),
        reinterpret_cast<float*>(args.scales), blockIdx.x, args.input_row_stride,
        args.code_row_stride, args.scale_row_stride);
}

extern "C" __global__ __launch_bounds__(256)
void shelf_activation_quantize_fp8_finegrained_k6144(
    ActivationQuantizeFp8FinegrainedKArgs args) {
    detail::activation_quantize_fp8_finegrained_vec_row_body<6144u, 6144u>(
        reinterpret_cast<const ::shelf::bf16_t*>(args.input),
        reinterpret_cast<uint8_t*>(args.codes),
        reinterpret_cast<float*>(args.scales), blockIdx.x, args.input_row_stride,
        args.code_row_stride, args.scale_row_stride);
}

extern "C" __global__ __launch_bounds__(256)
void shelf_activation_quantize_fp8_finegrained_k17408(
    ActivationQuantizeFp8FinegrainedKArgs args) {
    detail::activation_quantize_fp8_finegrained_vec_row_body<17408u, 17408u>(
        reinterpret_cast<const ::shelf::bf16_t*>(args.input),
        reinterpret_cast<uint8_t*>(args.codes),
        reinterpret_cast<float*>(args.scales), blockIdx.x, args.input_row_stride,
        args.code_row_stride, args.scale_row_stride);
}


}
