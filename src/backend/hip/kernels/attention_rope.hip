// vendored from PhaseShift (own work); namespace/type adapted only.
#include <shelf/backend/hip/kernels/rope.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace shelf::hipk {

namespace {

using bf16 = __hip_bfloat16;

constexpr uint32_t kRopeThreads = 256;
constexpr uint32_t kRopeMaxHalf = 128;

constexpr uint32_t kPairThreads = kRopePairBlock;
constexpr uint32_t kPairHeadDim = kRopePairHeadDim;
constexpr uint32_t kPairRotary = kRopePairRotary;
constexpr uint32_t kPairHalf = kPairRotary / 2u;

__global__ __launch_bounds__(kRopeThreads)
void rope_f32_bf16_kernel(
    const float* __restrict__ input, uint32_t input_row_stride,
    bf16* __restrict__ output, uint32_t output_row_stride,
    const uint32_t* __restrict__ positions,
    uint32_t features, uint32_t head_dim, uint32_t rotary, float theta,
    const double* __restrict__ inv_freq, float attention_factor) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t half = rotary >> 1;

    __shared__ float s_cos[kRopeMaxHalf];
    __shared__ float s_sin[kRopeMaxHalf];
    if (tid < half) {
        const uint32_t pos = positions != nullptr ? positions[row] : row;
        const double freq = inv_freq != nullptr
            ? inv_freq[tid]
            : 1.0 / pow(static_cast<double>(theta),
                        static_cast<double>(tid) / static_cast<double>(half));
        const double angle = static_cast<double>(pos) * freq;
        double sn = 0.0;
        double cs = 0.0;
        sincos(angle, &sn, &cs);
        s_cos[tid] = static_cast<float>(cs) * attention_factor;
        s_sin[tid] = static_cast<float>(sn) * attention_factor;
    }
    __syncthreads();

    const size_t in_base = static_cast<size_t>(row) * input_row_stride;
    const size_t out_base = static_cast<size_t>(row) * output_row_stride;
    for (uint32_t idx = tid; idx < features; idx += kRopeThreads) {
        const uint32_t head_col = idx % head_dim;
        const float x = input[in_base + idx];
        float y = x;
        if (head_col < rotary) {
            const uint32_t pi = head_col < half ? head_col : head_col - half;
            const uint32_t partner_col = head_col < half ? head_col + half : head_col - half;
            const float cs = s_cos[pi];
            const float sn = s_sin[pi];
            const float partner = input[in_base + (idx - head_col) + partner_col];
            y = head_col < half ? x * cs - partner * sn : x * cs + partner * sn;
        }
        output[out_base + idx] = __float2bfloat16(y);
    }
}

__global__ void rope_inv_freq_init_kernel(
    double* __restrict__ inv_freq, uint32_t half, float theta) {
    const uint32_t i = threadIdx.x;
    if (i >= half)
        return;
    const double theta_base = static_cast<double>(theta);
    inv_freq[i] = 1.0 / pow(theta_base,
                            static_cast<double>(i) / static_cast<double>(half));
}

__global__ void rope_inv_freq_yarn_init_kernel(
    double* __restrict__ inv_freq, uint32_t half, float theta,
    float factor, float original_len, float beta_fast, float beta_slow,
    uint32_t truncate) {
    const uint32_t i = threadIdx.x;
    if (i >= half)
        return;
    const double theta_base = static_cast<double>(theta);
    const double exponent_div = static_cast<double>(half);
    const double dim = static_cast<double>(half) * 2.0;
    const double f = static_cast<double>(factor);
    const double L = static_cast<double>(original_len);
    const double base = 1.0 / pow(theta_base,
                                  static_cast<double>(i) / exponent_div);
    const double pi2 = 6.283185307179586;
    const double cf = dim * log(L / (static_cast<double>(beta_fast) * pi2)) /
                      (2.0 * log(theta_base));
    const double cs = dim * log(L / (static_cast<double>(beta_slow) * pi2)) /
                      (2.0 * log(theta_base));
    const double low = truncate != 0u ? floor(cf) : cf;
    const double high = truncate != 0u ? ceil(cs) : cs;
    const double span = high - low;
    double t = (static_cast<double>(i) - low) / (span > 1e-10 ? span : 1e-10);
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    inv_freq[i] = base * ((1.0 - t) + t / f);
}

__device__ __forceinline__ void rope_f32_bf16_pair_body(
    const float* __restrict__ input, uint32_t input_row_stride,
    bf16* __restrict__ output, uint32_t output_row_stride,
    const uint32_t* __restrict__ positions,
    const double* __restrict__ inv_freq,
    uint32_t features, float attention_factor) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t heads = features >> 8;
    const size_t in_base = static_cast<size_t>(row) * input_row_stride;
    const size_t out_base = static_cast<size_t>(row) * output_row_stride;

    float cos_f = 0.0f;
    float sin_f = 0.0f;
    if (tid < kPairHalf) {
        const uint32_t pos = positions != nullptr ? positions[row] : row;
        const double freq = inv_freq[tid];
        const double angle = static_cast<double>(pos) * freq;
        double sn = 0.0;
        double cs = 0.0;
        sincos(angle, &sn, &cs);
        cos_f = static_cast<float>(cs) * attention_factor;
        sin_f = static_cast<float>(sn) * attention_factor;
    }

    if (tid < kPairHalf) {
        #pragma unroll 4
        for (uint32_t head = 0; head < heads; ++head) {
            const size_t in_off = in_base + static_cast<size_t>(head) * kPairHeadDim;
            const size_t out_off = out_base + static_cast<size_t>(head) * kPairHeadDim;
            const float a = input[in_off + tid];
            const float b = input[in_off + tid + kPairHalf];
            const float y0 = __fmaf_rn(a, cos_f, -__fmul_rn(b, sin_f));
            const float y1 = __fmaf_rn(b, cos_f, __fmul_rn(a, sin_f));
            output[out_off + tid] = __float2bfloat16(y0);
            output[out_off + tid + kPairHalf] = __float2bfloat16(y1);
        }
    } else {
        const uint32_t col = kPairRotary + (tid - kPairHalf) * 2u;
        #pragma unroll 4
        for (uint32_t head = 0; head < heads; ++head) {
            const size_t in_off = in_base + static_cast<size_t>(head) * kPairHeadDim + col;
            const size_t out_off = out_base + static_cast<size_t>(head) * kPairHeadDim + col;
            const float2 x = *reinterpret_cast<const float2*>(input + in_off);
            const uint32_t packed =
                static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(x.x))) |
                (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(x.y))) << 16u);
            *reinterpret_cast<uint32_t*>(output + out_off) = packed;
        }
    }
}

}  // namespace

extern "C" __global__ __launch_bounds__(kPairThreads)
void phaseshift_rope_f32_bf16_pair(::shelf::hipk::RopeF32Bf16PairArgs args) {
    rope_f32_bf16_pair_body(
        args.input, args.input_row_stride, reinterpret_cast<bf16*>(args.output),
        args.output_row_stride, args.positions, args.inv_freq, args.features,
        args.attention_factor);
}

bool rope_pair_shape_supported(
    const float* input, bf16_t* output,
    uint32_t input_row_stride, uint32_t output_row_stride,
    uint32_t features, uint32_t head_dim, uint32_t rotary_dim) {
    if (head_dim != kPairHeadDim || rotary_dim != kPairRotary)
        return false;
    if ((features & (kPairHeadDim - 1u)) != 0u)
        return false;
    if ((input_row_stride & 1u) != 0u || (output_row_stride & 1u) != 0u)
        return false;
    if ((reinterpret_cast<uintptr_t>(input) & 7u) != 0u)
        return false;
    if ((reinterpret_cast<uintptr_t>(output) & 3u) != 0u)
        return false;
    return true;
}

hipError_t launch_rope_inv_freq_init(
    double* inv_freq, uint32_t half, float theta, hipStream_t stream) {
    if (inv_freq == nullptr)
        return hipErrorInvalidValue;
    if (half == 0u || half > 32u)
        return hipErrorNotSupported;
    if (theta <= 0.0f)
        return hipErrorInvalidValue;
    rope_inv_freq_init_kernel<<<1, half, 0, stream>>>(inv_freq, half, theta);
    return hipGetLastError();
}

hipError_t launch_rope_inv_freq_yarn_init(
    double* inv_freq, uint32_t half, float theta, float factor,
    uint32_t original_max_position_embeddings, float beta_fast, float beta_slow,
    bool truncate, hipStream_t stream) {
    if (inv_freq == nullptr)
        return hipErrorInvalidValue;
    if (half == 0u || half > 32u)
        return hipErrorNotSupported;
    if (theta <= 0.0f)
        return hipErrorInvalidValue;
    if (!(factor > 1.0f) || !std::isfinite(factor))
        return hipErrorInvalidValue;
    if (original_max_position_embeddings == 0u)
        return hipErrorInvalidValue;
    if (!(beta_fast > 0.0f) || !(beta_slow > 0.0f))
        return hipErrorInvalidValue;
    rope_inv_freq_yarn_init_kernel<<<1, half, 0, stream>>>(
        inv_freq, half, theta, factor,
        static_cast<float>(original_max_position_embeddings), beta_fast, beta_slow,
        truncate ? 1u : 0u);
    return hipGetLastError();
}

hipError_t launch_rope_f32_bf16_generic(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || features == 0u || head_dim == 0u || features % head_dim != 0u)
        return hipErrorInvalidValue;
    if ((rotary_dim & 1u) != 0u || rotary_dim == 0u || rotary_dim > head_dim)
        return hipErrorInvalidValue;
    if (inv_freq == nullptr && theta <= 0.0f)
        return hipErrorInvalidValue;
    if (!(attention_factor > 0.0f) || !std::isfinite(attention_factor))
        return hipErrorInvalidValue;
    const uint32_t half = rotary_dim >> 1;
    if (half > kRopeMaxHalf)
        return hipErrorNotSupported;
    const double* table =
        (inv_freq != nullptr && half <= kRopeInvFreqTable) ? inv_freq : nullptr;
    rope_f32_bf16_kernel<<<dim3(rows), kRopeThreads, 0, stream>>>(
        input, input_row_stride, reinterpret_cast<bf16*>(output), output_row_stride,
        positions, features, head_dim, rotary_dim, theta, table, attention_factor);
    return hipGetLastError();
}

hipError_t launch_rope_f32_bf16_pair(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr || inv_freq == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || features == 0u || head_dim == 0u || features % head_dim != 0u)
        return hipErrorInvalidValue;
    if ((rotary_dim & 1u) != 0u || rotary_dim == 0u || rotary_dim > head_dim)
        return hipErrorInvalidValue;
    if (theta <= 0.0f)
        return hipErrorInvalidValue;
    if (!(attention_factor > 0.0f) || !std::isfinite(attention_factor))
        return hipErrorInvalidValue;
    if (!rope_pair_shape_supported(input, output, input_row_stride, output_row_stride,
                                   features, head_dim, rotary_dim))
        return hipErrorNotSupported;
    RopeF32Bf16PairArgs args{};
    args.input = input;
    args.input_row_stride = input_row_stride;
    args.output = output;
    args.output_row_stride = output_row_stride;
    args.positions = positions;
    args.inv_freq = inv_freq;
    args.features = features;
    args.attention_factor = attention_factor;
    phaseshift_rope_f32_bf16_pair<<<dim3(rows), kPairThreads, 0, stream>>>(
        args);
    return hipGetLastError();
}

hipError_t launch_rope_f32_bf16(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || features == 0u || head_dim == 0u || features % head_dim != 0u)
        return hipErrorInvalidValue;
    if ((rotary_dim & 1u) != 0u || rotary_dim == 0u || rotary_dim > head_dim)
        return hipErrorInvalidValue;
    if (inv_freq == nullptr && theta <= 0.0f)
        return hipErrorInvalidValue;
    if (!(attention_factor > 0.0f) || !std::isfinite(attention_factor))
        return hipErrorInvalidValue;
    if (inv_freq != nullptr &&
        rope_pair_shape_supported(input, output, input_row_stride, output_row_stride,
                                  features, head_dim, rotary_dim)) {
        RopeF32Bf16PairArgs pargs{};
        pargs.input = input;
        pargs.input_row_stride = input_row_stride;
        pargs.output = output;
        pargs.output_row_stride = output_row_stride;
        pargs.positions = positions;
        pargs.inv_freq = inv_freq;
        pargs.features = features;
        pargs.attention_factor = attention_factor;
        phaseshift_rope_f32_bf16_pair<<<dim3(rows), kPairThreads, 0, stream>>>(
            pargs);
        return hipGetLastError();
    }
    const uint32_t half = rotary_dim >> 1;
    if (half > kRopeMaxHalf)
        return hipErrorNotSupported;
    const double* table =
        (inv_freq != nullptr && half <= kRopeInvFreqTable) ? inv_freq : nullptr;
    rope_f32_bf16_kernel<<<dim3(rows), kRopeThreads, 0, stream>>>(
        input, input_row_stride, reinterpret_cast<bf16*>(output), output_row_stride,
        positions, features, head_dim, rotary_dim, theta, table, attention_factor);
    return hipGetLastError();
}

}  // namespace shelf::hipk
