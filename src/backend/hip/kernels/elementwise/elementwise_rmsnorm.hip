// vendored from PhaseShift (own work); namespace/type adapted only.
#include <shelf/backend/hip/kernels/elementwise/rmsnorm.h>
#include <shelf/backend/hip/kernels/e4m3_codec.h>
#include <shelf/quant/fp8_perchannel.hpp>
#include <shelf/quant/fp8_finegrained.hpp>
#include <shelf/backend/hip/kernels/detail/rmsnorm_device.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk {

extern "C" __global__ __launch_bounds__(256)
void phaseshift_rmsnorm_bf16_pf_oneplus(RmsNormBf16PfOnePlusArgs args);

extern "C" __global__ __launch_bounds__(256)
void phaseshift_rmsnorm_f32_pg_direct(RmsNormF32PgDirectArgs args);

extern "C" __global__ __launch_bounds__(256)
void phaseshift_rmsnorm_bf16_f32_pg_oneplus(RmsNormBf16F32PgOnePlusArgs args);

namespace {

template <RmsNormDataType InT, RmsNormDataType OutT, RmsNormDataType Wt,
          RmsNormWeightLayout WLayout, RmsNormWeightMode WMode>
__device__ __forceinline__ void rmsnorm_block_body(
    const void* input, const void* weight, void* output,
    uint32_t rows, uint32_t features, uint32_t group_size,
    uint32_t input_row_stride, uint32_t output_row_stride, float eps) {
    (void)rows;
    const uint32_t groups = features / group_size;
    const uint32_t work = blockIdx.x;
    const uint32_t row = static_cast<uint32_t>(work / groups);
    const uint32_t gi = static_cast<uint32_t>(work % groups);
    detail::rmsnorm_row_body<InT, OutT, Wt, WLayout, WMode>(
        input, weight, output, row, gi, group_size,
        input_row_stride, output_row_stride, eps);
}

template <RmsNormDataType InT, RmsNormDataType OutT, RmsNormDataType Wt,
          RmsNormWeightLayout WLayout, RmsNormWeightMode WMode>
__global__ __launch_bounds__(256)
void rmsnorm_impl(const void* input, const void* weight, void* output,
                  uint32_t rows, uint32_t features, uint32_t group_size,
                  uint32_t input_row_stride, uint32_t output_row_stride,
                  float eps) {
    rmsnorm_block_body<InT, OutT, Wt, WLayout, WMode>(
        input, weight, output, rows, features, group_size,
        input_row_stride, output_row_stride, eps);
}

template <RmsNormDataType I, RmsNormDataType O, RmsNormDataType W,
          RmsNormWeightLayout L, RmsNormWeightMode M>
hipError_t launch_one(const void* input, const void* weight, void* output,
                      uint32_t rows, uint32_t features, uint32_t group_size,
                      uint32_t input_row_stride, uint32_t output_row_stride,
                      float eps, hipStream_t stream) {
    const uint32_t grid = rmsnorm_bf16_pf_oneplus_grid(rows, features, group_size);
    if (grid == 0u) return hipSuccess;
    const uint32_t block = rmsnorm_bf16_pf_oneplus_block_size(group_size);
    if constexpr (I == RmsNormDataType::BF16 &&
                  O == RmsNormDataType::BF16 &&
                  W == RmsNormDataType::BF16 &&
                  L == RmsNormWeightLayout::PER_FEATURE &&
                  M == RmsNormWeightMode::ONE_PLUS) {
        RmsNormBf16PfOnePlusArgs args{};
        args.input = input;
        args.weight = weight;
        args.output = output;
        args.rows = rows;
        args.features = features;
        args.group_size = group_size;
        args.input_row_stride = input_row_stride;
        args.output_row_stride = output_row_stride;
        args.eps = eps;
        phaseshift_rmsnorm_bf16_pf_oneplus<<<grid, block, 0, stream>>>(args);
    } else if constexpr (I == RmsNormDataType::BF16 &&
                         O == RmsNormDataType::F32 &&
                         W == RmsNormDataType::BF16 &&
                         L == RmsNormWeightLayout::PER_GROUP &&
                         M == RmsNormWeightMode::ONE_PLUS) {
        RmsNormBf16F32PgOnePlusArgs args{};
        args.input = input;
        args.weight = weight;
        args.output = output;
        args.rows = rows;
        args.features = features;
        args.group_size = group_size;
        args.input_row_stride = input_row_stride;
        args.output_row_stride = output_row_stride;
        args.eps = eps;
        phaseshift_rmsnorm_bf16_f32_pg_oneplus<<<grid, block, 0, stream>>>(args);
    } else if constexpr (I == RmsNormDataType::F32 &&
                         O == RmsNormDataType::F32 &&
                         W == RmsNormDataType::BF16 &&
                         L == RmsNormWeightLayout::PER_GROUP &&
                         M == RmsNormWeightMode::DIRECT) {
        RmsNormF32PgDirectArgs args{};
        args.input = input;
        args.weight = weight;
        args.output = output;
        args.rows = rows;
        args.features = features;
        args.group_size = group_size;
        args.input_row_stride = input_row_stride;
        args.output_row_stride = output_row_stride;
        args.eps = eps;
        phaseshift_rmsnorm_f32_pg_direct<<<grid, block, 0, stream>>>(args);
    } else {
        rmsnorm_impl<I, O, W, L, M><<<grid, block, 0, stream>>>(
            input, weight, output, rows, features, group_size,
            input_row_stride, output_row_stride, eps);
    }
    return hipGetLastError();
}

}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_rmsnorm_bf16_pf_oneplus(
    RmsNormBf16PfOnePlusArgs args) {
    rmsnorm_block_body<RmsNormDataType::BF16, RmsNormDataType::BF16,
                       RmsNormDataType::BF16, RmsNormWeightLayout::PER_FEATURE,
                       RmsNormWeightMode::ONE_PLUS>(
        args.input, args.weight, args.output, args.rows, args.features,
        args.group_size, args.input_row_stride, args.output_row_stride,
        args.eps);
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_rmsnorm_bf16_f32_pg_oneplus(
    RmsNormBf16F32PgOnePlusArgs args) {
    rmsnorm_block_body<RmsNormDataType::BF16, RmsNormDataType::F32,
                       RmsNormDataType::BF16, RmsNormWeightLayout::PER_GROUP,
                       RmsNormWeightMode::ONE_PLUS>(
        args.input, args.weight, args.output, args.rows, args.features,
        args.group_size, args.input_row_stride, args.output_row_stride, args.eps);
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_rmsnorm_f32_pg_direct(RmsNormF32PgDirectArgs args) {
    rmsnorm_block_body<RmsNormDataType::F32, RmsNormDataType::F32,
                       RmsNormDataType::BF16, RmsNormWeightLayout::PER_GROUP,
                       RmsNormWeightMode::DIRECT>(
        args.input, args.weight, args.output, args.rows, args.features,
        args.group_size, args.input_row_stride, args.output_row_stride, args.eps);
}

uint32_t rmsnorm_bf16_pf_oneplus_block_size(uint32_t group_size) noexcept {
    const uint32_t chunks = (group_size + 3u) / 4u;
    return chunks <= 32u ? 32u : chunks <= 64u ? 64u : chunks <= 128u ? 128u
                                                                      : 256u;
}

uint32_t rmsnorm_bf16_pf_oneplus_grid(uint32_t rows, uint32_t features,
                                      uint32_t group_size) noexcept {
    if (group_size == 0u || features == 0u || features % group_size != 0u) {
        return 0u;
    }
    return rows * (features / group_size);
}

// ---- rmsnorm + fp8 活性化量子化の融合 --------------------------------------
// 消費側が fp8 GEMM のとき、norm の bf16 出力を読み直して量子化する pass を丸ごと
// 消せる（radiance は `rmsnorm_quant_fp8` として 1 op にしている）。
// 演算は既存の bf16→bf16・PER_FEATURE・ONE_PLUS 本体と同じ順で行い、量子化は
// **bf16 に丸めた後の値**から行う（现状の 2 段と 1 バイトも違わないようにするため）。
namespace {

constexpr std::uint32_t kRmsQThreads = 256u;
constexpr std::uint32_t kRmsQLanes = 32u;
constexpr std::uint32_t kRmsQWaves = kRmsQThreads / kRmsQLanes;
using shelf::quant::fp8::fine::kActGroupK;
using shelf::quant::fp8::kTileRows;
using shelf::quant::fp8::kSuperK;
using shelf::quant::fp8::amax_scale;

// quant/fp8 の tiled_offset と同一（16 行 × 64 K タイル）。片方を壊すと
// tests/unit/test_hip_gemm_fp8.cpp の readback が落ちる。
__device__ __forceinline__ std::size_t rmsq_code_offset(std::uint32_t row, std::uint32_t col,
                                                        std::uint32_t k_padded) {
    const std::uint32_t lane = (row % kTileRows) + kTileRows * ((col % 16u) / 8u);
    return static_cast<std::size_t>(row / kTileRows) *
               (static_cast<std::size_t>(k_padded) * kTileRows) +
           static_cast<std::size_t>(col / kSuperK) * (kTileRows * kSuperK) +
           static_cast<std::size_t>(lane) * 32u +
           static_cast<std::size_t>((col % kSuperK) / 16u) * 8u +
           static_cast<std::size_t>(col % 8u);
}

__global__ void __launch_bounds__(kRmsQThreads) rmsnorm_quant_fp8_kernel(   // block <= 256
    const detail::bf16* __restrict__ input, const detail::bf16* __restrict__ weight,
    detail::bf16* __restrict__ output, std::uint8_t* __restrict__ codes,
    float* __restrict__ scales, const std::uint32_t features,
    const std::uint32_t k_padded, const std::uint32_t input_row_stride,
    const std::uint32_t output_row_stride, const float eps) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31u;
    const std::uint32_t wave = tid >> 5u;
    const std::uint32_t num_waves = blockDim.x >> 5u;
    __shared__ float wave_sums[8];

    const std::size_t in_off = static_cast<std::size_t>(row) * input_row_stride;
    const std::size_t out_off = static_cast<std::size_t>(row) * output_row_stride;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (std::uint32_t base = tid * 4u; base < features; base += blockDim.x * 4u) {
        const detail::RmsF32x4 x = detail::rms_load_bf16_x4(input + in_off + base);
        sum0 += x.x0 * x.x0;
        sum1 += x.x1 * x.x1;
        sum0 += x.x2 * x.x2;
        sum1 += x.x3 * x.x3;
    }
    float total = detail::rms_wave_reduce_sum(sum0 + sum1);
    if (lane == 0u) wave_sums[wave] = total;
    __syncthreads();
    if (wave == 0u) {
        float bs = (lane < num_waves) ? wave_sums[lane] : 0.0f;
        bs = detail::rms_wave_reduce_sum(bs);
        if (lane == 0u) wave_sums[0] = rsqrtf(bs / static_cast<float>(features) + eps);
    }
    __syncthreads();
    const float inv = wave_sums[0];

    const std::uint32_t k_blocks = features / kActGroupK;
    for (std::uint32_t base = tid * 4u; base < features; base += blockDim.x * 4u) {
        const detail::RmsF32x4 x = detail::rms_load_bf16_x4(input + in_off + base);
        detail::RmsF32x4 y;
        y.x0 = x.x0 * inv;
        y.x1 = x.x1 * inv;
        y.x2 = x.x2 * inv;
        y.x3 = x.x3 * inv;
        const detail::bf16* wp = weight + base;
        y.x0 *= 1.0f + __bfloat162float(wp[0]);
        y.x1 *= 1.0f + __bfloat162float(wp[1]);
        y.x2 *= 1.0f + __bfloat162float(wp[2]);
        y.x3 *= 1.0f + __bfloat162float(wp[3]);
        detail::rms_store_bf16_x4(output + out_off + base, y);

        // bf16 に丸めた値で量子化する（本体が書き出したものをそのまま読むのと同じ）。
        float v0 = __bfloat162float(__float2bfloat16(y.x0));
        float v1 = __bfloat162float(__float2bfloat16(y.x1));
        float v2 = __bfloat162float(__float2bfloat16(y.x2));
        float v3 = __bfloat162float(__float2bfloat16(y.x3));
        float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        for (std::uint32_t off = 16u; off > 0u; off >>= 1u)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffull, amax, off));
        const float sc = amax_scale(amax);
        const float inv2 = (sc > 0.0f) ? 1.0f / sc : 0.0f;
        if (lane == 0u)
            scales[static_cast<std::size_t>(row) * k_blocks + base / kActGroupK] = sc;
        std::uint32_t packed = 0u;
        packed |= static_cast<std::uint32_t>(sc > 0.0f ? e4m3_encode_u8(v0 * inv2) : 0u);
        packed |= static_cast<std::uint32_t>(sc > 0.0f ? e4m3_encode_u8(v1 * inv2) : 0u) << 8u;
        packed |= static_cast<std::uint32_t>(sc > 0.0f ? e4m3_encode_u8(v2 * inv2) : 0u) << 16u;
        packed |= static_cast<std::uint32_t>(sc > 0.0f ? e4m3_encode_u8(v3 * inv2) : 0u) << 24u;
        *reinterpret_cast<std::uint32_t*>(codes + rmsq_code_offset(row, base, k_padded)) = packed;
    }
}

}  // namespace

hipError_t launch_rmsnorm_bf16_quantize(
    const void* input, const void* weight, void* output,
    std::uint8_t* codes, float* scales, std::uint32_t rows, std::uint32_t features,
    std::uint32_t input_row_stride, std::uint32_t output_row_stride, float eps,
    hipStream_t stream) {
    auto* in = reinterpret_cast<const detail::bf16*>(input);
    auto* w = reinterpret_cast<const detail::bf16*>(weight);
    auto* out = reinterpret_cast<detail::bf16*>(output);
    if (input == nullptr || weight == nullptr || output == nullptr ||
        codes == nullptr || scales == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || features == 0u || (features % kActGroupK) != 0u)
        return hipErrorNotSupported;
    if (shelf::quant::fp8::pad_k(features) != features) return hipErrorNotSupported;
    if ((input_row_stride & 3u) || (output_row_stride & 3u)) return hipErrorNotSupported;
    // block サイズは既存の oneplus 経路と**必ず同じ**にする。スレッド当たりの
    // 要素分担が変わると sumsq の合計順が変わり、1 ビット違った出力になる
    // （実測で 27B の logits がずれた: 固定 256 スレッドで走らせた版）。
    const std::uint32_t block = rmsnorm_bf16_pf_oneplus_block_size(features);
    rmsnorm_quant_fp8_kernel<<<dim3(rows), block, 0, stream>>>(
        in, w, out, codes, scales, features, features,
        input_row_stride, output_row_stride, eps);
    return hipGetLastError();
}

hipError_t launch_rmsnorm(
    const void* input,
    RmsNormDataType input_dtype,
    const void* weight,
    RmsNormDataType weight_dtype,
    RmsNormWeightLayout weight_layout,
    RmsNormWeightMode weight_mode,
    void* output,
    RmsNormDataType output_dtype,
    uint32_t rows,
    uint32_t features,
    uint32_t group_size,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    float eps,
    hipStream_t stream) {
    using T = RmsNormDataType;
    using L = RmsNormWeightLayout;
    using M = RmsNormWeightMode;
    if (group_size == 0u || features == 0u || features % group_size != 0u) {
        return hipErrorInvalidValue;
    }
    if (input_dtype != T::BF16 && input_dtype != T::F32) return hipErrorInvalidValue;
    if (output_dtype != T::BF16 && output_dtype != T::F32) return hipErrorInvalidValue;
    if (weight_dtype != T::BF16) return hipErrorInvalidValue;
    const bool in_b = input_dtype == T::BF16;
    const bool out_b = output_dtype == T::BF16;
    const bool pf = weight_layout == L::PER_FEATURE;
    const bool pg = weight_layout == L::PER_GROUP;
    const bool none = weight_layout == L::NONE;
    const bool op = weight_mode == M::ONE_PLUS;
    if (in_b && out_b) {
        if (pf) {
            if (op) return launch_one<T::BF16, T::BF16, T::BF16, L::PER_FEATURE, M::ONE_PLUS>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
            return launch_one<T::BF16, T::BF16, T::BF16, L::PER_FEATURE, M::DIRECT>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        }
        if (pg) {
            if (op) return launch_one<T::BF16, T::BF16, T::BF16, L::PER_GROUP, M::ONE_PLUS>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
            return launch_one<T::BF16, T::BF16, T::BF16, L::PER_GROUP, M::DIRECT>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        }
        if (none) return launch_one<T::BF16, T::BF16, T::BF16, L::NONE, M::ONE_PLUS>(
            input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        return hipErrorInvalidValue;
    }
    if (in_b && !out_b) {
        if (pf) {
            if (op) return launch_one<T::BF16, T::F32, T::BF16, L::PER_FEATURE, M::ONE_PLUS>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
            return launch_one<T::BF16, T::F32, T::BF16, L::PER_FEATURE, M::DIRECT>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        }
        if (pg) {
            if (op) return launch_one<T::BF16, T::F32, T::BF16, L::PER_GROUP, M::ONE_PLUS>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
            return launch_one<T::BF16, T::F32, T::BF16, L::PER_GROUP, M::DIRECT>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        }
        return hipErrorInvalidValue;
    }
    if (!in_b && out_b) {
        // f32 入力 → bf16 出力。本体は InT==F32 のとき store 前に
        // `rms_round_to_bf16_f32` を噛ませるので、「f32 で置いてから別 pass で
        // 変換する」と値は 1 バイト違わない（実 fixture で md5 一致を確認）。
        if (pf) {
            if (op) return launch_one<T::F32, T::BF16, T::BF16, L::PER_FEATURE, M::ONE_PLUS>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
            return launch_one<T::F32, T::BF16, T::BF16, L::PER_FEATURE, M::DIRECT>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        }
        if (pg) {
            if (op) return launch_one<T::F32, T::BF16, T::BF16, L::PER_GROUP, M::ONE_PLUS>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
            return launch_one<T::F32, T::BF16, T::BF16, L::PER_GROUP, M::DIRECT>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        }
        return hipErrorInvalidValue;
    }
    if (!in_b && !out_b) {
        if (pf) {
            if (op) return launch_one<T::F32, T::F32, T::BF16, L::PER_FEATURE, M::ONE_PLUS>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
            return launch_one<T::F32, T::F32, T::BF16, L::PER_FEATURE, M::DIRECT>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        }
        if (pg) {
            if (op) return launch_one<T::F32, T::F32, T::BF16, L::PER_GROUP, M::ONE_PLUS>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
            return launch_one<T::F32, T::F32, T::BF16, L::PER_GROUP, M::DIRECT>(
                input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        }
        if (none) return launch_one<T::F32, T::F32, T::BF16, L::NONE, M::DIRECT>(
            input, weight, output, rows, features, group_size, input_row_stride, output_row_stride, eps, stream);
        return hipErrorInvalidValue;
    }
    return hipErrorInvalidValue;
}

}  // namespace shelf::hipk
