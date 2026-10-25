// vendored from PhaseShift (own work); namespace/type adapted only.
#include <shelf/backend/hip/kernels/elementwise/rmsnorm.h>
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
