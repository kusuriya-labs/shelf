#include <shelf/kernels/optimized/elementwise.h>
#include "detail/vector_io.h"
#include "detail/residual_add_device.h"
#include "detail/target_swiglu_device.h"
#include "gdn/detail/gdn_prepare_device.h"
#include "detail/sigmoid_device.h"
#include "gdn/detail/gdn_post_device.h"
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::kernel {

using namespace ::shelf::kernel::detail;
constexpr uint32_t kThreads = 256u;
static_assert(kThreads == kElementwiseBlock);
static_assert(kVec == kElementwiseVec);

__device__ __forceinline__ uint32_t elementwise_row() {
    return blockIdx.y + blockIdx.z * gridDim.y;
}

namespace {
namespace op_concat_bf16 {

__global__ void impl(const bf16* a, uint32_t a_rs,
                     const bf16* b, uint32_t b_rs,
                     bf16* out, uint32_t out_rs,
                     uint32_t fa, uint32_t features, uint32_t rows) {
    const uint32_t row = elementwise_row();
    if (row >= rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= features) return;
    const bf16* a_row = a + static_cast<size_t>(row) * a_rs;
    const bf16* b_row = b + static_cast<size_t>(row) * b_rs;
    bf16* out_row = out + static_cast<size_t>(row) * out_rs;
    const bool from_a = (col0 + kVec <= fa);
    const bool from_b = (col0 >= fa);
    if (from_a || from_b) {
        const bf16* src_row = from_a ? a_row : b_row;
        const uint32_t src_col = from_a ? col0 : (col0 - fa);
        if (col0 + kVec <= features && aligned8(src_row + src_col) && aligned8(out_row + col0)) {
            store_bf16x4(out_row + col0, load_bf16x4(src_row + src_col));
            return;
        }
    }
    for (uint32_t i = 0u; i < kVec; ++i) {
        const uint32_t c = col0 + i;
        if (c >= features) break;
        out_row[c] = (c < fa) ? a_row[c] : b_row[c - fa];
    }
}

}  // namespace op_concat_bf16
}  // namespace

hipError_t launch_concat_bf16(
    const bf16_t* a, uint32_t a_row_stride,
    const bf16_t* b, uint32_t b_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features_a, uint32_t features_b, hipStream_t stream) {
    if (rows == 0u || features_a == 0u || features_b == 0u) return hipErrorInvalidValue;
    if (a == nullptr || b == nullptr || output == nullptr) return hipErrorInvalidValue;
    const uint32_t features = features_a + features_b;
    const dim3 grid = elementwise_grid(features, rows);
    op_concat_bf16::impl<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(a), a_row_stride,
        reinterpret_cast<const bf16*>(b), b_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride, features_a, features, rows);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void shelf_residual_add_bf16(ElementwiseArgs args) {
    const bf16* a = static_cast<const bf16*>(args.in0);
    const bf16* b = static_cast<const bf16*>(args.in1);
    bf16* out = static_cast<bf16*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = elementwise_row();
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= features) return;
    residual_add_bf16_body(a + static_cast<size_t>(row) * args.in0_row_stride,
                           b + static_cast<size_t>(row) * args.in1_row_stride,
                           out + static_cast<size_t>(row) * args.out0_row_stride,
                           col0, features);
}

hipError_t launch_residual_add_bf16(
    const bf16_t* a, uint32_t a_row_stride,
    const bf16_t* b, uint32_t b_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if (a == nullptr || b == nullptr || output == nullptr) return hipErrorInvalidValue;
    ElementwiseArgs args{};
    args.in0 = a;
    args.in0_row_stride = a_row_stride;
    args.in1 = b;
    args.in1_row_stride = b_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.rows = rows;
    const dim3 grid = elementwise_grid(features, rows);
    shelf_residual_add_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void shelf_scale_f32(ElementwiseArgs args) {
    const float* in = static_cast<const float*>(args.in0);
    float* out = static_cast<float*>(args.out0);
    const uint32_t features = args.features;
    const float scale = args.scalar_a;
    const uint32_t row = elementwise_row();
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= features) return;
    const float* in_row = in + static_cast<size_t>(row) * args.in0_row_stride;
    float* out_row = out + static_cast<size_t>(row) * args.out0_row_stride;
    if (col0 + kVec - 1u < features && aligned16(in_row + col0) && aligned16(out_row + col0)) {
        const F4 x = load_f32x4(in_row + col0);
        F4 y;
        y.v[0] = x.v[0] * scale;
        y.v[1] = x.v[1] * scale;
        y.v[2] = x.v[2] * scale;
        y.v[3] = x.v[3] * scale;
        store_f32x4(out_row + col0, y);
    } else {
        for (uint32_t i = 0u; i < kVec; ++i) {
            const uint32_t c = col0 + i;
            if (c >= features) break;
            out_row[c] = in_row[c] * scale;
        }
    }
}

__global__ void shelf_sigmoid_f32(ElementwiseArgs args) {
    const uint32_t row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kElementwiseVec;
    if (col0 >= args.features) return;
    const auto* in_row = static_cast<const float*>(args.in0) +
                         static_cast<std::uint64_t>(row) * args.in0_row_stride;
    auto* out_row = static_cast<float*>(args.out0) +
                    static_cast<std::uint64_t>(row) * args.out0_row_stride;
    for (uint32_t i = 0u; i < kElementwiseVec; ++i) {
        const uint32_t c = col0 + i;
        if (c >= args.features) break;
        out_row[c] = 1.0f / (1.0f + __expf(-in_row[c]));
    }
}

__global__ void shelf_sigmoid_bf16(ElementwiseArgs args) {
    const uint32_t row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kElementwiseVec;
    if (col0 >= args.features) return;
    const auto* in_row = reinterpret_cast<const __hip_bfloat16*>(args.in0) +
                         static_cast<std::uint64_t>(row) * args.in0_row_stride;
    auto* out_row = reinterpret_cast<__hip_bfloat16*>(args.out0) +
                    static_cast<std::uint64_t>(row) * args.out0_row_stride;
    for (uint32_t i = 0u; i < kElementwiseVec; ++i) {
        const uint32_t c = col0 + i;
        if (c >= args.features) break;
        const float x = __bfloat162float(in_row[c]);
        out_row[c] = __float2bfloat16(1.0f / (1.0f + __expf(-x)));
    }
}

__global__ void shelf_scale_bf16(ElementwiseArgs args) {
    const uint32_t row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kElementwiseVec;
    if (col0 >= args.features) return;
    const auto* in_row = reinterpret_cast<const __hip_bfloat16*>(args.in0) +
                         static_cast<std::uint64_t>(row) * args.in0_row_stride;
    auto* out_row = reinterpret_cast<__hip_bfloat16*>(args.out0) +
                    static_cast<std::uint64_t>(row) * args.out0_row_stride;
    for (uint32_t i = 0u; i < kElementwiseVec; ++i) {
        const uint32_t c = col0 + i;
        if (c >= args.features) break;
        out_row[c] = __float2bfloat16(__bfloat162float(in_row[c]) * args.scalar_a);
    }
}

hipError_t launch_sigmoid_f32(
    const float* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0u || features == 0u)
        return hipErrorInvalidValue;
    ElementwiseArgs args;
    args.in0 = input;
    args.in0_row_stride = input_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.rows = rows;
    shelf_sigmoid_f32<<<elementwise_grid(features, rows), kElementwiseBlock, 0,
                        stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_sigmoid_bf16(
    const bf16_t* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0u || features == 0u)
        return hipErrorInvalidValue;
    ElementwiseArgs args;
    args.in0 = input;
    args.in0_row_stride = input_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.rows = rows;
    shelf_sigmoid_bf16<<<elementwise_grid(features, rows), kElementwiseBlock, 0,
                         stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_scale_bf16(
    const bf16_t* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, float scale, hipStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0u || features == 0u) {
        return hipErrorInvalidValue;
    }
    ElementwiseArgs args;
    args.in0 = input;
    args.in0_row_stride = input_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.scalar_a = scale;
    args.rows = rows;
    const dim3 grid = elementwise_grid(features, rows);
    shelf_scale_bf16<<<grid, kElementwiseBlock, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_scale_f32(
    const float* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, float scale, hipStream_t stream) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if (input == nullptr || output == nullptr) return hipErrorInvalidValue;
    ElementwiseArgs args{};
    args.in0 = input;
    args.in0_row_stride = input_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.scalar_a = scale;
    args.rows = rows;
    const dim3 grid = elementwise_grid(features, rows);
    shelf_scale_f32<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void shelf_sigmoid_bf16_f32(ElementwiseArgs args) {
    const bf16* in = static_cast<const bf16*>(args.in0);
    float* out = static_cast<float*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = elementwise_row();
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= features) return;
    const bf16* in_row = in + static_cast<size_t>(row) * args.in0_row_stride;
    float* out_row = out + static_cast<size_t>(row) * args.out0_row_stride;
    if (col0 + kVec - 1u < features && aligned8(in_row + col0) && aligned16(out_row + col0)) {
        const F4 x = load_bf16x4(in_row + col0);
        F4 y;
        y.v[0] = detail::sigmoid_f32(x.v[0]);
        y.v[1] = detail::sigmoid_f32(x.v[1]);
        y.v[2] = detail::sigmoid_f32(x.v[2]);
        y.v[3] = detail::sigmoid_f32(x.v[3]);
        store_f32x4(out_row + col0, y);
    } else {
        for (uint32_t i = 0u; i < kVec; ++i) {
            const uint32_t c = col0 + i;
            if (c >= features) break;
            out_row[c] = detail::sigmoid_f32(__bfloat162float(in_row[c]));
        }
    }
}

hipError_t launch_sigmoid_bf16_f32(
    const bf16_t* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if (input == nullptr || output == nullptr) return hipErrorInvalidValue;
    ElementwiseArgs args{};
    args.in0 = input;
    args.in0_row_stride = input_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.rows = rows;
    const dim3 grid = elementwise_grid(features, rows);
    shelf_sigmoid_bf16_f32<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void shelf_silu_bf16_f32(ElementwiseArgs args) {
    const bf16* in = static_cast<const bf16*>(args.in0);
    float* out = static_cast<float*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = elementwise_row();
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= features) return;
    const bf16* in_row = in + static_cast<size_t>(row) * args.in0_row_stride;
    float* out_row = out + static_cast<size_t>(row) * args.out0_row_stride;
    if (col0 + kVec - 1u < features && aligned8(in_row + col0) && aligned16(out_row + col0)) {
        const F4 x = load_bf16x4(in_row + col0);
        F4 y;
        y.v[0] = detail::gdn_silu_f32(x.v[0]);
        y.v[1] = detail::gdn_silu_f32(x.v[1]);
        y.v[2] = detail::gdn_silu_f32(x.v[2]);
        y.v[3] = detail::gdn_silu_f32(x.v[3]);
        store_f32x4(out_row + col0, y);
    } else {
        for (uint32_t i = 0u; i < kVec; ++i) {
            const uint32_t c = col0 + i;
            if (c >= features) break;
            out_row[c] = detail::gdn_silu_f32(__bfloat162float(in_row[c]));
        }
    }
}

hipError_t launch_silu_bf16_f32(
    const bf16_t* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if (input == nullptr || output == nullptr) return hipErrorInvalidValue;
    ElementwiseArgs args{};
    args.in0 = input;
    args.in0_row_stride = input_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.rows = rows;
    const dim3 grid = elementwise_grid(features, rows);
    shelf_silu_bf16_f32<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void shelf_silu_f32_bf16(ElementwiseArgs args) {
    const float* in = static_cast<const float*>(args.in0);
    bf16* out = static_cast<bf16*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = elementwise_row();
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= features) return;
    const float* in_row = in + static_cast<size_t>(row) * args.in0_row_stride;
    bf16* out_row = out + static_cast<size_t>(row) * args.out0_row_stride;
    if (col0 + kVec - 1u < features && aligned16(in_row + col0) && aligned8(out_row + col0)) {
        const F4 x = load_f32x4(in_row + col0);
        F4 y;
        y.v[0] = detail::gdn_silu_f32(x.v[0]);
        y.v[1] = detail::gdn_silu_f32(x.v[1]);
        y.v[2] = detail::gdn_silu_f32(x.v[2]);
        y.v[3] = detail::gdn_silu_f32(x.v[3]);
        store_bf16x4(out_row + col0, y);
    } else {
        for (uint32_t i = 0u; i < kVec; ++i) {
            const uint32_t c = col0 + i;
            if (c >= features) break;
            out_row[c] = __float2bfloat16(detail::gdn_silu_f32(in_row[c]));
        }
    }
}

hipError_t launch_silu_f32_bf16(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if (input == nullptr || output == nullptr) return hipErrorInvalidValue;
    ElementwiseArgs args{};
    args.in0 = input;
    args.in0_row_stride = input_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.rows = rows;
    const dim3 grid = elementwise_grid(features, rows);
    shelf_silu_f32_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void shelf_mul_f32_f32_bf16(ElementwiseArgs args) {
    const float* a = static_cast<const float*>(args.in0);
    const float* b = static_cast<const float*>(args.in1);
    bf16* out = static_cast<bf16*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = elementwise_row();
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    const float* a_row = a + static_cast<size_t>(row) * args.in0_row_stride;
    const float* b_row = b + static_cast<size_t>(row) * args.in1_row_stride;
    bf16* out_row = out + static_cast<size_t>(row) * args.out0_row_stride;
    detail::gdn_mul_f32_bf16_body(a_row, b_row, out_row, col0, features);
}

hipError_t launch_mul_f32_f32_bf16(
    const float* a, uint32_t a_row_stride,
    const float* b, uint32_t b_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if (a == nullptr || b == nullptr || output == nullptr) return hipErrorInvalidValue;
    ElementwiseArgs args{};
    args.in0 = a;
    args.in0_row_stride = a_row_stride;
    args.in1 = b;
    args.in1_row_stride = b_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.rows = rows;
    const dim3 grid = elementwise_grid(features, rows);
    shelf_mul_f32_f32_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void shelf_split_bf16(ElementwiseArgs args) {
    const bf16* in = static_cast<const bf16*>(args.in0);
    bf16* o0 = static_cast<bf16*>(args.out0);
    bf16* o1 = static_cast<bf16*>(args.out1);
    const uint32_t f0 = args.features;
    const uint32_t hd = args.aux;
    const bool interleaved = args.reserved != 0u;
    const uint32_t row = elementwise_row();
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= f0) return;
    const bf16* in_row = in + static_cast<size_t>(row) * args.in0_row_stride;
    bf16* o0_row = o0 + static_cast<size_t>(row) * args.out0_row_stride;
    bf16* o1_row = o1 + static_cast<size_t>(row) * args.out1_row_stride;
    if (interleaved) {
        const uint32_t h = col0 / hd;
        const uint32_t o = col0 % hd;
        const size_t base0 = static_cast<size_t>(h) * 2u * hd + o;
        if (o + kVec <= hd && aligned8(in_row + base0) && aligned8(in_row + base0 + hd) &&
            aligned8(o0_row + col0) && aligned8(o1_row + col0)) {
            const F4 v0 = load_bf16x4(in_row + base0);
            const F4 v1 = load_bf16x4(in_row + base0 + hd);
            store_bf16x4(o0_row + col0, v0);
            store_bf16x4(o1_row + col0, v1);
        } else {
            for (uint32_t i = 0u; i < kVec; ++i) {
                const uint32_t c = col0 + i;
                if (c >= f0) break;
                const size_t base =
                    static_cast<size_t>(c / hd) * 2u * hd + (c % hd);
                o0_row[c] = in_row[base];
                o1_row[c] = in_row[base + hd];
            }
        }
    } else {
        if (col0 + kVec <= f0 && aligned8(in_row + col0) && aligned8(in_row + col0 + f0) &&
            aligned8(o0_row + col0) && aligned8(o1_row + col0)) {
            const F4 v0 = load_bf16x4(in_row + col0);
            const F4 v1 = load_bf16x4(in_row + col0 + f0);
            store_bf16x4(o0_row + col0, v0);
            store_bf16x4(o1_row + col0, v1);
        } else {
            for (uint32_t i = 0u; i < kVec; ++i) {
                const uint32_t c = col0 + i;
                if (c >= f0) break;
                o0_row[c] = in_row[c];
                o1_row[c] = in_row[c + f0];
            }
        }
    }
}

__global__ void scale_bf16_f32_kernel(const bf16* input, uint32_t in_stride,
    float* output, uint32_t out_stride, uint32_t rows, uint32_t features, float scale) {
    const uint32_t row = elementwise_row();
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < features)
        output[static_cast<size_t>(row) * out_stride + col] =
            __bfloat162float(input[static_cast<size_t>(row) * in_stride + col]) * scale;
}

hipError_t launch_scale_bf16_f32(const bf16_t* input, uint32_t in_stride,
    float* output, uint32_t out_stride, uint32_t rows, uint32_t features,
    float scale, hipStream_t stream) {
    if (!input || !output || !rows || !features) return hipErrorInvalidValue;
    const dim3 grid((features + kThreads - 1u) / kThreads, elementwise_grid_y(rows),
                    (rows + kElementwiseGridYMax - 1u) / kElementwiseGridYMax);
    scale_bf16_f32_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(input), in_stride, output, out_stride, rows, features, scale);
    return hipGetLastError();
}

__global__ void mul_bf16_kernel(
    const bf16* a, uint32_t as, const bf16* b, uint32_t bs,
    bf16* out, uint32_t os, uint32_t rows, uint32_t features) {
    const uint32_t row = elementwise_row();
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= features) return;
    const float av = __bfloat162float(a[static_cast<size_t>(row) * as + col]);
    const float bv = __bfloat162float(b[static_cast<size_t>(row) * bs + col]);
    out[static_cast<size_t>(row) * os + col] = __float2bfloat16(av * bv);
}

hipError_t launch_mul_bf16(
    const bf16_t* a, uint32_t as, const bf16_t* b, uint32_t bs,
    bf16_t* out, uint32_t os, uint32_t rows, uint32_t features, hipStream_t stream) {
    if (!a || !b || !out || !rows || !features) return hipErrorInvalidValue;
    const dim3 grid((features + kThreads - 1u) / kThreads, elementwise_grid_y(rows),
                    (rows + kElementwiseGridYMax - 1u) / kElementwiseGridYMax);
    mul_bf16_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(a), as, reinterpret_cast<const bf16*>(b), bs,
        reinterpret_cast<bf16*>(out), os, rows, features);
    return hipGetLastError();
}

__global__ void split_uneven_bf16_kernel(
    const bf16* input, uint32_t input_stride, bf16* a, uint32_t a_stride,
    bf16* b, uint32_t b_stride, uint32_t rows, uint32_t fa, uint32_t fb) {
    const uint32_t row = elementwise_row();
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= fa + fb) return;
    const bf16 v = input[static_cast<size_t>(row) * input_stride + col];
    if (col < fa) a[static_cast<size_t>(row) * a_stride + col] = v;
    else b[static_cast<size_t>(row) * b_stride + col - fa] = v;
}

hipError_t launch_split_uneven_bf16(
    const bf16_t* input, uint32_t input_stride, bf16_t* a, uint32_t a_stride,
    bf16_t* b, uint32_t b_stride, uint32_t rows, uint32_t fa, uint32_t fb,
    hipStream_t stream) {
    if (!input || !a || !b || !rows || !fa || !fb) return hipErrorInvalidValue;
    const uint32_t gx = (fa + fb + kThreads - 1u) / kThreads;
    const dim3 grid(gx, elementwise_grid_y(rows), (rows + kElementwiseGridYMax - 1u) / kElementwiseGridYMax);
    split_uneven_bf16_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(input), input_stride, reinterpret_cast<bf16*>(a), a_stride,
        reinterpret_cast<bf16*>(b), b_stride, rows, fa, fb);
    return hipGetLastError();
}

hipError_t launch_split_bf16(
    const bf16_t* input, uint32_t input_row_stride,
    bf16_t* o0, uint32_t o0_row_stride,
    bf16_t* o1, uint32_t o1_row_stride,
    uint32_t rows, uint32_t features,
    ElementwiseSplitLayout layout, uint32_t head_dim, hipStream_t stream) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if (input == nullptr || o0 == nullptr || o1 == nullptr) return hipErrorInvalidValue;
    const bool interleaved = (layout == ElementwiseSplitLayout::InterleavedHeads);
    if (interleaved) {
        if (head_dim == 0u || features % head_dim != 0u) return hipErrorInvalidValue;
    }
    ElementwiseArgs args{};
    args.in0 = input;
    args.in0_row_stride = input_row_stride;
    args.out0 = o0;
    args.out0_row_stride = o0_row_stride;
    args.out1 = o1;
    args.out1_row_stride = o1_row_stride;
    args.features = features;
    args.aux = head_dim;
    args.reserved = interleaved ? 1u : 0u;
    args.rows = rows;
    const dim3 grid = elementwise_grid(features, rows);
    shelf_split_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

__device__ __forceinline__ float swiglu_value(float gate, float up, bool bf16_intermediate) {
    if (!bf16_intermediate) return target_swiglu_f32(gate, up);
    const float activated = gate / (1.0f + __expf(-gate));
    return up * __bfloat162float(__float2bfloat16(activated));
}

extern "C" __global__ __launch_bounds__(256)
void shelf_swiglu_bf16(ElementwiseArgs args) {
    const bf16* gate = static_cast<const bf16*>(args.in0);
    const bf16* up = static_cast<const bf16*>(args.in1);
    bf16* out = static_cast<bf16*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = elementwise_row();
    if (row >= args.rows) return;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= features) return;
    const bf16* gate_row = gate + static_cast<size_t>(row) * args.in0_row_stride;
    const bf16* up_row = up + static_cast<size_t>(row) * args.in1_row_stride;
    bf16* out_row = out + static_cast<size_t>(row) * args.out0_row_stride;
    if (col0 + kVec - 1u < features && aligned8(gate_row + col0) && aligned8(up_row + col0) &&
        aligned8(out_row + col0)) {
        const F4 g = load_bf16x4(gate_row + col0);
        const F4 u = load_bf16x4(up_row + col0);
        F4 y;
        y.v[0] = swiglu_value(g.v[0], u.v[0], args.aux != 0u);
        y.v[1] = swiglu_value(g.v[1], u.v[1], args.aux != 0u);
        y.v[2] = swiglu_value(g.v[2], u.v[2], args.aux != 0u);
        y.v[3] = swiglu_value(g.v[3], u.v[3], args.aux != 0u);
        store_bf16x4(out_row + col0, y);
    } else {
        for (uint32_t i = 0u; i < kVec; ++i) {
            const uint32_t c = col0 + i;
            if (c >= features) break;
            const float gv = __bfloat162float(gate_row[c]);
            const float uv = __bfloat162float(up_row[c]);
            out_row[c] = __float2bfloat16(swiglu_value(gv, uv, args.aux != 0u));
        }
    }
}

hipError_t launch_swiglu_bf16(
    const bf16_t* gate, uint32_t gate_row_stride,
    const bf16_t* up, uint32_t up_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream, bool bf16_intermediate) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if (gate == nullptr || up == nullptr || output == nullptr) return hipErrorInvalidValue;
    ElementwiseArgs args{};
    args.in0 = gate;
    args.in0_row_stride = gate_row_stride;
    args.in1 = up;
    args.in1_row_stride = up_row_stride;
    args.out0 = output;
    args.out0_row_stride = output_row_stride;
    args.features = features;
    args.aux = bf16_intermediate ? 1u : 0u;
    args.rows = rows;
    const dim3 grid = elementwise_grid(features, rows);
    shelf_swiglu_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
