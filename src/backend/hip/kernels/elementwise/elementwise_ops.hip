// vendored from PhaseShift (own work); namespace/type adapted only.
#include <shelf/backend/hip/kernels/elementwise/elementwise.h>
#include <shelf/backend/hip/kernels/detail/vector_io.h>
#include <shelf/backend/hip/kernels/detail/residual_add_device.h>
#include <shelf/backend/hip/kernels/detail/target_swiglu_device.h>
#include <shelf/backend/hip/kernels/detail/elementwise_math.h>
#include <shelf/backend/hip/kernels/detail/sigmoid_device.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::hipk {

using namespace ::shelf::hipk::detail;
constexpr uint32_t kThreads = 256u;
static_assert(kThreads == kElementwiseBlock);
static_assert(kVec == kElementwiseVec);

namespace {
namespace op_concat_bf16 {

__global__ void impl(const bf16* a, uint32_t a_rs,
                     const bf16* b, uint32_t b_rs,
                     bf16* out, uint32_t out_rs,
                     uint32_t fa, uint32_t features) {
    const uint32_t row = blockIdx.y;
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
    const uint64_t gx =
        (static_cast<uint64_t>(features) + kThreads * kVec - 1u) / (kThreads * kVec);
    dim3 grid(static_cast<uint32_t>(gx), rows);
    op_concat_bf16::impl<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(a), a_row_stride,
        reinterpret_cast<const bf16*>(b), b_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride, features_a, features);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_residual_add_bf16(ElementwiseArgs args) {
    const bf16* a = static_cast<const bf16*>(args.in0);
    const bf16* b = static_cast<const bf16*>(args.in1);
    bf16* out = static_cast<bf16*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = blockIdx.y;
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
    dim3 grid(elementwise_grid_x(features), rows);
    phaseshift_residual_add_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_scale_f32(ElementwiseArgs args) {
    const float* in = static_cast<const float*>(args.in0);
    float* out = static_cast<float*>(args.out0);
    const uint32_t features = args.features;
    const float scale = args.scalar_a;
    const uint32_t row = blockIdx.y;
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
    dim3 grid(elementwise_grid_x(features), rows);
    phaseshift_scale_f32<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_sigmoid_bf16_f32(ElementwiseArgs args) {
    const bf16* in = static_cast<const bf16*>(args.in0);
    float* out = static_cast<float*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = blockIdx.y;
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
    dim3 grid(elementwise_grid_x(features), rows);
    phaseshift_sigmoid_bf16_f32<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_silu_bf16_f32(ElementwiseArgs args) {
    const bf16* in = static_cast<const bf16*>(args.in0);
    float* out = static_cast<float*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = blockIdx.y;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= features) return;
    const bf16* in_row = in + static_cast<size_t>(row) * args.in0_row_stride;
    float* out_row = out + static_cast<size_t>(row) * args.out0_row_stride;
    if (col0 + kVec - 1u < features && aligned8(in_row + col0) && aligned16(out_row + col0)) {
        const F4 x = load_bf16x4(in_row + col0);
        F4 y;
        y.v[0] = detail::silu_f32(x.v[0]);
        y.v[1] = detail::silu_f32(x.v[1]);
        y.v[2] = detail::silu_f32(x.v[2]);
        y.v[3] = detail::silu_f32(x.v[3]);
        store_f32x4(out_row + col0, y);
    } else {
        for (uint32_t i = 0u; i < kVec; ++i) {
            const uint32_t c = col0 + i;
            if (c >= features) break;
            out_row[c] = detail::silu_f32(__bfloat162float(in_row[c]));
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
    dim3 grid(elementwise_grid_x(features), rows);
    phaseshift_silu_bf16_f32<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_silu_f32_bf16(ElementwiseArgs args) {
    const float* in = static_cast<const float*>(args.in0);
    bf16* out = static_cast<bf16*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = blockIdx.y;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= features) return;
    const float* in_row = in + static_cast<size_t>(row) * args.in0_row_stride;
    bf16* out_row = out + static_cast<size_t>(row) * args.out0_row_stride;
    if (col0 + kVec - 1u < features && aligned16(in_row + col0) && aligned8(out_row + col0)) {
        const F4 x = load_f32x4(in_row + col0);
        F4 y;
        y.v[0] = detail::silu_f32(x.v[0]);
        y.v[1] = detail::silu_f32(x.v[1]);
        y.v[2] = detail::silu_f32(x.v[2]);
        y.v[3] = detail::silu_f32(x.v[3]);
        store_bf16x4(out_row + col0, y);
    } else {
        for (uint32_t i = 0u; i < kVec; ++i) {
            const uint32_t c = col0 + i;
            if (c >= features) break;
            out_row[c] = __float2bfloat16(detail::silu_f32(in_row[c]));
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
    dim3 grid(elementwise_grid_x(features), rows);
    phaseshift_silu_f32_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_mul_f32_f32_bf16(ElementwiseArgs args) {
    const float* a = static_cast<const float*>(args.in0);
    const float* b = static_cast<const float*>(args.in1);
    bf16* out = static_cast<bf16*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = blockIdx.y;
    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    const float* a_row = a + static_cast<size_t>(row) * args.in0_row_stride;
    const float* b_row = b + static_cast<size_t>(row) * args.in1_row_stride;
    bf16* out_row = out + static_cast<size_t>(row) * args.out0_row_stride;
    detail::mul_f32_bf16_body(a_row, b_row, out_row, col0, features);
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
    dim3 grid(elementwise_grid_x(features), rows);
    phaseshift_mul_f32_f32_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_split_bf16(ElementwiseArgs args) {
    const bf16* in = static_cast<const bf16*>(args.in0);
    bf16* o0 = static_cast<bf16*>(args.out0);
    bf16* o1 = static_cast<bf16*>(args.out1);
    const uint32_t f0 = args.features;
    const uint32_t hd = args.aux;
    const bool interleaved = args.reserved != 0u;
    const uint32_t row = blockIdx.y;
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
    dim3 grid(elementwise_grid_x(features), rows);
    phaseshift_split_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_swiglu_bf16(ElementwiseArgs args) {
    const bf16* gate = static_cast<const bf16*>(args.in0);
    const bf16* up = static_cast<const bf16*>(args.in1);
    bf16* out = static_cast<bf16*>(args.out0);
    const uint32_t features = args.features;
    const uint32_t row = blockIdx.y;
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
        y.v[0] = target_swiglu_f32(g.v[0], u.v[0]);
        y.v[1] = target_swiglu_f32(g.v[1], u.v[1]);
        y.v[2] = target_swiglu_f32(g.v[2], u.v[2]);
        y.v[3] = target_swiglu_f32(g.v[3], u.v[3]);
        store_bf16x4(out_row + col0, y);
    } else {
        for (uint32_t i = 0u; i < kVec; ++i) {
            const uint32_t c = col0 + i;
            if (c >= features) break;
            const float gv = __bfloat162float(gate_row[c]);
            const float uv = __bfloat162float(up_row[c]);
            out_row[c] = __float2bfloat16(target_swiglu_f32(gv, uv));
        }
    }
}

hipError_t launch_swiglu_bf16(
    const bf16_t* gate, uint32_t gate_row_stride,
    const bf16_t* up, uint32_t up_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream) {
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
    args.rows = rows;
    dim3 grid(elementwise_grid_x(features), rows);
    phaseshift_swiglu_bf16<<<grid, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::hipk
