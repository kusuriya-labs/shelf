// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/elementwise/elementwise_ops.hip.)
#include <shelf/backend/sycl/kernels/elementwise/elementwise.h>
#include <shelf/backend/sycl/kernels/detail/vector_io.h>
#include <shelf/backend/sycl/kernels/detail/residual_add_device.h>
#include <shelf/backend/sycl/kernels/detail/target_swiglu_device.h>
#include <shelf/backend/sycl/kernels/detail/elementwise_math.h>
#include <shelf/backend/sycl/kernels/detail/sigmoid_device.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

#include <cmath>
#include <cstdint>

namespace shelf::syclk {

using namespace ::shelf::syclk::detail;
constexpr uint32_t kThreads = 256u;
static_assert(kThreads == kElementwiseBlock);
static_assert(kVec == kElementwiseVec);

namespace {

// Shared launch shape: grid(elementwise_grid_x(features), rows) x block 256.
// `row` maps blockIdx.y, `col0` maps (blockIdx.x*256 + threadIdx.x) * kVec.
template <class BodyFn>
::shelf::Status launch_ew(const ::shelf::sycl::Stream& stream,
                          uint32_t rows, uint32_t features, BodyFn fn) {
    if (rows == 0u || features == 0u)
        return ::shelf::Status::invalid_argument("empty launch rejected", __FILE__, __LINE__);
    ::sycl::queue& q = stream.queue();
    const uint32_t gx = elementwise_grid_x(features);
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<2>{::sycl::range<2>{gx * kThreads, rows},
                                ::sycl::range<2>{kThreads, 1}},
            [=](::sycl::nd_item<2> it) {
                const uint32_t row = static_cast<uint32_t>(it.get_group(1));
                const uint32_t col0 = static_cast<uint32_t>(
                    (it.get_group(0) * kThreads + it.get_local_linear_id()) * kVec);
                fn(it, row, col0);
            });
    });
}

} // namespace

::shelf::Status launch_concat_bf16(
    const bf16_t* a, uint32_t a_row_stride,
    const bf16_t* b, uint32_t b_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features_a, uint32_t features_b,
    const ::shelf::sycl::Stream& stream) {
    if (features_b == 0u)
        return ::shelf::Status::invalid_argument("empty launch rejected", __FILE__, __LINE__);
    if (a == nullptr || b == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    const uint32_t features = features_a + features_b;
    const bf16* pa = reinterpret_cast<const bf16*>(a);
    const bf16* pb = reinterpret_cast<const bf16*>(b);
    bf16* po = reinterpret_cast<bf16*>(output);
    const uint32_t fa = features_a;
    const uint32_t ars = a_row_stride, brs = b_row_stride, ors = output_row_stride;
    return launch_ew(stream, rows, features,
        [=](::sycl::nd_item<2>&, uint32_t row, uint32_t col0) {
            if (col0 >= features) return;
            const bf16* a_row = pa + static_cast<size_t>(row) * ars;
            const bf16* b_row = pb + static_cast<size_t>(row) * brs;
            bf16* out_row = po + static_cast<size_t>(row) * ors;
            const bool from_a = (col0 + kVec <= fa);
            const bool from_b = (col0 >= fa);
            if (from_a || from_b) {
                const bf16* src_row = from_a ? a_row : b_row;
                const uint32_t src_col = from_a ? col0 : (col0 - fa);
                if (col0 + kVec <= features && aligned8(src_row + src_col) &&
                    aligned8(out_row + col0)) {
                    store_bf16x4(out_row + col0, load_bf16x4(src_row + src_col));
                    return;
                }
            }
            for (uint32_t i = 0u; i < kVec; ++i) {
                const uint32_t c = col0 + i;
                if (c >= features) break;
                out_row[c] = (c < fa) ? a_row[c] : b_row[c - fa];
            }
        });
}

::shelf::Status launch_residual_add_bf16(
    const bf16_t* a, uint32_t a_row_stride,
    const bf16_t* b, uint32_t b_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, const ::shelf::sycl::Stream& stream) {
    if (a == nullptr || b == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    const bf16* pa = reinterpret_cast<const bf16*>(a);
    const bf16* pb = reinterpret_cast<const bf16*>(b);
    bf16* po = reinterpret_cast<bf16*>(output);
    return launch_ew(stream, rows, features,
        [=](::sycl::nd_item<2>&, uint32_t row, uint32_t col0) {
            if (col0 >= features) return;
            residual_add_bf16_body(pa + static_cast<size_t>(row) * a_row_stride,
                                   pb + static_cast<size_t>(row) * b_row_stride,
                                   po + static_cast<size_t>(row) * output_row_stride,
                                   col0, features);
        });
}

::shelf::Status launch_scale_f32(
    const float* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, float scale, const ::shelf::sycl::Stream& stream) {
    if (input == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    return launch_ew(stream, rows, features,
        [=](::sycl::nd_item<2>&, uint32_t row, uint32_t col0) {
            if (col0 >= features) return;
            const float* in_row = input + static_cast<size_t>(row) * input_row_stride;
            float* out_row = output + static_cast<size_t>(row) * output_row_stride;
            if (col0 + kVec - 1u < features && aligned16(in_row + col0) &&
                aligned16(out_row + col0)) {
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
        });
}

::shelf::Status launch_sigmoid_bf16_f32(
    const bf16_t* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, const ::shelf::sycl::Stream& stream) {
    if (input == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    const bf16* in = reinterpret_cast<const bf16*>(input);
    return launch_ew(stream, rows, features,
        [=](::sycl::nd_item<2>&, uint32_t row, uint32_t col0) {
            if (col0 >= features) return;
            const bf16* in_row = in + static_cast<size_t>(row) * input_row_stride;
            float* out_row = output + static_cast<size_t>(row) * output_row_stride;
            if (col0 + kVec - 1u < features && aligned8(in_row + col0) &&
                aligned16(out_row + col0)) {
                const F4 x = load_bf16x4(in_row + col0);
                F4 y;
                y.v[0] = sigmoid_f32(x.v[0]);
                y.v[1] = sigmoid_f32(x.v[1]);
                y.v[2] = sigmoid_f32(x.v[2]);
                y.v[3] = sigmoid_f32(x.v[3]);
                store_f32x4(out_row + col0, y);
            } else {
                for (uint32_t i = 0u; i < kVec; ++i) {
                    const uint32_t c = col0 + i;
                    if (c >= features) break;
                    out_row[c] = sigmoid_f32(::shelf::bf16_to_f32(in_row[c]));
                }
            }
        });
}

::shelf::Status launch_silu_bf16_f32(
    const bf16_t* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, const ::shelf::sycl::Stream& stream) {
    if (input == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    const bf16* in = reinterpret_cast<const bf16*>(input);
    return launch_ew(stream, rows, features,
        [=](::sycl::nd_item<2>&, uint32_t row, uint32_t col0) {
            if (col0 >= features) return;
            const bf16* in_row = in + static_cast<size_t>(row) * input_row_stride;
            float* out_row = output + static_cast<size_t>(row) * output_row_stride;
            if (col0 + kVec - 1u < features && aligned8(in_row + col0) &&
                aligned16(out_row + col0)) {
                const F4 x = load_bf16x4(in_row + col0);
                F4 y;
                y.v[0] = silu_f32(x.v[0]);
                y.v[1] = silu_f32(x.v[1]);
                y.v[2] = silu_f32(x.v[2]);
                y.v[3] = silu_f32(x.v[3]);
                store_f32x4(out_row + col0, y);
            } else {
                for (uint32_t i = 0u; i < kVec; ++i) {
                    const uint32_t c = col0 + i;
                    if (c >= features) break;
                    out_row[c] = silu_f32(::shelf::bf16_to_f32(in_row[c]));
                }
            }
        });
}

::shelf::Status launch_silu_f32_bf16(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, const ::shelf::sycl::Stream& stream) {
    if (input == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    bf16* out = reinterpret_cast<bf16*>(output);
    return launch_ew(stream, rows, features,
        [=](::sycl::nd_item<2>&, uint32_t row, uint32_t col0) {
            if (col0 >= features) return;
            const float* in_row = input + static_cast<size_t>(row) * input_row_stride;
            bf16* out_row = out + static_cast<size_t>(row) * output_row_stride;
            if (col0 + kVec - 1u < features && aligned16(in_row + col0) &&
                aligned8(out_row + col0)) {
                const F4 x = load_f32x4(in_row + col0);
                F4 y;
                y.v[0] = silu_f32(x.v[0]);
                y.v[1] = silu_f32(x.v[1]);
                y.v[2] = silu_f32(x.v[2]);
                y.v[3] = silu_f32(x.v[3]);
                store_bf16x4(out_row + col0, y);
            } else {
                for (uint32_t i = 0u; i < kVec; ++i) {
                    const uint32_t c = col0 + i;
                    if (c >= features) break;
                    out_row[c] = ::shelf::f32_to_bf16(silu_f32(in_row[c]));
                }
            }
        });
}

::shelf::Status launch_mul_f32_f32_bf16(
    const float* a, uint32_t a_row_stride,
    const float* b, uint32_t b_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, const ::shelf::sycl::Stream& stream) {
    if (a == nullptr || b == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    bf16* out = reinterpret_cast<bf16*>(output);
    return launch_ew(stream, rows, features,
        [=](::sycl::nd_item<2>&, uint32_t row, uint32_t col0) {
            const float* a_row = a + static_cast<size_t>(row) * a_row_stride;
            const float* b_row = b + static_cast<size_t>(row) * b_row_stride;
            mul_f32_bf16_body(a_row, b_row, out + static_cast<size_t>(row) * output_row_stride,
                              col0, features);
        });
}

::shelf::Status launch_split_bf16(
    const bf16_t* input, uint32_t input_row_stride,
    bf16_t* o0, uint32_t o0_row_stride,
    bf16_t* o1, uint32_t o1_row_stride,
    uint32_t rows, uint32_t features,
    ElementwiseSplitLayout layout, uint32_t head_dim, const ::shelf::sycl::Stream& stream) {
    if (input == nullptr || o0 == nullptr || o1 == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    const bool interleaved = (layout == ElementwiseSplitLayout::InterleavedHeads);
    if (interleaved) {
        if (head_dim == 0u || features % head_dim != 0u)
            return ::shelf::Status::invalid_argument("bad split layout", __FILE__, __LINE__);
    }
    const bf16* in = reinterpret_cast<const bf16*>(input);
    bf16* p0 = reinterpret_cast<bf16*>(o0);
    bf16* p1 = reinterpret_cast<bf16*>(o1);
    const uint32_t f0 = features;
    const uint32_t hd = head_dim;
    return launch_ew(stream, rows, f0,
        [=](::sycl::nd_item<2>&, uint32_t row, uint32_t col0) {
            if (col0 >= f0) return;
            const bf16* in_row = in + static_cast<size_t>(row) * input_row_stride;
            bf16* o0_row = p0 + static_cast<size_t>(row) * o0_row_stride;
            bf16* o1_row = p1 + static_cast<size_t>(row) * o1_row_stride;
            if (interleaved) {
                const uint32_t h = col0 / hd;
                const uint32_t o = col0 % hd;
                const size_t base0 = static_cast<size_t>(h) * 2u * hd + o;
                if (o + kVec <= hd && aligned8(in_row + base0) &&
                    aligned8(in_row + base0 + hd) && aligned8(o0_row + col0) &&
                    aligned8(o1_row + col0)) {
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
                if (col0 + kVec <= f0 && aligned8(in_row + col0) &&
                    aligned8(in_row + col0 + f0) && aligned8(o0_row + col0) &&
                    aligned8(o1_row + col0)) {
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
        });
}

::shelf::Status launch_swiglu_bf16(
    const bf16_t* gate, uint32_t gate_row_stride,
    const bf16_t* up, uint32_t up_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, const ::shelf::sycl::Stream& stream) {
    if (gate == nullptr || up == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    const bf16* g = reinterpret_cast<const bf16*>(gate);
    const bf16* u = reinterpret_cast<const bf16*>(up);
    bf16* out = reinterpret_cast<bf16*>(output);
    return launch_ew(stream, rows, features,
        [=](::sycl::nd_item<2>&, uint32_t row, uint32_t col0) {
            if (col0 >= features) return;
            const bf16* gate_row = g + static_cast<size_t>(row) * gate_row_stride;
            const bf16* up_row = u + static_cast<size_t>(row) * up_row_stride;
            bf16* out_row = out + static_cast<size_t>(row) * output_row_stride;
            if (col0 + kVec - 1u < features && aligned8(gate_row + col0) &&
                aligned8(up_row + col0) && aligned8(out_row + col0)) {
                const F4 gv = load_bf16x4(gate_row + col0);
                const F4 uv = load_bf16x4(up_row + col0);
                F4 y;
                y.v[0] = target_swiglu_f32(gv.v[0], uv.v[0]);
                y.v[1] = target_swiglu_f32(gv.v[1], uv.v[1]);
                y.v[2] = target_swiglu_f32(gv.v[2], uv.v[2]);
                y.v[3] = target_swiglu_f32(gv.v[3], uv.v[3]);
                store_bf16x4(out_row + col0, y);
            } else {
                for (uint32_t i = 0u; i < kVec; ++i) {
                    const uint32_t c = col0 + i;
                    if (c >= features) break;
                    const float gfv = ::shelf::bf16_to_f32(gate_row[c]);
                    const float ufv = ::shelf::bf16_to_f32(up_row[c]);
                    out_row[c] = ::shelf::f32_to_bf16(target_swiglu_f32(gfv, ufv));
                }
            }
        });
}

} // namespace shelf::syclk
