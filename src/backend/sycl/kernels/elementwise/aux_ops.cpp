// shelf backend/sycl — auxiliary bf16 glue ops (SYCL port of
// backend/hip kernels/elementwise/aux_ops.hip).
#include <cstdint>

#include <shelf/backend/sycl/kernels/elementwise/aux_ops.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>
#include <shelf/foundation/numeric.hpp>

namespace shelf::syclk {

// out[r][d] = in[r][begin + d]; row-major bf16 with input row stride.
// (last-dim slice between two ops that need contiguous rows)
::shelf::Status launch_bf16_row_slice(const shelf::bf16_t* in, shelf::bf16_t* out,
                                      uint32_t rows, uint32_t cols, uint32_t begin,
                                      uint32_t in_row_stride,
                                      const ::shelf::sycl::Stream& stream) {
    const uint32_t total = rows * cols;
    const uint32_t grid = (total + 255u) / 256u;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid * 256u}, ::sycl::range<1>{256u}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t i = static_cast<uint32_t>(it.get_global_linear_id());
                if (i >= total) return;
                const uint32_t r = i / cols;
                const uint32_t d = i - r * cols;
                out[i] = in[static_cast<size_t>(r) * in_row_stride + begin + d];
            });
    });
}

// out = x * sigmoid(z), elementwise bf16.
::shelf::Status launch_bf16_sigmoid_gate(const shelf::bf16_t* x, const shelf::bf16_t* z,
                                         shelf::bf16_t* out, uint32_t n,
                                         const ::shelf::sycl::Stream& stream) {
    const uint32_t grid = (n + 255u) / 256u;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid * 256u}, ::sycl::range<1>{256u}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t i = static_cast<uint32_t>(it.get_global_linear_id());
                if (i >= n) return;
                const float xv = ::shelf::bf16_to_f32(x[i]);
                const float zv = ::shelf::bf16_to_f32(z[i]);
                const float s = 1.0f / (1.0f + ::sycl::exp(-zv));
                out[i] = shelf::bf16_t{f32_to_bf16_bits(xv * s)};
            });
    });
}

// out = a + b (bf16 residual stream, matches production bf16 residuals)
::shelf::Status launch_bf16_add(const shelf::bf16_t* a, const shelf::bf16_t* b,
                                shelf::bf16_t* out, uint32_t n,
                                const ::shelf::sycl::Stream& stream) {
    const uint32_t grid = (n + 255u) / 256u;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid * 256u}, ::sycl::range<1>{256u}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t i = static_cast<uint32_t>(it.get_global_linear_id());
                if (i >= n) return;
                const float av = ::shelf::bf16_to_f32(a[i]);
                const float bv = ::shelf::bf16_to_f32(b[i]);
                out[i] = shelf::bf16_t{f32_to_bf16_bits(av + bv)};
            });
    });
}

// f32 logits argmax -> token (single block reduction, vocab <= 1<<20)
::shelf::Status launch_f32_argmax(const float* logits, uint32_t n, uint32_t* out_tok,
                                  const ::shelf::sycl::Stream& stream) {
    constexpr uint32_t kBlock = 1024u;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        ::sycl::local_accessor<float, 1> best_v(::sycl::range<1>{kBlock}, cgh);
        ::sycl::local_accessor<uint32_t, 1> best_i(::sycl::range<1>{kBlock}, cgh);
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{kBlock}, ::sycl::range<1>{kBlock}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());
                uint32_t bv = 0;
                float bv0 = -__builtin_huge_valf();
                for (uint32_t i = tid; i < n; i += kBlock) {
                    const float v = logits[i];
                    if (v > bv0) { bv0 = v; bv = i; }
                }
                best_v[tid] = bv0;
                best_i[tid] = bv;
                it.barrier(::sycl::access::fence_space::local_space);
                for (uint32_t s = kBlock / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        if (best_v[tid + s] > best_v[tid]) {
                            best_v[tid] = best_v[tid + s];
                            best_i[tid] = best_i[tid + s];
                        }
                    }
                    it.barrier(::sycl::access::fence_space::local_space);
                }
                if (tid == 0) *out_tok = best_i[0];
            });
    });
}

// dtype conversions for activation streams
::shelf::Status launch_bf16_to_f32(const shelf::bf16_t* in, float* out, uint32_t n,
                                   const ::shelf::sycl::Stream& stream) {
    const uint32_t grid = (n + 255u) / 256u;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid * 256u}, ::sycl::range<1>{256u}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t i = static_cast<uint32_t>(it.get_global_linear_id());
                if (i >= n) return;
                out[i] = ::shelf::bf16_to_f32(in[i]);
            });
    });
}

::shelf::Status launch_f32_to_bf16(const float* in, shelf::bf16_t* out, uint32_t n,
                                   const ::shelf::sycl::Stream& stream) {
    const uint32_t grid = (n + 255u) / 256u;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid * 256u}, ::sycl::range<1>{256u}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t i = static_cast<uint32_t>(it.get_global_linear_id());
                if (i >= n) return;
                out[i] = ::shelf::f32_to_bf16(in[i]);
            });
    });
}

} // namespace shelf::syclk
