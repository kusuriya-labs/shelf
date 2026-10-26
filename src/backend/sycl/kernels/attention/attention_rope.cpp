// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/attention/attention_rope.hip.)
#include <shelf/backend/sycl/kernels/attention/rope.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace shelf::syclk {

namespace {

using bf16 = shelf::bf16_t;

constexpr uint32_t kRopeThreads = 256;
constexpr uint32_t kRopeMaxHalf = 128;

constexpr uint32_t kPairThreads = kRopePairBlock;
constexpr uint32_t kPairHeadDim = kRopePairHeadDim;
constexpr uint32_t kPairRotary = kRopePairRotary;
constexpr uint32_t kPairHalf = kPairRotary / 2u;

} // namespace

::shelf::Status launch_rope_inv_freq_init(
    double* inv_freq, uint32_t half, float theta, const ::shelf::sycl::Stream& stream) {
    if (inv_freq == nullptr)
        return ::shelf::Status::invalid_argument("null inv_freq", __FILE__, __LINE__);
    if (half == 0u || half > 32u)
        return ::shelf::Status::unsupported("inv_freq half out of range", __FILE__, __LINE__);
    if (theta <= 0.0f)
        return ::shelf::Status::invalid_argument("bad theta", __FILE__, __LINE__);
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::range<1>{half}, [=](::sycl::item<1> i) {
            const uint32_t idx = static_cast<uint32_t>(i.get_linear_id());
            const double theta_base = static_cast<double>(theta);
            inv_freq[idx] = 1.0 / ::sycl::pow(
                theta_base,
                static_cast<double>(idx) / static_cast<double>(half));
        });
    });
}

::shelf::Status launch_rope_inv_freq_yarn_init(
    double* inv_freq, uint32_t half, float theta, float factor,
    uint32_t original_max_position_embeddings, float beta_fast, float beta_slow,
    bool truncate, const ::shelf::sycl::Stream& stream) {
    if (inv_freq == nullptr)
        return ::shelf::Status::invalid_argument("null inv_freq", __FILE__, __LINE__);
    if (half == 0u || half > 32u)
        return ::shelf::Status::unsupported("inv_freq half out of range", __FILE__, __LINE__);
    if (theta <= 0.0f)
        return ::shelf::Status::invalid_argument("bad theta", __FILE__, __LINE__);
    if (!(factor > 1.0f) || !std::isfinite(factor))
        return ::shelf::Status::invalid_argument("bad factor", __FILE__, __LINE__);
    if (original_max_position_embeddings == 0u)
        return ::shelf::Status::invalid_argument("bad original_len", __FILE__, __LINE__);
    if (!(beta_fast > 0.0f) || !(beta_slow > 0.0f))
        return ::shelf::Status::invalid_argument("bad beta", __FILE__, __LINE__);
    const float original_len = static_cast<float>(original_max_position_embeddings);
    const uint32_t trunc = truncate ? 1u : 0u;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::range<1>{half}, [=](::sycl::item<1> i) {
            const uint32_t idx = static_cast<uint32_t>(i.get_linear_id());
            const double theta_base = static_cast<double>(theta);
            const double exponent_div = static_cast<double>(half);
            const double dim = static_cast<double>(half) * 2.0;
            const double f = static_cast<double>(factor);
            const double L = static_cast<double>(original_len);
            const double base = 1.0 / ::sycl::pow(
                theta_base, static_cast<double>(idx) / exponent_div);
            const double pi2 = 6.283185307179586;
            const double cf = dim * ::sycl::log(L / (static_cast<double>(beta_fast) * pi2)) /
                              (2.0 * ::sycl::log(theta_base));
            const double cs = dim * ::sycl::log(L / (static_cast<double>(beta_slow) * pi2)) /
                              (2.0 * ::sycl::log(theta_base));
            const double low = trunc != 0u ? ::sycl::floor(cf) : cf;
            const double high = trunc != 0u ? ::sycl::ceil(cs) : cs;
            const double span = high - low;
            double t = (static_cast<double>(idx) - low) / (span > 1e-10 ? span : 1e-10);
            t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
            inv_freq[idx] = base * ((1.0 - t) + t / f);
        });
    });
}

::shelf::Status launch_rope_f32_bf16_generic(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    const ::shelf::sycl::Stream& stream) {
    if (input == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (rows == 0u || features == 0u || head_dim == 0u || features % head_dim != 0u)
        return ::shelf::Status::invalid_argument("bad shape", __FILE__, __LINE__);
    if ((rotary_dim & 1u) != 0u || rotary_dim == 0u || rotary_dim > head_dim)
        return ::shelf::Status::invalid_argument("bad rotary_dim", __FILE__, __LINE__);
    if (inv_freq == nullptr && theta <= 0.0f)
        return ::shelf::Status::invalid_argument("need theta or inv_freq", __FILE__, __LINE__);
    if (!(attention_factor > 0.0f) || !std::isfinite(attention_factor))
        return ::shelf::Status::invalid_argument("bad attention_factor", __FILE__, __LINE__);
    const uint32_t half = rotary_dim >> 1;
    if (half > kRopeMaxHalf)
        return ::shelf::Status::unsupported("rotary half too large", __FILE__, __LINE__);
    const double* table =
        (inv_freq != nullptr && half <= kRopeInvFreqTable) ? inv_freq : nullptr;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        ::sycl::local_accessor<float, 1> s_cos(::sycl::range<1>{kRopeMaxHalf}, cgh);
        ::sycl::local_accessor<float, 1> s_sin(::sycl::range<1>{kRopeMaxHalf}, cgh);
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{rows * kRopeThreads},
                                ::sycl::range<1>{kRopeThreads}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t row = static_cast<uint32_t>(it.get_group(0));
                const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());

                if (tid < half) {
                    const uint32_t pos = positions != nullptr ? positions[row] : row;
                    const double freq = table != nullptr
                        ? table[tid]
                        : 1.0 / ::sycl::pow(static_cast<double>(theta),
                                           static_cast<double>(tid) /
                                               static_cast<double>(half));
                    const double angle = static_cast<double>(pos) * freq;
                    s_cos[tid] = static_cast<float>(::sycl::cos(angle)) * attention_factor;
                    s_sin[tid] = static_cast<float>(::sycl::sin(angle)) * attention_factor;
                }
                it.barrier(::sycl::access::fence_space::local_space);

                const size_t in_base = static_cast<size_t>(row) * input_row_stride;
                const size_t out_base = static_cast<size_t>(row) * output_row_stride;
                for (uint32_t idx = tid; idx < features; idx += kRopeThreads) {
                    const uint32_t head_col = idx % head_dim;
                    const float x = input[in_base + idx];
                    float y = x;
                    if (head_col < rotary_dim) {
                        const uint32_t pi = head_col < half ? head_col : head_col - half;
                        const uint32_t partner_col =
                            head_col < half ? head_col + half : head_col - half;
                        const float cs = s_cos[pi];
                        const float sn = s_sin[pi];
                        const float partner = input[in_base + (idx - head_col) + partner_col];
                        y = head_col < half ? x * cs - partner * sn
                                            : x * cs + partner * sn;
                    }
                    output[out_base + idx] = ::shelf::f32_to_bf16(y);
                }
            });
    });
}

::shelf::Status launch_rope_f32_bf16_pair(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    const ::shelf::sycl::Stream& stream) {
    if (input == nullptr || output == nullptr || inv_freq == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (rows == 0u || features == 0u || head_dim == 0u || features % head_dim != 0u)
        return ::shelf::Status::invalid_argument("bad shape", __FILE__, __LINE__);
    if ((rotary_dim & 1u) != 0u || rotary_dim == 0u || rotary_dim > head_dim)
        return ::shelf::Status::invalid_argument("bad rotary_dim", __FILE__, __LINE__);
    if (theta <= 0.0f)
        return ::shelf::Status::invalid_argument("bad theta", __FILE__, __LINE__);
    if (!(attention_factor > 0.0f) || !std::isfinite(attention_factor))
        return ::shelf::Status::invalid_argument("bad attention_factor", __FILE__, __LINE__);
    if (head_dim != kPairHeadDim || rotary_dim != kPairRotary)
        return ::shelf::Status::unsupported("pair kernel shape", __FILE__, __LINE__);
    if ((features & (kPairHeadDim - 1u)) != 0u)
        return ::shelf::Status::unsupported("pair kernel shape", __FILE__, __LINE__);
    if ((input_row_stride & 1u) != 0u || (output_row_stride & 1u) != 0u)
        return ::shelf::Status::unsupported("pair kernel strides", __FILE__, __LINE__);
    if ((reinterpret_cast<uintptr_t>(input) & 7u) != 0u)
        return ::shelf::Status::unsupported("pair kernel input alignment", __FILE__, __LINE__);
    if ((reinterpret_cast<uintptr_t>(output) & 3u) != 0u)
        return ::shelf::Status::unsupported("pair kernel output alignment", __FILE__, __LINE__);

    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{rows * kPairThreads},
                                ::sycl::range<1>{kPairThreads}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t row = static_cast<uint32_t>(it.get_group(0));
                const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());
                const uint32_t heads = features >> 8;
                const size_t in_base = static_cast<size_t>(row) * input_row_stride;
                const size_t out_base = static_cast<size_t>(row) * output_row_stride;

                float cos_f = 0.0f;
                float sin_f = 0.0f;
                if (tid < kPairHalf) {
                    const uint32_t pos = positions != nullptr ? positions[row] : row;
                    const double freq = inv_freq[tid];
                    const double angle = static_cast<double>(pos) * freq;
                    cos_f = static_cast<float>(::sycl::cos(angle)) * attention_factor;
                    sin_f = static_cast<float>(::sycl::sin(angle)) * attention_factor;
                }

                if (tid < kPairHalf) {
#pragma unroll 4
                    for (uint32_t head = 0; head < heads; ++head) {
                        const size_t in_off = in_base + static_cast<size_t>(head) * kPairHeadDim;
                        const size_t out_off = out_base + static_cast<size_t>(head) * kPairHeadDim;
                        const float a = input[in_off + tid];
                        const float b = input[in_off + tid + kPairHalf];
                        // __fmaf_rn(a, cos, -b*sin) / __fmaf_rn(b, cos, a*sin):
                        // hardware FMA (correctly rounded) matches on GPU.
                        const float y0 = ::sycl::fma(a, cos_f, -(b * sin_f));
                        const float y1 = ::sycl::fma(b, cos_f, a * sin_f);
                        output[out_off + tid] = ::shelf::f32_to_bf16(y0);
                        output[out_off + tid + kPairHalf] = ::shelf::f32_to_bf16(y1);
                    }
                } else {
                    const uint32_t col = kPairRotary + (tid - kPairHalf) * 2u;
#pragma unroll 4
                    for (uint32_t head = 0; head < heads; ++head) {
                        const size_t in_off = in_base + static_cast<size_t>(head) * kPairHeadDim + col;
                        const size_t out_off = out_base + static_cast<size_t>(head) * kPairHeadDim + col;
                        // pass-through: the HIP original packs two f32 -> two
                        // bf16 into one u32 store; scalar stores are
                        // bit-identical.
                        output[out_off + 0u] = ::shelf::f32_to_bf16(input[in_off + 0u]);
                        output[out_off + 1u] = ::shelf::f32_to_bf16(input[in_off + 1u]);
                    }
                }
            });
    });
}

::shelf::Status launch_rope_f32_bf16(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    const uint32_t* positions,
    uint32_t rows, uint32_t features, uint32_t head_dim, uint32_t rotary_dim,
    float theta, const double* inv_freq, float attention_factor,
    const ::shelf::sycl::Stream& stream) {
    if (input == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (rows == 0u || features == 0u || head_dim == 0u || features % head_dim != 0u)
        return ::shelf::Status::invalid_argument("bad shape", __FILE__, __LINE__);
    if ((rotary_dim & 1u) != 0u || rotary_dim == 0u || rotary_dim > head_dim)
        return ::shelf::Status::invalid_argument("bad rotary_dim", __FILE__, __LINE__);
    if (inv_freq == nullptr && theta <= 0.0f)
        return ::shelf::Status::invalid_argument("need theta or inv_freq", __FILE__, __LINE__);
    if (!(attention_factor > 0.0f) || !std::isfinite(attention_factor))
        return ::shelf::Status::invalid_argument("bad attention_factor", __FILE__, __LINE__);
    if (inv_freq != nullptr && head_dim == kPairHeadDim && rotary_dim == kPairRotary &&
        (features & (kPairHeadDim - 1u)) == 0u &&
        (input_row_stride & 1u) == 0u && (output_row_stride & 1u) == 0u &&
        (reinterpret_cast<uintptr_t>(input) & 7u) == 0u &&
        (reinterpret_cast<uintptr_t>(output) & 3u) == 0u) {
        return launch_rope_f32_bf16_pair(input, input_row_stride, output, output_row_stride,
                                         positions, rows, features, head_dim, rotary_dim,
                                         theta, inv_freq, attention_factor, stream);
    }
    return launch_rope_f32_bf16_generic(input, input_row_stride, output, output_row_stride,
                                        positions, rows, features, head_dim, rotary_dim,
                                        theta, inv_freq, attention_factor, stream);
}

} // namespace shelf::syclk
