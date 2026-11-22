#include <shelf/kernels/optimized/attention/paged_verify.h>
#include <shelf/kernels/fused/detail/octave_common.h>
#include <shelf/kernels/optimized/detail/lds_barrier_device.h>
#include <hip/hip_bf16.h>
#include <cfloat>

namespace shelf::kernel {
namespace {

using Fragment = short __attribute__((ext_vector_type(8)));
using Acc = float __attribute__((ext_vector_type(8)));
using HalfFragment = _Float16 __attribute__((ext_vector_type(8)));

template <bool half_precision> __device__ __forceinline__ Acc multiply(Fragment a, Fragment b, Acc c) {
    if constexpr (half_precision)
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(HalfFragment, a),
                                                                __builtin_bit_cast(HalfFragment, b), c);
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ short half_bits(float value) {
    return __builtin_bit_cast(short, static_cast<_Float16>(value));
}

__device__ __forceinline__ short pack(float value) {
    return __bfloat16_as_short(__float2bfloat16(value));
}

__device__ __forceinline__ Fragment fragment(const short* data, unsigned half) {
    Fragment result;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i)
        result[i] = data[half * 4 + (i & 3) + (i >> 2) * 8];
    return result;
}

__device__ __forceinline__ void unpack_c5(const uint8_t* source, unsigned dim, short (&values)[8]) {
    const unsigned low = detail::octave::load_word(source + dim / 2);
    const unsigned high = source[128 + dim / 8];
    using HalfPair = _Float16 __attribute__((ext_vector_type(2)));
#pragma unroll
    for (unsigned i = 0; i < 2; ++i) {
        unsigned nibbles = (low >> (16 * i)) & 0xffffu;
        nibbles = (nibbles | (nibbles << 8)) & 0x00ff00ffu;
        nibbles = (nibbles | (nibbles << 4)) & 0x0f0f0f0fu;
        const unsigned bits = ((high >> (4 * i)) & 15u) * 0x204081u;
        const unsigned bytes = detail::octave::octave_lut5x4_biased(nibbles | ((bits & 0x01010101u) << 4));
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const unsigned pair =
                __builtin_amdgcn_perm(0x64646464u, bytes, j == 0 ? 0x04010400u : 0x04030402u);
            const HalfPair decoded = __builtin_bit_cast(HalfPair, pair) - static_cast<_Float16>(1152);
            __builtin_memcpy(values + 4 * i + 2 * j, &decoded, sizeof(decoded));
        }
    }
}

template <bool compact5>
__global__ void rotate_queries(PagedAttentionCommonArgs a, short* rotated, bool compact) {
    const unsigned lane = threadIdx.x % 32;
    const unsigned job = blockIdx.x * 4 + threadIdx.x / 32;
    if (job >= a.q_heads * a.rows)
        return;
    const unsigned head = job % a.q_heads;
    const unsigned row = job / a.q_heads;
    const bf16_t* src = a.q + size_t(row) * a.q_row_stride + head * 256 + lane * 8;
    short* dst = rotated + size_t(job) * 256 + lane * 8;
    float x[8];
#pragma unroll
    for (unsigned e = 0; e < 8; ++e)
        x[e] = __uint_as_float(unsigned(src[e].data) << 16) * detail::octave::sign_of(lane * 8 + e, 11);
    detail::octave::inverse_wave(x, lane, compact);
#pragma unroll
    for (unsigned e = 0; e < 8; ++e)
        dst[e] = compact5 ? half_bits(x[e]) : pack(x[e]);
}

template <KVCacheDType dtype, unsigned waves, bool compact5 = false>
__global__ __launch_bounds__(waves * 32) void verify_pages(PagedAttentionCommonArgs a, PagedVerifyCache cache,
                                                           float* partials, unsigned splits) {
    __shared__ short keys[16][264];
    __shared__ short values[256][20];
    __shared__ float key_scale[16], value_scale[16];
    const unsigned tid = threadIdx.x, lane = tid & 31, wave = tid / 32;
    const unsigned col = lane & 15, half = lane / 16;
    const unsigned head = blockIdx.y;
    const unsigned rows = a.rows;
    const unsigned qid = wave * 16 + col;
    const unsigned row = min(qid / 6, rows - 1);
    const unsigned qhead = head * 6 + qid % 6;
    const unsigned slot = a.row_sequence_slots[0];
    const unsigned last = a.row_positions[rows - 1];
    const unsigned visible = a.row_positions[row];
    const unsigned pages = last / 16 + 1;
    const bool valid = slot < a.max_sequences && pages <= a.block_table_stride;
    const unsigned* table = a.block_tables + size_t(valid ? slot : 0u) * a.block_table_stride;
    const unsigned step = (pages + splits - 1) / splits;
    const unsigned begin = blockIdx.x * step, end = valid ? min(pages, begin + step) : 0u;
    const short* query = reinterpret_cast<const short*>(a.q) + size_t(row) * a.q_row_stride + qhead * 256;
    if constexpr (dtype == KVCacheDType::OCTAVE)
        query = reinterpret_cast<const short*>(partials +
                                               size_t(rows) * 24 * splits * kPagedVerifyPartialFloats) +
                (size_t(row) * 24 + qhead) * 256;
    Fragment q[16];
    Acc output[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) {
        q[i] = fragment(query + i * 16, half);
        output[i] = {};
    }
    float maximum = -FLT_MAX, sum = 0;
    for (unsigned pg = begin; pg < end; ++pg) {
        const unsigned phys = table[pg];
        if (phys >= a.num_pages)
            continue;
        detail::lds_barrier();
        for (unsigned item = tid; item < 16 * 32; item += waves * 32) {
            const unsigned token = item / 32, dim = (item & 31) * 8;
            const unsigned source_token = min(token, last - pg * 16);
            const size_t vec = ((size_t(a.layer) * a.num_pages + phys) * 16 + source_token) * 4 + head;
            short k[8], v[8];
            if constexpr (compact5) {
                constexpr unsigned bytes = octave_v_slot_bytes({OctaveKMode::Compact, 5, 5, 5});
                const auto* ks = static_cast<const uint8_t*>(cache.k) + vec * bytes;
                const auto* vs = static_cast<const uint8_t*>(cache.v) + vec * bytes;
                unpack_c5(ks, dim, k);
                unpack_c5(vs, dim, v);
                if (dim == 0) {
                    constexpr unsigned scale_offset = bytes - 2;
                    key_scale[token] = static_cast<_Float16>(detail::octave::half_at(ks + scale_offset) *
                                                             detail::octave::codebook_scale<5>());
                    value_scale[token] = static_cast<_Float16>(detail::octave::half_at(vs + scale_offset) *
                                                               detail::octave::codebook_scale<5>());
                }
            } else if constexpr (dtype == KVCacheDType::OCTAVE) {
                const bool compact = cache.format.k_mode == OctaveKMode::Compact;
                const unsigned block = compact ? 0 : dim / 64;
                const unsigned bits = block == 0 ? cache.format.k_rope_bits : cache.format.k_nope_bits;
                const unsigned kb = octave_k_slot_bytes(cache.format);
                const unsigned vb = octave_v_slot_bytes(cache.format);
                const auto* ks = static_cast<const uint8_t*>(cache.k) + vec * kb;
                const auto* vs = static_cast<const uint8_t*>(cache.v) + vec * vb;
                const unsigned offset =
                    block ? 8 * (cache.format.k_rope_bits + (block - 1) * cache.format.k_nope_bits) : 0;
                unsigned kc[8], vc[8];
                detail::octave::unpack8(ks + offset, compact ? dim : dim % 64, compact ? 256 : 64, bits, kc);
                detail::octave::unpack8(vs, dim, 256, cache.format.v_bits, vc);
                const float kscale = detail::octave::half_at(ks + kb - (compact ? 2 : 8) + 2 * block);
                const float vscale = detail::octave::half_at(vs + vb - 2);
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) {
                    k[e] = pack(cache.levels[bits * 64 + kc[e]] * kscale);
                    v[e] = pack(cache.levels[cache.format.v_bits * 64 + vc[e]] * vscale);
                }
            } else if constexpr (dtype == KVCacheDType::FP8_E4M3) {
                uint2 raw_k =
                    *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(cache.k) + vec * 256 + dim);
                uint2 raw_v =
                    *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(cache.v) + vec * 256 + dim);
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    const auto kp = __builtin_amdgcn_cvt_pk_f32_fp8(
                        int((i < 2 ? raw_k.x : raw_k.y) >> ((i % 2) * 16)), false);
                    const auto vp = __builtin_amdgcn_cvt_pk_f32_fp8(
                        int((i < 2 ? raw_v.x : raw_v.y) >> ((i % 2) * 16)), false);
                    k[i * 2] = pack(kp[0]);
                    k[i * 2 + 1] = pack(kp[1]);
                    v[i * 2] = pack(vp[0]);
                    v[i * 2 + 1] = pack(vp[1]);
                }
                if (dim == 0) {
                    key_scale[token] = cache.k_scale[vec];
                    value_scale[token] = cache.v_scale[vec];
                }
            } else {
                __builtin_memcpy(k, static_cast<const short*>(cache.k) + vec * 256 + dim, 16);
                __builtin_memcpy(v, static_cast<const short*>(cache.v) + vec * 256 + dim, 16);
            }
            __builtin_memcpy(keys[token] + dim, k, 16);
#pragma unroll
            for (unsigned e = 0; e < 8; ++e)
                values[dim + e][token] = v[e];
        }
        detail::lds_barrier();
        Acc scores{};
#pragma unroll
        for (unsigned i = 0; i < 16; ++i)
            scores = multiply<compact5>(fragment(keys[col] + i * 16, half), q[i], scores);
        float next = maximum;
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const unsigned t = half * 8 + e;
            scores[e] *= a.scale;
            if constexpr (compact5 || dtype == KVCacheDType::FP8_E4M3)
                scores[e] *= key_scale[t];
            if (pg * 16 + t > visible)
                scores[e] = -FLT_MAX;
            next = fmaxf(next, scores[e]);
        }
        next = fmaxf(next, __shfl_xor(next, 16));
        const float alpha = __expf(maximum - next);
        maximum = next;
        float subtotal = 0;
        short hi[8], lo[8];
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            float p = scores[e] == -FLT_MAX ? 0.f : __expf(scores[e] - next);
            subtotal += p;
            if constexpr (compact5 || dtype == KVCacheDType::FP8_E4M3)
                p *= value_scale[half * 8 + e];
            hi[e] = compact5 ? half_bits(p) : pack(p);
            lo[e] = pack(p - __bfloat162float(__short_as_bfloat16(hi[e])));
        }
        sum = sum * alpha + (subtotal + __shfl_xor(subtotal, 16));
        Fragment ph, pl;
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            const int other = __shfl_xor(int(half ? hi[e] : hi[e + 4]), 16);
            const int residual = __shfl_xor(int(half ? lo[e] : lo[e + 4]), 16);
            ph[e] = half ? other : hi[e];
            ph[e + 4] = half ? hi[e + 4] : other;
            pl[e] = half ? residual : lo[e];
            pl[e + 4] = half ? lo[e + 4] : residual;
        }
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) {
            const Fragment vf = fragment(values[i * 16 + col], half);
            output[i] *= alpha;
            output[i] = multiply<compact5>(vf, ph, output[i]);
            if constexpr (!compact5)
                output[i] = multiply<false>(vf, pl, output[i]);
        }
    }
    if (qid >= rows * 6)
        return;
    float* dst = partials + ((size_t(row) * 24 + qhead) * splits + blockIdx.x) * kPagedVerifyPartialFloats;
    if (half == 0) {
        dst[0] = maximum;
        dst[1] = sum;
    }
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
#pragma unroll
        for (unsigned e = 0; e < 8; ++e)
            dst[2 + i * 16 + half * 8 + e] = output[i][e];
}

template <bool rotate>
__global__ void verify_reduce(PagedAttentionCommonArgs a, const float* partials, unsigned splits) {
    __shared__ float weights[kPagedVerifySplits];
    __shared__ bool has_values;
    __shared__ float scratch[256];
    const unsigned tid = threadIdx.x, job = blockIdx.x;
    const float* src = partials + size_t(job) * splits * kPagedVerifyPartialFloats;
    if (tid < 32) {
        float m = -FLT_MAX;
        for (unsigned s = tid; s < splits; s += 32)
            m = fmaxf(m, src[s * kPagedVerifyPartialFloats]);
        for (unsigned step = 16; step; step >>= 1)
            m = fmaxf(m, __shfl_xor(m, step));
        float sum = 0;
        for (unsigned s = tid; s < splits; s += 32) {
            weights[s] = __expf(src[s * kPagedVerifyPartialFloats] - m);
            sum += src[s * kPagedVerifyPartialFloats + 1] * weights[s];
        }
        for (unsigned step = 16; step; step >>= 1)
            sum += __shfl_xor(sum, step);
        if (tid == 0)
            has_values = sum > 0.f;
        for (unsigned s = tid; s < splits; s += 32)
            weights[s] *= sum > 0 ? 1.f / sum : 0.f;
    }
    detail::lds_barrier();
    if (!has_values && !rotate)
        return;
    float value = 0;
    for (unsigned s = 0; s < splits; ++s)
        value += src[s * kPagedVerifyPartialFloats + 2 + tid] * weights[s];
    if constexpr (rotate) {
        detail::octave::wht(value, scratch, 256);
        value *= detail::octave::sign_of(tid, 29);
    }
    static_cast<float*>(a.output)[size_t(job / 24) * a.output_row_stride + (job % 24) * 256 + tid] = value;
}

}

hipError_t launch_paged_verify_split(const PagedAttentionCommonArgs& a, KVCacheDType dtype,
                                     PagedVerifyCache cache, float* partials, unsigned splits,
                                     hipStream_t stream) {
    if (!paged_verify_geometry(a) || !a.q || !a.block_tables || !a.row_positions || !a.row_sequence_slots ||
        !partials || !cache.k || !cache.v || splits == 0 || splits > kPagedVerifySplits ||
        (dtype == KVCacheDType::FP8_E4M3 && (!cache.k_scale || !cache.v_scale)) ||
        (dtype == KVCacheDType::OCTAVE && (!cache.levels || !octave_format_valid(cache.format))))
        return hipErrorInvalidValue;
    auto launch = [&]<KVCacheDType type, bool compact5 = false>() {
        if constexpr (type == KVCacheDType::OCTAVE) {
            short* rotated = reinterpret_cast<short*>(partials + size_t(a.rows) * a.q_heads * splits *
                                                                     kPagedVerifyPartialFloats);
            rotate_queries<compact5><<<(a.rows * a.q_heads + 3) / 4, 128, 0, stream>>>(
                a, rotated, cache.format.k_mode == OctaveKMode::Compact);
            if (auto err = hipGetLastError(); err != hipSuccess)
                return err;
        }
        if (a.rows > 8)
            verify_pages<type, 6, compact5><<<dim3(splits, 4), 192, 0, stream>>>(a, cache, partials, splits);
        else
            verify_pages<type, 3, compact5><<<dim3(splits, 4), 96, 0, stream>>>(a, cache, partials, splits);
        return hipGetLastError();
    };
    switch (dtype) {
    case KVCacheDType::OCTAVE:
        if (cache.format.k_mode == OctaveKMode::Compact && cache.format.k_nope_bits == 5 &&
            cache.format.v_bits == 5)
            return launch.template operator()<KVCacheDType::OCTAVE, true>();
        else
            return launch.template operator()<KVCacheDType::OCTAVE>();
    case KVCacheDType::FP8_E4M3:
        return launch.template operator()<KVCacheDType::FP8_E4M3>();
    case KVCacheDType::BF16:
        return launch.template operator()<KVCacheDType::BF16>();
    }
    return hipErrorInvalidValue;
}

hipError_t launch_paged_verify_reduce(const PagedAttentionCommonArgs& a, KVCacheDType dtype,
                                      const float* partials, unsigned splits, hipStream_t stream) {
    if (!paged_verify_geometry(a) || !a.output || !partials || splits == 0 || splits > kPagedVerifySplits)
        return hipErrorInvalidValue;
    if (dtype == KVCacheDType::OCTAVE)
        verify_reduce<true><<<a.rows * a.q_heads, 256, 0, stream>>>(a, partials, splits);
    else
        verify_reduce<false><<<a.rows * a.q_heads, 256, 0, stream>>>(a, partials, splits);
    return hipGetLastError();
}

}
