#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/kernels/optimized/detail/attention_paged_common.h>
#include <shelf/kernels/optimized/detail/lds_barrier_device.h>
#include <shelf/quantization/codec/e4m3_detail.h>
#include <shelf/kernels/kv_types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cfloat>
#include <type_traits>

namespace shelf::kernel {

namespace { namespace v_prefill_bf16 {

using bf16 = __hip_bfloat16;

using wbf8_t = short __attribute__((ext_vector_type(8)));
using wf8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kWaves = 8u;
constexpr uint32_t kThreads = kWaves * 32u;
constexpr uint32_t kTileRows = 32u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kSmaxPitch = 17u;
constexpr uint32_t kPitch = kHeadDim + 8u;
constexpr uint32_t kKsPerWarp = 4u;
constexpr uint32_t kChunksPerRow = kHeadDim / 8u;
constexpr uint32_t kChunksPerPage = kPageTokens * kChunksPerRow;
constexpr uint32_t kPPitch = 24u;

static_assert(kThreads == kPagedAttentionPrefillBlock);
static_assert(kTileRows == kPagedAttentionPrefillTileRows);

__device__ __forceinline__ unsigned short f2bf(float f) {
    return static_cast<unsigned short>((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16u);
}

__device__ __forceinline__ wbf8_t pack_frag(uint2 a, uint2 b) {
    union { unsigned w[4]; wbf8_t f; } u;
    u.w[0] = a.x; u.w[1] = a.y; u.w[2] = b.x; u.w[3] = b.y;
    return u.f;
}

__device__ __forceinline__ wbf8_t frag_nk(
    const unsigned short* src, uint32_t pitch, uint32_t base, uint32_t kbase, uint32_t lane) {
    const unsigned short* p = src + static_cast<size_t>(base + (lane & 15u)) * pitch
        + 4u * (lane >> 4u) + kbase;
    const uint2 a = *reinterpret_cast<const uint2*>(p);
    const uint2 b = *reinterpret_cast<const uint2*>(p + 8);
    return pack_frag(a, b);
}

__device__ __forceinline__ wbf8_t frag_kn(
    const unsigned short* src, uint32_t pitch, uint32_t base, uint32_t kbase, uint32_t lane) {
    const uint32_t idx = base + (lane & 15u);
    const uint32_t k0 = kbase + 4u * (lane >> 4u);
    union { unsigned short h[8]; wbf8_t f; } u;
    #pragma unroll
    for (uint32_t e = 0u; e < 4u; ++e) {
        u.h[e] = src[static_cast<size_t>(k0 + e) * pitch + idx];
        u.h[4u + e] = src[static_cast<size_t>(k0 + 8u + e) * pitch + idx];
    }
    return u.f;
}

__device__ __forceinline__ wf8_t wmma_bf16(wbf8_t a, wbf8_t b, wf8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ float group_max8(float x) {
    x = fmaxf(x, __shfl_xor(x, 4, 32));
    x = fmaxf(x, __shfl_xor(x, 2, 32));
    x = fmaxf(x, __shfl_xor(x, 1, 32));
    return x;
}

__device__ __forceinline__ float group_sum8(float x) {
    x += __shfl_xor(x, 4, 32);
    x += __shfl_xor(x, 2, 32);
    x += __shfl_xor(x, 1, 32);
    return x;
}

struct PrefillSharedBase {
    float s_part[2][kWaves / 2u][16][kSmaxPitch];
    unsigned short p_tile[kTileRows][kPPitch];
    unsigned short p_tile_lo[kTileRows][kPPitch];
    float row_max[kTileRows];
    float row_sum[kTileRows];
    float row_alpha[kTileRows];
    uint32_t vis[kTileRows];
    uint32_t rows_here;
    uint32_t visible_max;
    uint32_t page_count;
    uint32_t invalid;
    const uint32_t* block_table;
};

struct PrefillSharedV0 : PrefillSharedBase {
    unsigned short v_page[2][kPageTokens][kPitch];
};

struct PrefillSharedKV : PrefillSharedBase {
    unsigned short k_page[kPageTokens][kPitch];
    unsigned short v_page[1][kPageTokens][kPitch];
};

template <uint32_t kMode>
__device__ __forceinline__
void attention_paged_prefill_bf16_body(
    const PagedAttentionCommonArgs a,
    const bf16* k_pool,
    const bf16* v_pool,
    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const bf16* k_scale_pool,
    const bf16* v_scale_pool,
    uint32_t blocks_per_head,
    void* out_base) {
    const uint32_t q_heads = a.q_heads;
    const uint32_t kv_heads = a.kv_heads;
    const uint32_t q_per_kv = q_heads / kv_heads;
    const float scale = a.scale;

    const uint32_t row0 = blockIdx.x * kTileRows;
    const uint32_t q_head = blockIdx.y;
    const uint32_t kv_head = q_head / q_per_kv;

    if (row0 >= a.rows) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t hi = lane >> 4u;
    const uint32_t lo = lane & 15u;

    using PrefillShared = std::conditional_t<kMode == 0u, PrefillSharedV0, PrefillSharedKV>;
    __shared__ PrefillShared s;

    if (tid == 0u) {
        uint32_t rows = a.rows - row0;
        if (rows > kTileRows) rows = kTileRows;
        s.rows_here = rows;
        s.invalid = 0u;
        const uint32_t slot = a.row_sequence_slots[row0];
        uint32_t valid = (slot < a.max_sequences) ? 1u : 0u;
        uint32_t vis_max = 0u;
        if (valid != 0u) {
            vis_max = a.row_positions[row0 + rows - 1u] + 1u;
            const uint64_t cap = static_cast<uint64_t>(a.block_table_stride) * a.page_tokens;
            if (static_cast<uint64_t>(vis_max) > cap) valid = 0u;
        }
        if (valid == 0u) s.invalid = 1u;
        s.visible_max = vis_max;
        s.page_count = (vis_max + kPageTokens - 1u) / kPageTokens;
        s.block_table = a.block_tables
            + static_cast<size_t>(valid != 0u ? slot : 0u) * a.block_table_stride;
    }
    detail::lds_barrier();
    if (s.invalid != 0u) return;

    const uint32_t rows = s.rows_here;
    const uint32_t visible_max = s.visible_max;
    const uint32_t page_count = s.page_count;
    const uint32_t* block_table = s.block_table;

    if (tid < kTileRows) {
        s.vis[tid] = (tid < rows) ? (a.row_positions[row0 + tid] + 1u) : 0u;
        s.row_max[tid] = -FLT_MAX;
        s.row_sum[tid] = 0.0f;
        s.row_alpha[tid] = 0.0f;
    }

    const uint32_t mt = warp & 1u;
    const uint32_t kq = warp >> 1u;
    const uint32_t nq = warp >> 1u;

    const bf16* q_base = reinterpret_cast<const bf16*>(a.q)
        + static_cast<size_t>(row0 + mt * 16u) * a.q_row_stride
        + static_cast<size_t>(q_head) * kHeadDim;

    wbf8_t q_frag[4];
    #pragma unroll
    for (uint32_t ks = 0u; ks < 4u; ++ks) {
        const unsigned short* p = reinterpret_cast<const unsigned short*>(q_base)
            + static_cast<size_t>(lane & 15u) * a.q_row_stride
            + (kq * 4u + ks) * 16u + 4u * (lane >> 4u);
        q_frag[ks] = pack_frag(*reinterpret_cast<const uint2*>(p),
                               *reinterpret_cast<const uint2*>(p + 8));
    }

    wf8_t oacc[4];
    #pragma unroll
    for (uint32_t nt = 0u; nt < 4u; ++nt) oacc[nt] = wf8_t{};

    const uint64_t layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    const uint64_t head_off = static_cast<uint64_t>(kv_head) * kHeadDim;

    for (uint32_t page_index = 0u; page_index < page_count; ++page_index) {
        const uint32_t page = block_table[page_index];
        if (page >= a.num_pages) {
            if (tid == 0u) s.invalid = 1u;
            break;
        }
        const uint32_t page_start = page_index * kPageTokens;
        uint32_t tokens_here = visible_max - page_start;
        if (tokens_here > kPageTokens) tokens_here = kPageTokens;
        const uint64_t page_base = layer_base + static_cast<uint64_t>(page) * a.elems_per_page;
        const bf16* k_src = k_pool + page_base + head_off;
        const bf16* v_src = v_pool + page_base + head_off;
        if constexpr (kMode != 0u) {
            (void)k_src;
            (void)v_src;
        } else {
            (void)k_code_pool; (void)v_code_pool; (void)k_scale_pool; (void)v_scale_pool;
            (void)blocks_per_head;
        }
        const uint32_t vbuf = (kMode == 0u) ? (page_index & 1u) : 0u;
        auto store_chunk = [&](uint32_t t, uint32_t c) {
            uint4 kv;
            uint4 vv;
            {
                const size_t off = static_cast<size_t>(t) * a.elems_per_token + c * 8u;
                vv = *reinterpret_cast<const uint4*>(v_src + off);
                if constexpr (kMode != 0u)
                    kv = *reinterpret_cast<const uint4*>(k_src + off);
            }
            if constexpr (kMode != 0u)
                *reinterpret_cast<uint4*>(&s.k_page[t][c * 8u]) = kv;
            *reinterpret_cast<uint4*>(&s.v_page[vbuf][t][c * 8u]) = vv;
        };
        if (page_index + 1u < page_count) {
            for (uint32_t i = tid; i < kChunksPerPage; i += kThreads) {
                const uint32_t t = i / kChunksPerRow;
                const uint32_t c = i % kChunksPerRow;
                store_chunk(t, c);
            }
        } else {
            for (uint32_t i = tid; i < kChunksPerPage; i += kThreads) {
                const uint32_t t = i / kChunksPerRow;
                const uint32_t c = i % kChunksPerRow;
                if (t < tokens_here) {
                    store_chunk(t, c);
                } else {
                    if constexpr (kMode != 0u)
                        *reinterpret_cast<uint4*>(&s.k_page[t][c * 8u]) = make_uint4(0u, 0u, 0u, 0u);
                    *reinterpret_cast<uint4*>(&s.v_page[vbuf][t][c * 8u]) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        detail::lds_barrier();

        {
            wf8_t c = wf8_t{};
            #pragma unroll
            for (uint32_t ks = 0u; ks < kKsPerWarp; ++ks) {
                const uint32_t k0 = (kq * kKsPerWarp + ks) * 16u;
                wbf8_t fb;
                if constexpr (kMode == 0u)
                    fb = frag_nk(reinterpret_cast<const unsigned short*>(k_src), a.elems_per_token, 0u, k0, lane);
                else
                    fb = frag_nk(&s.k_page[0][0], kPitch, 0u, k0, lane);
                c = wmma_bf16(q_frag[ks], fb, c);
            }
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e)
                s.s_part[mt][kq][8u * hi + e][lo] = c[e];
        }
        detail::lds_barrier();

        {
            const uint32_t r = tid >> 3u;
            const uint32_t tg = tid & 7u;
            const uint32_t pmt = r >> 4u;
            const uint32_t rr = r & 15u;
            const uint32_t ta = tg;
            const uint32_t tb = tg + 8u;
            float sa = s.s_part[pmt][0][rr][ta] + s.s_part[pmt][1][rr][ta]
                + s.s_part[pmt][2][rr][ta] + s.s_part[pmt][3][rr][ta];
            float sb = s.s_part[pmt][0][rr][tb] + s.s_part[pmt][1][rr][tb]
                + s.s_part[pmt][2][rr][tb] + s.s_part[pmt][3][rr][tb];
            const uint32_t visr = s.vis[r];
            const uint32_t tok_a = page_start + ta;
            const uint32_t tok_b = page_start + tb;
            const uint32_t mask_row = row0 + r;
            const bool va = tok_a < visr &&
                paged_token_visible(a.token_mask, a.mask_stride, a.mask_tokens,
                                    mask_row, tok_a);
            const bool vb = tok_b < visr &&
                paged_token_visible(a.token_mask, a.mask_stride, a.mask_tokens,
                                    mask_row, tok_b);
            sa = va ? sa * scale : -FLT_MAX;
            sb = vb ? sb * scale : -FLT_MAX;
            const float pmax = group_max8(fmaxf(sa, sb));
            const float m_old = s.row_max[r];
            const float m_new = fmaxf(m_old, pmax);
            const float alpha = __expf(m_old - m_new);
            const float pa = va ? __expf(sa - m_new) : 0.0f;
            const float pb = vb ? __expf(sb - m_new) : 0.0f;
            const float psum = group_sum8(pa + pb);
            const unsigned short phi = f2bf(pa);
            const unsigned short plo =
                f2bf(pa - __bfloat162float(__ushort_as_bfloat16(phi)));
            const unsigned short pphi = f2bf(pb);
            const unsigned short pplo =
                f2bf(pb - __bfloat162float(__ushort_as_bfloat16(pphi)));
            s.p_tile[r][ta] = phi;
            s.p_tile_lo[r][ta] = plo;
            s.p_tile[r][tb] = pphi;
            s.p_tile_lo[r][tb] = pplo;
            if (tg == 0u) {
                s.row_max[r] = m_new;
                s.row_sum[r] = s.row_sum[r] * alpha + psum;
                s.row_alpha[r] = alpha;
            }
        }
        detail::lds_barrier();

        {
            const wbf8_t fa = frag_nk(&s.p_tile[0][0], kPPitch, mt * 16u, 0u, lane);
            const wbf8_t fa_lo = frag_nk(&s.p_tile_lo[0][0], kPPitch, mt * 16u, 0u, lane);
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e) {
                const float al = s.row_alpha[mt * 16u + 8u * hi + e];
                #pragma unroll
                for (uint32_t nt = 0u; nt < 4u; ++nt) oacc[nt][e] *= al;
            }
            #pragma unroll
            for (uint32_t nt = 0u; nt < 4u; ++nt) {
                const wbf8_t fb = frag_kn(&s.v_page[vbuf][0][0], kPitch, nq * 64u + nt * 16u, 0u, lane);
                oacc[nt] = wmma_bf16(fa, fb, oacc[nt]);
                oacc[nt] = wmma_bf16(fa_lo, fb, oacc[nt]);
            }
        }
        if constexpr (kMode != 0u) detail::lds_barrier();
    }

    if (s.invalid != 0u) return;

    if (tid < kTileRows && tid < rows) s.row_sum[tid] = 1.0f / s.row_sum[tid];
    detail::lds_barrier();

    const uint32_t r_base = mt * 16u + 8u * hi;
    #pragma unroll
    for (uint32_t nt = 0u; nt < 4u; ++nt) {
        const uint32_t dim = nq * 64u + nt * 16u + lo;
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            const uint32_t r = r_base + e;
            if (r < rows) {
                const float v = oacc[nt][e] * s.row_sum[r];
                const size_t o = static_cast<size_t>(row0 + r) * a.output_row_stride
                    + static_cast<size_t>(q_head) * kHeadDim + dim;
                if (a.output_dtype == PagedAttentionOutputDType::F32) {
                    reinterpret_cast<float*>(out_base)[o] = v;
                } else {
                    reinterpret_cast<bf16*>(out_base)[o] = __float2bfloat16(v);
                }
            }
        }
    }
}

template <uint32_t kMode>
__global__ __launch_bounds__(kThreads, 2)
void attention_paged_prefill_bf16_impl(
    const PagedAttentionCommonArgs a,
    const bf16* k_pool,
    const bf16* v_pool,
    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const bf16* k_scale_pool,
    const bf16* v_scale_pool,
    uint32_t blocks_per_head,
    void* out_base) {
    attention_paged_prefill_bf16_body<kMode>(
        a, k_pool, v_pool, k_code_pool, v_code_pool, k_scale_pool, v_scale_pool,
        blocks_per_head, out_base);
}

} }  // namespace v_prefill_bf16

extern "C" __global__ __launch_bounds__(kPagedAttentionPrefillBlock)
void shelf_attention_paged_prefill_bf16(
    ::shelf::kernel::PagedAttentionBf16Args args) {
    using namespace v_prefill_bf16;
    attention_paged_prefill_bf16_body<0u>(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool),
        nullptr, nullptr, nullptr, nullptr, 0u,
        args.common.output);
}

hipError_t launch_attention_paged_prefill_bf16(
    const PagedAttentionCommonArgs& common,
    const bf16_t* k_pool,
    const bf16_t* v_pool,
    hipStream_t stream) {
    using namespace v_prefill_bf16;
    if (common.rows == 0u || common.head_dim != kHeadDim) return hipErrorInvalidValue;
    if (common.page_tokens != kPageTokens) return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    if (common.block_table_stride == 0u || common.num_pages == 0u) return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return hipErrorInvalidValue;
    const uint32_t tiles = (common.rows + kTileRows - 1u) / kTileRows;
    const PagedAttentionCommonArgs a = common;
    attention_paged_prefill_bf16_impl<0u><<<dim3(tiles, common.q_heads), kThreads, 0, stream>>>(
        a, reinterpret_cast<const bf16*>(k_pool), reinterpret_cast<const bf16*>(v_pool),
        nullptr, nullptr, nullptr, nullptr, 0u,
        common.output);
    return hipGetLastError();
}

namespace { namespace v_prefill_fp8 {

using bf16 = __hip_bfloat16;
using fp8 = ::shelf::kernel::fp8e4m3_storage_t;
using ::shelf::quantization::codec::device::e4m3_encode_u8;


using wf8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kWaves = 8u;
constexpr uint32_t kThreads = kWaves * 32u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kTileRows = 64u;
constexpr uint32_t kMTiles = kTileRows / 16u;
constexpr uint32_t kDimHalves = kWaves / kMTiles;
constexpr uint32_t kDimsPerHalf = kHeadDim / kDimHalves;
constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kSmaxPitch = 17u;
constexpr uint32_t kKsPerWarp = kDimsPerHalf / 16u;
constexpr uint32_t kOfrags = kDimsPerHalf / 16u;
constexpr uint32_t kChunksPerRow = kHeadDim / 8u;
constexpr uint32_t kChunksPerPage = kPageTokens * kChunksPerRow;
constexpr uint32_t kPPitch = 24u;
constexpr uint32_t kRowsPerPass = kThreads / 8u;

using i32x2_t = int __attribute__((ext_vector_type(2)));
constexpr uint32_t kPitch8 = kHeadDim + 8u;
constexpr uint32_t kVPitch = 20u;
constexpr float kE4m3Max = 448.0f;
constexpr float kPQuantMinScale = 1e-30f;

__device__ __forceinline__ i32x2_t pack_frag_fp8(uint32_t lo, uint32_t hi) {
    union { unsigned w[2]; i32x2_t f; } u;
    u.w[0] = lo; u.w[1] = hi;
    return u.f;
}

__device__ __forceinline__ i32x2_t frag_kn_fp8(
    const unsigned char* src, uint32_t pitch, uint32_t row, uint32_t kbase, uint32_t lane) {
    const unsigned char* q = src + static_cast<size_t>(row) * pitch + kbase + 4u * (lane >> 4u);
    return pack_frag_fp8(*reinterpret_cast<const uint32_t*>(q),
                         *reinterpret_cast<const uint32_t*>(q + 8u));
}

__device__ __forceinline__ wf8_t wmma_fp8(i32x2_t a, i32x2_t b, wf8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}


__device__ __forceinline__ float bf2f(unsigned short h) {
    return __builtin_bit_cast(float, static_cast<unsigned>(h) << 16u);
}

__device__ __forceinline__ float group_max8(float x) {
    x = fmaxf(x, __shfl_xor(x, 4, 32));
    x = fmaxf(x, __shfl_xor(x, 2, 32));
    x = fmaxf(x, __shfl_xor(x, 1, 32));
    return x;
}

__device__ __forceinline__ float group_sum8(float x) {
    x += __shfl_xor(x, 4, 32);
    x += __shfl_xor(x, 2, 32);
    x += __shfl_xor(x, 1, 32);
    return x;
}

struct PrefillShared {
    unsigned char k_page[kPageTokens][kPitch8];
    unsigned char v_page[kHeadDim][kVPitch];
    float s_part[kMTiles][kDimHalves][16][kSmaxPitch];
    unsigned char p_tile[kTileRows][kPPitch];
    float p_scale[kTileRows];
    float row_max[kTileRows];
    float row_sum[kTileRows];
    float row_alpha[kTileRows];
    float q_scale[kTileRows];
    uint32_t vis[kTileRows];
    uint32_t rows_here;
    uint32_t visible_max;
    uint32_t page_count;
    uint32_t invalid;
    const uint32_t* block_table;
};

__global__ __launch_bounds__(kThreads, 2)
void attention_paged_prefill_fp8_impl(
    const PagedAttentionCommonArgs a,
    const fp8* k_pool,
    const fp8* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    void* out_base) {
    const uint32_t q_heads = a.q_heads;
    const uint32_t kv_heads = a.kv_heads;
    const uint32_t q_per_kv = q_heads / kv_heads;
    const float scale = a.scale;

    const uint32_t row0 = blockIdx.x * kTileRows;
    const uint32_t q_head = blockIdx.y;
    const uint32_t kv_head = q_head / q_per_kv;

    if (row0 >= a.rows) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t hi = lane >> 4u;
    const uint32_t lo = lane & 15u;

    __shared__ PrefillShared s;

    if (tid == 0u) {
        uint32_t rows = a.rows - row0;
        if (rows > kTileRows) rows = kTileRows;
        s.rows_here = rows;
        s.invalid = 0u;
        const uint32_t slot = a.row_sequence_slots[row0];
        uint32_t valid = (slot < a.max_sequences) ? 1u : 0u;
        uint32_t vis_max = 0u;
        if (valid != 0u) {
            vis_max = a.row_positions[row0 + rows - 1u] + 1u;
            const uint64_t cap = static_cast<uint64_t>(a.block_table_stride) * a.page_tokens;
            if (static_cast<uint64_t>(vis_max) > cap) valid = 0u;
        }
        if (valid == 0u) s.invalid = 1u;
        s.visible_max = vis_max;
        s.page_count = (vis_max + kPageTokens - 1u) / kPageTokens;
        s.block_table = a.block_tables
            + static_cast<size_t>(valid != 0u ? slot : 0u) * a.block_table_stride;
    }
    detail::lds_barrier();
    if (s.invalid != 0u) return;

    const uint32_t rows = s.rows_here;
    const uint32_t visible_max = s.visible_max;
    const uint32_t page_count = s.page_count;
    const uint32_t* block_table = s.block_table;

    if (tid < kTileRows) {
        s.vis[tid] = (tid < rows) ? (a.row_positions[row0 + tid] + 1u) : 0u;
        s.row_max[tid] = -FLT_MAX;
        s.row_sum[tid] = 0.0f;
        s.row_alpha[tid] = 0.0f;
    }

    const uint32_t mt = warp % kMTiles;
    const uint32_t kq = warp / kMTiles;
    const uint32_t nq = kq;

    const bf16* q_base = reinterpret_cast<const bf16*>(a.q)
        + static_cast<size_t>(row0 + mt * 16u) * a.q_row_stride
        + static_cast<size_t>(q_head) * kHeadDim;

    const size_t scale_layer_base =
        static_cast<size_t>(a.layer) * a.num_pages * kPageTokens * a.kv_heads;

    {
        const uint32_t qseg = tid & 7u;
        for (uint32_t qrow = tid >> 3u; qrow < kTileRows; qrow += kRowsPerPass) {
            float local_max = 0.0f;
            if (qrow < rows) {
                const bf16* qr = reinterpret_cast<const bf16*>(a.q)
                    + static_cast<size_t>(row0 + qrow) * a.q_row_stride
                    + static_cast<size_t>(q_head) * kHeadDim;
                #pragma unroll
                for (uint32_t i = 0u; i < 32u; ++i) {
                    local_max = fmaxf(local_max, fabsf(__bfloat162float(qr[qseg * 32u + i])));
                }
            }
            local_max = fmaxf(local_max, __shfl_xor(local_max, 4, 32));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 2, 32));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 1, 32));
            if (qseg == 0u)
                s.q_scale[qrow] = (qrow < rows && local_max > 0.0f) ? local_max / kE4m3Max : 1.0f;
        }
    }
    detail::lds_barrier();

    i32x2_t q_frag[kKsPerWarp];
    {
        const float inv_qs = 1.0f / s.q_scale[mt * 16u + (lane & 15u)];
        #pragma unroll
        for (uint32_t ks = 0u; ks < kKsPerWarp; ++ks) {
            const unsigned short* p = reinterpret_cast<const unsigned short*>(q_base)
                + static_cast<size_t>(lane & 15u) * a.q_row_stride
                + (kq * kKsPerWarp + ks) * 16u + 4u * (lane >> 4u);
            const uint2 w0 = *reinterpret_cast<const uint2*>(p);
            const uint2 w1 = *reinterpret_cast<const uint2*>(p + 8);
            const unsigned short h[8] = {
                static_cast<unsigned short>(w0.x), static_cast<unsigned short>(w0.x >> 16u),
                static_cast<unsigned short>(w0.y), static_cast<unsigned short>(w0.y >> 16u),
                static_cast<unsigned short>(w1.x), static_cast<unsigned short>(w1.x >> 16u),
                static_cast<unsigned short>(w1.y), static_cast<unsigned short>(w1.y >> 16u)};
            uint32_t lo = 0u;
            uint32_t hi = 0u;
            #pragma unroll
            for (uint32_t e = 0u; e < 4u; ++e)
                lo |= static_cast<uint32_t>(e4m3_encode_u8(bf2f(h[e]) * inv_qs)) << (8u * e);
            #pragma unroll
            for (uint32_t e = 0u; e < 4u; ++e)
                hi |= static_cast<uint32_t>(e4m3_encode_u8(bf2f(h[4u + e]) * inv_qs)) << (8u * e);
            q_frag[ks] = pack_frag_fp8(lo, hi);
        }
    }

    wf8_t oacc[kOfrags];
    #pragma unroll
    for (uint32_t nt = 0u; nt < kOfrags; ++nt) oacc[nt] = wf8_t{};

    const uint64_t layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    const uint64_t head_off = static_cast<uint64_t>(kv_head) * kHeadDim;

    for (uint32_t page_index = 0u; page_index < page_count; ++page_index) {
        const uint32_t page = block_table[page_index];
        if (page >= a.num_pages) {
            if (tid == 0u) s.invalid = 1u;
            break;
        }
        const uint32_t page_start = page_index * kPageTokens;
        uint32_t tokens_here = visible_max - page_start;
        if (tokens_here > kPageTokens) tokens_here = kPageTokens;
        const uint64_t page_base = layer_base + static_cast<uint64_t>(page) * a.elems_per_page;
        const size_t scale_page_base =
            scale_layer_base + static_cast<size_t>(page) * kPageTokens * a.kv_heads;
        const fp8* k_src = k_pool + page_base + head_off;
        const fp8* v_src = v_pool + page_base + head_off;
        auto store_chunk = [&](uint32_t t, uint32_t c) {
            const size_t off = static_cast<size_t>(t) * a.elems_per_token + c * 8u;
            const uint2 pk = *reinterpret_cast<const uint2*>(k_src + off);
            const uint2 pv = *reinterpret_cast<const uint2*>(v_src + off);
            *reinterpret_cast<uint2*>(&s.k_page[t][c * 8u]) = pk;
            const unsigned char* vbytes = reinterpret_cast<const unsigned char*>(&pv);
            #pragma unroll
            for (uint32_t j = 0u; j < 8u; ++j)
                s.v_page[c * 8u + j][t] = vbytes[j];
        };
        if (page_index + 1u < page_count) {
            for (uint32_t i = tid; i < kChunksPerPage; i += kThreads)
                store_chunk(i / kChunksPerRow, i % kChunksPerRow);
        } else {
            for (uint32_t i = tid; i < kChunksPerPage; i += kThreads) {
                const uint32_t t = i / kChunksPerRow;
                const uint32_t c = i % kChunksPerRow;
                if (t < tokens_here) {
                    store_chunk(t, c);
                } else {
                    *reinterpret_cast<uint2*>(&s.k_page[t][c * 8u]) = make_uint2(0u, 0u);
                    #pragma unroll
                    for (uint32_t j = 0u; j < 8u; ++j)
                        s.v_page[c * 8u + j][t] = 0u;
                }
            }
        }
        detail::lds_barrier();

        {
            wf8_t c = wf8_t{};
            #pragma unroll
            for (uint32_t ks = 0u; ks < kKsPerWarp; ++ks) {
                const uint32_t k0 = (kq * kKsPerWarp + ks) * 16u;
                const i32x2_t fb = frag_kn_fp8(&s.k_page[0][0], kPitch8, lane & 15u, k0, lane);
                c = wmma_fp8(q_frag[ks], fb, c);
            }
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e)
                s.s_part[mt][kq][8u * hi + e][lo] = c[e];
        }
        detail::lds_barrier();

        {
            const uint32_t tg = tid & 7u;
            const uint32_t ta = tg;
            const uint32_t tb = tg + 8u;
            for (uint32_t r = tid >> 3u; r < kTileRows; r += kRowsPerPass) {
                const uint32_t pmt = r >> 4u;
                const uint32_t rr = r & 15u;
                float sa = s.s_part[pmt][0][rr][ta];
                float sb = s.s_part[pmt][0][rr][tb];
                #pragma unroll
                for (uint32_t kh = 1u; kh < kDimHalves; ++kh) {
                    sa += s.s_part[pmt][kh][rr][ta];
                    sb += s.s_part[pmt][kh][rr][tb];
                }
                const uint32_t visr = s.vis[r];
                const bool va = (page_start + ta) < visr;
                const bool vb = (page_start + tb) < visr;
                const size_t sa_idx = scale_page_base + static_cast<size_t>(ta) * a.kv_heads + kv_head;
                const size_t sb_idx = scale_page_base + static_cast<size_t>(tb) * a.kv_heads + kv_head;
                const float qsc = s.q_scale[r];
                sa = va ? sa * qsc * scale * k_scale_pool[sa_idx] : -FLT_MAX;
                sb = vb ? sb * qsc * scale * k_scale_pool[sb_idx] : -FLT_MAX;
                const float pmax = group_max8(fmaxf(sa, sb));
                const float m_old = s.row_max[r];
                const float m_new = fmaxf(m_old, pmax);
                const float alpha = __expf(m_old - m_new);
                const float pa = va ? __expf(sa - m_new) : 0.0f;
                const float pb = vb ? __expf(sb - m_new) : 0.0f;
                const float psum = group_sum8(pa + pb);
                const float pva = pa * v_scale_pool[sa_idx];
                const float pvb = pb * v_scale_pool[sb_idx];
                const float pmax_v = group_max8(fmaxf(pva, pvb));
                const bool p_live = pmax_v > kPQuantMinScale;
                const float inv_pv = p_live ? 1.0f / pmax_v : 0.0f;
                s.p_tile[r][ta] = e4m3_encode_u8(pva * inv_pv);
                s.p_tile[r][tb] = e4m3_encode_u8(pvb * inv_pv);
                if (tg == 0u) s.p_scale[r] = p_live ? pmax_v : 0.0f;
                if (tg == 0u) {
                    s.row_max[r] = m_new;
                    s.row_sum[r] = s.row_sum[r] * alpha + psum;
                    s.row_alpha[r] = alpha;
                }
            }
        }
        detail::lds_barrier();

        {
            const i32x2_t fa = frag_kn_fp8(&s.p_tile[0][0], kPPitch,
                                          mt * 16u + (lane & 15u), 0u, lane);
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e) {
                const float al = s.row_alpha[mt * 16u + 8u * hi + e];
                #pragma unroll
                for (uint32_t nt = 0u; nt < kOfrags; ++nt) oacc[nt][e] *= al;
            }
            #pragma unroll
            for (uint32_t nt = 0u; nt < kOfrags; ++nt) {
                const i32x2_t fb = frag_kn_fp8(&s.v_page[0][0], kVPitch,
                                              nq * kDimsPerHalf + nt * 16u + (lane & 15u), 0u, lane);
                const wf8_t c = wmma_fp8(fa, fb, wf8_t{});
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++e)
                    oacc[nt][e] += c[e] * s.p_scale[mt * 16u + 8u * hi + e];
            }
        }
        detail::lds_barrier();
    }

    if (s.invalid != 0u) return;

    if (tid < kTileRows && tid < rows) s.row_sum[tid] = 1.0f / s.row_sum[tid];
    detail::lds_barrier();

    const uint32_t r_base = mt * 16u + 8u * hi;
    #pragma unroll
    for (uint32_t nt = 0u; nt < kOfrags; ++nt) {
        const uint32_t dim = nq * kDimsPerHalf + nt * 16u + lo;
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            const uint32_t r = r_base + e;
            if (r < rows) {
                const float v = oacc[nt][e] * s.row_sum[r];
                const size_t o = static_cast<size_t>(row0 + r) * a.output_row_stride
                    + static_cast<size_t>(q_head) * kHeadDim + dim;
                if (a.output_dtype == PagedAttentionOutputDType::F32) {
                    reinterpret_cast<float*>(out_base)[o] = v;
                } else {
                    reinterpret_cast<bf16*>(out_base)[o] = __float2bfloat16(v);
                }
            }
        }
    }
}

} }  // namespace v_prefill_fp8

hipError_t launch_attention_paged_prefill_fp8_e4m3(
    const PagedAttentionCommonArgs& common,
    const ::shelf::kernel::fp8e4m3_storage_t* k_pool,
    const ::shelf::kernel::fp8e4m3_storage_t* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    hipStream_t stream) {
    using namespace v_prefill_fp8;
    if (common.rows == 0u || common.head_dim != kHeadDim) return hipErrorInvalidValue;
    if (common.page_tokens != kPageTokens) return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    if (common.block_table_stride == 0u || common.num_pages == 0u) return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return hipErrorInvalidValue;
    const uint32_t tiles = (common.rows + kTileRows - 1u) / kTileRows;
    const PagedAttentionCommonArgs a = common;
    attention_paged_prefill_fp8_impl<<<dim3(tiles, common.q_heads), kThreads, 0, stream>>>(
        a, k_pool, v_pool, k_scale_pool, v_scale_pool, common.output);
    return hipGetLastError();
}

}  // namespace shelf::kernel
