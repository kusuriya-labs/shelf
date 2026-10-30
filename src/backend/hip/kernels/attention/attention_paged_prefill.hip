// vendored from PhaseShift (own work); namespace/type adapted only.
//
// prefill 用 paged attention（bf16 KV・head_dim 256・page_tokens 16）。
// decode 用の `attention_paged_decode.hip` とは構造が違い、Q を 32 行タイルで
// 処理して KV ページを LDS に張替えながら online softmax を回す。
//   * Q×K^T は bf16 WMMA（16x16x16）。K はプールから直接断片を作る（mode 0）。
//   * softmax 後の確率は bf16 の hi/lo 二成分で持ち、P×V を WMMA で二重に累加する
//     （bf16 一成分だと確率の丸め誤差が PV にそのまま乗るため）。
//   * causal 判定は行ごとの可见最大位置（`row_positions`）から決める。
// 受理条件（launcher 側で検査し、満たさなければ decode 経路へ戻す）:
//   rows > 1、head_dim == 256、page_tokens == 16、q_heads % kv_heads == 0。
#include <shelf/backend/hip/kernels/attention/paged_attention.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>
#include <cfloat>

namespace shelf::hipk {
namespace {
namespace prefill_bf16 {

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

struct PrefillShared {
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
    unsigned short v_page[2][kPageTokens][kPitch];
};

__device__ __forceinline__
void attention_paged_prefill_bf16_body(
    const PagedAttentionCommonArgs a,
    const bf16* k_pool,
    const bf16* v_pool,
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
    __syncthreads();
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
        const uint32_t vbuf = page_index & 1u;
        auto store_chunk = [&](uint32_t t, uint32_t c) {
            const size_t off = static_cast<size_t>(t) * a.elems_per_token + c * 8u;
            const uint4 vv = *reinterpret_cast<const uint4*>(v_src + off);
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
                    *reinterpret_cast<uint4*>(&s.v_page[vbuf][t][c * 8u]) =
                        make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        {   // S = Q K^T（K はプールから直接断片を作る。ページは 16 行なので LDS に張らない）
            wf8_t c = wf8_t{};
            #pragma unroll
            for (uint32_t ks = 0u; ks < kKsPerWarp; ++ks) {
                const uint32_t k0 = (kq * kKsPerWarp + ks) * 16u;
                const wbf8_t fb = frag_nk(reinterpret_cast<const unsigned short*>(k_src),
                                          a.elems_per_token, 0u, k0, lane);
                c = wmma_bf16(q_frag[ks], fb, c);
            }
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e)
                s.s_part[mt][kq][8u * hi + e][lo] = c[e];
        }
        __syncthreads();

        {   // 4 つの kq 波の部分和を畳み、online softmax。P は bf16 の hi/lo で保持
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
            const bool va = (page_start + ta) < visr;
            const bool vb = (page_start + tb) < visr;
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
        __syncthreads();

        {   // O = P V（P の hi/lo を同じ V 断片で二重累加 = bf16 の丸めを打ち消す）
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
                const wbf8_t fb = frag_kn(&s.v_page[vbuf][0][0], kPitch,
                                          nq * 64u + nt * 16u, 0u, lane);
                oacc[nt] = wmma_bf16(fa, fb, oacc[nt]);
                oacc[nt] = wmma_bf16(fa_lo, fb, oacc[nt]);
            }
        }
    }

    if (s.invalid != 0u) return;

    if (tid < kTileRows && tid < rows) s.row_sum[tid] = 1.0f / s.row_sum[tid];
    __syncthreads();

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

__global__ __launch_bounds__(kThreads, 2)
void attention_paged_prefill_bf16_impl(const PagedAttentionCommonArgs a,
                                       const bf16* k_pool,
                                       const bf16* v_pool,
                                       void* out_base) {
    attention_paged_prefill_bf16_body(a, k_pool, v_pool, out_base);
}

}  // namespace prefill_bf16
}  // namespace

extern "C" __global__ __launch_bounds__(kPagedAttentionPrefillBlock)
void phaseshift_attention_paged_prefill_bf16(
    ::shelf::hipk::PagedAttentionBf16Args args) {
    using namespace prefill_bf16;
    attention_paged_prefill_bf16_body(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool), args.common.output);
}

bool attention_paged_prefill_bf16_supported(const PagedAttentionCommonArgs& common) {
    using namespace prefill_bf16;
    return common.rows > 1u &&
           common.head_dim == kHeadDim &&
           common.page_tokens == kPageTokens &&
           common.q_heads > 0u && common.kv_heads > 0u &&
           common.q_heads % common.kv_heads == 0u &&
           common.block_table_stride > 0u && common.num_pages > 0u &&
           common.q != nullptr && common.output != nullptr &&
           common.row_sequence_slots != nullptr && common.row_positions != nullptr &&
           common.block_tables != nullptr;
}

hipError_t launch_attention_paged_prefill_bf16(
    const PagedAttentionCommonArgs& common,
    const bf16_t* k_pool,
    const bf16_t* v_pool,
    hipStream_t stream) {
    using namespace prefill_bf16;
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
    attention_paged_prefill_bf16_impl<<<dim3(tiles, common.q_heads), kThreads, 0, stream>>>(
        common, reinterpret_cast<const bf16*>(k_pool), reinterpret_cast<const bf16*>(v_pool),
        common.output);
    return hipGetLastError();
}

}  // namespace shelf::hipk
