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
#include <cstdlib>
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

__device__ __forceinline__ uint32_t octave_load_word(const uint8_t* source) {
    uint32_t value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

__device__ __forceinline__ uint32_t octave_unpack_plane_direct(
    const uint8_t* source, uint32_t index, uint32_t bits) {
    const uint32_t fields = 32u / bits;
    const uint32_t word = octave_load_word(source + (index / fields) * 4u);
    return (word >> ((index % fields) * bits)) & ((1u << bits) - 1u);
}

__device__ __forceinline__ uint32_t octave_unpack_direct(
    const uint8_t* source, uint32_t index, uint32_t dimensions,
    uint32_t bits) {
    if (bits >= 4u) {
        uint32_t code = octave_unpack_plane_direct(source, index, 4u);
        source += dimensions / 2u;
        if (bits == 5u)
            code |= octave_unpack_plane_direct(source, index, 1u) << 4u;
        if (bits == 6u)
            code |= octave_unpack_plane_direct(source, index, 2u) << 4u;
        return code;
    }
    uint32_t code = octave_unpack_plane_direct(source, index, 2u);
    source += dimensions / 4u;
    return code | (octave_unpack_plane_direct(source, index, 1u) << 2u);
}

// A wave lane always owns an aligned group of eight dimensions.  Decode the
// corresponding bit-plane words once instead of reloading the same word for
// every scalar code.
__device__ __forceinline__ void octave_unpack8_direct(
    const uint8_t* source, uint32_t base, uint32_t dimensions, uint32_t bits,
    uint32_t (&codes)[8]) {
    if (bits >= 4u) {
        const uint32_t low = octave_load_word(source + (base / 8u) * 4u);
        source += dimensions / 2u;
        uint32_t high = 0u;
        if (bits == 5u)
            high = octave_load_word(source + (base / 32u) * 4u);
        else if (bits == 6u)
            high = octave_load_word(source + (base / 16u) * 4u);
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            codes[e] = (low >> (4u * e)) & 15u;
            if (bits == 5u)
                codes[e] |= ((high >> ((base & 31u) + e)) & 1u) << 4u;
            else if (bits == 6u)
                codes[e] |= ((high >> (2u * ((base & 15u) + e))) & 3u) << 4u;
        }
        return;
    }
    const uint32_t low = octave_load_word(source + (base / 16u) * 4u);
    source += dimensions / 4u;
    const uint32_t high = octave_load_word(source + (base / 32u) * 4u);
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e)
        codes[e] = ((low >> (2u * ((base & 15u) + e))) & 3u) |
                   (((high >> ((base & 31u) + e)) & 1u) << 2u);
}

template <uint32_t kBits>
__device__ __forceinline__ int32_t octave_codebook_int8_prefill(uint32_t code) {
    const uint32_t bank = code >> 3u;
    const uint32_t selector = code & 7u;
    uint32_t value;
    if constexpr (kBits == 3u) {
        value = __builtin_amdgcn_perm(0x7f4f2d0eu, 0xf2d3b181u, selector);
    } else if constexpr (kBits == 4u) {
        const uint32_t v0 = __builtin_amdgcn_perm(0xfaeee1d4u, 0xc6b5a081u, selector);
        const uint32_t v1 = __builtin_amdgcn_perm(0x7f604b3au, 0x2c1f1206u, selector);
        value = bank == 0u ? v0 : v1;
    } else if constexpr (kBits == 5u) {
        const uint32_t v0 = __builtin_amdgcn_perm(0xd1cac3bau, 0xb1a69781u, selector);
        const uint32_t v1 = __builtin_amdgcn_perm(0xfdf8f3eeu, 0xe8e3ddd7u, selector);
        const uint32_t v2 = __builtin_amdgcn_perm(0x29231d18u, 0x120d0803u, selector);
        const uint32_t v3 = __builtin_amdgcn_perm(0x7f695a4fu, 0x463d362fu, selector);
        value = bank == 0u ? v0 : bank == 1u ? v1 : bank == 2u ? v2 : v3;
    } else {
        const uint32_t v0 = __builtin_amdgcn_perm(0xbcb7b2acu, 0xa59d9281u, selector);
        const uint32_t v1 = __builtin_amdgcn_perm(0xd8d5d2cfu, 0xcbc8c4c0u, selector);
        const uint32_t v2 = __builtin_amdgcn_perm(0xeceae8e5u, 0xe2e0dddau, selector);
        const uint32_t v3 = __builtin_amdgcn_perm(0xfffdfaf8u, 0xf6f3f1efu, selector);
        const uint32_t v4 = __builtin_amdgcn_perm(0x110f0d0au, 0x08060301u, selector);
        const uint32_t v5 = __builtin_amdgcn_perm(0x2623201eu, 0x1b181614u, selector);
        const uint32_t v6 = __builtin_amdgcn_perm(0x403c3835u, 0x312e2b28u, selector);
        const uint32_t v7 = __builtin_amdgcn_perm(0x7f6e635bu, 0x544e4944u, selector);
        value = bank == 0u ? v0 : bank == 1u ? v1 : bank == 2u ? v2
              : bank == 3u ? v3 : bank == 4u ? v4 : bank == 5u ? v5
              : bank == 6u ? v6 : v7;
    }
    return static_cast<int32_t>(static_cast<int8_t>(value & 0xffu));
}

template <uint32_t kBits>
__device__ __forceinline__ constexpr float octave_codebook_scale_prefill() {
    if constexpr (kBits == 3u) return 0.0169444544f;
    if constexpr (kBits == 4u) return 0.0215164533f;
    if constexpr (kBits == 5u) return 0.0256750590f;
    return 0.0294811393f;
}

__device__ __forceinline__ float octave_sign(uint32_t index, uint32_t salt) {
    uint32_t x = index * 2654435761u + salt * 40503u + 12345u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x & 1u) != 0u ? -1.0f : 1.0f;
}

__device__ __forceinline__ float octave_half_at(const uint8_t* source) {
    const uint16_t bits = static_cast<uint16_t>(source[0]) |
                          static_cast<uint16_t>(source[1]) << 8u;
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ void octave_wht8(float (&x)[8]) {
    #pragma unroll
    for (uint32_t half = 1u; half < 8u; half <<= 1u) {
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            if ((i & half) == 0u) {
                const float a = x[i], b = x[i + half];
                x[i] = a + b;
                x[i + half] = a - b;
            }
        }
    }
}

__device__ __forceinline__ void octave_inverse_wave(
    float (&x)[8], uint32_t lane, bool compact) {
    octave_wht8(x);
    const uint32_t limit = compact ? 16u : 4u;
    #pragma unroll
    for (uint32_t mask = 1u; mask <= 16u; mask <<= 1u) {
        if (mask > limit) break;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            const float other = __shfl_xor(x[i], mask, 32);
            x[i] = (lane & mask) != 0u ? other - x[i] : x[i] + other;
        }
    }
    const float norm = compact ? 0.0625f : 0.125f;
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) x[i] *= norm;
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
    const float* octave_levels,
    OctaveKVCacheFormat octave_format,
    void* out_base,
    uint32_t kv_block_start,
    float* state_max,
    float* state_sum) {
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
        if constexpr (kMode == 3u) {
            const uint32_t block_end = min(
                vis_max, kv_block_start + kOctavePrefillKvBlockTokens);
            s.page_count = block_end > kv_block_start
                ? (block_end - kv_block_start + kPageTokens - 1u) /
                      kPageTokens
                : 0u;
        } else {
            s.page_count = (vis_max + kPageTokens - 1u) / kPageTokens;
        }
        s.block_table = a.block_tables
            + static_cast<size_t>(valid != 0u ? slot : 0u) * a.block_table_stride;
    }
    detail::lds_barrier();
    if (s.invalid != 0u) return;
    if constexpr (kMode == 3u) {
        if (s.page_count == 0u) return;
    }

    const uint32_t rows = s.rows_here;
    const uint32_t visible_max = s.visible_max;
    const uint32_t page_count = s.page_count;
    const uint32_t* block_table = s.block_table;

    if (tid < kTileRows) {
        s.vis[tid] = (tid < rows) ? (a.row_positions[row0 + tid] + 1u) : 0u;
        const size_t state = static_cast<size_t>(row0 + tid) * q_heads + q_head;
        s.row_max[tid] = kMode == 3u && kv_block_start != 0u && tid < rows
            ? state_max[state] : -FLT_MAX;
        s.row_sum[tid] = kMode == 3u && kv_block_start != 0u && tid < rows
            ? state_sum[state] : 0.0f;
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
    for (uint32_t nt = 0u; nt < 4u; ++nt) {
        oacc[nt] = wf8_t{};
        if constexpr (kMode == 3u) {
            if (kv_block_start != 0u) {
                const uint32_t r_base = mt * 16u + 8u * hi;
                const uint32_t dim = nq * 64u + nt * 16u + lo;
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++e) {
                    const uint32_t r = r_base + e;
                    if (r < rows) {
                        const size_t o = static_cast<size_t>(row0 + r) *
                            a.output_row_stride +
                            static_cast<size_t>(q_head) * kHeadDim + dim;
                        oacc[nt][e] = a.output_dtype ==
                                PagedAttentionOutputDType::F32
                            ? reinterpret_cast<const float*>(out_base)[o]
                            : __bfloat162float(
                                  reinterpret_cast<const bf16*>(out_base)[o]);
                    }
                }
            }
        }
    }

    const uint64_t layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    const uint64_t head_off = static_cast<uint64_t>(kv_head) * kHeadDim;

    for (uint32_t page_index = 0u; page_index < page_count; ++page_index) {
        const uint32_t page = kMode == 3u ? page_index : block_table[page_index];
        if constexpr (kMode != 3u) {
            if (page >= a.num_pages) {
                if (tid == 0u) s.invalid = 1u;
                break;
            }
        }
        const uint32_t page_start = (kMode == 3u ? kv_block_start : 0u) +
            page_index * kPageTokens;
        uint32_t tokens_here = visible_max - page_start;
        if (tokens_here > kPageTokens) tokens_here = kPageTokens;
        const uint64_t page_base = kMode == 3u
            ? static_cast<uint64_t>(page_index) * kPageTokens *
                  a.elems_per_token
            : layer_base + static_cast<uint64_t>(page) * a.elems_per_page;
        const bf16* k_src = kMode == 2u ? nullptr : k_pool + page_base + head_off;
        const bf16* v_src = kMode == 2u ? nullptr : v_pool + page_base + head_off;
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
        if constexpr (kMode == 2u) {
            const uint32_t k_slot_bytes = octave_k_slot_bytes(octave_format);
            const uint32_t v_slot_bytes = octave_v_slot_bytes(octave_format);
            const uint64_t vectors_per_layer = static_cast<uint64_t>(a.num_pages) *
                kPageTokens * a.kv_heads;
            for (uint32_t token = warp; token < kPageTokens; token += kWaves) {
                const bool active = token < tokens_here;
                const uint64_t vector = static_cast<uint64_t>(a.layer) * vectors_per_layer +
                    (static_cast<uint64_t>(page) * kPageTokens + token) * a.kv_heads +
                    kv_head;
                const uint8_t* ks = k_code_pool + vector * k_slot_bytes;
                const uint8_t* vs = v_code_pool + vector * v_slot_bytes;
                float kval[8];
                float vval[8];
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++e) {
                    const uint32_t dimension = lane * 8u + e;
                    const uint32_t segment = octave_format.k_mode == OctaveKMode::Compact
                        ? 256u : 64u;
                    const uint32_t block = dimension / segment;
                    const uint32_t local = dimension & (segment - 1u);
                    const uint32_t bits = octave_format.k_mode == OctaveKMode::Compact || block != 0u
                        ? octave_format.k_nope_bits : octave_format.k_rope_bits;
                    const uint32_t code_offset = octave_format.k_mode == OctaveKMode::Compact || block == 0u
                        ? 0u
                        : octave_plane_bytes(64u, octave_format.k_rope_bits) +
                              (block - 1u) * octave_plane_bytes(64u, octave_format.k_nope_bits);
                    const uint32_t scale_count = octave_format.k_mode == OctaveKMode::Compact ? 1u : 4u;
                    const float kscale = active ? octave_half_at(
                        ks + k_slot_bytes - scale_count * 2u + block * 2u) : 0.0f;
                    const uint32_t kcode = active
                        ? octave_unpack_direct(ks + code_offset, local, segment, bits) : 0u;
                    const uint32_t vcode = active
                        ? octave_unpack_direct(vs, dimension, 256u, octave_format.v_bits) : 0u;
                    kval[e] = active ? octave_levels[bits * 64u + kcode] * kscale : 0.0f;
                    vval[e] = active ? octave_levels[octave_format.v_bits * 64u + vcode] *
                        octave_half_at(vs + v_slot_bytes - 2u) : 0.0f;
                }
                octave_inverse_wave(
                    kval, lane, octave_format.k_mode == OctaveKMode::Compact);
                octave_inverse_wave(vval, lane, true);
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++e) {
                    const uint32_t dimension = lane * 8u + e;
                    s.k_page[token][dimension] = f2bf(
                        kval[e] * octave_sign(dimension, 11u));
                    s.v_page[vbuf][token][dimension] = f2bf(
                        vval[e] * octave_sign(dimension, 29u));
                }
            }
        } else if (page_index + 1u < page_count) {
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
                paged_token_visible(a.block_mask, a.mask_stride, a.mask_blocks,
                                    a.mask_block_shift, mask_row, tok_a);
            const bool vb = tok_b < visr &&
                paged_token_visible(a.block_mask, a.mask_stride, a.mask_blocks,
                                    a.mask_block_shift, mask_row, tok_b);
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

    const bool final_block = kMode != 3u ||
        kv_block_start + kOctavePrefillKvBlockTokens >= visible_max;
    if constexpr (kMode == 3u) {
        if (tid < kTileRows && tid < rows) {
            const size_t state = static_cast<size_t>(row0 + tid) * q_heads + q_head;
            state_max[state] = s.row_max[tid];
            state_sum[state] = s.row_sum[tid];
        }
    }
    if (final_block && tid < kTileRows && tid < rows)
        s.row_sum[tid] = 1.0f / s.row_sum[tid];
    detail::lds_barrier();

    const uint32_t r_base = mt * 16u + 8u * hi;
    #pragma unroll
    for (uint32_t nt = 0u; nt < 4u; ++nt) {
        const uint32_t dim = nq * 64u + nt * 16u + lo;
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            const uint32_t r = r_base + e;
            if (r < rows) {
                const float v = final_block
                    ? oacc[nt][e] * s.row_sum[r] : oacc[nt][e];
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
    const float* octave_levels,
    OctaveKVCacheFormat octave_format,
    void* out_base,
    uint32_t kv_block_start = 0u,
    float* state_max = nullptr,
    float* state_sum = nullptr) {
    attention_paged_prefill_bf16_body<kMode>(
        a, k_pool, v_pool, k_code_pool, v_code_pool, k_scale_pool, v_scale_pool,
        blocks_per_head, octave_levels, octave_format, out_base,
        kv_block_start, state_max, state_sum);
}

} }  // namespace v_prefill_bf16

namespace { namespace v_prefill_octave {

using namespace v_prefill_bf16;

// One workgroup owns one KV head and a causal query-row tile.  The M dimension
// is flattened as [query-head-within-GQA, query-row], so the packed K/V page is
// decoded once and shared by all six Qwen3.8 query heads.  The default 32-row
// tile therefore performs 192 rows of WMMA work per unpack.  The former
// 10-row/64-flat-row kernel remains available for controlled A/B measurements.
// The WMMA fragment layout follows shelf's gfx12 paged-prefill kernel
// (lane & 15, lane >> 4 and eight accumulator elements).  The packed-page
// tiling is structurally based on vLLM's Apache-2.0
// csrc/attention/octave_prefill_rdna3.cu, but this path is independently
// implemented for gfx1201 wave32.
constexpr uint32_t kLegacyQueryRows = 10u;
constexpr uint32_t kLegacyFlatRows = 64u;
constexpr uint32_t kLegacyTileWaves = 8u;
constexpr uint32_t kLegacyPPitch = kPPitch;
constexpr uint32_t kWideQueryRows = 32u;
constexpr uint32_t kWideFlatRows = 192u;
constexpr uint32_t kWideTileWaves = 24u;
constexpr uint32_t kWidePPitch = 16u;
constexpr uint32_t kOctavePitch = kHeadDim;
constexpr uint32_t kOctaveTokens = 16u;
constexpr uint32_t kOctaveTokenHalves = kOctaveTokens / 16u;
constexpr uint32_t kOctaveSmaxPitch = kOctaveTokens + 1u;

using hf8_t = _Float16 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned short octave_f2h(float value) {
    return __half_as_ushort(__float2half(value));
}

__device__ __forceinline__ hf8_t octave_frag_nk_f16(
    const unsigned short* source, uint32_t pitch, uint32_t row_base,
    uint32_t kbase, uint32_t lane) {
    const unsigned short* p = source +
        static_cast<size_t>(row_base + (lane & 15u)) * pitch + kbase +
        4u * (lane >> 4u);
    union { unsigned short h[8]; hf8_t f; } result;
    #pragma unroll
    for (uint32_t e = 0u; e < 4u; ++e) {
        result.h[e] = p[e];
        result.h[4u + e] = p[8u + e];
    }
    return result.f;
}

__device__ __forceinline__ hf8_t octave_frag_kn_f16(
    const unsigned short* source, uint32_t pitch, uint32_t column_base,
    uint32_t kbase, uint32_t lane) {
    const uint32_t column = column_base + (lane & 15u);
    const uint32_t k0 = kbase + 4u * (lane >> 4u);
    union { unsigned short h[8]; hf8_t f; } result;
    #pragma unroll
    for (uint32_t e = 0u; e < 4u; ++e) {
        result.h[e] = source[static_cast<size_t>(k0 + e) * pitch + column];
        result.h[4u + e] =
            source[static_cast<size_t>(k0 + 8u + e) * pitch + column];
    }
    return result.f;
}

__device__ __forceinline__ wf8_t octave_wmma_f16(
    hf8_t a, hf8_t b, wf8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

template <uint32_t kTileFlatRows, uint32_t kTileWaves,
          uint32_t kTilePPitch>
struct OctavePrefillShared {
    static constexpr uint32_t kFlatTiles = kTileFlatRows / 16u;
    static constexpr uint32_t kDimParts = kTileWaves / kFlatTiles;
    unsigned short k_page[kOctaveTokens][kOctavePitch];
    unsigned short v_page[kOctaveTokens][kOctavePitch];
    float s_part[kFlatTiles][kDimParts][16][kOctaveSmaxPitch];
    unsigned short p_tile[kTileFlatRows][kTilePPitch];
    unsigned short p_tile_lo[kTileFlatRows][kTilePPitch];
    float k_scale[kOctaveTokens];
    float v_scale[kOctaveTokens];
    float row_max[kTileFlatRows];
    float row_sum[kTileFlatRows];
    float row_alpha[kTileFlatRows];
    uint32_t vis[kTileFlatRows];
    uint32_t rows_here;
    uint32_t flat_rows;
    uint32_t visible_max;
    uint32_t page_count;
    uint32_t invalid;
    const uint32_t* block_table;
};

__global__ __launch_bounds__(kThreads, 2)
void attention_paged_prefill_octave_rotate_q(
    PagedAttentionCommonArgs a, bool compact, void* output_base) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t vector = blockIdx.x * kWaves + wave;
    const uint32_t vectors = a.rows * a.q_heads;
    if (vector >= vectors) return;
    const uint32_t row = vector / a.q_heads;
    const uint32_t q_head = vector % a.q_heads;
    const bf16* source = reinterpret_cast<const bf16*>(a.q) +
        static_cast<size_t>(row) * a.q_row_stride +
        static_cast<size_t>(q_head) * kHeadDim + lane * 8u;
    float value[8];
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e) {
        const uint32_t dimension = lane * 8u + e;
        value[e] = __bfloat162float(source[e]) * octave_sign(dimension, 11u);
    }
    octave_inverse_wave(value, lane, compact);
    const size_t base = static_cast<size_t>(row) * a.output_row_stride +
        static_cast<size_t>(q_head) * kHeadDim + lane * 8u;
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e) {
        if (a.output_dtype == PagedAttentionOutputDType::F32)
            reinterpret_cast<float*>(output_base)[base + e] = value[e];
        else
            reinterpret_cast<bf16*>(output_base)[base + e] =
                __float2bfloat16(value[e]);
    }
}

template <uint32_t kTileQueryRows>
__device__ __forceinline__ hf8_t octave_flat_q_frag(
    const PagedAttentionCommonArgs& a, const void* output_base,
    uint32_t row0, uint32_t kv_head, uint32_t q_per_kv,
    uint32_t flat_base, uint32_t kbase, uint32_t lane) {
    const uint32_t flat = flat_base + (lane & 15u);
    const uint32_t local_q_head = flat / kTileQueryRows;
    const uint32_t local_row = flat % kTileQueryRows;
    const bool active = local_q_head < q_per_kv && row0 + local_row < a.rows;
    const size_t base = static_cast<size_t>(row0 + local_row) * a.output_row_stride +
        static_cast<size_t>(kv_head * q_per_kv + local_q_head) * kHeadDim +
        kbase + 4u * (lane >> 4u);
    union { unsigned short h[8]; hf8_t f; } result;
    #pragma unroll
    for (uint32_t e = 0u; e < 4u; ++e) {
        if (active && a.output_dtype == PagedAttentionOutputDType::F32) {
            result.h[e] = octave_f2h(reinterpret_cast<const float*>(output_base)[base + e]);
            result.h[4u + e] = octave_f2h(
                reinterpret_cast<const float*>(output_base)[base + 8u + e]);
        } else if (active) {
            result.h[e] = octave_f2h(__bfloat162float(
                reinterpret_cast<const bf16*>(output_base)[base + e]));
            result.h[4u + e] = octave_f2h(__bfloat162float(
                reinterpret_cast<const bf16*>(output_base)[base + 8u + e]));
        } else {
            result.h[e] = 0u;
            result.h[4u + e] = 0u;
        }
    }
    return result.f;
}

__global__ __launch_bounds__(32)
void attention_paged_prefill_octave_unpack_block(
    const PagedAttentionCommonArgs a,
    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const float* octave_levels,
    OctaveKVCacheFormat octave_format,
    uint32_t block_start,
    bf16* k_scratch,
    bf16* v_scratch) {
    const uint32_t lane = threadIdx.x;
    const uint32_t token = blockIdx.x / a.kv_heads;
    const uint32_t kv_head = blockIdx.x % a.kv_heads;
    const uint32_t position = block_start + token;
    const uint32_t slot = a.row_sequence_slots[0];
    const uint32_t visible = a.row_positions[a.rows - 1u] + 1u;
    bool active = position < visible && slot < a.max_sequences;
    uint32_t page = 0u;
    uint32_t page_offset = 0u;
    if (active) {
        const uint32_t logical_page = position / kPageTokens;
        page_offset = position % kPageTokens;
        if (logical_page >= a.block_table_stride) {
            active = false;
        } else {
            page = a.block_tables[static_cast<size_t>(slot) *
                a.block_table_stride + logical_page];
            active = page < a.num_pages;
        }
    }

    const uint32_t k_slot_bytes = octave_k_slot_bytes(octave_format);
    const uint32_t v_slot_bytes = octave_v_slot_bytes(octave_format);
    const uint64_t vectors_per_layer = static_cast<uint64_t>(a.num_pages) *
        kPageTokens * a.kv_heads;
    const uint64_t vector = static_cast<uint64_t>(a.layer) *
        vectors_per_layer +
        (static_cast<uint64_t>(page) * kPageTokens + page_offset) *
            a.kv_heads + kv_head;
    const uint8_t* ks = k_code_pool + vector * k_slot_bytes;
    const uint8_t* vs = v_code_pool + vector * v_slot_bytes;
    const uint32_t dimension_base = lane * 8u;
    const uint32_t segment = octave_format.k_mode == OctaveKMode::Compact
        ? 256u : 64u;
    const uint32_t kblock = dimension_base / segment;
    const uint32_t local_base = dimension_base & (segment - 1u);
    const uint32_t bits = octave_format.k_mode == OctaveKMode::Compact ||
            kblock != 0u
        ? octave_format.k_nope_bits : octave_format.k_rope_bits;
    const uint32_t code_offset = octave_format.k_mode ==
            OctaveKMode::Compact || kblock == 0u
        ? 0u
        : octave_plane_bytes(64u, octave_format.k_rope_bits) +
              (kblock - 1u) * octave_plane_bytes(
                  64u, octave_format.k_nope_bits);
    const uint32_t scale_count = octave_format.k_mode ==
            OctaveKMode::Compact
        ? 1u : 4u;
    const float kscale = active
        ? octave_half_at(ks + k_slot_bytes - scale_count * 2u +
              kblock * 2u)
        : 0.0f;
    const float vscale = active
        ? octave_half_at(vs + v_slot_bytes - 2u) : 0.0f;
    uint32_t kcodes[8];
    uint32_t vcodes[8];
    if (active) {
        octave_unpack8_direct(
            ks + code_offset, local_base, segment, bits, kcodes);
        octave_unpack8_direct(
            vs, dimension_base, 256u, octave_format.v_bits, vcodes);
    } else {
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            kcodes[e] = 0u;
            vcodes[e] = 0u;
        }
    }
    float kval[8];
    float vval[8];
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e) {
        kval[e] = active
            ? octave_levels[bits * 64u + kcodes[e]] * kscale : 0.0f;
        vval[e] = active
            ? octave_levels[octave_format.v_bits * 64u + vcodes[e]] * vscale
            : 0.0f;
    }
    octave_inverse_wave(
        kval, lane, octave_format.k_mode == OctaveKMode::Compact);
    octave_inverse_wave(vval, lane, true);
    const size_t output =
        (static_cast<size_t>(token) * a.kv_heads + kv_head) * kHeadDim +
        dimension_base;
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e) {
        const uint32_t dimension = dimension_base + e;
        k_scratch[output + e] = __float2bfloat16(
            kval[e] * octave_sign(dimension, 11u));
        v_scratch[output + e] = __float2bfloat16(
            vval[e] * octave_sign(dimension, 29u));
    }
}

template <uint32_t kFixedBits, uint32_t kTileQueryRows,
          uint32_t kTileFlatRows, uint32_t kTileWaves,
          uint32_t kTilePPitch, uint32_t kMinBlocks>
__global__ __launch_bounds__(kTileWaves * 32u, kMinBlocks)
void attention_paged_prefill_octave_rotated(
    const PagedAttentionCommonArgs a,
    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const float* octave_levels,
    OctaveKVCacheFormat octave_format,
    void* out_base) {
    static_assert(kTileFlatRows % 16u == 0u);
    static_assert(kTileWaves % (kTileFlatRows / 16u) == 0u);
    static_assert(kTilePPitch >= 16u);
    using Shared = OctavePrefillShared<
        kTileFlatRows, kTileWaves, kTilePPitch>;
    static_assert(sizeof(Shared) <= 64u * 1024u);
    constexpr uint32_t kFlatTiles = kTileFlatRows / 16u;
    constexpr uint32_t kDimParts = kTileWaves / kFlatTiles;
    constexpr uint32_t kDimsPerPart = kHeadDim / kDimParts;
    constexpr uint32_t kKSteps = kDimsPerPart / 16u;
    constexpr uint32_t kTileThreads = kTileWaves * 32u;
    __shared__ Shared s;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t hi = lane >> 4u;
    const uint32_t lo = lane & 15u;
    const uint32_t q_per_kv = a.q_heads / a.kv_heads;
    const uint32_t row0 = blockIdx.x * kTileQueryRows;
    const uint32_t kv_head = blockIdx.y;
    if (row0 >= a.rows) return;

    if (tid == 0u) {
        s.rows_here = min(kTileQueryRows, a.rows - row0);
        s.flat_rows = q_per_kv * kTileQueryRows;
        s.invalid = s.flat_rows > kTileFlatRows ? 1u : 0u;
        const uint32_t slot = a.row_sequence_slots[row0];
        if (slot >= a.max_sequences) s.invalid = 1u;
        s.visible_max = 0u;
        s.page_count = 0u;
        s.block_table = a.block_tables;
        if (s.invalid == 0u) {
            s.visible_max = a.row_positions[row0 + s.rows_here - 1u] + 1u;
            const uint64_t capacity =
                static_cast<uint64_t>(a.block_table_stride) * a.page_tokens;
            if (static_cast<uint64_t>(s.visible_max) > capacity) {
                s.invalid = 1u;
            } else {
                s.page_count = (s.visible_max + kPageTokens - 1u) / kPageTokens;
                s.block_table = a.block_tables +
                    static_cast<size_t>(slot) * a.block_table_stride;
            }
        }
    }
    detail::lds_barrier();
    if (s.invalid != 0u) return;
    const uint32_t rows = s.rows_here;
    const uint32_t flat_rows = s.flat_rows;
    if (tid < kTileFlatRows) {
        const uint32_t local_row = tid % kTileQueryRows;
        const bool active = tid < flat_rows && local_row < rows;
        s.vis[tid] = active ? a.row_positions[row0 + local_row] + 1u : 0u;
        s.row_max[tid] = -FLT_MAX;
        s.row_sum[tid] = 0.0f;
        s.row_alpha[tid] = 0.0f;
    }

    const uint32_t mt = wave % kFlatTiles;
    const uint32_t kq = wave / kFlatTiles;
    hf8_t q_frag[kKSteps];
    #pragma unroll
    for (uint32_t ks = 0u; ks < kKSteps; ++ks)
        q_frag[ks] = octave_flat_q_frag<kTileQueryRows>(
            a, out_base, row0, kv_head, q_per_kv, mt * 16u,
            (kq * kKSteps + ks) * 16u, lane);

    wf8_t oacc[kKSteps];
    #pragma unroll
    for (uint32_t nt = 0u; nt < kKSteps; ++nt) oacc[nt] = wf8_t{};

    const uint32_t k_slot_bytes = octave_k_slot_bytes(octave_format);
    const uint32_t v_slot_bytes = octave_v_slot_bytes(octave_format);
    const uint64_t vectors_per_layer = static_cast<uint64_t>(a.num_pages) *
        kPageTokens * a.kv_heads;

    for (uint32_t tile_start = 0u; tile_start < s.visible_max;
         tile_start += kOctaveTokens) {
        const uint32_t tokens_here =
            min(kOctaveTokens, s.visible_max - tile_start);

        // Decode codebook values directly in rotated space.  No per-token
        // inverse WHT or sign restoration occurs here.
        for (uint32_t token = wave; token < kOctaveTokens;
             token += kTileWaves) {
            bool active = token < tokens_here;
            const uint32_t position = tile_start + token;
            const uint32_t logical_page = position / kPageTokens;
            const uint32_t page_offset = position % kPageTokens;
            uint32_t page = active ? s.block_table[logical_page] : 0u;
            if (active && page >= a.num_pages) {
                active = false;
                if (lane == 0u) s.invalid = 1u;
                page = 0u;
            }
            const uint64_t vector = static_cast<uint64_t>(a.layer) * vectors_per_layer +
                (static_cast<uint64_t>(page) * kPageTokens + page_offset) * a.kv_heads +
                kv_head;
            const uint8_t* ks = k_code_pool + vector * k_slot_bytes;
            const uint8_t* vs = v_code_pool + vector * v_slot_bytes;
            const float vscale = active
                ? octave_half_at(vs + v_slot_bytes - 2u) : 0.0f;
            const uint32_t dimension_base = lane * 8u;
            const uint32_t segment = octave_format.k_mode == OctaveKMode::Compact
                ? 256u : 64u;
            const uint32_t block = dimension_base / segment;
            const uint32_t local_base = dimension_base & (segment - 1u);
            const uint32_t bits = octave_format.k_mode == OctaveKMode::Compact || block != 0u
                ? octave_format.k_nope_bits : octave_format.k_rope_bits;
            const uint32_t code_offset = octave_format.k_mode == OctaveKMode::Compact || block == 0u
                ? 0u
                : octave_plane_bytes(64u, octave_format.k_rope_bits) +
                      (block - 1u) * octave_plane_bytes(64u, octave_format.k_nope_bits);
            const uint32_t scale_count = octave_format.k_mode == OctaveKMode::Compact ? 1u : 4u;
            const float kscale = active ? octave_half_at(
                ks + k_slot_bytes - scale_count * 2u + block * 2u) : 0.0f;
            if constexpr (kFixedBits != 0u) {
                if (lane == 0u) {
                    s.k_scale[token] = kscale;
                    s.v_scale[token] = vscale;
                }
            }
            uint32_t kcodes[8];
            uint32_t vcodes[8];
            if (active) {
                octave_unpack8_direct(
                    ks + code_offset, local_base, segment, bits, kcodes);
                octave_unpack8_direct(
                    vs, dimension_base, 256u, octave_format.v_bits, vcodes);
            } else {
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++e) {
                    kcodes[e] = 0u;
                    vcodes[e] = 0u;
                }
            }
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e) {
                const uint32_t dimension = dimension_base + e;
                if constexpr (kFixedBits != 0u) {
                    s.k_page[token][dimension] = octave_f2h(active
                        ? static_cast<float>(octave_codebook_int8_prefill<kFixedBits>(
                              kcodes[e]))
                        : 0.0f);
                    s.v_page[token][dimension] = octave_f2h(active
                        ? static_cast<float>(octave_codebook_int8_prefill<kFixedBits>(
                              vcodes[e]))
                        : 0.0f);
                } else {
                    s.k_page[token][dimension] = octave_f2h(active
                        ? octave_levels[bits * 64u + kcodes[e]] * kscale : 0.0f);
                    s.v_page[token][dimension] = octave_f2h(active
                        ? octave_levels[octave_format.v_bits * 64u + vcodes[e]] * vscale : 0.0f);
                }
            }
        }
        detail::lds_barrier();

        #pragma unroll
        for (uint32_t token_half = 0u; token_half < kOctaveTokenHalves;
             ++token_half) {
            wf8_t score_acc = wf8_t{};
            #pragma unroll
            for (uint32_t ks = 0u; ks < kKSteps; ++ks) {
                const uint32_t k0 = (kq * kKSteps + ks) * 16u;
                score_acc = octave_wmma_f16(
                    q_frag[ks],
                    octave_frag_nk_f16(&s.k_page[0][0], kOctavePitch,
                                       token_half * 16u, k0, lane),
                    score_acc);
            }
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e)
                s.s_part[mt][kq][8u * hi + e][token_half * 16u + lo] =
                    score_acc[e];
            detail::lds_barrier();

            {
                const uint32_t tg = tid & 7u;
                const uint32_t pa_index = tg;
                const uint32_t pb_index = tg + 8u;
                const uint32_t ta = token_half * 16u + pa_index;
                const uint32_t tb = token_half * 16u + pb_index;
                for (uint32_t flat = tid >> 3u; flat < kTileFlatRows;
                     flat += kTileThreads / 8u) {
                const uint32_t pmt = flat >> 4u;
                const uint32_t rr = flat & 15u;
                float sa = 0.0f;
                float sb = 0.0f;
                #pragma unroll
                for (uint32_t part = 0u; part < kDimParts; ++part) {
                    sa += s.s_part[pmt][part][rr][ta];
                    sb += s.s_part[pmt][part][rr][tb];
                }
                const uint32_t local_row = flat % kTileQueryRows;
                const uint32_t mask_row = row0 + local_row;
                const uint32_t vis = s.vis[flat];
                const uint32_t tok_a = tile_start + ta;
                const uint32_t tok_b = tile_start + tb;
                const bool va = tok_a < vis &&
                    paged_token_visible(a.block_mask, a.mask_stride, a.mask_blocks,
                                        a.mask_block_shift, mask_row, tok_a);
                const bool vb = tok_b < vis &&
                    paged_token_visible(a.block_mask, a.mask_stride, a.mask_blocks,
                                        a.mask_block_shift, mask_row, tok_b);
                if constexpr (kFixedBits != 0u) {
                    constexpr float codebook_scale =
                        octave_codebook_scale_prefill<kFixedBits>();
                    sa *= s.k_scale[ta] * codebook_scale;
                    sb *= s.k_scale[tb] * codebook_scale;
                }
                sa = va ? sa * a.scale : -FLT_MAX;
                sb = vb ? sb * a.scale : -FLT_MAX;
                const float tile_max = group_max8(fmaxf(sa, sb));
                const float old_max = s.row_max[flat];
                const float next_max = fmaxf(old_max, tile_max);
                const float alpha = __expf(old_max - next_max);
                const float pa = va ? __expf(sa - next_max) : 0.0f;
                const float pb = vb ? __expf(sb - next_max) : 0.0f;
                const float tile_sum = group_sum8(pa + pb);
                float pva = pa;
                float pvb = pb;
                if constexpr (kFixedBits != 0u) {
                    constexpr float codebook_scale =
                        octave_codebook_scale_prefill<kFixedBits>();
                    pva *= s.v_scale[ta] * codebook_scale;
                    pvb *= s.v_scale[tb] * codebook_scale;
                }
                const unsigned short pa_hi = octave_f2h(pva);
                const unsigned short pb_hi = octave_f2h(pvb);
                s.p_tile[flat][pa_index] = pa_hi;
                s.p_tile[flat][pb_index] = pb_hi;
                s.p_tile_lo[flat][pa_index] = octave_f2h(
                    pva - __half2float(__ushort_as_half(pa_hi)));
                s.p_tile_lo[flat][pb_index] = octave_f2h(
                    pvb - __half2float(__ushort_as_half(pb_hi)));
                if (tg == 0u) {
                    s.row_max[flat] = next_max;
                    s.row_sum[flat] = s.row_sum[flat] * alpha + tile_sum;
                    s.row_alpha[flat] = alpha;
                }
            }
            }
            detail::lds_barrier();

            const hf8_t probability = octave_frag_nk_f16(
                &s.p_tile[0][0], kTilePPitch, mt * 16u, 0u, lane);
            const hf8_t probability_lo = octave_frag_nk_f16(
                &s.p_tile_lo[0][0], kTilePPitch, mt * 16u, 0u, lane);
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e) {
                const float alpha = s.row_alpha[mt * 16u + 8u * hi + e];
                #pragma unroll
                for (uint32_t nt = 0u; nt < kKSteps; ++nt)
                    oacc[nt][e] *= alpha;
            }
            #pragma unroll
            for (uint32_t nt = 0u; nt < kKSteps; ++nt) {
                const hf8_t value = octave_frag_kn_f16(
                    &s.v_page[0][0], kOctavePitch,
                    kq * kDimsPerPart + nt * 16u, token_half * 16u, lane);
                oacc[nt] = octave_wmma_f16(probability, value, oacc[nt]);
                oacc[nt] = octave_wmma_f16(probability_lo, value, oacc[nt]);
            }
            detail::lds_barrier();
        }
    }

    if (s.invalid != 0u) return;
    if (tid < kTileFlatRows)
        s.row_sum[tid] = s.row_sum[tid] > 0.0f ? 1.0f / s.row_sum[tid] : 0.0f;
    detail::lds_barrier();

    const uint32_t flat_base = mt * 16u + 8u * hi;
    #pragma unroll
    for (uint32_t nt = 0u; nt < kKSteps; ++nt) {
        const uint32_t dimension = kq * kDimsPerPart + nt * 16u + lo;
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            const uint32_t flat = flat_base + e;
            const uint32_t local_q_head = flat / kTileQueryRows;
            const uint32_t local_row = flat % kTileQueryRows;
            if (flat < flat_rows && local_row < rows) {
                const float rotated = oacc[nt][e] * s.row_sum[flat];
                const size_t output = static_cast<size_t>(row0 + local_row) *
                    a.output_row_stride +
                    static_cast<size_t>(kv_head * q_per_kv + local_q_head) *
                        kHeadDim + dimension;
                if (a.output_dtype == PagedAttentionOutputDType::F32)
                    reinterpret_cast<float*>(out_base)[output] = rotated;
                else
                    reinterpret_cast<bf16*>(out_base)[output] =
                        __float2bfloat16(rotated);
            }
        }
    }
}

__global__ __launch_bounds__(kThreads, 2)
void attention_paged_prefill_octave_inverse_output(
    PagedAttentionCommonArgs a, void* output_base) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t vector = blockIdx.x * kWaves + wave;
    const uint32_t vectors = a.rows * a.q_heads;
    if (vector >= vectors) return;
    const uint32_t row = vector / a.q_heads;
    const uint32_t q_head = vector % a.q_heads;
    const size_t base = static_cast<size_t>(row) * a.output_row_stride +
        static_cast<size_t>(q_head) * kHeadDim;
    float value[8];
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e) {
        const size_t index = base + lane * 8u + e;
        value[e] = a.output_dtype == PagedAttentionOutputDType::F32
            ? reinterpret_cast<const float*>(output_base)[index]
            : __bfloat162float(reinterpret_cast<const bf16*>(output_base)[index]);
    }
    octave_inverse_wave(value, lane, true);
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e) {
        const uint32_t dimension = lane * 8u + e;
        const size_t index = base + dimension;
        const float restored = value[e] * octave_sign(dimension, 29u);
        if (a.output_dtype == PagedAttentionOutputDType::F32)
            reinterpret_cast<float*>(output_base)[index] = restored;
        else
            reinterpret_cast<bf16*>(output_base)[index] =
                __float2bfloat16(restored);
    }
}

template <uint32_t kTileQueryRows, uint32_t kTileFlatRows,
          uint32_t kTileWaves, uint32_t kTilePPitch,
          uint32_t kMinBlocks>
hipError_t launch_attention_paged_prefill_octave_tile(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool,
    const uint8_t* v_pool,
    const float* levels,
    OctaveKVCacheFormat format,
    hipStream_t stream) {
    const uint32_t tiles =
        (common.rows + kTileQueryRows - 1u) / kTileQueryRows;
    const dim3 grid(tiles, common.kv_heads);
    constexpr uint32_t kTileThreads = kTileWaves * 32u;
    if (format.k_mode == OctaveKMode::Compact &&
        format.k_nope_bits == format.v_bits && format.v_bits == 3u) {
        attention_paged_prefill_octave_rotated<
            3u, kTileQueryRows, kTileFlatRows, kTileWaves,
            kTilePPitch, kMinBlocks><<<grid, kTileThreads, 0, stream>>>(
                common, k_pool, v_pool, levels, format, common.output);
    } else if (format.k_mode == OctaveKMode::Compact &&
               format.k_nope_bits == format.v_bits && format.v_bits == 4u) {
        attention_paged_prefill_octave_rotated<
            4u, kTileQueryRows, kTileFlatRows, kTileWaves,
            kTilePPitch, kMinBlocks><<<grid, kTileThreads, 0, stream>>>(
                common, k_pool, v_pool, levels, format, common.output);
    } else if (format.k_mode == OctaveKMode::Compact &&
               format.k_nope_bits == format.v_bits && format.v_bits == 5u) {
        attention_paged_prefill_octave_rotated<
            5u, kTileQueryRows, kTileFlatRows, kTileWaves,
            kTilePPitch, kMinBlocks><<<grid, kTileThreads, 0, stream>>>(
                common, k_pool, v_pool, levels, format, common.output);
    } else if (format.k_mode == OctaveKMode::Compact &&
               format.k_nope_bits == format.v_bits && format.v_bits == 6u) {
        attention_paged_prefill_octave_rotated<
            6u, kTileQueryRows, kTileFlatRows, kTileWaves,
            kTilePPitch, kMinBlocks><<<grid, kTileThreads, 0, stream>>>(
                common, k_pool, v_pool, levels, format, common.output);
    } else {
        attention_paged_prefill_octave_rotated<
            0u, kTileQueryRows, kTileFlatRows, kTileWaves,
            kTilePPitch, kMinBlocks><<<grid, kTileThreads, 0, stream>>>(
                common, k_pool, v_pool, levels, format, common.output);
    }
    return hipGetLastError();
}

} }  // namespace v_prefill_octave

extern "C" __global__ __launch_bounds__(kPagedAttentionPrefillBlock)
void shelf_attention_paged_prefill_bf16(
    ::shelf::kernel::PagedAttentionBf16Args args) {
    using namespace v_prefill_bf16;
    attention_paged_prefill_bf16_body<0u>(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool),
        nullptr, nullptr, nullptr, nullptr, 0u,
        nullptr, {},
        args.common.output, 0u, nullptr, nullptr);
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
        nullptr, {},
        common.output);
    return hipGetLastError();
}

hipError_t launch_attention_paged_prefill_octave(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool,
    const uint8_t* v_pool,
    const float* levels,
    OctaveKVCacheFormat format,
    hipStream_t stream) {
    using namespace v_prefill_bf16;
    if (common.rows == 0u || common.head_dim != kHeadDim ||
        common.page_tokens != kPageTokens || common.q_heads == 0u ||
        common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u ||
        common.block_table_stride == 0u || common.num_pages == 0u ||
        common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        levels == nullptr || !octave_format_valid(format))
        return hipErrorInvalidValue;
    const char* legacy_value = std::getenv("SHELF_OCTAVE_PREFILL_LEGACY");
    const bool legacy = legacy_value != nullptr && legacy_value[0] == '1' &&
                        legacy_value[1] == '\0';
    if (common.q_heads / common.kv_heads >
            v_prefill_octave::kWideFlatRows /
                v_prefill_octave::kWideQueryRows) {
        const uint32_t tiles = (common.rows + kTileRows - 1u) / kTileRows;
        attention_paged_prefill_bf16_impl<2u>
            <<<dim3(tiles, common.q_heads), kThreads, 0, stream>>>(
                common, nullptr, nullptr, k_pool, v_pool, nullptr, nullptr, 0u,
                levels, format, common.output);
    } else {
        const uint64_t vectors = static_cast<uint64_t>(common.rows) * common.q_heads;
        const uint32_t transform_blocks = static_cast<uint32_t>(
            (vectors + v_prefill_bf16::kWaves - 1u) / v_prefill_bf16::kWaves);
        v_prefill_octave::attention_paged_prefill_octave_rotate_q
            <<<transform_blocks, kThreads, 0, stream>>>(
                common, format.k_mode == OctaveKMode::Compact, common.output);
        hipError_t tile_error = hipSuccess;
        if (legacy) {
            tile_error =
                v_prefill_octave::launch_attention_paged_prefill_octave_tile<
                v_prefill_octave::kLegacyQueryRows,
                v_prefill_octave::kLegacyFlatRows,
                v_prefill_octave::kLegacyTileWaves,
                v_prefill_octave::kLegacyPPitch, 2u>(
                    common, k_pool, v_pool, levels, format, stream);
        } else {
            tile_error =
                v_prefill_octave::launch_attention_paged_prefill_octave_tile<
                v_prefill_octave::kWideQueryRows,
                v_prefill_octave::kWideFlatRows,
                v_prefill_octave::kWideTileWaves,
                v_prefill_octave::kWidePPitch, 1u>(
                    common, k_pool, v_pool, levels, format, stream);
        }
        if (tile_error != hipSuccess) return tile_error;
        v_prefill_octave::attention_paged_prefill_octave_inverse_output
            <<<transform_blocks, kThreads, 0, stream>>>(common, common.output);
    }
    return hipGetLastError();
}

hipError_t launch_attention_paged_prefill_octave_splitkv(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool,
    const uint8_t* v_pool,
    const float* levels,
    OctaveKVCacheFormat format,
    void* scratch,
    uint64_t scratch_bytes,
    uint32_t max_visible_tokens,
    hipStream_t stream) {
    const char* legacy_value = std::getenv("SHELF_OCTAVE_PREFILL_LEGACY");
    const bool legacy = legacy_value != nullptr && legacy_value[0] == '1' &&
                        legacy_value[1] == '\0';
    if (legacy)
        return launch_attention_paged_prefill_octave(
            common, k_pool, v_pool, levels, format, stream);
    if (common.rows == 0u || common.head_dim != v_prefill_bf16::kHeadDim ||
        common.page_tokens != v_prefill_bf16::kPageTokens ||
        common.q_heads == 0u || common.kv_heads == 0u ||
        common.q_heads % common.kv_heads != 0u ||
        common.block_table_stride == 0u || common.num_pages == 0u ||
        common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        levels == nullptr || scratch == nullptr || max_visible_tokens == 0u ||
        !octave_format_valid(format))
        return hipErrorInvalidValue;
    const uint64_t required = octave_prefill_splitkv_scratch_bytes(
        common.rows, common.q_heads, common.kv_heads, common.head_dim);
    if (scratch_bytes < required) return hipErrorOutOfMemory;

    const uint64_t decoded_elems =
        static_cast<uint64_t>(kOctavePrefillKvBlockTokens) *
        common.kv_heads * common.head_dim;
    auto* k_scratch = reinterpret_cast<v_prefill_bf16::bf16*>(scratch);
    auto* v_scratch = k_scratch + decoded_elems;
    auto* state_max = reinterpret_cast<float*>(v_scratch + decoded_elems);
    auto* state_sum = state_max +
        static_cast<uint64_t>(common.rows) * common.q_heads;
    const uint32_t tiles =
        (common.rows + v_prefill_bf16::kTileRows - 1u) /
        v_prefill_bf16::kTileRows;
    const dim3 attention_grid(tiles, common.q_heads);
    const uint32_t unpack_grid =
        kOctavePrefillKvBlockTokens * common.kv_heads;

    for (uint32_t block_start = 0u; block_start < max_visible_tokens;
         block_start += kOctavePrefillKvBlockTokens) {
        v_prefill_octave::attention_paged_prefill_octave_unpack_block
            <<<unpack_grid, 32u, 0, stream>>>(
                common, k_pool, v_pool, levels, format, block_start,
                k_scratch, v_scratch);
        v_prefill_bf16::attention_paged_prefill_bf16_impl<3u>
            <<<attention_grid, v_prefill_bf16::kThreads, 0, stream>>>(
                common, k_scratch, v_scratch, nullptr, nullptr, nullptr,
                nullptr, 0u, nullptr, {}, common.output, block_start,
                state_max, state_sum);
    }
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
