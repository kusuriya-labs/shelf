#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/kernels/optimized/detail/attention_paged_common.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdlib>

namespace shelf::kernel {
namespace {

__device__ __forceinline__ float sign_of(uint32_t index, uint32_t salt) {
    uint32_t x = index * 2654435761u + salt * 40503u + 12345u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x & 1u) != 0u ? -1.0f : 1.0f;
}

__device__ __forceinline__ uint32_t load_word(const uint8_t* source) {
    uint32_t word;
    __builtin_memcpy(&word, source, sizeof(word));
    return word;
}

__device__ __forceinline__ uint32_t unpack_plane(const uint8_t* source,
                                                 uint32_t index,
                                                 uint32_t plane_bits) {
    const uint32_t fields = 32u / plane_bits;
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;
    const uint32_t leader = lane - lane % fields;
    uint32_t word = lane == leader
        ? load_word(source + (index / fields) * 4u) : 0u;
    word = __shfl(word, leader);
    return (word >> ((index % fields) * plane_bits)) & ((1u << plane_bits) - 1u);
}

__device__ __forceinline__ uint32_t unpack_code(const uint8_t* source,
                                                uint32_t index,
                                                uint32_t dimensions,
                                                uint32_t bits) {
    if (bits >= 4u) {
        uint32_t code = unpack_plane(source, index, 4u);
        source += dimensions / 2u;
        if (bits == 5u) code |= unpack_plane(source, index, 1u) << 4u;
        if (bits == 6u) code |= unpack_plane(source, index, 2u) << 4u;
        return code;
    }
    uint32_t code = unpack_plane(source, index, 2u);
    source += dimensions / 4u;
    return code | (unpack_plane(source, index, 1u) << 2u);
}

__device__ __forceinline__ void wht(float& value, float* shared,
                                    uint32_t segment) {
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t half = 1u; half < segment; half <<= 1u) {
        shared[tid] = value;
        __syncthreads();
        const float other = shared[tid ^ half];
        __syncthreads();
        value = (tid & half) != 0u ? other - value : value + other;
    }
    value *= rsqrtf(static_cast<float>(segment));
}

__global__ __launch_bounds__(256)
void dequant(const uint8_t* k_pool, const uint8_t* v_pool, const float* levels,
             bf16_t* k_scratch, bf16_t* v_scratch, OctaveKVCacheFormat format,
             uint32_t layer, uint32_t num_pages, uint32_t page_tokens,
             uint32_t kv_heads) {
    __shared__ float shared[256];
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint64_t vectors_per_layer = static_cast<uint64_t>(num_pages) * page_tokens * kv_heads;
    const uint64_t source_vector = static_cast<uint64_t>(layer) * vectors_per_layer + blockIdx.x;
    const uint32_t k_slot_bytes = octave_k_slot_bytes(format);
    const uint32_t v_slot_bytes = octave_v_slot_bytes(format);
    const uint8_t* k_source = k_pool + source_vector * k_slot_bytes;
    const uint8_t* v_source = v_pool + source_vector * v_slot_bytes;
    const uint64_t output = static_cast<uint64_t>(blockIdx.x) * 256u + tid;

    const uint32_t k_segment = format.k_mode == OctaveKMode::Compact ? 256u : 64u;
    const uint32_t block = tid / k_segment;
    const uint32_t local = tid & (k_segment - 1u);
    const uint32_t k_bits = format.k_mode == OctaveKMode::Compact || block != 0u
        ? format.k_nope_bits : format.k_rope_bits;
    const uint32_t k_code_offset = format.k_mode == OctaveKMode::Compact || block == 0u
        ? 0u
        : octave_plane_bytes(64u, format.k_rope_bits) +
              (block - 1u) * octave_plane_bytes(64u, format.k_nope_bits);
    const uint32_t k_code = unpack_code(k_source + k_code_offset, local,
                                        k_segment, k_bits);
    const __half* k_scales = reinterpret_cast<const __half*>(
        k_source + k_slot_bytes - (format.k_mode == OctaveKMode::Compact ? 2u : 8u));
    float k_value = levels[k_bits * 64u + k_code] * __half2float(k_scales[block]);
    wht(k_value, shared, k_segment);
    k_value *= sign_of(tid, 11u);
    k_scratch[output] = bf16_t{__bfloat16_as_ushort(__float2bfloat16(k_value))};
    __syncthreads();

    const uint32_t v_code = unpack_code(v_source, tid, 256u, format.v_bits);
    const __half* v_scale = reinterpret_cast<const __half*>(v_source + v_slot_bytes - 2u);
    float v_value = levels[format.v_bits * 64u + v_code] * __half2float(*v_scale);
    wht(v_value, shared, 256u);
    v_value *= sign_of(tid, 29u);
    v_scratch[output] = bf16_t{__bfloat16_as_ushort(__float2bfloat16(v_value))};
}

constexpr uint32_t kOctaveQPerKv = 6u;
constexpr uint32_t kOctaveTokenLanes = 4u;
constexpr uint32_t kOctaveThreads = 256u * kOctaveTokenLanes;
constexpr uint32_t kOctaveMaxSplits = 64u;

__device__ __forceinline__ void wht_query(float& value, float* shared,
                                          uint32_t segment) {
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t half = 1u; half < segment; half <<= 1u) {
        if (tid < 256u) shared[tid] = value;
        __syncthreads();
        const float other = tid < 256u ? shared[tid ^ half] : 0.0f;
        __syncthreads();
        if (tid < 256u)
            value = (tid & half) != 0u ? other - value : value + other;
    }
    if (tid < 256u) value *= rsqrtf(static_cast<float>(segment));
}

__device__ __forceinline__ float load_scale(const uint8_t* source) {
    const uint16_t bits = static_cast<uint16_t>(source[0]) |
                          static_cast<uint16_t>(source[1]) << 8u;
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float load_k_rotated(
    const uint8_t* source, uint32_t dimension, const float* levels,
    OctaveKVCacheFormat format) {
    const uint32_t segment = format.k_mode == OctaveKMode::Compact ? 256u : 64u;
    const uint32_t block = dimension / segment;
    const uint32_t local = dimension & (segment - 1u);
    const uint32_t bits = format.k_mode == OctaveKMode::Compact || block != 0u
        ? format.k_nope_bits : format.k_rope_bits;
    const uint32_t code_offset = format.k_mode == OctaveKMode::Compact || block == 0u
        ? 0u
        : octave_plane_bytes(64u, format.k_rope_bits) +
              (block - 1u) * octave_plane_bytes(64u, format.k_nope_bits);
    const uint32_t code = unpack_code(source + code_offset, local, segment, bits);
    const uint32_t scale_offset = octave_k_slot_bytes(format) -
        (format.k_mode == OctaveKMode::Compact ? 2u : 8u) + block * 2u;
    return levels[bits * 64u + code] * load_scale(source + scale_offset);
}

__device__ __forceinline__ float load_v_rotated(
    const uint8_t* source, uint32_t dimension, const float* levels,
    OctaveKVCacheFormat format) {
    const uint32_t code = unpack_code(source, dimension, 256u, format.v_bits);
    return levels[format.v_bits * 64u + code] *
           load_scale(source + octave_v_slot_bytes(format) - 2u);
}

__global__ __launch_bounds__(kOctaveThreads)
void octave_split_legacy(PagedAttentionCommonArgs args, const uint8_t* k_pool,
                         const uint8_t* v_pool, const float* levels,
                         OctaveKVCacheFormat format, float* partials,
                         uint32_t splits, uint32_t q_head_base,
                         uint32_t q_per_kv_total) {
    __shared__ float q_shared[kOctaveQPerKv * 256u];
    __shared__ float k_values[kOctaveThreads];
    __shared__ float scores[kOctaveTokenLanes * kOctaveQPerKv];
    __shared__ float lane_accumulators[kOctaveTokenLanes * kOctaveQPerKv * 256u];
    __shared__ float lane_max[kOctaveTokenLanes * kOctaveQPerKv];
    __shared__ float lane_sum_values[kOctaveTokenLanes * kOctaveQPerKv];
    __shared__ float level_shared[7u * 64u];
    __shared__ uint32_t row_valid;
    __shared__ uint32_t visible;
    __shared__ uint32_t ancestor_begin;
    __shared__ uint32_t ancestor_count;
    __shared__ const uint32_t* block_table;
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t dimension = tid & 255u;
    const uint32_t token_lane = tid >> 8u;
    const uint32_t split = blockIdx.x % splits;
    const uint32_t task = blockIdx.x / splits;
    const uint32_t kv_head = task % args.kv_heads;
    const uint32_t row = task / args.kv_heads;
    if (row >= args.rows) return;
    const uint32_t q_per_kv = q_per_kv_total;
    const uint32_t group_heads =
        q_per_kv - q_head_base < kOctaveQPerKv ? q_per_kv - q_head_base
                                               : kOctaveQPerKv;

    for (uint32_t i = tid; i < 7u * 64u; i += kOctaveThreads)
        level_shared[i] = levels[i];

    if (tid == 0u) {
        const uint32_t slot = args.row_sequence_slots[row];
        row_valid = slot < args.max_sequences ? 1u : 0u;
        visible = 0u;
        ancestor_begin = 0u;
        ancestor_count = 0u;
        if (row_valid != 0u) {
            block_table = args.block_tables + static_cast<size_t>(slot) * args.block_table_stride;
            if (args.tree_anc_offsets == nullptr) {
                visible = args.row_positions[row] + 1u;
            } else {
                ancestor_begin = args.tree_anc_offsets[row];
                ancestor_count = args.tree_anc_offsets[row + 1u] - ancestor_begin;
                visible = args.tree_committed_end_rows != nullptr
                    ? args.tree_committed_end_rows[row] : args.tree_committed_end;
            }
            const uint64_t capacity = static_cast<uint64_t>(args.block_table_stride) * args.page_tokens;
            if (visible > capacity) row_valid = 0u;
        }
    }
    __syncthreads();
    if (row_valid == 0u) return;

    const uint32_t q_segment = format.k_mode == OctaveKMode::Compact ? 256u : 64u;
    for (uint32_t qh = 0u; qh < group_heads; ++qh) {
        float q_value = 0.0f;
        if (tid < 256u) {
            const uint32_t q_head = kv_head * q_per_kv + q_head_base + qh;
            const bf16_t* q_source = args.q +
                static_cast<uint64_t>(row) * args.q_row_stride +
                static_cast<uint64_t>(q_head) * 256u;
            q_value = __uint_as_float(
                static_cast<uint32_t>(q_source[tid].data) << 16u) *
                sign_of(tid, 11u);
        }
        wht_query(q_value, q_shared + qh * 256u, q_segment);
        if (tid < 256u) q_shared[qh * 256u + tid] = q_value;
        __syncthreads();
    }

    const uint32_t page_count = (visible + args.page_tokens - 1u) / args.page_tokens;
    const uint32_t pages_per_split = (page_count + splits - 1u) / splits;
    const uint32_t page_begin = split * pages_per_split;
    const uint32_t page_limit = page_begin + pages_per_split;
    const uint32_t page_end = page_count < page_limit ? page_count : page_limit;
    const uint32_t k_slot_bytes = octave_k_slot_bytes(format);
    const uint32_t v_slot_bytes = octave_v_slot_bytes(format);
    const uint64_t vectors_per_layer = static_cast<uint64_t>(args.num_pages) *
        args.page_tokens * args.kv_heads;
    float run_max[kOctaveQPerKv];
    float run_sum[kOctaveQPerKv];
    float accumulator[kOctaveQPerKv];
    for (uint32_t qh = 0u; qh < group_heads; ++qh) {
        run_max[qh] = -FLT_MAX;
        run_sum[qh] = 0.0f;
        accumulator[qh] = 0.0f;
    }

    const uint32_t token_begin = page_begin * args.page_tokens;
    const uint32_t committed_end = page_end * args.page_tokens < visible
        ? page_end * args.page_tokens : visible;
    const uint32_t committed_count = committed_end > token_begin
        ? committed_end - token_begin : 0u;
    const uint32_t extra_count = args.tree_anc_offsets != nullptr && split == 0u
        ? ancestor_count : 0u;
    const uint32_t item_count = committed_count + extra_count;
    const uint32_t rounds =
        (item_count + kOctaveTokenLanes - 1u) / kOctaveTokenLanes;
    for (uint32_t round = 0u; round < rounds; ++round) {
        const uint32_t item = round * kOctaveTokenLanes + token_lane;
        bool active = item < item_count;
        uint32_t position = 0u;
        if (active) {
            position = item < committed_count
                ? token_begin + item
                : args.tree_anc_positions[ancestor_begin + item - committed_count];
            active = paged_token_visible(
                args.block_mask, args.mask_stride, args.mask_blocks,
                args.mask_block_shift, row, position);
        }
        const uint32_t logical_page = position / args.page_tokens;
        const uint32_t offset = position % args.page_tokens;
        const uint32_t page = active ? block_table[logical_page] : 0u;
        active = active && page < args.num_pages;
        const uint64_t vector = static_cast<uint64_t>(args.layer) * vectors_per_layer +
            (static_cast<uint64_t>(page) * args.page_tokens + offset) * args.kv_heads + kv_head;
        const uint8_t* k_source = k_pool + vector * k_slot_bytes;
        const uint8_t* v_source = v_pool + vector * v_slot_bytes;
        const float k_value = active
            ? load_k_rotated(k_source, dimension, level_shared, format) : 0.0f;
        const float v_value = active
            ? load_v_rotated(v_source, dimension, level_shared, format) : 0.0f;
        k_values[tid] = k_value;
        __syncthreads();
        const uint32_t qh_wave = dimension >> 5u;
        const uint32_t wave_lane = dimension & 31u;
        if (qh_wave < group_heads) {
            float score = 0.0f;
            for (uint32_t d = wave_lane; d < 256u; d += 32u)
                score += q_shared[qh_wave * 256u + d] *
                         k_values[token_lane * 256u + d];
            for (uint32_t offset = 16u; offset != 0u; offset >>= 1u)
                score += __shfl_down(score, offset);
            if (wave_lane == 0u)
                scores[token_lane * group_heads + qh_wave] = score * args.scale;
        }
        __syncthreads();
        for (uint32_t qh = 0u; qh < group_heads; ++qh) {
            const float score = scores[token_lane * group_heads + qh];
            if (active) {
                const float next_max = fmaxf(run_max[qh], score);
                const float alpha = __expf(run_max[qh] - next_max);
                const float weight = __expf(score - next_max);
                run_sum[qh] = run_sum[qh] * alpha + weight;
                accumulator[qh] = accumulator[qh] * alpha + weight * v_value;
                run_max[qh] = next_max;
            }
        }
    }

    const uint64_t partial_stride = static_cast<uint64_t>(q_per_kv) * 258u;
    float* base = partials +
        ((static_cast<uint64_t>(row) * args.kv_heads + kv_head) * splits + split) * partial_stride;
    for (uint32_t qh = 0u; qh < group_heads; ++qh) {
        lane_accumulators[(token_lane * group_heads + qh) * 256u + dimension] =
            accumulator[qh];
        if (dimension == 0u) {
            lane_max[token_lane * group_heads + qh] = run_max[qh];
            lane_sum_values[token_lane * group_heads + qh] = run_sum[qh];
        }
    }
    __syncthreads();
    if (token_lane != 0u) return;
    for (uint32_t qh = 0u; qh < group_heads; ++qh) {
        float combined_max = -FLT_MAX;
        for (uint32_t lane = 0u; lane < kOctaveTokenLanes; ++lane)
            combined_max = fmaxf(combined_max, lane_max[lane * group_heads + qh]);
        float combined_sum = 0.0f;
        float combined_accumulator = 0.0f;
        for (uint32_t lane = 0u; lane < kOctaveTokenLanes; ++lane) {
            const float lane_weight = __expf(
                lane_max[lane * group_heads + qh] - combined_max);
            combined_sum += lane_weight * lane_sum_values[lane * group_heads + qh];
            combined_accumulator += lane_weight *
                lane_accumulators[(lane * group_heads + qh) * 256u + dimension];
        }
        if (dimension == 0u) {
            base[q_head_base + qh] = combined_max;
            base[q_per_kv + q_head_base + qh] = combined_sum;
        }
        base[2u * q_per_kv + (q_head_base + qh) * 256u + dimension] =
            combined_accumulator;
    }
}

// gfx12 WMMA fragments follow the shelf paged-prefill layout: lane l owns
// matrix column l % 16 and two groups of four K elements selected by l / 16.
// The packed-code tiling and scale placement follow the Apache-2.0 vLLM
// octave_decode_wmma_rdna3.cu design, but gfx11 DPP/lane-pair fragments are
// deliberately not reused here.
using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kOctaveWmmaThreads = 256u;
constexpr uint32_t kOctaveWmmaTokens = 32u;
constexpr uint32_t kOctaveWmmaPitch = 264u;
constexpr uint32_t kOctaveWmmaQueries = 16u;
constexpr uint32_t kOctaveWmmaProbabilityPitch = 40u;

__device__ __forceinline__ f16x8_t octave_frag_nk(
    const _Float16* source, uint32_t pitch, uint32_t row_base,
    uint32_t k_base, uint32_t lane) {
    const _Float16* p = source +
        static_cast<size_t>(row_base + (lane & 15u)) * pitch +
        k_base + 4u * (lane >> 4u);
    f16x8_t fragment;
    #pragma unroll
    for (uint32_t e = 0u; e < 4u; ++e) {
        fragment[e] = p[e];
        fragment[4u + e] = p[8u + e];
    }
    return fragment;
}

__device__ __forceinline__ _Float16 octave_biased_byte_to_half(uint8_t value) {
    return __builtin_bit_cast(_Float16, static_cast<uint16_t>(0x6400u + value)) -
           static_cast<_Float16>(1152.0f);
}

__device__ __forceinline__ f16x8_t octave_frag_nk_u8(
    const uint8_t* source, uint32_t pitch, uint32_t row_base,
    uint32_t k_base, uint32_t lane) {
    const uint8_t* p = source +
        static_cast<size_t>(row_base + (lane & 15u)) * pitch +
        k_base + 4u * (lane >> 4u);
    f16x8_t fragment;
    #pragma unroll
    for (uint32_t e = 0u; e < 4u; ++e) {
        fragment[e] = octave_biased_byte_to_half(p[e]);
        fragment[4u + e] = octave_biased_byte_to_half(p[8u + e]);
    }
    return fragment;
}

__device__ __forceinline__ f16x8_t octave_frag_kn_u8(
    const uint8_t* source, uint32_t pitch, uint32_t column_base,
    uint32_t k_base, uint32_t lane) {
    const uint32_t column = column_base + (lane & 15u);
    const uint32_t k0 = k_base + 4u * (lane >> 4u);
    f16x8_t fragment;
    #pragma unroll
    for (uint32_t e = 0u; e < 4u; ++e) {
        fragment[e] = octave_biased_byte_to_half(
            source[static_cast<size_t>(k0 + e) * pitch + column]);
        fragment[4u + e] = octave_biased_byte_to_half(
            source[static_cast<size_t>(k0 + 8u + e) * pitch + column]);
    }
    return fragment;
}

__device__ __forceinline__ f32x8_t octave_wmma(
    f16x8_t a, f16x8_t b, f32x8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ void octave_wht8(float (&x)[8]) {
    #pragma unroll
    for (uint32_t half = 1u; half < 8u; half <<= 1u) {
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            if ((i & half) == 0u) {
                const float a = x[i];
                const float b = x[i + half];
                x[i] = a + b;
                x[i + half] = a - b;
            }
        }
    }
}

__device__ __forceinline__ void octave_wht_query_wave(
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

struct OctaveWmmaShared {
    _Float16 q[kOctaveWmmaQueries][kOctaveWmmaPitch];
    uint8_t k[kOctaveWmmaTokens][kOctaveWmmaPitch];
    uint8_t v[kOctaveWmmaTokens][kOctaveWmmaPitch];
    _Float16 probability[kOctaveWmmaQueries][kOctaveWmmaProbabilityPitch];
    _Float16 k_scale[kOctaveWmmaTokens][4u];
    _Float16 v_scale[kOctaveWmmaTokens];
    float score_part[4u][kOctaveWmmaQueries][kOctaveWmmaTokens];
    float score[kOctaveWmmaQueries][kOctaveWmmaTokens];
    float alpha[kOctaveWmmaQueries];
    uint64_t k_source[kOctaveWmmaTokens];
    uint64_t v_source[kOctaveWmmaTokens];
    uint32_t active[kOctaveWmmaTokens];
    uint32_t chunk_active;
    uint32_t row_valid;
    uint32_t visible;
    uint32_t ancestor_begin;
    uint32_t ancestor_count;
    const uint32_t* block_table;
};

template <uint32_t kBits>
__device__ __forceinline__ int32_t octave_codebook_int8_fixed(uint32_t code) {
    // Eight signed bytes are selected by one v_perm_b32. Wider codebooks use
    // the high code bits to choose a register bank; no LUT memory is touched.
    uint32_t value = 0u;
    const uint32_t bank = code >> 3u;
    const uint32_t selector = code & 7u;
    if constexpr (kBits == 3u) {
        value = __builtin_amdgcn_perm(0x7f4f2d0eu, 0xf2d3b181u, selector);
    } else if constexpr (kBits == 4u) {
        const uint32_t v0 = __builtin_amdgcn_perm(
            0xfaeee1d4u, 0xc6b5a081u, selector);
        const uint32_t v1 = __builtin_amdgcn_perm(
            0x7f604b3au, 0x2c1f1206u, selector);
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
__device__ __forceinline__ float octave_codebook_scale_fixed() {
    if constexpr (kBits == 3u) return 0.0169444544f;
    if constexpr (kBits == 4u) return 0.0215164533f;
    if constexpr (kBits == 5u) return 0.0256750590f;
    return 0.0294811393f;
}

template <uint32_t kBits>
__device__ __forceinline__ uint8_t octave_level_biased_fixed(
    const uint8_t* source, uint32_t dimension, uint32_t segment) {
    const uint32_t code = unpack_code(source, dimension, segment, kBits);
    return static_cast<uint8_t>(octave_codebook_int8_fixed<kBits>(code) + 128);
}

__device__ __forceinline__ uint32_t octave_lut3x4_biased(uint32_t codes) {
    return __builtin_amdgcn_perm(0x7f4f2d0eu, 0xf2d3b181u, codes) ^
           0x80808080u;
}

__device__ __forceinline__ uint32_t octave_lut4x4_biased(uint32_t codes) {
    const uint32_t selector = codes & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(
        0xfaeee1d4u, 0xc6b5a081u, selector);
    const uint32_t hi = __builtin_amdgcn_perm(
        0x7f604b3au, 0x2c1f1206u, selector);
    const uint32_t sign = codes & 0x08080808u;
    const uint32_t mask = (sign << 5u) - (sign >> 3u);
    return ((hi & mask) | (lo & ~mask)) ^ 0x80808080u;
}

__device__ __forceinline__ uint32_t octave_byte_mask(uint32_t bits,
                                                      uint32_t shift) {
    return (bits << (8u - shift)) - (bits >> shift);
}

__device__ __forceinline__ uint32_t octave_lut5x4_biased(uint32_t codes) {
    const uint32_t selector = codes & 0x07070707u;
    const uint32_t v0 = __builtin_amdgcn_perm(0xd1cac3bau, 0xb1a69781u, selector);
    const uint32_t v1 = __builtin_amdgcn_perm(0xfdf8f3eeu, 0xe8e3ddd7u, selector);
    const uint32_t v2 = __builtin_amdgcn_perm(0x29231d18u, 0x120d0803u, selector);
    const uint32_t v3 = __builtin_amdgcn_perm(0x7f695a4fu, 0x463d362fu, selector);
    const uint32_t m3 = octave_byte_mask(codes & 0x08080808u, 3u);
    const uint32_t m4 = octave_byte_mask(codes & 0x10101010u, 4u);
    const uint32_t lo = (v1 & m3) | (v0 & ~m3);
    const uint32_t hi = (v3 & m3) | (v2 & ~m3);
    return ((hi & m4) | (lo & ~m4)) ^ 0x80808080u;
}

__device__ __forceinline__ uint32_t octave_lut6x4_biased(uint32_t codes) {
    const uint32_t selector = codes & 0x07070707u;
    const uint32_t v0 = __builtin_amdgcn_perm(0xbcb7b2acu, 0xa59d9281u, selector);
    const uint32_t v1 = __builtin_amdgcn_perm(0xd8d5d2cfu, 0xcbc8c4c0u, selector);
    const uint32_t v2 = __builtin_amdgcn_perm(0xeceae8e5u, 0xe2e0dddau, selector);
    const uint32_t v3 = __builtin_amdgcn_perm(0xfffdfaf8u, 0xf6f3f1efu, selector);
    const uint32_t v4 = __builtin_amdgcn_perm(0x110f0d0au, 0x08060301u, selector);
    const uint32_t v5 = __builtin_amdgcn_perm(0x2623201eu, 0x1b181614u, selector);
    const uint32_t v6 = __builtin_amdgcn_perm(0x403c3835u, 0x312e2b28u, selector);
    const uint32_t v7 = __builtin_amdgcn_perm(0x7f6e635bu, 0x544e4944u, selector);
    const uint32_t m3 = octave_byte_mask(codes & 0x08080808u, 3u);
    const uint32_t m4 = octave_byte_mask(codes & 0x10101010u, 4u);
    const uint32_t m5 = octave_byte_mask(codes & 0x20202020u, 5u);
    const uint32_t a0 = (v1 & m3) | (v0 & ~m3);
    const uint32_t a1 = (v3 & m3) | (v2 & ~m3);
    const uint32_t b0 = (v5 & m3) | (v4 & ~m3);
    const uint32_t b1 = (v7 & m3) | (v6 & ~m3);
    const uint32_t lo = (a1 & m4) | (a0 & ~m4);
    const uint32_t hi = (b1 & m4) | (b0 & ~m4);
    return ((hi & m5) | (lo & ~m5)) ^ 0x80808080u;
}

template <uint32_t kBits>
__device__ __forceinline__ void octave_unpack_segment_cooperative(
    const uint64_t* sources, const uint32_t* active, uint8_t* destination,
    uint32_t destination_pitch, uint32_t source_offset,
    uint32_t destination_offset, uint32_t dimensions) {
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    if constexpr (kBits == 4u || kBits == 5u || kBits == 6u) {
        const uint32_t groups = dimensions / 8u;
        for (uint32_t item = tid; item < kOctaveWmmaTokens * groups;
             item += kOctaveWmmaThreads) {
            const uint32_t token = item / groups;
            const uint32_t group = item - token * groups;
            uint32_t lo = 0x80808080u;
            uint32_t hi = 0x80808080u;
            if (active[token] != 0u) {
                const uint8_t* source =
                    reinterpret_cast<const uint8_t*>(sources[token]) +
                    source_offset + group * 4u;
                const uint32_t raw = load_word(source);
                uint32_t high = 0u;
                if constexpr (kBits == 5u) {
                    const uint32_t word = load_word(
                        reinterpret_cast<const uint8_t*>(sources[token]) + source_offset +
                        dimensions / 2u + (group >> 2u) * 4u);
                    high = (word >> ((group & 3u) * 8u)) & 0xffu;
                } else if constexpr (kBits == 6u) {
                    const uint32_t word = load_word(
                        reinterpret_cast<const uint8_t*>(sources[token]) + source_offset +
                        dimensions / 2u + (group >> 1u) * 4u);
                    high = (word >> ((group & 1u) * 16u)) & 0xffffu;
                }
                uint32_t c0 = 0u;
                uint32_t c1 = 0u;
                #pragma unroll
                for (uint32_t j = 0u; j < 4u; ++j) {
                    uint32_t a = (raw >> (4u * j)) & 15u;
                    uint32_t b = (raw >> (16u + 4u * j)) & 15u;
                    if constexpr (kBits == 5u) {
                        a |= ((high >> j) & 1u) << 4u;
                        b |= ((high >> (4u + j)) & 1u) << 4u;
                    } else if constexpr (kBits == 6u) {
                        a |= ((high >> (2u * j)) & 3u) << 4u;
                        b |= ((high >> (8u + 2u * j)) & 3u) << 4u;
                    }
                    c0 |= a << (8u * j);
                    c1 |= b << (8u * j);
                }
                if constexpr (kBits == 4u) {
                    lo = octave_lut4x4_biased(c0);
                    hi = octave_lut4x4_biased(c1);
                } else if constexpr (kBits == 5u) {
                    lo = octave_lut5x4_biased(c0);
                    hi = octave_lut5x4_biased(c1);
                } else {
                    lo = octave_lut6x4_biased(c0);
                    hi = octave_lut6x4_biased(c1);
                }
            }
            uint8_t* out = destination + static_cast<size_t>(token) *
                destination_pitch + destination_offset + group * 8u;
            *reinterpret_cast<uint32_t*>(out) = lo;
            *reinterpret_cast<uint32_t*>(out + 4u) = hi;
        }
    } else if constexpr (kBits == 3u) {
        const uint32_t groups = dimensions / 16u;
        for (uint32_t item = tid; item < kOctaveWmmaTokens * groups;
             item += kOctaveWmmaThreads) {
            const uint32_t token = item / groups;
            const uint32_t group = item - token * groups;
            uint32_t low = 0u;
            uint32_t high = 0u;
            if (active[token] != 0u) {
                const uint8_t* source =
                    reinterpret_cast<const uint8_t*>(sources[token]) + source_offset;
                low = load_word(source + group * 4u);
                high = load_word(source + dimensions / 4u + (group >> 1u) * 4u);
                high >>= (group & 1u) * 16u;
            }
            uint8_t* out = destination + static_cast<size_t>(token) *
                destination_pitch + destination_offset + group * 16u;
            #pragma unroll
            for (uint32_t i = 0u; i < 4u; ++i) {
                uint32_t codes = 0u;
                #pragma unroll
                for (uint32_t j = 0u; j < 4u; ++j) {
                    const uint32_t dimension = i + 4u * j;
                    const uint32_t code = ((low >> (2u * dimension)) & 3u) |
                                          (((high >> dimension) & 1u) << 2u);
                    codes |= code << (8u * j);
                }
                const uint32_t decoded = active[token] != 0u
                    ? octave_lut3x4_biased(codes) : 0x80808080u;
                out[i] = static_cast<uint8_t>(decoded);
                out[i + 4u] = static_cast<uint8_t>(decoded >> 8u);
                out[i + 8u] = static_cast<uint8_t>(decoded >> 16u);
                out[i + 12u] = static_cast<uint8_t>(decoded >> 24u);
            }
        }
    } else {
        for (uint32_t item = tid; item < kOctaveWmmaTokens * dimensions;
             item += kOctaveWmmaThreads) {
            const uint32_t token = item / dimensions;
            const uint32_t dimension = item - token * dimensions;
            const uint8_t* source =
                reinterpret_cast<const uint8_t*>(sources[token]) + source_offset;
            destination[static_cast<size_t>(token) * destination_pitch +
                        destination_offset + dimension] = active[token] != 0u
                ? octave_level_biased_fixed<kBits>(source, dimension, dimensions)
                : 128u;
        }
    }
}

template <bool kCompact, uint32_t kRopeBits, uint32_t kNopeBits,
          uint32_t kVBits>
__global__ __launch_bounds__(kOctaveWmmaThreads, 1)
void octave_split_wmma(PagedAttentionCommonArgs args, const uint8_t* k_pool,
                       const uint8_t* v_pool, const float* levels,
                       OctaveKVCacheFormat format, float* partials,
                       uint32_t splits, uint32_t q_head_base,
                       uint32_t q_per_kv_total) {
    __shared__ OctaveWmmaShared s;
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t split = blockIdx.x % splits;
    const uint32_t task = blockIdx.x / splits;
    const uint32_t kv_head = task % args.kv_heads;
    const uint32_t row = task / args.kv_heads;
    if (row >= args.rows) return;
    const uint32_t q_per_kv = q_per_kv_total;
    const uint32_t group_heads =
        q_per_kv - q_head_base < kOctaveQPerKv ? q_per_kv - q_head_base
                                               : kOctaveQPerKv;

    (void)levels;
    for (uint32_t i = tid; i < kOctaveWmmaQueries * kOctaveWmmaPitch;
         i += kOctaveWmmaThreads)
        s.q[0][i] = static_cast<_Float16>(0.0f);

    if (tid == 0u) {
        const uint32_t slot = args.row_sequence_slots[row];
        s.row_valid = slot < args.max_sequences ? 1u : 0u;
        s.visible = 0u;
        s.ancestor_begin = 0u;
        s.ancestor_count = 0u;
        s.block_table = args.block_tables;
        if (s.row_valid != 0u) {
            s.block_table = args.block_tables +
                static_cast<size_t>(slot) * args.block_table_stride;
            if (args.tree_anc_offsets == nullptr) {
                s.visible = args.row_positions[row] + 1u;
            } else {
                s.ancestor_begin = args.tree_anc_offsets[row];
                s.ancestor_count = args.tree_anc_offsets[row + 1u] -
                                   s.ancestor_begin;
                s.visible = args.tree_committed_end_rows != nullptr
                    ? args.tree_committed_end_rows[row]
                    : args.tree_committed_end;
            }
            const uint64_t capacity =
                static_cast<uint64_t>(args.block_table_stride) * args.page_tokens;
            if (s.visible > capacity) s.row_valid = 0u;
        }
    }
    __syncthreads();
    if (s.row_valid == 0u) return;

    if (wave < group_heads) {
        float qv[8];
        const uint32_t q_head = kv_head * q_per_kv + q_head_base + wave;
        const bf16_t* source = args.q +
            static_cast<uint64_t>(row) * args.q_row_stride +
            static_cast<uint64_t>(q_head) * 256u + lane * 8u;
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            const uint32_t dimension = lane * 8u + e;
            qv[e] = __uint_as_float(static_cast<uint32_t>(source[e].data) << 16u) *
                    sign_of(dimension, 11u);
        }
        octave_wht_query_wave(qv, lane, kCompact);
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e)
            s.q[wave][lane * 8u + e] = static_cast<_Float16>(qv[e]);
    }
    __syncthreads();

    const uint32_t page_count =
        (s.visible + args.page_tokens - 1u) / args.page_tokens;
    const uint32_t pages_per_split = (page_count + splits - 1u) / splits;
    const uint32_t page_begin = split * pages_per_split;
    const uint32_t page_end = min(page_begin + pages_per_split, page_count);
    const uint32_t token_begin = page_begin * args.page_tokens;
    const uint32_t committed_end = min(page_end * args.page_tokens, s.visible);
    const uint32_t committed_count = committed_end > token_begin
        ? committed_end - token_begin : 0u;
    const uint32_t extra_count = args.tree_anc_offsets != nullptr && split == 0u
        ? s.ancestor_count : 0u;
    const uint32_t item_count = committed_count + extra_count;
    const uint32_t k_slot_bytes = octave_k_slot_bytes(format);
    const uint32_t v_slot_bytes = octave_v_slot_bytes(format);
    const uint64_t vectors_per_layer = static_cast<uint64_t>(args.num_pages) *
        args.page_tokens * args.kv_heads;

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    f32x8_t output_accumulator[4];
    #pragma unroll
    for (uint32_t i = 0u; i < 4u; ++i) output_accumulator[i] = f32x8_t{};

    for (uint32_t base_item = 0u; base_item < item_count;
         base_item += kOctaveWmmaTokens) {
        if (tid < kOctaveWmmaTokens) {
            const uint32_t item = base_item + tid;
            bool active = item < item_count;
            uint32_t position = 0u;
            if (active) {
                position = item < committed_count
                    ? token_begin + item
                    : args.tree_anc_positions[s.ancestor_begin + item - committed_count];
                active = paged_token_visible(
                    args.block_mask, args.mask_stride, args.mask_blocks,
                    args.mask_block_shift, row, position);
            }
            const uint32_t logical_page = position / args.page_tokens;
            const uint32_t offset = position % args.page_tokens;
            const uint32_t page = active ? s.block_table[logical_page] : 0u;
            active = active && page < args.num_pages;
            const uint64_t vector = static_cast<uint64_t>(args.layer) *
                vectors_per_layer +
                (static_cast<uint64_t>(page) * args.page_tokens + offset) *
                    args.kv_heads + kv_head;
            const uint8_t* ks = k_pool + vector * k_slot_bytes;
            const uint8_t* vs = v_pool + vector * v_slot_bytes;
            s.k_source[tid] = reinterpret_cast<uint64_t>(ks);
            s.v_source[tid] = reinterpret_cast<uint64_t>(vs);
            s.active[tid] = active ? 1u : 0u;
            constexpr uint32_t scale_count = kCompact ? 1u : 4u;
            const uint32_t scale_offset = k_slot_bytes - scale_count * 2u;
            #pragma unroll
            for (uint32_t block = 0u; block < 4u; ++block) {
                float value = active && block < scale_count
                    ? load_scale(ks + scale_offset + block * 2u) : 0.0f;
                if constexpr (kCompact) {
                    value *= octave_codebook_scale_fixed<kNopeBits>();
                } else {
                    value *= block == 0u
                        ? octave_codebook_scale_fixed<kRopeBits>()
                        : octave_codebook_scale_fixed<kNopeBits>();
                }
                s.k_scale[tid][block] = static_cast<_Float16>(value);
            }
            s.v_scale[tid] = static_cast<_Float16>(active
                ? load_scale(vs + v_slot_bytes - 2u) *
                      octave_codebook_scale_fixed<kVBits>()
                : 0.0f);
        }
        __syncthreads();
        if (tid == 0u) {
            uint32_t any = 0u;
            for (uint32_t i = 0u; i < kOctaveWmmaTokens; ++i)
                any |= s.active[i];
            s.chunk_active = any;
        }
        __syncthreads();
        if (s.chunk_active == 0u) continue;

        if constexpr (kCompact) {
            octave_unpack_segment_cooperative<kNopeBits>(
                s.k_source, s.active, &s.k[0][0], kOctaveWmmaPitch,
                0u, 0u, 256u);
        } else {
            octave_unpack_segment_cooperative<kRopeBits>(
                s.k_source, s.active, &s.k[0][0], kOctaveWmmaPitch,
                0u, 0u, 64u);
            constexpr uint32_t nope_bytes = octave_plane_bytes(64u, kNopeBits);
            constexpr uint32_t rope_bytes = octave_plane_bytes(64u, kRopeBits);
            #pragma unroll
            for (uint32_t block = 1u; block < 4u; ++block) {
                octave_unpack_segment_cooperative<kNopeBits>(
                    s.k_source, s.active, &s.k[0][0], kOctaveWmmaPitch,
                    rope_bytes + (block - 1u) * nope_bytes, block * 64u, 64u);
            }
        }
        octave_unpack_segment_cooperative<kVBits>(
            s.v_source, s.active, &s.v[0][0], kOctaveWmmaPitch,
            0u, 0u, 256u);
        __syncthreads();

        const bool qk_wave = wave == 0u || wave >= 5u;
        if (qk_wave) {
            const uint32_t segment_index = wave == 0u ? 0u : wave - 4u;
            #pragma unroll
            for (uint32_t token_tile = 0u; token_tile < 2u; ++token_tile) {
                f32x8_t score_acc = f32x8_t{};
                #pragma unroll
                for (uint32_t step = 0u; step < 4u; ++step) {
                    const uint32_t ks = segment_index * 4u + step;
                    const f16x8_t qf = octave_frag_nk(
                        &s.q[0][0], kOctaveWmmaPitch, 0u, ks * 16u, lane);
                    const f16x8_t kf = octave_frag_nk_u8(
                        &s.k[0][0], kOctaveWmmaPitch, token_tile * 16u,
                        ks * 16u, lane);
                    score_acc = octave_wmma(qf, kf, score_acc);
                }
                const uint32_t query = (lane >> 4u) * 8u;
                const uint32_t token = token_tile * 16u + (lane & 15u);
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++e)
                    s.score_part[segment_index][query + e][token] = score_acc[e];
            }
        }
        __syncthreads();

        if (wave == 0u && lane < kOctaveWmmaQueries) {
            if (lane < group_heads) {
                float tile_max = -FLT_MAX;
                #pragma unroll
                for (uint32_t token = 0u; token < kOctaveWmmaTokens; ++token) {
                    float value = 0.0f;
                    #pragma unroll
                    for (uint32_t segment_index = 0u; segment_index < 4u;
                         ++segment_index) {
                        const uint32_t scale_index = kCompact ? 0u : segment_index;
                        value += s.score_part[segment_index][lane][token] *
                                 static_cast<float>(s.k_scale[token][scale_index]);
                    }
                    value *= args.scale;
                    s.score[lane][token] = value;
                    if (s.active[token] != 0u) tile_max = fmaxf(tile_max, value);
                }
                const float next_max = fmaxf(running_max, tile_max);
                const float alpha = running_max == -FLT_MAX
                    ? 0.0f : __expf(running_max - next_max);
                float tile_sum = 0.0f;
                #pragma unroll
                for (uint32_t token = 0u; token < kOctaveWmmaTokens; ++token) {
                    const float probability = s.active[token] != 0u
                        ? __expf(s.score[lane][token] - next_max) : 0.0f;
                    tile_sum += probability;
                    s.probability[lane][token] = static_cast<_Float16>(
                        probability * static_cast<float>(s.v_scale[token]));
                }
                running_sum = running_sum * alpha + tile_sum;
                running_max = next_max;
                s.alpha[lane] = alpha;
            } else {
                s.alpha[lane] = 0.0f;
                #pragma unroll
                for (uint32_t token = 0u; token < kOctaveWmmaTokens; ++token)
                    s.probability[lane][token] = static_cast<_Float16>(0.0f);
            }
        }
        __syncthreads();

        if (wave >= 1u && wave <= 4u) {
            const uint32_t row_base = (lane >> 4u) * 8u;
            #pragma unroll
            for (uint32_t nt = 0u; nt < 4u; ++nt)
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++e)
                    output_accumulator[nt][e] *= s.alpha[row_base + e];
            const f16x8_t pf = octave_frag_nk(
                &s.probability[0][0], kOctaveWmmaProbabilityPitch,
                0u, 0u, lane);
            const uint32_t dimension_base = (wave - 1u) * 64u;
            #pragma unroll
            for (uint32_t nt = 0u; nt < 4u; ++nt) {
                output_accumulator[nt] = octave_wmma(
                    pf,
                    octave_frag_kn_u8(&s.v[0][0], kOctaveWmmaPitch,
                                      dimension_base + nt * 16u, 0u, lane),
                    output_accumulator[nt]);
                const f16x8_t pf_hi = octave_frag_nk(
                    &s.probability[0][0], kOctaveWmmaProbabilityPitch,
                    0u, 16u, lane);
                output_accumulator[nt] = octave_wmma(
                    pf_hi,
                    octave_frag_kn_u8(&s.v[0][0], kOctaveWmmaPitch,
                                      dimension_base + nt * 16u, 16u, lane),
                    output_accumulator[nt]);
            }
        }
        __syncthreads();
    }

    const uint64_t partial_stride = static_cast<uint64_t>(q_per_kv) * 258u;
    float* partial = partials +
        ((static_cast<uint64_t>(row) * args.kv_heads + kv_head) * splits + split) *
            partial_stride;
    if (wave == 0u && lane < group_heads) {
        partial[q_head_base + lane] = running_max;
        partial[q_per_kv + q_head_base + lane] = running_sum;
    }
    if (wave >= 1u && wave <= 4u) {
        const uint32_t query_base = (lane >> 4u) * 8u;
        const uint32_t dimension_base = (wave - 1u) * 64u + (lane & 15u);
        #pragma unroll
        for (uint32_t nt = 0u; nt < 4u; ++nt) {
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e) {
                const uint32_t query = query_base + e;
                if (query < group_heads) {
                    partial[2u * q_per_kv + (q_head_base + query) * 256u +
                            dimension_base + nt * 16u] = output_accumulator[nt][e];
                }
            }
        }
    }
}

__global__ __launch_bounds__(256)
void octave_reduce(PagedAttentionCommonArgs args, const float* partials,
                   uint32_t splits) {
    __shared__ float shared[256];
    __shared__ float weights[kOctaveMaxSplits];
    __shared__ float total_sum;
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t q_head = blockIdx.x % args.q_heads;
    const uint32_t row = blockIdx.x / args.q_heads;
    if (row >= args.rows) return;
    const uint32_t q_per_kv = args.q_heads / args.kv_heads;
    const uint32_t kv_head = q_head / q_per_kv;
    const uint32_t qh = q_head % q_per_kv;
    const uint64_t partial_stride = static_cast<uint64_t>(q_per_kv) * 258u;
    const float* base = partials +
        (static_cast<uint64_t>(row) * args.kv_heads + kv_head) * splits * partial_stride;
    if (tid == 0u) {
        float global_max = -FLT_MAX;
        for (uint32_t split = 0u; split < splits; ++split)
            global_max = fmaxf(global_max, base[split * partial_stride + qh]);
        float sum = 0.0f;
        for (uint32_t split = 0u; split < splits; ++split) {
            const float weight = __expf(base[split * partial_stride + qh] - global_max);
            weights[split] = weight;
            sum += weight * base[split * partial_stride + q_per_kv + qh];
        }
        total_sum = sum;
    }
    __syncthreads();
    float value = 0.0f;
    for (uint32_t split = 0u; split < splits; ++split)
        value += weights[split] *
            base[split * partial_stride + 2u * q_per_kv + qh * 256u + tid];
    value = total_sum > 0.0f ? value / total_sum : 0.0f;
    wht(value, shared, 256u);
    value *= sign_of(tid, 29u);
    const uint64_t output = static_cast<uint64_t>(row) * args.output_row_stride +
                            static_cast<uint64_t>(q_head) * 256u + tid;
    if (args.output_dtype == PagedAttentionOutputDType::F32)
        reinterpret_cast<float*>(args.output)[output] = value;
    else
        reinterpret_cast<bf16_t*>(args.output)[output] =
            bf16_t{__bfloat16_as_ushort(__float2bfloat16(value))};
}

}  // namespace

hipError_t launch_octave_kv_dequant(
    const uint8_t* k_pool, const uint8_t* v_pool, const float* levels,
    bf16_t* k_scratch, bf16_t* v_scratch, OctaveKVCacheFormat format,
    uint32_t layer, uint32_t num_pages, uint32_t page_tokens,
    uint32_t kv_heads, hipStream_t stream) {
    if (k_pool == nullptr || v_pool == nullptr || levels == nullptr ||
        k_scratch == nullptr || v_scratch == nullptr ||
        !octave_format_valid(format) || num_pages == 0u || page_tokens == 0u ||
        kv_heads == 0u)
        return hipErrorInvalidValue;
    const uint64_t vectors = static_cast<uint64_t>(num_pages) * page_tokens * kv_heads;
    if (vectors > UINT32_MAX) return hipErrorInvalidValue;
    dequant<<<static_cast<uint32_t>(vectors), 256u, 0u, stream>>>(
        k_pool, v_pool, levels, k_scratch, v_scratch, format, layer,
        num_pages, page_tokens, kv_heads);
    return hipGetLastError();
}

hipError_t launch_attention_paged_octave_split(
    const PagedAttentionCommonArgs& common, const uint8_t* k_pool,
    const uint8_t* v_pool, const float* levels, OctaveKVCacheFormat format,
    float* partials, uint32_t splits, hipStream_t stream) {
    if (common.rows == 0u || common.head_dim != 256u || common.q_heads == 0u ||
        common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u ||
        k_pool == nullptr || v_pool == nullptr ||
        levels == nullptr || partials == nullptr || splits == 0u ||
        splits > kOctaveMaxSplits ||
        !octave_format_valid(format))
        return hipErrorInvalidValue;
    const uint64_t grid = static_cast<uint64_t>(common.rows) * common.kv_heads * splits;
    if (grid > UINT32_MAX) return hipErrorInvalidValue;
    static const bool legacy = [] {
        const char* value = std::getenv("SHELF_OCTAVE_DECODE_LEGACY");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    for (uint32_t q_base = 0u; q_base < q_per_kv; q_base += kOctaveQPerKv) {
        if (legacy) {
            octave_split_legacy<<<static_cast<uint32_t>(grid), kOctaveThreads, 0u, stream>>>(
                common, k_pool, v_pool, levels, format, partials, splits, q_base, q_per_kv);
        } else if (format.k_mode == OctaveKMode::Compact &&
                   format.k_nope_bits == 3u && format.v_bits == 3u) {
            octave_split_wmma<true, 3u, 3u, 3u>
                <<<static_cast<uint32_t>(grid), kOctaveWmmaThreads, 0u, stream>>>(
                    common, k_pool, v_pool, levels, format, partials, splits, q_base,
                    q_per_kv);
        } else if (format.k_mode == OctaveKMode::Compact &&
                   format.k_nope_bits == 4u && format.v_bits == 4u) {
            octave_split_wmma<true, 4u, 4u, 4u>
                <<<static_cast<uint32_t>(grid), kOctaveWmmaThreads, 0u, stream>>>(
                    common, k_pool, v_pool, levels, format, partials, splits, q_base,
                    q_per_kv);
        } else if (format.k_mode == OctaveKMode::Compact &&
                   format.k_nope_bits == 5u && format.v_bits == 5u) {
            octave_split_wmma<true, 5u, 5u, 5u>
                <<<static_cast<uint32_t>(grid), kOctaveWmmaThreads, 0u, stream>>>(
                    common, k_pool, v_pool, levels, format, partials, splits, q_base,
                    q_per_kv);
        } else if (format.k_mode == OctaveKMode::Compact &&
                   format.k_nope_bits == 6u && format.v_bits == 6u) {
            octave_split_wmma<true, 6u, 6u, 6u>
                <<<static_cast<uint32_t>(grid), kOctaveWmmaThreads, 0u, stream>>>(
                    common, k_pool, v_pool, levels, format, partials, splits, q_base,
                    q_per_kv);
        } else if (format.k_mode == OctaveKMode::Block &&
                   format.k_rope_bits == 4u && format.k_nope_bits == 3u &&
                   format.v_bits == 3u) {
            octave_split_wmma<false, 4u, 3u, 3u>
                <<<static_cast<uint32_t>(grid), kOctaveWmmaThreads, 0u, stream>>>(
                    common, k_pool, v_pool, levels, format, partials, splits, q_base,
                    q_per_kv);
        } else if (format.k_mode == OctaveKMode::Block &&
                   format.k_rope_bits == 4u && format.k_nope_bits == 3u &&
                   format.v_bits == 4u) {
            octave_split_wmma<false, 4u, 3u, 4u>
                <<<static_cast<uint32_t>(grid), kOctaveWmmaThreads, 0u, stream>>>(
                    common, k_pool, v_pool, levels, format, partials, splits, q_base,
                    q_per_kv);
        } else {
            // Uncommon mixed bit-width descriptors keep the exact scalar path;
            // the production descriptors above use the gfx12 WMMA path.
            octave_split_legacy<<<static_cast<uint32_t>(grid), kOctaveThreads, 0u, stream>>>(
                common, k_pool, v_pool, levels, format, partials, splits, q_base, q_per_kv);
        }
    }
    return hipGetLastError();
}

hipError_t launch_attention_paged_octave_reduce(
    const PagedAttentionCommonArgs& common, const float* partials,
    uint32_t splits, hipStream_t stream) {
    if (common.rows == 0u || common.head_dim != 256u || common.q_heads == 0u ||
        common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u ||
        common.output == nullptr || partials == nullptr ||
        splits == 0u || splits > kOctaveMaxSplits)
        return hipErrorInvalidValue;
    const uint64_t grid = static_cast<uint64_t>(common.rows) * common.q_heads;
    if (grid > UINT32_MAX) return hipErrorInvalidValue;
    octave_reduce<<<static_cast<uint32_t>(grid), 256u, 0u, stream>>>(
        common, partials, splits);
    return hipGetLastError();
}

}  // namespace shelf::kernel
