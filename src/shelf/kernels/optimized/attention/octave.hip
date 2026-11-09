#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/kernels/optimized/detail/attention_paged_common.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cfloat>

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
void octave_split(PagedAttentionCommonArgs args, const uint8_t* k_pool,
                  const uint8_t* v_pool, const float* levels,
                  OctaveKVCacheFormat format, float* partials,
                  uint32_t splits) {
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
    const uint32_t q_per_kv = args.q_heads / args.kv_heads;

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
    for (uint32_t qh = 0u; qh < q_per_kv; ++qh) {
        float q_value = 0.0f;
        if (tid < 256u) {
            const uint32_t q_head = kv_head * q_per_kv + qh;
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
    for (uint32_t qh = 0u; qh < q_per_kv; ++qh) {
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
        if (qh_wave < q_per_kv) {
            float score = 0.0f;
            for (uint32_t d = wave_lane; d < 256u; d += 32u)
                score += q_shared[qh_wave * 256u + d] *
                         k_values[token_lane * 256u + d];
            for (uint32_t offset = 16u; offset != 0u; offset >>= 1u)
                score += __shfl_down(score, offset);
            if (wave_lane == 0u)
                scores[token_lane * q_per_kv + qh_wave] = score * args.scale;
        }
        __syncthreads();
        for (uint32_t qh = 0u; qh < q_per_kv; ++qh) {
            const float score = scores[token_lane * q_per_kv + qh];
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
    for (uint32_t qh = 0u; qh < q_per_kv; ++qh) {
        lane_accumulators[(token_lane * q_per_kv + qh) * 256u + dimension] =
            accumulator[qh];
        if (dimension == 0u) {
            lane_max[token_lane * q_per_kv + qh] = run_max[qh];
            lane_sum_values[token_lane * q_per_kv + qh] = run_sum[qh];
        }
    }
    __syncthreads();
    if (token_lane != 0u) return;
    for (uint32_t qh = 0u; qh < q_per_kv; ++qh) {
        float combined_max = -FLT_MAX;
        for (uint32_t lane = 0u; lane < kOctaveTokenLanes; ++lane)
            combined_max = fmaxf(combined_max, lane_max[lane * q_per_kv + qh]);
        float combined_sum = 0.0f;
        float combined_accumulator = 0.0f;
        for (uint32_t lane = 0u; lane < kOctaveTokenLanes; ++lane) {
            const float lane_weight = __expf(
                lane_max[lane * q_per_kv + qh] - combined_max);
            combined_sum += lane_weight * lane_sum_values[lane * q_per_kv + qh];
            combined_accumulator += lane_weight *
                lane_accumulators[(lane * q_per_kv + qh) * 256u + dimension];
        }
        if (dimension == 0u) {
            base[qh] = combined_max;
            base[q_per_kv + qh] = combined_sum;
        }
        base[2u * q_per_kv + qh * 256u + dimension] = combined_accumulator;
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
    if (common.rows == 0u || common.head_dim != 256u || common.q_heads != 24u ||
        common.kv_heads != 4u || k_pool == nullptr || v_pool == nullptr ||
        levels == nullptr || partials == nullptr || splits == 0u ||
        splits > kOctaveMaxSplits ||
        !octave_format_valid(format))
        return hipErrorInvalidValue;
    const uint64_t grid = static_cast<uint64_t>(common.rows) * common.kv_heads * splits;
    if (grid > UINT32_MAX) return hipErrorInvalidValue;
    octave_split<<<static_cast<uint32_t>(grid), kOctaveThreads, 0u, stream>>>(
        common, k_pool, v_pool, levels, format, partials, splits);
    return hipGetLastError();
}

hipError_t launch_attention_paged_octave_reduce(
    const PagedAttentionCommonArgs& common, const float* partials,
    uint32_t splits, hipStream_t stream) {
    if (common.rows == 0u || common.head_dim != 256u || common.q_heads != 24u ||
        common.kv_heads != 4u || common.output == nullptr || partials == nullptr ||
        splits == 0u || splits > kOctaveMaxSplits)
        return hipErrorInvalidValue;
    const uint64_t grid = static_cast<uint64_t>(common.rows) * common.q_heads;
    if (grid > UINT32_MAX) return hipErrorInvalidValue;
    octave_reduce<<<static_cast<uint32_t>(grid), 256u, 0u, stream>>>(
        common, partials, splits);
    return hipGetLastError();
}

}  // namespace shelf::kernel
