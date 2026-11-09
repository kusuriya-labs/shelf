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
    return static_cast<uint32_t>(source[0]) |
           (static_cast<uint32_t>(source[1]) << 8u) |
           (static_cast<uint32_t>(source[2]) << 16u) |
           (static_cast<uint32_t>(source[3]) << 24u);
}

__device__ __forceinline__ uint32_t unpack_plane(const uint8_t* source,
                                                 uint32_t index,
                                                 uint32_t plane_bits) {
    const uint32_t fields = 32u / plane_bits;
    const uint32_t word = load_word(source + (index / fields) * 4u);
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

__device__ __forceinline__ float block_sum(float value, float* shared) {
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    shared[tid] = value;
    for (uint32_t half = 128u; half != 0u; half >>= 1u) {
        __syncthreads();
        if (tid < half) shared[tid] += shared[tid + half];
    }
    __syncthreads();
    return shared[0];
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

__global__ __launch_bounds__(256)
void octave_split(PagedAttentionCommonArgs args, const uint8_t* k_pool,
                  const uint8_t* v_pool, const float* levels,
                  OctaveKVCacheFormat format, float* partials,
                  uint32_t splits) {
    __shared__ float q_shared[256];
    __shared__ float reduction[256];
    __shared__ uint32_t row_valid;
    __shared__ uint32_t visible;
    __shared__ uint32_t ancestor_begin;
    __shared__ uint32_t ancestor_count;
    __shared__ const uint32_t* block_table;
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t split = blockIdx.x % splits;
    const uint32_t task = blockIdx.x / splits;
    const uint32_t q_head = task % args.q_heads;
    const uint32_t row = task / args.q_heads;
    if (row >= args.rows) return;
    const uint32_t q_per_kv = args.q_heads / args.kv_heads;
    const uint32_t kv_head = q_head / q_per_kv;
    const uint32_t qh = q_head % q_per_kv;

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

    const bf16_t* q_source = args.q + static_cast<uint64_t>(row) * args.q_row_stride +
                              static_cast<uint64_t>(q_head) * 256u;
    float q_value = __uint_as_float(static_cast<uint32_t>(q_source[tid].data) << 16u) *
                    sign_of(tid, 11u);
    const uint32_t q_segment = format.k_mode == OctaveKMode::Compact ? 256u : 64u;
    wht(q_value, q_shared, q_segment);
    q_shared[tid] = q_value;
    __syncthreads();

    const uint32_t page_count = (visible + args.page_tokens - 1u) / args.page_tokens;
    const uint32_t pages_per_split = (page_count + splits - 1u) / splits;
    const uint32_t page_begin = split * pages_per_split;
    const uint32_t page_limit = page_begin + pages_per_split;
    const uint32_t page_end = page_count < page_limit ? page_count : page_limit;
    const uint32_t k_slot_bytes = octave_k_slot_bytes(format);
    const uint32_t v_slot_bytes = octave_v_slot_bytes(format);
    const uint64_t vectors_per_layer = static_cast<uint64_t>(args.num_pages) *
        args.page_tokens * args.kv_heads;
    float run_max = -FLT_MAX;
    float run_sum = 0.0f;
    float accumulator = 0.0f;

    for (uint32_t logical_page = page_begin; logical_page < page_end; ++logical_page) {
        const uint32_t page = block_table[logical_page];
        if (page >= args.num_pages) return;
        const uint32_t page_start = logical_page * args.page_tokens;
        const uint32_t remaining = visible - page_start;
        const uint32_t tokens = args.page_tokens < remaining ? args.page_tokens : remaining;
        for (uint32_t offset = 0u; offset < tokens; ++offset) {
            const uint32_t position = page_start + offset;
            if (!paged_token_visible(args.block_mask, args.mask_stride, args.mask_blocks,
                                     args.mask_block_shift, row, position))
                continue;
            const uint64_t vector = static_cast<uint64_t>(args.layer) * vectors_per_layer +
                (static_cast<uint64_t>(page) * args.page_tokens + offset) * args.kv_heads + kv_head;
            const uint8_t* k_source = k_pool + vector * k_slot_bytes;
            const uint8_t* v_source = v_pool + vector * v_slot_bytes;
            const float score = block_sum(
                q_shared[tid] * load_k_rotated(k_source, tid, levels, format), reduction) * args.scale;
            const float next_max = fmaxf(run_max, score);
            const float alpha = __expf(run_max - next_max);
            const float weight = __expf(score - next_max);
            run_sum = run_sum * alpha + weight;
            accumulator = accumulator * alpha +
                          weight * load_v_rotated(v_source, tid, levels, format);
            run_max = next_max;
        }
    }

    if (args.tree_anc_offsets != nullptr && split == 0u) {
        for (uint32_t ai = 0u; ai < ancestor_count; ++ai) {
            const uint32_t position = args.tree_anc_positions[ancestor_begin + ai];
            if (!paged_token_visible(args.block_mask, args.mask_stride, args.mask_blocks,
                                     args.mask_block_shift, row, position))
                continue;
            const uint32_t logical_page = position / args.page_tokens;
            const uint32_t offset = position % args.page_tokens;
            const uint32_t page = block_table[logical_page];
            if (page >= args.num_pages) return;
            const uint64_t vector = static_cast<uint64_t>(args.layer) * vectors_per_layer +
                (static_cast<uint64_t>(page) * args.page_tokens + offset) * args.kv_heads + kv_head;
            const uint8_t* k_source = k_pool + vector * k_slot_bytes;
            const uint8_t* v_source = v_pool + vector * v_slot_bytes;
            const float score = block_sum(
                q_shared[tid] * load_k_rotated(k_source, tid, levels, format), reduction) * args.scale;
            const float next_max = fmaxf(run_max, score);
            const float alpha = __expf(run_max - next_max);
            const float weight = __expf(score - next_max);
            run_sum = run_sum * alpha + weight;
            accumulator = accumulator * alpha +
                          weight * load_v_rotated(v_source, tid, levels, format);
            run_max = next_max;
        }
    }

    const uint64_t partial_stride = static_cast<uint64_t>(q_per_kv) * 258u;
    float* base = partials +
        ((static_cast<uint64_t>(row) * args.kv_heads + kv_head) * splits + split) * partial_stride;
    if (tid == 0u) {
        base[qh] = run_max;
        base[q_per_kv + qh] = run_sum;
    }
    base[2u * q_per_kv + qh * 256u + tid] = accumulator;
}

__global__ __launch_bounds__(256)
void octave_reduce(PagedAttentionCommonArgs args, const float* partials,
                   uint32_t splits) {
    __shared__ float shared[256];
    __shared__ float weights[16];
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
        levels == nullptr || partials == nullptr || splits == 0u || splits > 16u ||
        !octave_format_valid(format))
        return hipErrorInvalidValue;
    const uint64_t grid = static_cast<uint64_t>(common.rows) * common.q_heads * splits;
    if (grid > UINT32_MAX) return hipErrorInvalidValue;
    octave_split<<<static_cast<uint32_t>(grid), 256u, 0u, stream>>>(
        common, k_pool, v_pool, levels, format, partials, splits);
    return hipGetLastError();
}

hipError_t launch_attention_paged_octave_reduce(
    const PagedAttentionCommonArgs& common, const float* partials,
    uint32_t splits, hipStream_t stream) {
    if (common.rows == 0u || common.head_dim != 256u || common.q_heads != 24u ||
        common.kv_heads != 4u || common.output == nullptr || partials == nullptr ||
        splits == 0u || splits > 16u)
        return hipErrorInvalidValue;
    const uint64_t grid = static_cast<uint64_t>(common.rows) * common.q_heads;
    if (grid > UINT32_MAX) return hipErrorInvalidValue;
    octave_reduce<<<static_cast<uint32_t>(grid), 256u, 0u, stream>>>(
        common, partials, splits);
    return hipGetLastError();
}

}  // namespace shelf::kernel
