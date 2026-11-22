#include <shelf/kernels/fused/kv_append_octave.h>

#include <shelf/core/memory/types.h>
#include <shelf/kernels/kv_types.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel {
namespace {
namespace op_kv_append_octave {

using bf16 = __hip_bfloat16;

__device__ __forceinline__ float sign_of(uint32_t index, uint32_t salt) {
    uint32_t x = index * 2654435761u + salt * 40503u + 12345u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x & 1u) != 0u ? -1.0f : 1.0f;
}

__device__ __forceinline__ float segmented_sum(float value, float* shared,
                                                uint32_t segment) {
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t half = segment >> 1u; half >= 32u; half >>= 1u) {
        shared[tid] = value;
        __syncthreads();
        if ((tid & (segment - 1u)) < half) value += shared[tid + half];
        __syncthreads();
    }
    for (uint32_t half = 16u; half != 0u; half >>= 1u)
        value += __shfl_xor(value, half);
    if ((tid & (segment - 1u)) == 0u) shared[tid] = value;
    __syncthreads();
    const float result = shared[tid & ~(segment - 1u)];
    __syncthreads();
    return result;
}

__device__ __forceinline__ void wht(float& value, float* shared,
                                    uint32_t segment) {
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t half = 1u; half < 32u; half <<= 1u) {
        const float other = __shfl_xor(value, half);
        value = (tid & half) != 0u ? other - value : value + other;
    }
    for (uint32_t half = 32u; half < segment; half <<= 1u) {
        shared[tid] = value;
        __syncthreads();
        const float other = shared[tid ^ half];
        __syncthreads();
        value = (tid & half) != 0u ? other - value : value + other;
    }
    value *= rsqrtf(static_cast<float>(segment));
}

__device__ __forceinline__ uint32_t nearest_code(float value, const float* levels,
                                                 uint32_t bits) {
    const uint32_t count = 1u << bits;
    const float* table = levels + bits * 64u;
    uint32_t lower = 0u;
    for (uint32_t step = count >> 1u; step != 0u; step >>= 1u) {
        const uint32_t mid = lower + step;
        if (table[mid] < value) lower = mid;
    }
    const uint32_t upper = min(lower + 1u, count - 1u);
    return fabsf(value - table[upper]) < fabsf(value - table[lower]) ? upper : lower;
}

__device__ __forceinline__ void pack_plane(const uint8_t* codes, uint8_t* dst,
                                           uint32_t dimensions, uint32_t shift,
                                           uint32_t plane_bits) {
    const uint32_t fields = 32u / plane_bits;
    const uint32_t words = dimensions / fields;
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x) & (dimensions - 1u);
    if (tid < words) {
        const uint32_t mask = (1u << plane_bits) - 1u;
        uint32_t word = 0u;
        for (uint32_t field = 0u; field < fields; ++field) {
            const uint32_t code = codes[tid * fields + field];
            word |= ((code >> shift) & mask) << (field * plane_bits);
        }
        uint8_t* bytes = dst + tid * 4u;
        bytes[0] = static_cast<uint8_t>(word);
        bytes[1] = static_cast<uint8_t>(word >> 8u);
        bytes[2] = static_cast<uint8_t>(word >> 16u);
        bytes[3] = static_cast<uint8_t>(word >> 24u);
    }
}

__device__ __forceinline__ void pack_codes(const uint8_t* codes, uint8_t* dst,
                                           uint32_t dimensions, uint32_t bits) {
    if (bits >= 4u) {
        pack_plane(codes, dst, dimensions, 0u, 4u);
        dst += dimensions / 2u;
        if (bits == 5u) pack_plane(codes, dst, dimensions, 4u, 1u);
        if (bits == 6u) pack_plane(codes, dst, dimensions, 4u, 2u);
    } else {
        pack_plane(codes, dst, dimensions, 0u, 2u);
        dst += dimensions / 4u;
        pack_plane(codes, dst, dimensions, 2u, 1u);
    }
}

__device__ __forceinline__ void quantize_segment(float value, uint32_t segment,
                                                 uint32_t bits, uint32_t salt,
                                                 const float* levels, float* shared,
                                                 uint8_t* codes, __half* scales) {
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t local = tid & (segment - 1u);
    const uint32_t segment_index = tid / segment;
    float rotated = value * sign_of(tid, salt);
    wht(rotated, shared, segment);
    const float ss = segmented_sum(rotated * rotated, shared, segment);
    const float rms = sqrtf(ss / static_cast<float>(segment));
    const uint32_t code = rms > 0.0f ? nearest_code(rotated / rms, levels, bits) : 0u;
    codes[tid] = static_cast<uint8_t>(code);
    const float q = rms > 0.0f ? levels[bits * 64u + code] : 0.0f;
    const float dot = segmented_sum(rotated * q, shared, segment);
    if (local == 0u) scales[segment_index] = __float2half(dot > 0.0f ? ss / dot : 0.0f);
}

__global__ __launch_bounds__(256)
void impl(KvAppendCommonArgs args, uint8_t* k_pool, uint8_t* v_pool,
          const float* levels, OctaveKVCacheFormat format) {
    __shared__ float shared[256];
    __shared__ uint8_t codes[256];
    __shared__ uint32_t page;
    __shared__ uint32_t offset;
    __shared__ uint32_t valid;
    const uint32_t row = blockIdx.x / args.kv_heads;
    const uint32_t head = blockIdx.x % args.kv_heads;
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    if (tid == 0u) {
        valid = 1u;
        const uint32_t slot = args.row_sequence_slots[row];
        if (slot >= args.max_sequences) {
            valid = 0u;
        } else {
            const uint32_t position = args.row_positions[row];
            const uint32_t logical = position / args.page_tokens;
            offset = position % args.page_tokens;
            if (logical >= args.block_table_stride) {
                valid = 0u;
            } else {
                page = args.block_tables[static_cast<size_t>(slot) * args.block_table_stride + logical];
                if (page >= args.num_pages) valid = 0u;
            }
        }
    }
    __syncthreads();
    if (valid == 0u) return;

    const uint32_t k_slot_bytes = octave_k_slot_bytes(format);
    const uint32_t v_slot_bytes = octave_v_slot_bytes(format);
    const uint64_t slot_index =
        ((static_cast<uint64_t>(args.layer) * args.num_pages + page) * args.page_tokens + offset) *
            args.kv_heads + head;
    uint8_t* k_dst = k_pool + slot_index * k_slot_bytes;
    uint8_t* v_dst = v_pool + slot_index * v_slot_bytes;
    const bf16* k_src = reinterpret_cast<const bf16*>(args.k_input) +
        static_cast<size_t>(row) * args.k_row_stride + static_cast<size_t>(head) * 256u;
    const bf16* v_src = reinterpret_cast<const bf16*>(args.v_input) +
        static_cast<size_t>(row) * args.v_row_stride + static_cast<size_t>(head) * 256u;

    const uint32_t k_segment = format.k_mode == OctaveKMode::Compact ? 256u : 64u;
    const uint32_t k_bits = format.k_mode == OctaveKMode::Compact || tid >= 64u
        ? format.k_nope_bits : format.k_rope_bits;
    quantize_segment(__bfloat162float(k_src[tid]), k_segment, k_bits, 11u,
                     levels, shared, codes,
                     reinterpret_cast<__half*>(k_dst + k_slot_bytes -
                         (format.k_mode == OctaveKMode::Compact ? 2u : 8u)));
    __syncthreads();
    if (format.k_mode == OctaveKMode::Compact) {
        pack_codes(codes, k_dst, 256u, format.k_nope_bits);
    } else {
        const uint32_t block = tid >> 6u;
        const uint32_t block_bits = block == 0u ? format.k_rope_bits : format.k_nope_bits;
        const uint32_t block_offset = block == 0u
            ? 0u
            : octave_plane_bytes(64u, format.k_rope_bits) +
                  (block - 1u) * octave_plane_bytes(64u, format.k_nope_bits);
        if ((tid & 63u) < 48u)
            pack_codes(codes + block * 64u, k_dst + block_offset, 64u, block_bits);
    }
    __syncthreads();

    quantize_segment(__bfloat162float(v_src[tid]), 256u, format.v_bits, 29u,
                     levels, shared, codes,
                     reinterpret_cast<__half*>(v_dst + v_slot_bytes - 2u));
    __syncthreads();
    pack_codes(codes, v_dst, 256u, format.v_bits);
}

}  // namespace op_kv_append_octave
}  // namespace

hipError_t launch_kv_append_octave(const KvAppendCommonArgs& common,
                                   uint8_t* k_pool, uint8_t* v_pool,
                                   const float* levels,
                                   OctaveKVCacheFormat format,
                                   hipStream_t stream) {
    if (common.rows == 0u || common.kv_heads == 0u || common.head_dim != 256u ||
        common.page_tokens == 0u || common.block_table_stride == 0u ||
        common.num_pages == 0u || common.k_input == nullptr || common.v_input == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        levels == nullptr || !octave_format_valid(format))
        return hipErrorInvalidValue;
    const uint64_t grid = static_cast<uint64_t>(common.rows) * common.kv_heads;
    op_kv_append_octave::impl<<<dim3(static_cast<uint32_t>(grid)), 256u, 0u, stream>>>(
        common, k_pool, v_pool, levels, format);
    return hipGetLastError();
}


}  // namespace shelf::kernel
