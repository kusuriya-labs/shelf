#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

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

}  // namespace shelf::kernel
