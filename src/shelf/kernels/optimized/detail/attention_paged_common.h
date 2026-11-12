#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstddef>

namespace shelf::kernel {

namespace {

using bf16 = __hip_bfloat16;

struct PagedTokenAddress {
    uint64_t element_base;
    uint32_t page_id;
    uint32_t page_offset;
    bool valid;
};

// A null mask is the ordinary causal case. A block past the mask's covered range is not
// visible: the mask is the complete selection, not a prefix hint.
__device__ __forceinline__ bool paged_token_visible(const uint8_t* mask, uint32_t stride,
                                                    uint32_t blocks, uint32_t block_shift,
                                                    uint32_t row, uint32_t pos) {
    if (mask == nullptr) return true;
    const uint32_t block = pos >> block_shift;
    if (block >= blocks) return false;
    const size_t byte = static_cast<size_t>(row) * stride + (block >> 3u);
    return ((mask[byte] >> (block & 7u)) & 1u) != 0u;
}

__device__ __forceinline__ void paged_build_page_bits(
    const uint8_t* mask, uint32_t stride, uint32_t block_shift, uint32_t row0,
    uint32_t rows, uint32_t visible_max, uint32_t page_tokens, uint32_t page_count,
    uint32_t words, uint32_t* page_bits) {
    for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) page_bits[i] = 0u;
    __syncthreads();
    if (mask == nullptr || rows == 0u) return;
    const uint32_t visible_blocks = (visible_max + (1u << block_shift) - 1u) >> block_shift;
    const uint32_t mask_words = (visible_blocks + 31u) >> 5u;
    for (uint32_t r = 0u; r < rows; ++r) {
        const uint8_t* row_mask =
            mask + static_cast<size_t>(row0 + r) * stride;
        for (uint32_t wi = threadIdx.x; wi < mask_words; wi += blockDim.x) {
            uint32_t bits = reinterpret_cast<const uint32_t*>(row_mask)[wi];
            while (bits != 0u) {
                const uint32_t block =
                    wi * 32u + static_cast<uint32_t>(__ffs(bits) - 1);
                const uint32_t page = (block << block_shift) / page_tokens;
                if (page < page_count)
                    atomicOr(&page_bits[page >> 5u], 1u << (page & 31u));
                bits &= bits - 1u;
            }
        }
    }
    __syncthreads();
}

// Same as paged_build_page_bits, but the bits are indexed relative to page_begin and the
// scan only covers the mask blocks that intersect [token_begin, token_end). The split-KV
// octave prefill calls this once per KV block so the mask does not turn the loop into a
// scan over the whole context.
__device__ __forceinline__ void paged_build_page_bits_range(
    const uint8_t* mask, uint32_t stride, uint32_t mask_blocks, uint32_t block_shift,
    uint32_t row0, uint32_t rows, uint32_t token_begin, uint32_t token_end,
    uint32_t page_tokens, uint32_t page_begin, uint32_t page_count, uint32_t words,
    uint32_t* page_bits) {
    for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) page_bits[i] = 0u;
    __syncthreads();
    if (mask == nullptr || rows == 0u || token_end <= token_begin) return;
    const uint32_t block_begin = token_begin >> block_shift;
    uint32_t block_end = (token_end + (1u << block_shift) - 1u) >> block_shift;
    if (block_end > mask_blocks) block_end = mask_blocks;
    if (block_end <= block_begin) return;
    const uint32_t word_begin = block_begin >> 5u;
    const uint32_t word_end = (block_end + 31u) >> 5u;
    for (uint32_t r = 0u; r < rows; ++r) {
        const uint8_t* row_mask = mask + static_cast<size_t>(row0 + r) * stride;
        for (uint32_t wi = word_begin + threadIdx.x; wi < word_end; wi += blockDim.x) {
            uint32_t bits = reinterpret_cast<const uint32_t*>(row_mask)[wi];
            if (wi == word_begin && (block_begin & 31u) != 0u)
                bits &= ~((1u << (block_begin & 31u)) - 1u);
            if (wi + 1u == word_end && (block_end & 31u) != 0u)
                bits &= (1u << (block_end & 31u)) - 1u;
            while (bits != 0u) {
                const uint32_t block =
                    wi * 32u + static_cast<uint32_t>(__ffs(bits) - 1);
                const uint32_t page = (block << block_shift) / page_tokens;
                if (page >= page_begin && page - page_begin < page_count)
                    atomicOr(&page_bits[(page - page_begin) >> 5u],
                             1u << ((page - page_begin) & 31u));
                bits &= bits - 1u;
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ bool paged_page_selected(const uint32_t* page_bits,
                                                    uint32_t page_index) {
    return ((page_bits[page_index >> 5u] >> (page_index & 31u)) & 1u) != 0u;
}

__device__ __forceinline__ PagedTokenAddress
resolve_paged_token(
    uint32_t p,
    const uint32_t* block_table,
    uint32_t block_table_stride,
    uint32_t page_tokens,
    uint32_t num_pages,
    uint32_t layer,
    uint32_t elems_per_layer,
    uint32_t elems_per_page,
    uint32_t elems_per_token) {
    PagedTokenAddress r;
    r.element_base = 0;
    r.page_id = 0;
    r.page_offset = 0;
    r.valid = false;
    const uint32_t lb = p / page_tokens;
    const uint32_t off = p % page_tokens;
    if (lb >= block_table_stride) return r;
    const uint32_t page = block_table[lb];
    if (page >= num_pages) return r;
    r.page_id = page;
    r.page_offset = off;
    r.element_base =
        static_cast<uint64_t>(layer) * elems_per_layer +
        static_cast<uint64_t>(page) * elems_per_page +
        static_cast<uint64_t>(off) * elems_per_token;
    r.valid = true;
    return r;
}

}  // namespace

}  // namespace shelf::kernel
