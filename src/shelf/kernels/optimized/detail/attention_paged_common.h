#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstddef>

namespace ps::kernel {

namespace {

using bf16 = __hip_bfloat16;

struct PagedTokenAddress {
    uint64_t element_base;
    uint32_t page_id;
    uint32_t page_offset;
    bool valid;
};

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

}  // namespace ps::kernel
