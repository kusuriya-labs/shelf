#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/kernels/kv_types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::kernel {

constexpr uint32_t kPagedAttentionHeadLimit = 256u;
inline constexpr uint32_t kPagedAttentionBlock = 256u;
inline constexpr uint32_t kPagedAttentionSplitBlock = 128u;
inline constexpr uint32_t kPagedAttentionPrefillBlock = 256u;
inline constexpr uint32_t kPagedAttentionPrefillTileRows = 32u;

// The split-KV decode kernels are instantiated per (kv dtype, q_per_kv). A geometry outside
// the table runs the non-split decode kernel, which walks the whole visible range in one
// workgroup per (row, q_head). The predicates are per dtype because the split and reduce
// kernels are separate instantiations: planning splits for a dtype that has no instantiation
// makes the launch fail with hipErrorInvalidValue instead of falling back.
inline constexpr bool paged_attention_bf16_split_supports(uint32_t q_per_kv) {
    return q_per_kv == 4u || q_per_kv == 6u || q_per_kv == 8u;
}

inline constexpr bool paged_attention_fp8_split_supports(uint32_t q_per_kv) {
    return q_per_kv == 4u || q_per_kv == 6u || q_per_kv == 8u;
}

inline constexpr const char* kPagedAttentionBf16Symbol =
    "shelf_attention_paged_bf16";
inline constexpr const char* kPagedAttentionPrefillBf16Symbol =
    "shelf_attention_paged_prefill_bf16";
inline constexpr const char* kPagedAttentionSplitQ4FullSymbol =
    "shelf_attention_paged_bf16_split_q4_full";
inline constexpr const char* kPagedAttentionReduceQ4FullQh1Symbol =
    "shelf_attention_paged_bf16_reduce_q4_full_qh1";
inline constexpr const char* kPagedAttentionReduceQ4FullQh2Symbol =
    "shelf_attention_paged_bf16_reduce_q4_full_qh2";
inline constexpr const char* kPagedAttentionReduceQ4FullBaseSymbol =
    "shelf_attention_paged_bf16_reduce_q4_full_base";
inline constexpr const char* kPagedAttentionSplitOtherSymbol =
    "shelf_attention_paged_bf16_split_other";
inline constexpr const char* kPagedAttentionReduceOtherSymbol =
    "shelf_attention_paged_bf16_reduce_other";

uint32_t paged_attention_split_reduce_qh_per_wg();

enum class PagedAttentionOutputDType : uint8_t {
    BF16,
    F32,
};

struct PagedAttentionCommonArgs {
    const bf16_t* q = nullptr;

    void* output = nullptr;

    PagedAttentionOutputDType
        output_dtype = PagedAttentionOutputDType::BF16;

    uint32_t q_row_stride = 0;
    uint32_t output_row_stride = 0;

    const uint32_t* row_sequence_slots = nullptr;
    const uint32_t* row_positions = nullptr;

    const uint32_t* tree_anc_offsets = nullptr;
    const uint32_t* tree_anc_positions = nullptr;
    uint32_t tree_committed_end = 0u;
    const uint32_t* tree_committed_end_rows = nullptr;

    const uint32_t* block_tables = nullptr;

    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;

    uint32_t rows = 0;

    uint32_t layer = 0;

    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;

    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;

    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;

    float scale = 0.0f;

    // Optional per-query block mask: one bit per indexer block, in byte order with the
    // low bit of each byte covering the lower block index. A token is visible only when
    // its block's bit is set. `mask_stride` is the row stride in bytes, `mask_blocks` is
    // the number of covered blocks and `mask_block_shift` is log2 of the tokens per block.
    // When null, attention is causal over the row position.
    const uint8_t* block_mask = nullptr;
    uint32_t mask_stride = 0;
    uint32_t mask_blocks = 0;
    uint32_t mask_block_shift = 0;
};

struct PagedAttentionBf16Args {
    PagedAttentionCommonArgs common{};
    const bf16_t* k_pool = nullptr;
    const bf16_t* v_pool = nullptr;
};

static_assert(sizeof(PagedAttentionCommonArgs) == 168);
static_assert(alignof(PagedAttentionCommonArgs) == 8);
static_assert(offsetof(PagedAttentionCommonArgs, q) == 0);
static_assert(offsetof(PagedAttentionCommonArgs, output) == 8);
static_assert(offsetof(PagedAttentionCommonArgs, output_dtype) == 16);
static_assert(offsetof(PagedAttentionCommonArgs, q_row_stride) == 20);
static_assert(offsetof(PagedAttentionCommonArgs, output_row_stride) == 24);
static_assert(offsetof(PagedAttentionCommonArgs, row_sequence_slots) == 32);
static_assert(offsetof(PagedAttentionCommonArgs, row_positions) == 40);
static_assert(offsetof(PagedAttentionCommonArgs, tree_anc_offsets) == 48);
static_assert(offsetof(PagedAttentionCommonArgs, tree_anc_positions) == 56);
static_assert(offsetof(PagedAttentionCommonArgs, tree_committed_end) == 64);
static_assert(offsetof(PagedAttentionCommonArgs, block_tables) == 80);
static_assert(offsetof(PagedAttentionCommonArgs, block_table_stride) == 88);
static_assert(offsetof(PagedAttentionCommonArgs, max_sequences) == 92);
static_assert(offsetof(PagedAttentionCommonArgs, rows) == 96);
static_assert(offsetof(PagedAttentionCommonArgs, layer) == 100);
static_assert(offsetof(PagedAttentionCommonArgs, page_tokens) == 104);
static_assert(offsetof(PagedAttentionCommonArgs, num_pages) == 108);
static_assert(offsetof(PagedAttentionCommonArgs, num_attention_layers) == 112);
static_assert(offsetof(PagedAttentionCommonArgs, q_heads) == 116);
static_assert(offsetof(PagedAttentionCommonArgs, kv_heads) == 120);
static_assert(offsetof(PagedAttentionCommonArgs, head_dim) == 124);
static_assert(offsetof(PagedAttentionCommonArgs, elems_per_token) == 128);
static_assert(offsetof(PagedAttentionCommonArgs, elems_per_page) == 132);
static_assert(offsetof(PagedAttentionCommonArgs, elems_per_layer) == 136);
static_assert(offsetof(PagedAttentionCommonArgs, scale) == 140);
static_assert(offsetof(PagedAttentionCommonArgs, block_mask) == 144);
static_assert(offsetof(PagedAttentionCommonArgs, mask_stride) == 152);
static_assert(offsetof(PagedAttentionCommonArgs, mask_blocks) == 156);
static_assert(offsetof(PagedAttentionCommonArgs, mask_block_shift) == 160);
static_assert(sizeof(PagedAttentionBf16Args) == 184);
static_assert(alignof(PagedAttentionBf16Args) == 8);
static_assert(offsetof(PagedAttentionBf16Args, k_pool) == 168);
static_assert(offsetof(PagedAttentionBf16Args, v_pool) == 176);

struct PagedAttentionSplitArgs {
    PagedAttentionCommonArgs common{};
    const bf16_t* k_pool = nullptr;
    const bf16_t* v_pool = nullptr;
    float* partials = nullptr;
    uint32_t splits = 0;
};

struct PagedAttentionSplitReduceArgs {
    PagedAttentionCommonArgs common{};
    const float* partials = nullptr;
    uint32_t splits = 0;
};

static_assert(sizeof(PagedAttentionSplitArgs) == 200);
static_assert(alignof(PagedAttentionSplitArgs) == 8);
static_assert(offsetof(PagedAttentionSplitArgs, k_pool) == 168);
static_assert(offsetof(PagedAttentionSplitArgs, v_pool) == 176);
static_assert(offsetof(PagedAttentionSplitArgs, partials) == 184);
static_assert(offsetof(PagedAttentionSplitArgs, splits) == 192);
static_assert(sizeof(PagedAttentionSplitReduceArgs) == 184);
static_assert(alignof(PagedAttentionSplitReduceArgs) == 8);
static_assert(offsetof(PagedAttentionSplitReduceArgs, partials) == 168);
static_assert(offsetof(PagedAttentionSplitReduceArgs, splits) == 176);

struct PagedAttentionFp8SplitArgs {
    PagedAttentionCommonArgs common{};
    const ::shelf::kernel::fp8e4m3_storage_t* k_pool = nullptr;
    const ::shelf::kernel::fp8e4m3_storage_t* v_pool = nullptr;
    const float* k_scale_pool = nullptr;
    const float* v_scale_pool = nullptr;
    float* partials = nullptr;
    uint32_t splits = 0;
};

static_assert(sizeof(PagedAttentionFp8SplitArgs) == 216);
static_assert(alignof(PagedAttentionFp8SplitArgs) == 8);
static_assert(offsetof(PagedAttentionFp8SplitArgs, k_pool) == 168);
static_assert(offsetof(PagedAttentionFp8SplitArgs, v_pool) == 176);
static_assert(offsetof(PagedAttentionFp8SplitArgs, k_scale_pool) == 184);
static_assert(offsetof(PagedAttentionFp8SplitArgs, v_scale_pool) == 192);
static_assert(offsetof(PagedAttentionFp8SplitArgs, partials) == 200);
static_assert(offsetof(PagedAttentionFp8SplitArgs, splits) == 208);

hipError_t
launch_attention_paged_bf16(
    const PagedAttentionCommonArgs& common,

    const bf16_t* k_pool,
    const bf16_t* v_pool,

    hipStream_t stream);

hipError_t
launch_attention_paged_prefill_bf16(
    const PagedAttentionCommonArgs& common,

    const bf16_t* k_pool,
    const bf16_t* v_pool,

    hipStream_t stream);

hipError_t
launch_attention_paged_prefill_fp8_e4m3(
    const PagedAttentionCommonArgs& common,

    const ::shelf::kernel::fp8e4m3_storage_t* k_pool,
    const ::shelf::kernel::fp8e4m3_storage_t* v_pool,

    const float* k_scale_pool,
    const float* v_scale_pool,

    hipStream_t stream);

hipError_t
launch_attention_paged_fp8_e4m3(
    const PagedAttentionCommonArgs& common,

    const ::shelf::kernel::fp8e4m3_storage_t* k_pool,
    const ::shelf::kernel::fp8e4m3_storage_t* v_pool,

    const float* k_scale_pool,
    const float* v_scale_pool,

    hipStream_t stream);

hipError_t
launch_attention_paged_bf16_split(
    const PagedAttentionCommonArgs& common,

    const bf16_t* k_pool,
    const bf16_t* v_pool,

    float* partials,
    uint32_t splits,

    hipStream_t stream);

hipError_t
launch_attention_paged_bf16_split_reduce(
    const PagedAttentionCommonArgs& common,

    const float* partials,
    uint32_t splits,

    hipStream_t stream);

hipError_t
launch_attention_paged_fp8_split(
    const PagedAttentionCommonArgs& common,

    const ::shelf::kernel::fp8e4m3_storage_t* k_pool,
    const ::shelf::kernel::fp8e4m3_storage_t* v_pool,

    const float* k_scale_pool,
    const float* v_scale_pool,

    float* partials,
    uint32_t splits,

    hipStream_t stream);

}  // namespace shelf::kernel
