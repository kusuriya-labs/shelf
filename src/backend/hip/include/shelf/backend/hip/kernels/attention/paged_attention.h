// vendored from PhaseShift (own work); namespace/type adapted only.
#pragma once

#include <shelf/backend/hip/kernels/attention/kv_types.h>
#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk {

using shelf::bf16_t;

constexpr uint32_t kPagedAttentionHeadLimit = 256u;
inline constexpr uint32_t kPagedAttentionBlock = 256u;
inline constexpr uint32_t kPagedAttentionSplitBlock = 128u;
inline constexpr uint32_t kPagedAttentionPrefillBlock = 256u;
inline constexpr uint32_t kPagedAttentionPrefillTileRows = 32u;

inline constexpr const char* kPagedAttentionBf16Symbol =
    "phaseshift_attention_paged_bf16";
inline constexpr const char* kPagedAttentionPrefillBf16Symbol =
    "phaseshift_attention_paged_prefill_bf16";
inline constexpr const char* kPagedAttentionSplitQ4FullSymbol =
    "phaseshift_attention_paged_bf16_split_q4_full";
inline constexpr const char* kPagedAttentionReduceQ4FullQh1Symbol =
    "phaseshift_attention_paged_bf16_reduce_q4_full_qh1";
inline constexpr const char* kPagedAttentionReduceQ4FullQh2Symbol =
    "phaseshift_attention_paged_bf16_reduce_q4_full_qh2";
inline constexpr const char* kPagedAttentionReduceQ4FullBaseSymbol =
    "phaseshift_attention_paged_bf16_reduce_q4_full_base";
inline constexpr const char* kPagedAttentionSplitOtherSymbol =
    "phaseshift_attention_paged_bf16_split_other";
inline constexpr const char* kPagedAttentionReduceOtherSymbol =
    "phaseshift_attention_paged_bf16_reduce_other";

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
};

struct PagedAttentionBf16Args {
    PagedAttentionCommonArgs common{};
    const bf16_t* k_pool = nullptr;
    const bf16_t* v_pool = nullptr;
};

static_assert(sizeof(PagedAttentionCommonArgs) == 144);
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
static_assert(sizeof(PagedAttentionBf16Args) == 160);
static_assert(alignof(PagedAttentionBf16Args) == 8);
static_assert(offsetof(PagedAttentionBf16Args, k_pool) == 144);
static_assert(offsetof(PagedAttentionBf16Args, v_pool) == 152);

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

static_assert(sizeof(PagedAttentionSplitArgs) == 176);
static_assert(alignof(PagedAttentionSplitArgs) == 8);
static_assert(offsetof(PagedAttentionSplitArgs, k_pool) == 144);
static_assert(offsetof(PagedAttentionSplitArgs, v_pool) == 152);
static_assert(offsetof(PagedAttentionSplitArgs, partials) == 160);
static_assert(offsetof(PagedAttentionSplitArgs, splits) == 168);
static_assert(sizeof(PagedAttentionSplitReduceArgs) == 160);
static_assert(alignof(PagedAttentionSplitReduceArgs) == 8);
static_assert(offsetof(PagedAttentionSplitReduceArgs, partials) == 144);
static_assert(offsetof(PagedAttentionSplitReduceArgs, splits) == 152);

struct PagedAttentionFp8SplitArgs {
    PagedAttentionCommonArgs common{};
    const ::shelf::hipk::fp8e4m3_storage_t* k_pool = nullptr;
    const ::shelf::hipk::fp8e4m3_storage_t* v_pool = nullptr;
    const float* k_scale_pool = nullptr;
    const float* v_scale_pool = nullptr;
    float* partials = nullptr;
    uint32_t splits = 0;
};

static_assert(sizeof(PagedAttentionFp8SplitArgs) == 192);
static_assert(alignof(PagedAttentionFp8SplitArgs) == 8);
static_assert(offsetof(PagedAttentionFp8SplitArgs, k_pool) == 144);
static_assert(offsetof(PagedAttentionFp8SplitArgs, v_pool) == 152);
static_assert(offsetof(PagedAttentionFp8SplitArgs, k_scale_pool) == 160);
static_assert(offsetof(PagedAttentionFp8SplitArgs, v_scale_pool) == 168);
static_assert(offsetof(PagedAttentionFp8SplitArgs, partials) == 176);
static_assert(offsetof(PagedAttentionFp8SplitArgs, splits) == 184);

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

    const ::shelf::hipk::fp8e4m3_storage_t* k_pool,
    const ::shelf::hipk::fp8e4m3_storage_t* v_pool,

    const float* k_scale_pool,
    const float* v_scale_pool,

    hipStream_t stream);

hipError_t
launch_attention_paged_fp8_e4m3(
    const PagedAttentionCommonArgs& common,

    const ::shelf::hipk::fp8e4m3_storage_t* k_pool,
    const ::shelf::hipk::fp8e4m3_storage_t* v_pool,

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

    const ::shelf::hipk::fp8e4m3_storage_t* k_pool,
    const ::shelf::hipk::fp8e4m3_storage_t* v_pool,

    const float* k_scale_pool,
    const float* v_scale_pool,

    float* partials,
    uint32_t splits,

    hipStream_t stream);

}  // namespace shelf::hipk
