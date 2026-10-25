// vendored from PhaseShift (own work); namespace/type adapted only.
#pragma once

#include <shelf/backend/hip/kernels/attention/kv_types.h>
#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk {

using shelf::bf16_t;

struct KvAppendCommonArgs {
    const bf16_t* k_input = nullptr;
    const bf16_t* v_input = nullptr;

    uint32_t k_row_stride = 0;
    uint32_t v_row_stride = 0;

    const uint32_t* row_sequence_slots = nullptr;
    const uint32_t* row_positions = nullptr;

    const uint32_t* block_tables = nullptr;

    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;

    uint32_t rows = 0;

    uint32_t layer = 0;

    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;

    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;

    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
};

inline constexpr const char* kKvAppendBf16Symbol =
    "phaseshift_kv_append_bf16";

struct KvAppendBf16Args {
    KvAppendCommonArgs common{};
    bf16_t* k_pool = nullptr;
    bf16_t* v_pool = nullptr;
};

static_assert(sizeof(KvAppendCommonArgs) == 96);
static_assert(alignof(KvAppendCommonArgs) == 8);
static_assert(offsetof(KvAppendCommonArgs, k_input) == 0);
static_assert(offsetof(KvAppendCommonArgs, v_input) == 8);
static_assert(offsetof(KvAppendCommonArgs, k_row_stride) == 16);
static_assert(offsetof(KvAppendCommonArgs, v_row_stride) == 20);
static_assert(offsetof(KvAppendCommonArgs, row_sequence_slots) == 24);
static_assert(offsetof(KvAppendCommonArgs, row_positions) == 32);
static_assert(offsetof(KvAppendCommonArgs, block_tables) == 40);
static_assert(offsetof(KvAppendCommonArgs, block_table_stride) == 48);
static_assert(offsetof(KvAppendCommonArgs, max_sequences) == 52);
static_assert(offsetof(KvAppendCommonArgs, rows) == 56);
static_assert(offsetof(KvAppendCommonArgs, layer) == 60);
static_assert(offsetof(KvAppendCommonArgs, page_tokens) == 64);
static_assert(offsetof(KvAppendCommonArgs, num_pages) == 68);
static_assert(offsetof(KvAppendCommonArgs, num_attention_layers) == 72);
static_assert(offsetof(KvAppendCommonArgs, kv_heads) == 76);
static_assert(offsetof(KvAppendCommonArgs, head_dim) == 80);
static_assert(offsetof(KvAppendCommonArgs, elems_per_token) == 84);
static_assert(offsetof(KvAppendCommonArgs, elems_per_page) == 88);
static_assert(offsetof(KvAppendCommonArgs, elems_per_layer) == 92);
static_assert(sizeof(KvAppendBf16Args) == 112);
static_assert(alignof(KvAppendBf16Args) == 8);
static_assert(offsetof(KvAppendBf16Args, k_pool) == 96);
static_assert(offsetof(KvAppendBf16Args, v_pool) == 104);

hipError_t
launch_kv_append_bf16(
    const KvAppendCommonArgs& common,

    bf16_t* k_pool,
    bf16_t* v_pool,

    hipStream_t stream);

hipError_t
launch_kv_append_fp8_e4m3(
    const KvAppendCommonArgs& common,

    ::shelf::hipk::fp8e4m3_storage_t* k_pool,
    ::shelf::hipk::fp8e4m3_storage_t* v_pool,

    float* k_scale_pool,
    float* v_scale_pool,

    hipStream_t stream);

}  // namespace shelf::hipk
