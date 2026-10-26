// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/attention/kv_append.h.)
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/backend/sycl/kernels/attention/kv_types.h>
#include <shelf/backend/sycl/sycl_api.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

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

struct KvAppendBf16Args {
    KvAppendCommonArgs common{};
    bf16_t* k_pool = nullptr;
    bf16_t* v_pool = nullptr;
};

static_assert(sizeof(KvAppendCommonArgs) == 96);
static_assert(alignof(KvAppendCommonArgs) == 8);
static_assert(sizeof(KvAppendBf16Args) == 112);

::shelf::Status launch_kv_append_bf16(
    const KvAppendCommonArgs& common,
    bf16_t* k_pool,
    bf16_t* v_pool,
    const ::shelf::sycl::Stream& stream);

::shelf::Status launch_kv_append_fp8_e4m3(
    const KvAppendCommonArgs& common,
    ::shelf::syclk::fp8e4m3_storage_t* k_pool,
    ::shelf::syclk::fp8e4m3_storage_t* v_pool,
    float* k_scale_pool,
    float* v_scale_pool,
    const ::shelf::sycl::Stream& stream);

} // namespace shelf::syclk
