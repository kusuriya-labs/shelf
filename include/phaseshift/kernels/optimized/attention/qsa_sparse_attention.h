#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

enum class QsaSparseAttentionOutputDType : uint8_t {
    F32 = 0,
    BF16 = 1,
};

struct QsaSparseAttentionArgs {
    const bf16_t* q = nullptr;
    uint32_t q_row_stride = 0;

    void* output = nullptr;
    QsaSparseAttentionOutputDType output_dtype = QsaSparseAttentionOutputDType::F32;
    uint32_t output_row_stride = 0;

    const uint32_t* row_sequence_slots = nullptr;
    const uint32_t* row_positions = nullptr;

    const uint8_t* mask = nullptr;
    uint32_t mask_row_stride = 0;

    const uint32_t* block_tables = nullptr;
    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;

    uint32_t rows = 0;
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;

    uint32_t layer = 0;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;

    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;

    const bf16_t* k_pool = nullptr;
    const bf16_t* v_pool = nullptr;

    float scale = 0.0f;
    hipStream_t stream = nullptr;
};

hipError_t launch_qsa_sparse_attention_bf16(const QsaSparseAttentionArgs& args);

}
