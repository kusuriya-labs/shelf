#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/kernels/kv_types.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

struct KvRelocateArgs {
    const uint32_t* src_positions = nullptr;
    const uint32_t* dst_positions = nullptr;

    uint32_t rows = 0;
    uint32_t slot = 0;

    const uint32_t* block_tables = nullptr;

    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;

    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;

    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;

    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
};

struct KvRelocateBf16Args {
    KvRelocateArgs common{};
    bf16_t* k_pool = nullptr;
    bf16_t* v_pool = nullptr;
};

struct KvRelocateFp8Args {
    KvRelocateArgs common{};
    ::ps::kernel::fp8e4m3_storage_t* k_pool = nullptr;
    ::ps::kernel::fp8e4m3_storage_t* v_pool = nullptr;
    float* k_scale_pool = nullptr;
    float* v_scale_pool = nullptr;
};

hipError_t launch_kv_relocate_bf16(const KvRelocateBf16Args& args, hipStream_t stream);

hipError_t launch_kv_relocate_fp8_e4m3(const KvRelocateFp8Args& args, hipStream_t stream);

}  // namespace ps::kernel
