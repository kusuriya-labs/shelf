#pragma once

#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

struct GdnConv1dArgs {
    const bf16_t* input = nullptr;
    float* output = nullptr;
    const bf16_t* conv_weight = nullptr;
    bf16_t* conv_state = nullptr;

    const ::ps::runtime::DeviceRequestDescriptor* requests = nullptr;

    uint32_t num_requests = 0;
    uint32_t actual_rows = 0;
    uint32_t max_request_rows = 0;

    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;

    uint32_t conv_dim = 0;
    uint32_t conv_history = 3;
    uint32_t state_index = 0;
    uint32_t max_sequences = 0;

    uint64_t conv_slot_stride = 0;
    uint64_t conv_layer_stride = 0;
    uint64_t conv_history_stride = 0;

    bf16_t* conv_history_store = nullptr;
    uint64_t conv_history_row_stride = 0;
    uint32_t capture_rows = 0;
    uint32_t capture_verify_only = 0;
    const int32_t* row_parent = nullptr;
};

inline constexpr const char* kGdnConv1dSymbol =
    "phaseshift_gdn_conv1d_bf16_f32";

struct GdnConv1dAqlArgs {
    const void* input = nullptr;
    void* output = nullptr;
    const void* conv_weight = nullptr;
    void* conv_state = nullptr;
    const void* requests = nullptr;
    uint32_t actual_rows = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t conv_dim = 0;
    uint32_t state_index = 0;
    uint32_t max_sequences = 0;
    uint64_t conv_slot_stride = 0;
    uint64_t conv_layer_stride = 0;
    uint64_t conv_history_stride = 0;
    uint32_t row_tile = 0;
    void* conv_history_store = nullptr;
    uint64_t conv_history_row_stride = 0;
    uint32_t capture_rows = 0;
    uint32_t capture_verify_only = 0;
};

static_assert(sizeof(GdnConv1dAqlArgs) == 120);
static_assert(alignof(GdnConv1dAqlArgs) == 8);
static_assert(offsetof(GdnConv1dAqlArgs, input) == 0);
static_assert(offsetof(GdnConv1dAqlArgs, output) == 8);
static_assert(offsetof(GdnConv1dAqlArgs, conv_weight) == 16);
static_assert(offsetof(GdnConv1dAqlArgs, conv_state) == 24);
static_assert(offsetof(GdnConv1dAqlArgs, requests) == 32);
static_assert(offsetof(GdnConv1dAqlArgs, actual_rows) == 40);
static_assert(offsetof(GdnConv1dAqlArgs, input_row_stride) == 44);
static_assert(offsetof(GdnConv1dAqlArgs, output_row_stride) == 48);
static_assert(offsetof(GdnConv1dAqlArgs, conv_dim) == 52);
static_assert(offsetof(GdnConv1dAqlArgs, state_index) == 56);
static_assert(offsetof(GdnConv1dAqlArgs, max_sequences) == 60);
static_assert(offsetof(GdnConv1dAqlArgs, conv_slot_stride) == 64);
static_assert(offsetof(GdnConv1dAqlArgs, conv_layer_stride) == 72);
static_assert(offsetof(GdnConv1dAqlArgs, conv_history_stride) == 80);
static_assert(offsetof(GdnConv1dAqlArgs, row_tile) == 88);
static_assert(offsetof(GdnConv1dAqlArgs, conv_history_store) == 96);
static_assert(offsetof(GdnConv1dAqlArgs, conv_history_row_stride) == 104);
static_assert(offsetof(GdnConv1dAqlArgs, capture_rows) == 112);
static_assert(offsetof(GdnConv1dAqlArgs, capture_verify_only) == 116);

inline constexpr uint32_t kGdnConv1dThreads = 256u;

constexpr uint32_t gdn_conv1d_blocks_x(uint32_t conv_dim) {
    return (conv_dim + kGdnConv1dThreads - 1u) / kGdnConv1dThreads;
}

constexpr uint32_t gdn_conv1d_resolved_row_tile(uint32_t row_tile,
                                                uint32_t actual_rows) {
    return row_tile == 0u ? actual_rows : row_tile;
}

constexpr uint32_t gdn_conv1d_tiles(uint32_t row_tile, uint32_t tiled_rows) {
    return row_tile == 0u ? 1u : (tiled_rows + row_tile - 1u) / row_tile;
}

uint32_t gdn_conv1d_row_tile();

hipError_t launch_gdn_conv1d_bf16_f32(
    const GdnConv1dArgs& args,
    hipStream_t stream);

}  // namespace ps::kernel
