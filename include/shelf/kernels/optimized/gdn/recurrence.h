#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

struct GdnRecurrenceArgs {
    const float* q = nullptr;
    const float* k = nullptr;
    const bf16_t* v = nullptr;
    const bf16_t* a = nullptr;
    const bf16_t* b = nullptr;

    const bf16_t* dt_bias = nullptr;
    const bf16_t* a_log = nullptr;

    fp16_t* recurrent_state = nullptr;
    float* output = nullptr;

    const ::ps::runtime::DeviceRequestDescriptor* requests = nullptr;

    uint32_t num_requests = 0;
    uint32_t actual_rows = 0;

    uint32_t q_row_stride = 0;
    uint32_t k_row_stride = 0;
    uint32_t v_row_stride = 0;
    uint32_t a_row_stride = 0;
    uint32_t b_row_stride = 0;
    uint32_t output_row_stride = 0;

    uint32_t state_index = 0;
    uint32_t max_sequences = 0;
    uint32_t num_gdn_states = 0;

    uint32_t key_heads = 0;
    uint32_t num_v_heads = 0;

    uint32_t head_k = 0;
    uint32_t head_v = 0;

    uint64_t recurrent_slot_stride = 0;
    uint64_t recurrent_layer_stride = 0;

    fp16_t* recurrent_history = nullptr;
    uint64_t recurrent_history_row_stride = 0;
    uint32_t capture_rows = 0;

    int32_t row_override = -1;
    uint32_t capture_verify_only = 0;
    const int32_t* row_parent = nullptr;

    float* compact_delta = nullptr;
    float* compact_k = nullptr;
    float* compact_a = nullptr;
    uint64_t compact_delta_layer_stride = 0;
    uint64_t compact_k_layer_stride = 0;
    uint64_t compact_a_layer_stride = 0;
    uint32_t compact_rows = 0;
    const int32_t* compact_row_indices = nullptr;
    uint32_t compact_row_capacity = 0;
};

hipError_t
launch_gdn_recurrence_f32(
    const GdnRecurrenceArgs& args,
    hipStream_t stream);

hipError_t
launch_gdn_recurrence_f32_wmma(
    const GdnRecurrenceArgs& args,
    hipStream_t stream);

hipError_t
launch_gdn_recurrence_f32_wmma_serial(
    const GdnRecurrenceArgs& args,
    hipStream_t stream);

inline constexpr const char* kGdnRecurrenceWmmaLossySymbol =
    "shelf_gdn_recurrence_wmma_lossy";
inline constexpr const char* kGdnRecurrenceWmmaExactSymbol =
    "shelf_gdn_recurrence_wmma_exact";

inline constexpr const char* kGdnRecurrenceDecodeRowsExactR2Symbol =
    "shelf_gdn_recurrence_decode_rows_exact_r2";
inline constexpr const char* kGdnRecurrenceDecodeRowsExactR4Symbol =
    "shelf_gdn_recurrence_decode_rows_exact_r4";
inline constexpr const char* kGdnRecurrenceDecodeRowsExactR8Symbol =
    "shelf_gdn_recurrence_decode_rows_exact_r8";

inline constexpr const char* kGdnRecurrenceWmmaSerialLossySymbol =
    "shelf_gdn_recurrence_wmma_serial_lossy";
inline constexpr const char* kGdnRecurrenceWmmaSerialExactSymbol =
    "shelf_gdn_recurrence_wmma_serial_exact";

inline constexpr uint32_t kGdnRecurrenceWmmaThreads = 256u;
inline constexpr uint32_t kGdnRecurrenceWmmaK = 128u;
inline constexpr uint32_t kGdnRecurrenceWmmaVGroup = 64u;

constexpr uint32_t gdn_recurrence_wmma_grid_x(uint32_t num_requests,
                                              uint32_t num_v_heads,
                                              uint32_t head_v) {
    if (num_requests == 0u || num_v_heads == 0u || head_v < kGdnRecurrenceWmmaVGroup)
        return 0u;
    return num_requests * num_v_heads * (head_v / kGdnRecurrenceWmmaVGroup);
}

bool gdn_recurrence_lossy_enabled();

bool gdn_recurrence_decode_rows_supported(const GdnRecurrenceArgs& args);

hipError_t
launch_gdn_recurrence_f32_wmma_decode_rows_exact(
    const GdnRecurrenceArgs& args,
    hipStream_t stream);

bool gdn_recurrence_commit_supported(const GdnRecurrenceArgs& args);

hipError_t
launch_gdn_recurrence_f32_wmma_commit(
    const GdnRecurrenceArgs& args,
    hipStream_t stream);

}  // namespace ps::kernel
