#pragma once
#include <shelf/runtime/execution/execution_types.h>
#include <shelf/runtime/request_handle.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace shelf::runtime {

struct DeviceSamplingParams {
    uint64_t seed = 0;
    uint64_t sample_index = 0;
    const uint32_t* token_bitmask = nullptr;
    float temperature = 0.0f;
    float top_p = 1.0f;
    float min_p = 0.0f;
    uint32_t top_k = 0;
    // Distance in uint32_t words between output-row masks; zero shares one mask.
    uint32_t token_bitmask_row_stride = 0;
    uint8_t mode = 0;
    uint8_t reserved[3] = {};
};

static_assert(sizeof(DeviceSamplingParams) == 48);
static_assert(alignof(DeviceSamplingParams) == 8);
static_assert(std::is_standard_layout_v<DeviceSamplingParams>);
static_assert(std::is_trivially_copyable_v<DeviceSamplingParams>);

struct DeviceRequestDescriptor {
    RequestHandle request_handle{};

    uint32_t row_begin = 0;
    uint32_t row_count = 0;
    uint32_t prefix_length = 0;
    uint32_t sequence_length = 0;
    uint32_t output_index = 0xFFFFFFFFu;

    ExecutionClass execution_class = ExecutionClass::DECODE;

    uint8_t compute_logits = 0;
    uint8_t output_count = 1;

    DeviceSamplingParams sampling{};
};

static_assert(sizeof(DeviceRequestDescriptor) == 80);
static_assert(alignof(DeviceRequestDescriptor) == 8);
static_assert(std::is_standard_layout_v<DeviceRequestDescriptor>);
static_assert(std::is_trivially_copyable_v<DeviceRequestDescriptor>);

struct DeviceBatchContext {
    uint32_t actual_rows = 0;
    uint32_t num_requests = 0;
    uint32_t num_outputs = 0;
    uint32_t num_decode_requests = 0;
    uint32_t num_verify_requests = 0;
    uint32_t num_prefill_requests = 0;
    const int32_t* token_ids = nullptr;
    const DeviceRequestDescriptor* requests = nullptr;
    uint32_t* row_sequence_slots = nullptr;
    uint32_t* row_positions = nullptr;
    const uint32_t* rope_positions = nullptr;
    const double* rope_inv_freq = nullptr;
    uint32_t rope_table_stride = 0;
    uint32_t rope_table_count = 0;
    uint32_t* output_rows = nullptr;
    DeviceSamplingParams* output_sampling_params = nullptr;
    hipDeviceptr_t device_ptr = nullptr;
};

struct DeviceBatchContextStorage {
    gpu::Tensor token_ids;
    gpu::Tensor requests;
    gpu::Tensor row_sequence_slots;
    gpu::Tensor row_positions;
    gpu::Tensor rope_positions;
    gpu::Tensor output_rows;
    gpu::Tensor output_sampling_params;
    gpu::Tensor device_struct;
    uint32_t max_requests = 0;
    uint32_t max_tokens = 0;
};

Result<DeviceBatchContextStorage> create_device_batch_context(gpu::GpuArena& arena, uint32_t max_requests, uint32_t max_tokens);
Status destroy_device_batch_context(DeviceBatchContextStorage& storage) noexcept;
Status launch_prepare_batch_descriptor(DeviceBatchContext* ctx, uint32_t max_requests, hipStream_t stream);

// A descriptor with no requests: every thread returns without touching the batch. Launching
// it once loads the kernel, so the first real batch is not the first launch of its module.
void warm_prepare_batch_descriptor(DeviceBatchContext* ctx, hipStream_t stream);

}
