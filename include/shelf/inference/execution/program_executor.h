#pragma once
#include <shelf/core/status.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/program/program.h>
#include <shelf/runtime/staging_layout.h>
#include <shelf/runtime/tp/tp_execution.h>
#include <shelf/inference/execution/spec_decode_context.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace shelf::execution {

constexpr uint64_t kDecodeAttnPartialBytes = 32u * 1024u * 1024u;

inline constexpr uint32_t kHostExternalValueCapacity = 16u;

struct HostResolvedValue {
    void* ptr = nullptr;
    uint32_t row_stride = 0;
    uint32_t feature_count = 0;
    ::shelf::runtime::ValueDType dtype = ::shelf::runtime::ValueDType::BF16;
    ::shelf::runtime::ValueRowDomain row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
};

struct ValueTraceSink {
    std::function<void(uint32_t value_id, const void* device_ptr, uint32_t row_stride,
                       uint32_t feature_count, ::shelf::runtime::ValueDType dtype,
                       uint32_t rows, hipStream_t stream)> on_value = nullptr;
};

struct LayerInputOverride {
    ::shelf::runtime::ValueId value_id{};
    const std::uint16_t* host_bf16 = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t hidden = 0;
};

struct HostExecutionContext {
    ::shelf::runtime::DeviceBatchContext* batch_context = nullptr;    ::shelf::kernel::ModelDispatchStateView model_state{};
    const ::shelf::runtime::WeightSlot* weights = nullptr;
    uint32_t weight_count = 0;
    const ::shelf::runtime::StaticParameterSlot* parameters = nullptr;
    uint32_t parameter_count = 0;
    const ::shelf::runtime::StaticParameterSlot* host_parameters = nullptr;
    uint32_t host_parameter_count = 0;
    const uint32_t* row_positions = nullptr;
    const uint32_t* rope_positions = nullptr;
    const double* rope_inv_freq = nullptr;
    uint32_t rope_table_stride = 0;
    uint32_t rope_table_count = 0;
    const uint32_t* row_sequence_slots = nullptr;
    ::shelf::execution::SpecDecodeContext spec{};
    const void* const* external_inputs = nullptr;
    uint32_t external_input_count = 0;
    void* const* external_outputs = nullptr;
    uint32_t external_output_count = 0;
    const void* external_input_storage[kHostExternalValueCapacity] = {};
    void* external_output_storage[kHostExternalValueCapacity] = {};
    uint8_t* workspace = nullptr;
    uint64_t workspace_bytes = 0;
    float* scratch = nullptr;
    uint32_t scratch_floats = 0;
    uint8_t* decode_attn_partials = nullptr;
    uint32_t decode_attn_partial_bytes = 0;
    void* exl3_xrot = nullptr;
    uint32_t exl3_xrot_bytes = 0;
    void* exl3_dense = nullptr;
    uint64_t exl3_dense_bytes = 0;
    void* exl3_accum = nullptr;
    uint64_t exl3_accum_bytes = 0;
    void* exl3_rowsum = nullptr;
    uint64_t exl3_rowsum_bytes = 0;
    float* qsa_scores = nullptr;
    float* qsa_topk_scores = nullptr;
    int32_t* qsa_topk_indices = nullptr;
    uint32_t* error_word = nullptr;
    // Row capacity the split-K choice is tuned for when this run is a speculative verify:
    // a verify's row count follows how many requests are active, and the split count
    // changes the K reduction order. 0 tunes on the actual row count.
    uint32_t splitk_tuning_rows = 0;
    uint32_t bucket_m = 0;
    void* staging = nullptr;
    uint64_t staging_bytes = 0;
    void* host_staging = nullptr;
    uint32_t actual_rows = 0;
    uint32_t actual_outputs = 0;
    uint32_t actual_sampled_outputs = 0;
    uint32_t actual_stochastic_outputs = 0;
    uint32_t num_requests = 0;
    uint32_t num_decode_requests = 0;
    uint32_t num_verify_requests = 0;
    uint32_t num_prefill_requests = 0;
    const ::shelf::runtime::DeviceSamplingParams* output_sampling_params = nullptr;
    const ::shelf::runtime::DeviceRequestDescriptor* device_requests = nullptr;
    uint32_t* device_output_rows = nullptr;
    bool stochastic_topk_eligible = false;
    uint32_t stochastic_top_k = 0;
    int32_t* stochastic_topk_ids = nullptr;
    float* stochastic_topk_logits = nullptr;
    void* stochastic_topk_scratch = nullptr;
    uint32_t stochastic_topk_partitions = 0;
    uint32_t stochastic_topk_scratch_bytes = 0;
    uint32_t* stochastic_topk_active_counts = nullptr;
    ::shelf::runtime::DispatchStagingPool* staging_pool = nullptr;
    const ::shelf::runtime::ProgramStagingMeta* program_meta = nullptr;
    ValueTraceSink* value_trace = nullptr;
    ::shelf::runtime::ExecutionRole role = ::shelf::runtime::ExecutionRole::Decode;
    const ::shelf::runtime::DeviceRequestDescriptor* staged_requests = nullptr;
    uint32_t staged_request_count = 0;
    uint32_t staged_verify_request_count = 0;
    uint32_t tp_peer_value = 0;
    void* tp_peer_out = nullptr;
    bool tp_peer_armed = false;
    bool tp_peer_written = false;
    const LayerInputOverride* layer_overrides = nullptr;
    uint32_t layer_override_count = 0;
    LayerInputOverride layer_override_storage[8] = {};
    std::uint8_t layer_override_injected[8] = {};
};

Status execute_program(
    const ::shelf::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream);

// Prepare the device state a program run starts from. The plan scratch of a grouped expert
// FFN is accumulated into by the router, so a run has to start from zeros; the scan then
// leaves the counters zero, which is what lets every layer of one run share a single scratch.
// The owner of a program run calls this once, before it chooses an execution path, so a
// multi-rank run cannot take a path that skips it.
Status prepare_program_run(
    const ::shelf::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream);

// Launches dispatches [begin, end) and the rank segments of a TP schedule. Neither prepares the
// run: the caller that starts the program does.
Status execute_program_range(
    const ::shelf::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream,
    size_t begin,
    size_t end);

Status execute_program_tp_ranges(
    const ::shelf::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream,
    const ::shelf::runtime::TpExecutionSchedule& schedule,
    ::shelf::runtime::TpBarrierHook* barrier,
    int rank);

Result<HostResolvedValue> resolve_host_value(
    const ::shelf::runtime::Program& program,
    ::shelf::runtime::ValueId id,
    const HostExecutionContext& ctx);

Status upload_host_weight_table(
    const ::shelf::runtime::WeightTableView& weights,
    ::shelf::runtime::WeightSlot** out,
    hipStream_t stream);

Status upload_host_parameter_table(
    const ::shelf::runtime::StaticParameterTableView& params,
    ::shelf::runtime::StaticParameterSlot** out,
    hipStream_t stream);

}
