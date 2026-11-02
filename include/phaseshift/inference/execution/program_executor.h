#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/runtime/staging_layout.h>
#include <phaseshift/runtime/tp/tp_execution.h>
#include <phaseshift/inference/execution/spec_decode_context.h>
#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ps::execution {

constexpr uint64_t kDecodeAttnPartialBytes = 32u * 1024u * 1024u;

inline constexpr uint32_t kHostExternalValueCapacity = 16u;

struct HostResolvedValue {
    void* ptr = nullptr;
    uint32_t row_stride = 0;
    uint32_t feature_count = 0;
    ::ps::runtime::ValueDType dtype = ::ps::runtime::ValueDType::BF16;
    ::ps::runtime::ValueRowDomain row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
};

struct ValueTraceSink {
    std::function<void(uint32_t value_id, const void* device_ptr, uint32_t row_stride,
                       uint32_t feature_count, ::ps::runtime::ValueDType dtype,
                       uint32_t rows, hipStream_t stream)> on_value = nullptr;
};

struct LayerInputOverride {
    ::ps::runtime::ValueId value_id{};
    const std::uint16_t* host_bf16 = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t hidden = 0;
};

struct HostExecutionContext {
    ::ps::runtime::DeviceBatchContext* batch_context = nullptr;    ::ps::kernel::ModelDispatchStateView model_state{};
    const ::ps::runtime::WeightSlot* weights = nullptr;
    uint32_t weight_count = 0;
    const ::ps::runtime::StaticParameterSlot* parameters = nullptr;
    uint32_t parameter_count = 0;
    const ::ps::runtime::StaticParameterSlot* host_parameters = nullptr;
    uint32_t host_parameter_count = 0;
    const uint32_t* row_positions = nullptr;
    const uint32_t* rope_positions = nullptr;
    const double* rope_inv_freq = nullptr;
    const uint32_t* row_sequence_slots = nullptr;
    ::ps::execution::SpecDecodeContext spec{};
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
    uint32_t* error_word = nullptr;
    uint32_t bucket_m = 0;
    void* staging = nullptr;
    uint64_t staging_bytes = 0;
    void* host_staging = nullptr;
    uint32_t actual_rows = 0;
    uint32_t actual_outputs = 0;
    uint32_t actual_sampled_outputs = 0;
    uint32_t actual_stochastic_outputs = 0;
    const ::ps::runtime::DeviceSamplingParams* output_sampling_params = nullptr;
    const ::ps::runtime::DeviceRequestDescriptor* device_requests = nullptr;
    uint32_t* device_output_rows = nullptr;
    bool stochastic_topk_eligible = false;
    uint32_t stochastic_top_k = 0;
    int32_t* stochastic_topk_ids = nullptr;
    float* stochastic_topk_logits = nullptr;
    void* stochastic_topk_scratch = nullptr;
    uint32_t stochastic_topk_partitions = 0;
    uint32_t stochastic_topk_scratch_bytes = 0;
    uint32_t* stochastic_topk_active_counts = nullptr;
    ::ps::runtime::DispatchStagingPool* staging_pool = nullptr;
    const ::ps::runtime::ProgramStagingMeta* program_meta = nullptr;
    ValueTraceSink* value_trace = nullptr;
    ::ps::runtime::ExecutionRole role = ::ps::runtime::ExecutionRole::Decode;
    const ::ps::runtime::DeviceRequestDescriptor* staged_requests = nullptr;
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
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream);

Status execute_program_range(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream,
    size_t begin,
    size_t end);

Status execute_program_tp_ranges(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream,
    const ::ps::runtime::TpExecutionSchedule& schedule,
    ::ps::runtime::TpBarrierHook* barrier,
    int rank);

Result<HostResolvedValue> resolve_host_value(
    const ::ps::runtime::Program& program,
    ::ps::runtime::ValueId id,
    const HostExecutionContext& ctx);

Status upload_host_weight_table(
    const ::ps::runtime::WeightTableView& weights,
    ::ps::runtime::WeightSlot** out,
    hipStream_t stream);

Status upload_host_parameter_table(
    const ::ps::runtime::StaticParameterTableView& params,
    ::ps::runtime::StaticParameterSlot** out,
    hipStream_t stream);

}
