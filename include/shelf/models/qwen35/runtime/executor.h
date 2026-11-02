#pragma once
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen35/model/tensor_parallel_context.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/runtime/execution_types.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/runtime/gdn_spec_history.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/program/program.h>
#include <shelf/runtime/staging_layout.h>
#include <shelf/runtime/tp/tp_execution.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <shelf/models/qwen35/runtime/decode_backend.h>
#include <shelf/runtime/stream_bridge.h>

namespace ps {
namespace qwen35 {


constexpr uint32_t kMaxTargetHiddenTaps = 8;

enum class DecodeStatus : uint8_t {
    Pending = 0,
    Complete = 1,
    Error = 2,
};

struct ExecutorConfig {
    uint32_t max_scheduled_tokens = 0;
    uint32_t max_scheduled_requests = 0;
    uint32_t max_scheduled_output_rows = 0;
    std::array<uint32_t, kMaxTargetHiddenTaps> target_hidden_taps{};
    uint32_t target_hidden_tap_count = 0;
    uint32_t tp_combine_f32 = 0;
    runtime::DecodeBackend backend = runtime::DecodeBackend::Hip;
    const Qwen35TensorParallelContext* tp = nullptr;
    EffectiveRopeProfile rope_profile;
};

using ::ps::runtime::DispatchStagingPool;
using ::ps::runtime::ProgramStagingMeta;

struct GraphCacheSlot {
    hipGraphExec_t exec = nullptr;
    uint32_t token_count = 0;
    uint32_t num_outputs = 0;
    uint32_t num_sampled = 0;
    uint32_t num_stochastic = 0;
    uint32_t attention_plan_key = 0;
    uint8_t exec_role = 0;
    uint8_t has_spec_history = 0;
    uint32_t capture_rows = 0;
    const void* gdn_conv_history = nullptr;
    const void* gdn_recurrent_history = nullptr;
    const void* token_device_src = nullptr;
    bool valid = false;
};

struct Executor {
    Executor();
    ~Executor() noexcept;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    Executor(Executor&& other) noexcept;
    Executor& operator=(Executor&& other) noexcept;
    bool is_shutdown() const noexcept;
    const Qwen35Model* model = nullptr;

    SequenceSlotPool* sequence_pool = nullptr;
    GdnStatePool* gdn_state_pool = nullptr;
    PagedKVPool* kv_pool = nullptr;

    ExecutorConfig config;

    ::ps::runtime::DeviceBatchContextStorage batch_context_storage;
    ::ps::runtime::DeviceBatchContext* batch_context = nullptr;
    runtime::ValueTraceSink* value_trace = nullptr;
    void* host_request_staging = nullptr;
    size_t host_staging_bytes = 0;
    void* host_pending_staging = nullptr;
    size_t host_pending_staging_bytes = 0;
    void* host_sampled_staging = nullptr;
    size_t host_sampled_staging_bytes = 0;
    void* host_status_staging = nullptr;
    uint64_t next_submission_id = 0;
    uint64_t active_submission_id = 0;

    ::ps::runtime::ProgramSet program_set;
    bool program_set_ready = false;
    std::vector<::ps::runtime::TpExecutionSchedule> tp_schedules;
    ::ps::runtime::TpBarrierHook* tp_barrier = nullptr;
    int tp_rank = 0;
    gpu::Tensor execution_status;
    gpu::Tensor execution_workspace;
    ::ps::runtime::WeightSlot* host_weight_table = nullptr;
    ::ps::runtime::StaticParameterSlot* host_parameter_table = nullptr;
    std::vector<::ps::runtime::StaticParameterSlot> host_parameter_slots;
    ::ps::kernel::ModelDispatchStateView host_model_state{};
    void* dispatch_staging = nullptr;
    uint64_t dispatch_staging_bytes = 0;
    void* host_dispatch_staging = nullptr;
    DispatchStagingPool staging_pool;
    std::vector<ProgramStagingMeta> program_staging_meta;
    std::array<GraphCacheSlot, ::ps::runtime::ProgramSet::kRowBucketCount> graph_cache{};
    void* host_token_graph_staging = nullptr;
    size_t host_token_graph_staging_bytes = 0;
    ::ps::runtime::DeviceBatchContext* host_ctx_graph_staging = nullptr;
    bool graph_failed = false;

    gpu::Tensor input_ids;

    gpu::Tensor rope_inv_freq;

    gpu::Tensor output_hidden;
    gpu::Tensor logits;
    gpu::Tensor sampled_tokens;
    gpu::Tensor token_hidden;
    std::array<gpu::Tensor, kMaxTargetHiddenTaps> dflash_target_hidden;
    gpu::Tensor stochastic_topk_ids;
    gpu::Tensor stochastic_topk_logits;
    gpu::Tensor stochastic_topk_scratch;
    gpu::Tensor stochastic_topk_active_counts;
    uint32_t stochastic_topk_partitions = 0;
    std::size_t stochastic_topk_scratch_bytes = 0;

    std::vector<std::uint32_t> layer_override_value_ids;
    std::vector<std::uint32_t> layer_override_rows;
    std::vector<std::uint32_t> layer_override_hidden;
    std::vector<std::vector<std::uint16_t>> layer_override_host;

    void* keepalive = nullptr;

    ::ps::runtime::StreamSignal completion_signal{};
};

Result<Executor> create_model_executor(
    const Qwen35Model& model,
    SequenceSlotPool& sequence_pool,
    GdnStatePool& gdn_state_pool,
    PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const ExecutorConfig& config);

DecodeStatus poll_decode(const Executor& executor, uint64_t generation) noexcept;

Status executor_shutdown(Executor& ex) noexcept;

struct BatchExecutionOutput {
    uint32_t num_outputs = 0;
    gpu::Tensor final_hidden;
    gpu::Tensor logits;
    gpu::Tensor sampled_tokens;
    uint32_t num_token_rows = 0;
    gpu::Tensor token_hidden;
};

struct PendingRequestCommit {
    PagedSequenceState* sequence = nullptr;
    ::ps::runtime::RequestHandle handle{};
    uint32_t committed_tokens = 0;
};

struct PendingBatch {
    hipEvent_t done = nullptr;
    BatchExecutionOutput output;
    const PendingRequestCommit* commits = nullptr;
    uint32_t commit_count = 0;
    uint64_t submission_id = 0;
    uint64_t generation = 0;
    PendingBatch() = default;
    ~PendingBatch() noexcept;
    PendingBatch(const PendingBatch&) = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;
    PendingBatch(PendingBatch&& other) noexcept;
    PendingBatch& operator=(PendingBatch&& other) noexcept;
    void destroy() noexcept;
    bool has_event() const noexcept {
        return done != nullptr;
    }
};

struct ExecuteBatchOptions {
    runtime::GdnSpecHistoryDeviceView gdn_spec_history{};
    runtime::GdnCompactLogDeviceView gdn_compact{};
    const uint32_t* tree_anc_offsets = nullptr;
    const uint32_t* tree_anc_positions = nullptr;
    uint32_t tree_committed_end = 0u;
    const uint32_t* tree_committed_end_rows = nullptr;
    const uint32_t* rope_positions = nullptr;
    const int32_t* tree_row_parent = nullptr;
};

Result<PendingBatch> submit_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream);

Result<PendingBatch> submit_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream,
    const ExecuteBatchOptions& options);

Result<BatchExecutionOutput> complete_batch(
    Executor& executor,
    PendingBatch&& pending,
    hipStream_t stream);

Result<BatchExecutionOutput> execute_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream);

Result<BatchExecutionOutput> execute_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream,
    const ExecuteBatchOptions& options);

}
}
