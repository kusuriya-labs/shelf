#pragma once
#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/models/qwen35/model/tensor_parallel_context.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/runtime/execution_types.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/inference/state/qsa_state_pool.h>
#include <shelf/inference/state/ple_conv_state_pool.h>
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
#include <functional>
#include <memory>
#include <vector>
#include <shelf/models/qwen35/runtime/decode_backend.h>
#include <shelf/runtime/stream_bridge.h>

namespace shelf {
namespace qwen35 {


constexpr uint32_t kMaxTargetHiddenTaps = 8;

enum class DecodeStatus : uint8_t {
    Pending = 0,
    Complete = 1,
    Error = 2,
};

struct ExecutorConfig {
    uint32_t max_batch_rows = 0;
    uint32_t max_scheduled_requests = 0;
    uint32_t max_scheduled_output_rows = 0;
    std::array<uint32_t, kMaxTargetHiddenTaps> target_hidden_taps{};
    uint32_t target_hidden_tap_count = 0;
    uint32_t tp_combine_f32 = 0;
    // Maximum cached context the model-level graphs size their state from. Qwen4Exp uses
    // it for the QSA indexer capacity; the Qwen35 families ignore it.
    uint32_t max_context_tokens = 0;
    runtime::DecodeBackend backend = runtime::DecodeBackend::Hip;
    const Qwen35TensorParallelContext* tp = nullptr;
    EffectiveRopeProfile rope_profile;
    // Empty retains the legacy single-table profile and theta.
    std::vector<EffectiveRopeProfile> rope_profiles;
    PipelineStage stage;
    bool emit_normed_token_hidden = false;
    bool external_hidden_input = false;
    // 0 means the family's hidden size. Qwen4Exp's MTP block consumes the pre-mixer wide
    // residual (hc * hidden), so both its hidden input and its returned residual differ
    // from the token-row hidden.
    std::uint32_t external_hidden_width = 0;
    std::uint32_t normed_token_hidden_width = 0;
    // Largest rotated-domain linear this rank runs, used to size the EXL3 staging planes.
    // An empty shape keeps the model-independent upper bound.
    shelf::weights::Exl3LinearShape exl3_linear_shape;
};

using ::shelf::runtime::DispatchStagingPool;
using ::shelf::runtime::ProgramStagingMeta;

// What the runtime reads from a loaded model while it runs: the backbone geometry, and the
// embedding table the drafter looks tokens up in. Both are the same whatever a layer's FFN
// is, so the executor needs nothing else from the model's weights after lowering.
struct RuntimeModelView {
    const shelf::qwen_hybrid::QwenHybridConfig* config = nullptr;
    const shelf::weights::MatrixWeight* embed_tokens = nullptr;
};

// One layer's input replaced with host data (SHELF_LAYER_INPUT_OVERRIDE). It names a layer
// before the graph exists, so the value id is resolved from the lowered form later.
struct LayerOverrideSource {
    uint32_t layer = 0;
    uint32_t rows = 0;
    uint32_t hidden = 0;
    std::vector<std::uint16_t> host;
};

// Everything the executor needs to run a model: the family-neutral lowered form, the view of
// the loaded model it reads while running, and the layer input overrides. A model family
// lowers its own weights and hands the result here, so the executor names no family.
struct ExecutorModelInput {
    RuntimeModelView model;
    shelf::qwen_hybrid::HybridLoweredPrimitives lowered;
    std::vector<LayerOverrideSource> layer_overrides;

    // Qwen4Exp model-level state that is not part of the GDN / KV pools.
    bool uses_qsa_state = false;
    std::uint32_t qsa_layers = 0;
    std::uint32_t qsa_head_dim = 0;
    std::uint32_t qsa_max_blocks = 0;
    std::uint32_t qsa_compress_ratio = 0;
    bool uses_ple_conv_state = false;
    std::uint32_t ple_conv_layers = 0;
    std::uint32_t ple_conv_channels = 0;
};

// Prepares the extra external input (the PLE embedding rows) for one batch. The executor
// owns no PLE policy: the family that has PLE supplies this hook.
using ExternalInputPreparer =
    std::function<Status(const ScheduledBatch&, std::vector<bf16_t>&)>;

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
    uint64_t gdn_conv_row_stride = 0;
    uint64_t gdn_recurrent_row_stride = 0;
    ::shelf::execution::GdnCompactLogDeviceView gdn_compact{};
    const void* token_device_src = nullptr;
    bool valid = false;
};

// True when one program runs a linear carried in the rotated-domain trellis form. The
// executor's staging planes for that form are large, so they exist only for a program that
// uses them: everything else takes that arena space as KV pool.
bool program_needs_exl3_workspace(const ::shelf::runtime::Program& program) noexcept;

// True when any program of the set does. This is what sizes the execution workspace.
bool program_set_needs_exl3_workspace(const ::shelf::runtime::ProgramSet& programs) noexcept;

// True when one program runs the QSA selector, which keeps its score and top-k state in
// the execution workspace.
bool program_needs_qsa_workspace(const ::shelf::runtime::Program& program) noexcept;

bool program_set_needs_qsa_workspace(const ::shelf::runtime::ProgramSet& programs) noexcept;

struct Executor {
    Executor();
    ~Executor() noexcept;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    Executor(Executor&& other) noexcept;
    Executor& operator=(Executor&& other) noexcept;
    bool is_shutdown() const noexcept;
    RuntimeModelView model;

    SequenceSlotPool* sequence_pool = nullptr;
    GdnStatePool* gdn_state_pool = nullptr;
    PagedKVPool* kv_pool = nullptr;
    std::unique_ptr<::shelf::state::QsaKeyStatePool> qsa_state_pool;
    std::unique_ptr<::shelf::state::PleConvStatePool> ple_conv_state_pool;
    ExternalInputPreparer external_input_preparer;

    ExecutorConfig config;

    ::shelf::runtime::DeviceBatchContextStorage batch_context_storage;
    ::shelf::runtime::DeviceBatchContext* batch_context = nullptr;
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

    ::shelf::runtime::ProgramSet program_set;
    bool program_set_ready = false;
    std::vector<::shelf::runtime::TpExecutionSchedule> tp_schedules;
    ::shelf::runtime::TpBarrierHook* tp_barrier = nullptr;
    int tp_rank = 0;
    gpu::Tensor execution_status;
    gpu::Tensor execution_workspace;
    ::shelf::runtime::WeightSlot* host_weight_table = nullptr;
    ::shelf::runtime::StaticParameterSlot* host_parameter_table = nullptr;
    std::vector<::shelf::runtime::StaticParameterSlot> host_parameter_slots;
    ::shelf::kernel::ModelDispatchStateView host_model_state{};
    void* dispatch_staging = nullptr;
    uint64_t dispatch_staging_bytes = 0;
    void* host_dispatch_staging = nullptr;
    DispatchStagingPool staging_pool;
    std::vector<ProgramStagingMeta> program_staging_meta;
    std::array<GraphCacheSlot, ::shelf::runtime::ProgramSet::kRowBucketCount> graph_cache{};
    void* host_token_graph_staging = nullptr;
    size_t host_token_graph_staging_bytes = 0;
    ::shelf::runtime::DeviceBatchContext* host_ctx_graph_staging = nullptr;
    bool graph_failed = false;

    gpu::Tensor input_ids;
    gpu::Tensor input_hidden;

    gpu::Tensor rope_inv_freq;
    uint32_t rope_table_stride = 0;
    uint32_t rope_table_count = 0;

    gpu::Tensor output_hidden;
    gpu::Tensor logits;
    gpu::Tensor sampled_tokens;
    gpu::Tensor token_hidden;
    gpu::Tensor normed_token_hidden;
    gpu::Tensor pipeline_input_hidden;
    gpu::Tensor pipeline_output_hidden;
    std::array<gpu::Tensor, kMaxTargetHiddenTaps> dflash_target_hidden;
    gpu::Tensor stochastic_topk_ids;
    gpu::Tensor stochastic_topk_logits;
    gpu::Tensor stochastic_topk_scratch;
    gpu::Tensor stochastic_topk_active_counts;
    uint32_t stochastic_topk_partitions = 0;
    std::size_t stochastic_topk_scratch_bytes = 0;

    std::vector<std::uint32_t> layer_output_layers;
    std::vector<std::uint32_t> layer_output_value_ids;
    std::vector<std::string> trace_names;
    std::vector<std::uint32_t> trace_value_ids;
    std::vector<std::uint32_t> layer_override_value_ids;
    std::vector<std::uint32_t> layer_override_rows;
    std::vector<std::uint32_t> layer_override_hidden;
    std::vector<std::vector<std::uint16_t>> layer_override_host;

    ::shelf::runtime::StreamSignal completion_signal{};
};

// The layer input overrides named in the environment, as host data. `hidden` is the model's
// hidden width, which fixes the row count of each override file.
std::vector<LayerOverrideSource> read_layer_override_sources(uint32_t hidden);

Result<Executor> create_executor(
    ExecutorModelInput&& input,
    SequenceSlotPool& sequence_pool,
    GdnStatePool& gdn_state_pool,
    PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const ExecutorConfig& config);

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
    ::shelf::runtime::RequestHandle handle{};
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
    ::shelf::execution::QsaSpecHistoryDeviceView qsa_spec_history{};
    ::shelf::execution::PleConvSpecHistoryDeviceView ple_conv_history{};
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
