#pragma once
#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
#include <shelf/runtime/rope_profile.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/models/common/qwen_hybrid/tensor_parallel_context.h>
#include <shelf/runtime/rope_profile.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/elementwise_dispatch.h>
#include <shelf/inference/execution/embedding_dispatch.h>
#include <shelf/inference/execution/gdn_conv_dispatch.h>
#include <shelf/inference/execution/gdn_recurrence_dispatch.h>
#include <shelf/inference/execution/kv_append_dispatch.h>
#include <shelf/inference/execution/kv_calib_dump.h>
#include <shelf/inference/execution/l2_normalize_dispatch.h>
#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/paged_attention_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/rope_dispatch.h>
#include <shelf/inference/execution/sampling_dispatch.h>
#include <shelf/inference/execution/spec_decode_context.h>
#include <shelf/inference/scheduling/scheduled_batch.h>
#include <shelf/inference/scheduling/sampling_params.h>
#include <shelf/inference/scheduling/schedule_plan.h>
#include <shelf/inference/scheduling/kv_capacity_manager.h>
#include <shelf/inference/scheduling/kv_banker.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/kernels/kv_types.h>
#include <shelf/inference/state/qsa_state_pool.h>
#include <shelf/inference/state/ple_conv_state_pool.h>
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
#include <span>
#include <vector>
#include <shelf/runtime/decode_backend.h>
#include <shelf/runtime/stream_bridge.h>

namespace shelf {
namespace runtime {
using ::shelf::execution::DecodePerfStats;
using ::shelf::execution::KernelDispatchCounters;
using ::shelf::execution::GdnCompactLogDeviceView;
using ::shelf::execution::GdnRecurrenceResolvedArgs;
using ::shelf::execution::GdnSpecHistoryDeviceView;
using ::shelf::execution::HostExecutionContext;
using ::shelf::execution::HostResolvedValue;
using ::shelf::execution::KvAppendResolvedArgs;
using ::shelf::execution::LayerInputOverride;
using ::shelf::execution::OptimizedLaunchKind;
using ::shelf::execution::OptimizedLaunchResult;
using ::shelf::execution::PagedAttentionPlan;
using ::shelf::execution::PagedAttentionResolvedArgs;
using ::shelf::execution::PhysicalActivationQuantizeLaunch;
using ::shelf::execution::PhysicalBf16Launch;
using ::shelf::execution::PhysicalBf16Variant;
using ::shelf::execution::PhysicalElementwiseLaunch;
using ::shelf::execution::PhysicalEmbeddingLaunch;
using ::shelf::execution::PhysicalGdnConv1dLaunch;
using ::shelf::execution::PhysicalGdnRecurrenceLaunch;
using ::shelf::execution::PhysicalGdnRecurrenceVariant;
using ::shelf::execution::PhysicalKvAppendLaunch;
using ::shelf::execution::PhysicalKvAppendVariant;
using ::shelf::execution::PhysicalL2NormalizeLaunch;
using ::shelf::execution::PhysicalPagedAttentionKernel;
using ::shelf::execution::PhysicalPagedAttentionPlan;
using ::shelf::execution::PhysicalPagedAttentionVariant;
using ::shelf::execution::PhysicalResolveStatus;
using ::shelf::execution::PhysicalRmsNormLaunch;
using ::shelf::execution::PhysicalRmsNormVariant;
using ::shelf::execution::PhysicalRopeLaunch;
using ::shelf::execution::PhysicalRopeVariant;
using ::shelf::execution::KernelMode;
using ::shelf::execution::SemPath;
using ::shelf::execution::SpecDecodeContext;
using ::shelf::execution::ValueTraceSink;
using ::shelf::execution::execute_program;
using ::shelf::execution::execute_program_range;
using ::shelf::execution::execute_program_tp_ranges;
using ::shelf::execution::kDecodeAttnPartialBytes;
using ::shelf::execution::kHostExternalValueCapacity;
using ::shelf::execution::kPagedAttentionPrefillMinRows;
using ::shelf::execution::kPagedAttentionSplitMinVisible;
using ::shelf::execution::kPhysicalPagedAttentionMaxKernels;
using ::shelf::execution::kPhysicalPagedAttentionMaxPrefillRegions;
using ::shelf::execution::kv_calib_dump_maybe;
using ::shelf::execution::launch_kv_append_physical;
using ::shelf::execution::launch_paged_attention_physical;
using ::shelf::execution::launch_rope_physical;
using ::shelf::execution::make_paged_attention_plan_key;
using ::shelf::execution::paged_attention_plan_signature;
using ::shelf::execution::perf_stats_dump;
using ::shelf::execution::perf_stats_enabled;
using ::shelf::execution::perf_stats_instance;
using ::shelf::execution::perf_stats_reset;
using ::shelf::execution::plan_paged_attention;
using ::shelf::execution::read_kernel_mode;
using ::shelf::execution::record_degraded_dispatch;
using ::shelf::execution::record_fallback_dispatch;
using ::shelf::execution::record_logical_dispatch;
using ::shelf::execution::record_optimized_dispatch;
using ::shelf::execution::record_optimized_op;
using ::shelf::execution::record_physical_launch;
using ::shelf::execution::resolve_activation_quantize_physical;
using ::shelf::execution::resolve_bf16_physical;
using ::shelf::execution::resolve_elementwise_physical;
using ::shelf::execution::resolve_embedding_physical;
using ::shelf::execution::resolve_gdn_conv1d_physical;
using ::shelf::execution::resolve_gdn_recurrence_args;
using ::shelf::execution::resolve_gdn_recurrence_physical;
using ::shelf::execution::resolve_host_value;
using ::shelf::execution::resolve_kv_append_args;
using ::shelf::execution::resolve_kv_append_physical;
using ::shelf::execution::resolve_l2_normalize_physical;
using ::shelf::execution::resolve_paged_attention_args;
using ::shelf::execution::resolve_paged_attention_physical;
using ::shelf::execution::resolve_paged_attention_physical_static_plan;
using ::shelf::execution::resolve_paged_attention_physical_static_prefill;
using ::shelf::execution::resolve_rmsnorm_physical;
using ::shelf::execution::resolve_rope_physical;
using ::shelf::execution::rmsnorm_kd;
using ::shelf::execution::semantic_timing_begin;
using ::shelf::execution::semantic_timing_enabled;
using ::shelf::execution::semantic_timing_end;
using ::shelf::execution::semantic_timing_finish;
using ::shelf::execution::semantic_timing_start;
using ::shelf::execution::try_launch_activation_quantize;
using ::shelf::execution::try_launch_elementwise;
using ::shelf::execution::try_launch_embedding;
using ::shelf::execution::try_launch_gdn_conv1d;
using ::shelf::execution::try_launch_gdn_recurrence;
using ::shelf::execution::try_launch_kv_append;
using ::shelf::execution::try_launch_l2_normalize;
using ::shelf::execution::try_launch_optimized;
using ::shelf::execution::try_launch_paged_attention;
using ::shelf::execution::try_launch_rmsnorm;
using ::shelf::execution::try_launch_rope;
using ::shelf::execution::try_launch_sampling;
using ::shelf::execution::upload_host_parameter_table;
using ::shelf::execution::upload_host_weight_table;
using ::shelf::scheduling::ScheduledRequest;
using ::shelf::scheduling::ScheduledBatch;
using ::shelf::scheduling::TokenIdsLocation;
using ::shelf::scheduling::validate_scheduled_batch;
using ::shelf::scheduling::SchedulePlan;
using ::shelf::scheduling::SamplingConfig;
using ::shelf::scheduling::SamplingMode;
using ::shelf::scheduling::SamplingRequest;
using ::shelf::scheduling::validate_sampling_config;
using ::shelf::scheduling::make_sampling_request;
using ::shelf::scheduling::sampling_config_is_default;
using ::shelf::scheduling::KVBankerProcess;
using ::shelf::scheduling::KVBankerState;
using ::shelf::scheduling::validate_banker_state;
using ::shelf::scheduling::is_safe;
using ::shelf::scheduling::can_add_process;
using ::shelf::scheduling::can_grant_pages;
using ::shelf::scheduling::max_safe_grant_pages;
using ::shelf::scheduling::KVAdmissionPolicy;
using ::shelf::scheduling::KVCapacitySnapshot;
using ::shelf::scheduling::KVCapacityManager;
using ::shelf::scheduling::compute_max_kv_tokens;
using ::shelf::scheduling::kv_pages_for_tokens;
using ::shelf::scheduling::additional_kv_pages;
using TensorParallelContext = ::shelf::qwen_hybrid::TensorParallelContext;

using ::shelf::execution::HostExecutionContext;
using ::shelf::execution::kHostExternalValueCapacity;
using ::shelf::kernel::KVCacheDType;
using ::shelf::scheduling::ScheduledBatch;
using ::shelf::scheduling::ScheduledRequest;
using ::shelf::state::GdnStatePool;
using ::shelf::state::GdnStatePoolLayout;
using ::shelf::state::PagedKVPool;
using ::shelf::state::PagedSequenceState;
using ::shelf::state::SequenceSlotId;
using ::shelf::state::SequenceSlotPool;

constexpr uint32_t kMaxTargetHiddenTaps = 8;

enum class DecodeStatus : uint8_t {
    Pending = 0,
    Complete = 1,
    Error = 2,
};

using GeometryValidator = std::function<Status(
    const shelf::qwen_hybrid::QwenHybridConfig&,
    const TensorParallelContext*,
    const ::shelf::qwen_hybrid::PipelineStage&,
    const GdnStatePoolLayout&,
    const PagedKVPool&)>;

struct ExecutorConfig {
    GeometryValidator validate_geometry;
    uint32_t max_batch_rows = 0;
    uint32_t max_scheduled_requests = 0;
    uint32_t max_scheduled_output_rows = 0;
    std::array<uint32_t, kMaxTargetHiddenTaps> target_hidden_taps{};
    uint32_t target_hidden_tap_count = 0;
    uint32_t tp_combine_f32 = 0;
    // Maximum cached context the model-level graphs size their state from. Qwen4Exp uses
    // it for the QSA indexer capacity; other families ignore it.
    uint32_t max_context_tokens = 0;
    ::shelf::runtime::DecodeBackend backend = ::shelf::runtime::DecodeBackend::Hip;
    const TensorParallelContext* tp = nullptr;
    EffectiveRopeProfile rope_profile;
    // Empty retains the legacy single-table profile and theta.
    std::vector<EffectiveRopeProfile> rope_profiles;
    ::shelf::qwen_hybrid::PipelineStage stage;
    // Width of the hidden a pipeline stage receives and emits. 0 uses the model's own
    // hidden size. A family whose stage boundary carries a wider stream (Qwen4Exp's
    // hyper-connection stack) states it here so the transfer buffers are sized for it.
    std::uint32_t pipeline_hidden_width = 0;
    bool emit_normed_token_hidden = false;
    bool external_hidden_input = false;
    // 0 means the family's hidden size. Qwen4Exp's MTP block consumes the pre-mixer wide
    // residual (hc * hidden), so both its hidden input and its returned residual differ
    // from the token-row hidden.
    std::uint32_t external_hidden_width = 0;
    // Bytes per mask row when the program takes a third external input (drafter window).
    std::uint32_t external_mask_bytes = 0;
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

// Re-seeds family state that is derived from the consumed token history (the Qwen4Exp
// PLE n-gram history) after a prefix-cache restore advances a sequence past tokens this
// process never scheduled.
using ExternalPrefixRestoreHandler =
    std::function<Status(::shelf::state::SequenceSlotId, std::span<const int32_t>)>;

// Stateless gather of the extra external input for a run of tokens that follow
// `prefix_tail`. The family supplies it so a prefill driver can compute the next chunk's
// rows on a worker thread while the current chunk runs on the GPU.
using ExternalInputStatelessFn =
    std::function<Status(std::span<const std::int32_t> prefix_tail,
                         std::span<const std::int32_t> tokens,
                         std::vector<bf16_t>& out)>;

struct GraphCacheSlot {
    hipGraphExec_t exec = nullptr;
    uint32_t token_count = 0;
    uint32_t num_outputs = 0;
    uint32_t num_sampled = 0;
    uint32_t num_stochastic = 0;
    uint32_t sampling_top_k = 0;
    uint32_t attention_plan_key = 0;
    uint8_t exec_role = 0;
    uint8_t has_spec_history = 0;
    uint8_t has_token_bitmask = 0;
    uint32_t capture_rows = 0;
    const void* gdn_conv_history = nullptr;
    const void* gdn_recurrent_history = nullptr;
    const void* gdn_recurrent_capture_mask = nullptr;
    uint64_t gdn_conv_row_stride = 0;
    uint64_t gdn_recurrent_row_stride = 0;
    ::shelf::execution::GdnCompactLogDeviceView gdn_compact{};
    const void* token_device_src = nullptr;
    const void* tree_anc_offsets = nullptr;
    const void* tree_anc_positions = nullptr;
    const void* tree_committed_end_rows = nullptr;
    const void* tree_row_parent = nullptr;
    const void* rope_positions = nullptr;
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
    void bind_kv_pool(PagedKVPool& pool) {
        kv_pool = &pool;
        auto ms = host_model_state;
        if (pool.dtype() == KVCacheDType::FP8_E4M3) {
            ms.kv_fp8 = make_paged_kv_fp8_view(pool);
        } else if (pool.dtype() == KVCacheDType::OCTAVE) {
            ms.kv_octave = make_paged_kv_octave_view(pool);
        } else {
            ms.kv_bf16 = make_paged_kv_bf16_view(pool);
        }
        ms.kv_dtype = static_cast<uint32_t>(pool.dtype());
        host_model_state = ms;
    }
    bool uses_qsa_state = false;
    bool uses_ple_conv_state = false;
    std::unique_ptr<::shelf::state::QsaKeyStatePool> qsa_state_pool;
    std::unique_ptr<::shelf::state::PleConvStatePool> ple_conv_state_pool;
    ExternalInputPreparer external_input_preparer;
    ExternalPrefixRestoreHandler external_prefix_restore;
    ExternalInputStatelessFn external_input_stateless;
    // Runs after a batch has completed and before the next one is submitted. The Qwen4Exp
    // expert cache uses it to update residency from the routing the batch produced.
    std::function<Status(hipStream_t)> after_batch;

    ExecutorConfig config;

    ::shelf::runtime::DeviceBatchContextStorage batch_context_storage;
    ::shelf::runtime::DeviceBatchContext* batch_context = nullptr;
    ::shelf::execution::ValueTraceSink* value_trace = nullptr;
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
    std::vector<::shelf::execution::TpSegmentGraphCache> tp_segment_graphs;
    bool tp_segment_graph_off = false;
    ::shelf::runtime::TpBarrierHook* tp_barrier = nullptr;
    int tp_rank = 0;
    ::shelf::runtime::TpBarrierHook* ep_barrier = nullptr;
    int ep_rank = 0;
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
    std::array<GraphCacheSlot, 6> dflash_verify_graph_cache{};
    uint32_t* graph_token_bitmask_staging = nullptr;
    void* host_token_graph_staging = nullptr;
    size_t host_token_graph_staging_bytes = 0;
    ::shelf::runtime::DeviceBatchContext* host_ctx_graph_staging = nullptr;
    bool graph_failed = false;

    gpu::Tensor input_ids;
    gpu::Tensor input_hidden;
    // Optional per-row block mask for the drafter-wide attention (MTP sliding window).
    gpu::Tensor input_mask;

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
    ::shelf::execution::GdnSpecHistoryDeviceView gdn_spec_history{};
    ::shelf::execution::GdnCompactLogDeviceView gdn_compact{};
    ::shelf::execution::QsaSpecHistoryDeviceView qsa_spec_history{};
    ::shelf::execution::PleConvSpecHistoryDeviceView ple_conv_history{};
    const uint32_t* tree_anc_offsets = nullptr;
    const uint32_t* tree_anc_positions = nullptr;
    uint32_t tree_committed_end = 0u;
    const uint32_t* tree_committed_end_rows = nullptr;
    const uint32_t* rope_positions = nullptr;
    const int32_t* tree_row_parent = nullptr;
    const bf16_t* prepared_external_input = nullptr;
    uint32_t prepared_external_input_rows = 0u;
};

struct QsaSpecReplayRequest {
    const ::shelf::execution::QsaSpecHistoryDeviceView* history = nullptr;
    const int32_t* row_map = nullptr;
    const uint32_t* positions = nullptr;
    uint32_t rows = 0u;
    ::shelf::state::SequenceSlotId slot = 0u;
};

Status replay_qsa_spec_state(Executor& executor, const QsaSpecReplayRequest& request,
                             hipStream_t stream);

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
