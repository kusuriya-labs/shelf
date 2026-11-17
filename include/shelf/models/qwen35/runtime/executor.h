#pragma once

#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/runtime/executor.h>

namespace shelf {
namespace qwen35 {
using ::shelf::state::GdnStatePool;
using ::shelf::state::GdnStatePoolLayout;
using ::shelf::state::PagedKVPool;
using ::shelf::state::PagedSequenceState;
using ::shelf::state::SequenceSlotId;
using ::shelf::state::SequenceSlotPool;


using ::shelf::runtime::kMaxTargetHiddenTaps;
using ::shelf::runtime::program_needs_exl3_workspace;
using ::shelf::runtime::program_needs_qsa_workspace;
using ::shelf::runtime::program_set_needs_exl3_workspace;
using ::shelf::runtime::program_set_needs_qsa_workspace;
using Qwen35TensorParallelContext = ::shelf::runtime::TensorParallelContext;

using ::shelf::runtime::complete_batch;
using ::shelf::runtime::create_executor;
using ::shelf::runtime::execute_batch;
using ::shelf::runtime::executor_shutdown;
using ::shelf::runtime::poll_decode;
using ::shelf::runtime::read_layer_override_sources;
using ::shelf::runtime::replay_qsa_spec_state;
using ::shelf::runtime::submit_batch;

using ::shelf::runtime::BatchExecutionOutput;
using ::shelf::runtime::DecodeStatus;
using ::shelf::runtime::ExecuteBatchOptions;
using ::shelf::runtime::Executor;
using ::shelf::runtime::ExecutorConfig;
using ::shelf::runtime::ExecutorModelInput;
using ::shelf::runtime::ExternalInputPreparer;
using ::shelf::runtime::ExternalPrefixRestoreHandler;
using ::shelf::runtime::GraphCacheSlot;
using ::shelf::runtime::LayerOverrideSource;
using ::shelf::runtime::PendingBatch;
using ::shelf::runtime::PendingRequestCommit;
using ::shelf::runtime::QsaSpecReplayRequest;
using ::shelf::runtime::RuntimeModelView;

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
using ::shelf::kernel::KVCacheDType;
using ::shelf::state::GdnStatePool;
using ::shelf::state::GdnStatePoolLayout;
using ::shelf::state::PagedKVPool;
using ::shelf::state::PagedSequenceState;
using ::shelf::state::SequenceSlotId;
using ::shelf::state::SequenceSlotPool;
using ::shelf::runtime::DispatchStagingPool;
using ::shelf::runtime::ProgramStagingMeta;

using ::shelf::runtime::complete_batch;
using ::shelf::runtime::create_executor;
using ::shelf::runtime::execute_batch;
using ::shelf::runtime::executor_shutdown;
using ::shelf::runtime::kMaxTargetHiddenTaps;
using ::shelf::runtime::poll_decode;
using ::shelf::runtime::program_needs_exl3_workspace;
using ::shelf::runtime::program_needs_qsa_workspace;
using ::shelf::runtime::program_set_needs_exl3_workspace;
using ::shelf::runtime::program_set_needs_qsa_workspace;
using ::shelf::runtime::read_layer_override_sources;
using ::shelf::runtime::replay_qsa_spec_state;
using ::shelf::runtime::submit_batch;

Result<Executor> create_model_executor(
    const Qwen35Model& model,
    SequenceSlotPool& sequence_pool,
    GdnStatePool& gdn_state_pool,
    PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const ExecutorConfig& config);

}  // namespace runtime

using ::shelf::qwen35::runtime::create_model_executor;

}  // namespace qwen35
}  // namespace shelf
