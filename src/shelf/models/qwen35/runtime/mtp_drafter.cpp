#include <shelf/models/qwen35/runtime/mtp_drafter.h>

#include <shelf/core/gpu/scoped_device.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/models/qwen35/runtime/spec_decode.h>
#include <shelf/models/qwen35/runtime/tp_batch_hook.h>

#include <algorithm>
#include <chrono>
#include <vector>

namespace shelf::qwen35::runtime {

namespace {

constexpr uint32_t kMtpKvLayers = 1u;

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) noexcept {
    return (value + divisor - 1u) / divisor;
}

shelf::qwen_hybrid::QwenHybridConfig mtp_executor_config(
    const shelf::qwen_hybrid::QwenHybridConfig& source) {
    shelf::qwen_hybrid::QwenHybridConfig config = source;
    config.num_hidden_layers = 1;
    config.layer_types = {1};
    config.full_attention_interval = 1;
    return config;
}

ScheduledBatch make_batch(state::PagedSequenceState& sequence,
                          std::vector<ScheduledRequest>& requests,
                          const int32_t* tokens,
                          uint32_t num_tokens,
                          uint32_t prefix_tokens,
                          uint32_t num_output_rows,
                          bool compute_logits,
                          bool sample,
                          ::shelf::runtime::ExecutionClass execution_class,
                          bool speculative_verify,
                          const SamplingConfig* sampling,
                          uint64_t sampling_index) {
    ScheduledRequest request;
    request.sequence = &sequence;
    request.handle = sequence.request_handle();
    request.execution_class = execution_class;
    request.token_begin = 0u;
    request.num_tokens = num_tokens;
    request.prefix_tokens = prefix_tokens;
    request.compute_logits = compute_logits;
    request.sample = compute_logits && sample;
    request.num_output_rows = num_output_rows;
    if (sampling != nullptr) {
        request.sampling = *sampling;
        request.sampling_index = sampling_index;
    }
    requests.clear();
    requests.push_back(request);

    ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.token_ids_location = shelf::qwen35::TokenIdsLocation::Host;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests =
        execution_class == ::shelf::runtime::ExecutionClass::DECODE ? 1u : 0u;
    batch.num_verify_requests =
        (speculative_verify && num_tokens > 1u) ? 1u : 0u;
    batch.num_prefill_requests =
        (execution_class == ::shelf::runtime::ExecutionClass::PREFILL && num_tokens > 1u)
            ? 1u
            : 0u;
    batch.speculative_verify = speculative_verify;
    return batch;
}

const gpu::Tensor& drafter_hidden_output(const MtpDrafter& drafter) {
    return drafter.wide_hidden ? drafter.executor.normed_token_hidden
                               : drafter.executor.token_hidden;
}

Result<int32_t> copy_sampled_token(const gpu::Tensor& sampled, hipStream_t stream) {
    int32_t token = -1;
    const hipError_t err =
        hipMemcpyAsync(&token, sampled.data<int32_t>(), sizeof(int32_t),
                       hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        return Status::hip_error("mtp sampled token copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        return Status::hip_error("mtp sampled token sync", hipGetErrorString(hipGetLastError()),
                                 __FILE__, __LINE__);
    }
    return token;
}

Status copy_hidden_to_input(MtpDrafter& drafter, const gpu::Tensor& source,
                            uint32_t source_row_offset, uint32_t rows,
                            hipStream_t stream) {
    if (rows == 0u) return Status::make_ok();
    const std::size_t hidden = drafter.hidden_width;
    void* dst = drafter.executor.input_hidden.data<void>();
    if (dst == nullptr) {
        return Status::invalid_state("mtp drafter has no hidden input buffer", __FILE__,
                                     __LINE__);
    }
    const std::size_t offset =
        static_cast<std::size_t>(source_row_offset) * hidden * sizeof(bf16_t);
    const hipError_t err = hipMemcpyAsync(
        dst, static_cast<const std::uint8_t*>(source.data<void>()) + offset,
        static_cast<std::size_t>(rows) * hidden * sizeof(bf16_t),
        hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("mtp hidden input copy", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

Status cache_target_hidden(MtpDrafter& drafter, const Executor& target,
                           uint32_t row_base, uint32_t rows, hipStream_t stream) {
    if (rows == 0u) return Status::make_ok();
    if (drafter.hidden_cache.ndim() == 0) {
        return Status::invalid_state("mtp drafter has no hidden cache", __FILE__,
                                     __LINE__);
    }
    if (rows > drafter.hidden_cache.dim(0)) {
        return Status::out_of_range("mtp hidden cache rows exceeded", __FILE__, __LINE__);
    }
    auto scope = shelf::gpu::ScopedDevice::create(drafter.target_device);
    if (!scope.ok()) return scope.status();
    const std::size_t width = drafter.hidden_width;
    const hipError_t err = hipMemcpyAsync(
        drafter.hidden_cache.data<void>(),
        static_cast<const std::uint8_t*>(target.normed_token_hidden.data<void>()) +
            static_cast<std::size_t>(row_base) * width * sizeof(bf16_t),
        static_cast<std::size_t>(rows) * width * sizeof(bf16_t),
        hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("mtp hidden cache copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status rewind_sequence(state::PagedSequenceState& sequence, uint32_t position,
                       uint32_t required_blocks, hipStream_t stream) {
    if (required_blocks < sequence.block_table.size()) {
        Status st = rollback_sequence_append(sequence, required_blocks, stream);
        if (!st.ok()) return st;
    }
    sequence.position = position;
    return Status::make_ok();
}

Result<BatchExecutionOutput> run_mtp_forward(MtpDrafter& drafter,
                                             std::vector<int32_t>& tokens,
                                             uint32_t prefix,
                                             bool compute_logits,
                                             hipStream_t stream) {
    if (tokens.empty()) {
        return Status::invalid_argument("mtp forward has no tokens", __FILE__, __LINE__);
    }
    if (prefix != drafter.sequence->position) {
        return Status::invalid_state("mtp sequence position diverged from prefix", __FILE__,
                                     __LINE__);
    }
    const uint32_t rows = static_cast<uint32_t>(tokens.size());
    const ::shelf::runtime::ExecutionClass execution_class =
        rows == 1u ? ::shelf::runtime::ExecutionClass::DECODE
                   : ::shelf::runtime::ExecutionClass::PREFILL;
    std::vector<ScheduledRequest> requests;
    ScheduledBatch batch = make_batch(
        *drafter.sequence, requests, tokens.data(), rows, prefix, 1u, compute_logits,
        true, execution_class, false, nullptr, 0u);
    return execute_batch(drafter.executor, batch, stream);
}

MtpGdnRankState rank0_state_view(const MtpDrafter& drafter) {
    MtpGdnRankState state;
    state.pool = drafter.target_pool;
    state.stream = drafter.target_stream;
    state.device = drafter.target_device;
    state.conv_snapshot = drafter.gdn_conv_snapshot;
    state.recurrent_snapshot = drafter.gdn_recurrent_snapshot;
    state.conv_bytes = drafter.gdn_conv_bytes;
    state.recurrent_bytes = drafter.gdn_recurrent_bytes;
    state.history = drafter.gdn_history;
    state.compact_delta = drafter.compact_delta;
    state.compact_k = drafter.compact_k;
    state.compact_a = drafter.compact_a;
    state.compact_delta_layer_stride = drafter.compact_delta_layer_stride;
    state.compact_k_layer_stride = drafter.compact_k_layer_stride;
    state.compact_a_layer_stride = drafter.compact_a_layer_stride;
    state.commit_request = drafter.commit_request;
    state.commit_request_host = drafter.commit_request_host;
    state.gdn_key_heads = drafter.gdn_key_heads;
    state.target = drafter.target_executor;
    state.qsa_partial_history = drafter.qsa_partial_history;
    state.qsa_count_history = drafter.qsa_count_history;
    state.qsa_partial_state_stride = drafter.qsa_partial_state_stride;
    state.qsa_count_state_stride = drafter.qsa_count_state_stride;
    state.qsa_history_rows = drafter.qsa_history_rows;
    state.ple_conv_history = drafter.ple_conv_history;
    state.ple_conv_state_stride = drafter.ple_conv_state_stride;
    state.ple_conv_history_rows = drafter.ple_conv_history_rows;
    return state;
}

std::vector<MtpGdnRankState> all_rank_states(const MtpDrafter& drafter) {
    std::vector<MtpGdnRankState> states;
    states.reserve(1u + drafter.extra_ranks.size());
    states.push_back(rank0_state_view(drafter));
    for (const MtpGdnRankState& state : drafter.extra_ranks) states.push_back(state);
    return states;
}

ExecuteBatchOptions verify_options_for(const MtpGdnRankState& state, uint32_t rows) {
    ExecuteBatchOptions options;
    options.gdn_spec_history = gdn_spec_history_view(state.history, rows);
    options.gdn_compact.delta = state.compact_delta;
    options.gdn_compact.k = state.compact_k;
    options.gdn_compact.a = state.compact_a;
    options.gdn_compact.delta_layer_stride = state.compact_delta_layer_stride;
    options.gdn_compact.k_layer_stride = state.compact_k_layer_stride;
    options.gdn_compact.a_layer_stride = state.compact_a_layer_stride;
    options.qsa_spec_history.partial = state.qsa_partial_history;
    options.qsa_spec_history.counts = state.qsa_count_history;
    options.qsa_spec_history.rows = state.qsa_history_rows;
    options.qsa_spec_history.partial_state_stride = state.qsa_partial_state_stride;
    options.qsa_spec_history.count_state_stride = state.qsa_count_state_stride;
    options.ple_conv_history.store = state.ple_conv_history;
    options.ple_conv_history.rows = state.ple_conv_history_rows;
    options.ple_conv_history.state_stride = state.ple_conv_state_stride;
    return options;
}

std::vector<ExecuteBatchOptions> build_verify_options(const MtpDrafter& drafter,
                                                      uint32_t rows) {
    std::vector<MtpGdnRankState> states = all_rank_states(drafter);
    std::vector<ExecuteBatchOptions> options;
    options.reserve(states.size());
    for (const MtpGdnRankState& state : states) {
        options.push_back(verify_options_for(state, rows));
    }
    return options;
}

Result<BatchExecutionOutput> run_target(MtpDrafter& drafter,
                                        Executor& target,
                                        state::PagedSequenceState& sequence,
                                        std::vector<int32_t>& tokens,
                                        uint32_t prefix,
                                        bool compute_logits,
                                        bool sample,
                                        uint32_t output_rows,
                                        ::shelf::runtime::ExecutionClass execution_class,
                                        bool speculative_verify,
                                        const SamplingConfig& sampling,
                                        uint64_t sampling_index,
                                        const std::vector<ExecuteBatchOptions>* per_rank,
                                        hipStream_t stream) {
    if (tokens.empty()) {
        return Status::invalid_argument("target forward has no tokens", __FILE__, __LINE__);
    }
    if (prefix != sequence.position) {
        return Status::invalid_state("target sequence position diverged from prefix",
                                     __FILE__, __LINE__);
    }
    std::vector<ScheduledRequest> requests;
    ScheduledBatch batch = make_batch(
        sequence, requests, tokens.data(), static_cast<uint32_t>(tokens.size()), prefix,
        output_rows, compute_logits, sample, execution_class, speculative_verify,
        compute_logits ? &sampling : nullptr, sampling_index);
    if (drafter.tp_hook != nullptr) {
        if (per_rank != nullptr && !per_rank->empty()) {
            return drafter.tp_hook->on_execute(batch, per_rank->data());
        }
        return drafter.tp_hook->on_execute(batch);
    }
    if (per_rank != nullptr && !per_rank->empty()) {
        return execute_batch(target, batch, stream, (*per_rank)[0]);
    }
    return execute_batch(target, batch, stream);
}

Status allocate_qsa_history(Executor* target, gpu::GpuArena& arena, uint32_t rows,
                           bf16_t*& partial, std::uint32_t*& counts,
                           std::uint64_t& partial_state_stride,
                           std::uint64_t& count_state_stride) {
    if (target == nullptr || target->qsa_state_pool == nullptr || rows == 0u) {
        return Status::make_ok();
    }
    const auto view = target->qsa_state_pool->device_view();
    if (view.num_qsa_states == 0u || view.compress_ratio == 0u || view.head_dim == 0u) {
        return Status::make_ok();
    }
    partial_state_stride =
        static_cast<std::uint64_t>(rows) * view.compress_ratio * view.head_dim;
    count_state_stride = static_cast<std::uint64_t>(rows) * 2u;
    const std::size_t partial_bytes = static_cast<std::size_t>(view.num_qsa_states) *
        partial_state_stride * sizeof(bf16_t);
    const std::size_t count_bytes = static_cast<std::size_t>(view.num_qsa_states) *
        count_state_stride * sizeof(std::uint32_t);
    auto pv = arena.allocate_aligned(partial_bytes, 256u);
    if (!pv.ok()) return pv.status();
    partial = static_cast<bf16_t*>(pv.release().data());
    auto cv = arena.allocate_aligned(count_bytes, 256u);
    if (!cv.ok()) return cv.status();
    counts = static_cast<std::uint32_t*>(cv.release().data());
    return Status::make_ok();
}

Status restore_rank_qsa(const MtpGdnRankState& state, SequenceSlotId slot, uint32_t row) {
    if (state.target == nullptr || state.target->qsa_state_pool == nullptr ||
        state.qsa_partial_history == nullptr || state.qsa_count_history == nullptr) {
        return Status::make_ok();
    }
    ::shelf::execution::QsaSpecHistoryDeviceView view;
    view.partial = state.qsa_partial_history;
    view.counts = state.qsa_count_history;
    view.rows = state.qsa_history_rows;
    view.partial_state_stride = state.qsa_partial_state_stride;
    view.count_state_stride = state.qsa_count_state_stride;
    auto scope = shelf::gpu::ScopedDevice::create(state.device);
    if (!scope.ok()) return scope.status();
    return restore_qsa_spec_state(*state.target->qsa_state_pool, slot, view, row,
                                  state.stream);
}

Status allocate_ple_conv_history(Executor* target, gpu::GpuArena& arena, uint32_t rows,
                                bf16_t*& store, std::uint64_t& state_stride) {
    if (target == nullptr || target->ple_conv_state_pool == nullptr || rows == 0u) {
        return Status::make_ok();
    }
    const auto view = target->ple_conv_state_pool->device_view();
    if (view.channels == 0u || view.history_stride == 0u) return Status::make_ok();
    const std::size_t history_elems = static_cast<std::size_t>(view.history_stride) *
                                      ::shelf::kernel::kPleConvHistory;
    state_stride = static_cast<std::uint64_t>(rows) * history_elems;
    const std::size_t bytes = static_cast<std::size_t>(
        target->ple_conv_state_pool->num_ple_states()) * state_stride * sizeof(bf16_t);
    auto view_alloc = arena.allocate_aligned(bytes, 256u);
    if (!view_alloc.ok()) return view_alloc.status();
    store = static_cast<bf16_t*>(view_alloc.release().data());
    return Status::make_ok();
}

Status restore_rank_ple(const MtpGdnRankState& state, SequenceSlotId slot, uint32_t row) {
    if (state.target == nullptr || state.target->ple_conv_state_pool == nullptr ||
        state.ple_conv_history == nullptr) {
        return Status::make_ok();
    }
    ::shelf::execution::PleConvSpecHistoryDeviceView view;
    view.store = state.ple_conv_history;
    view.rows = state.ple_conv_history_rows;
    view.state_stride = state.ple_conv_state_stride;
    auto scope = shelf::gpu::ScopedDevice::create(state.device);
    if (!scope.ok()) return scope.status();
    return restore_ple_conv_spec_state(*state.target->ple_conv_state_pool, slot, view,
                                       row, state.stream);
}

}  // namespace

std::size_t mtp_drafter_reserve_bytes(const shelf::qwen_hybrid::QwenHybridConfig& config,
                                      const MtpDrafterConfig& options,
                                      std::size_t token_row_intermediate,
                                      std::size_t grouped_executor_bytes,
                                      bool exl3_linear_workspace) noexcept {
    if (options.max_seq_len == 0u || options.page_tokens == 0u) {
        return 0u;
    }
    shelf::qwen_hybrid::QwenHybridConfig mtp = mtp_executor_config(config);
    const uint32_t pages =
        ceil_div_u32(options.max_seq_len + 1u, options.page_tokens) + 1u;
    const std::size_t kv = state::PagedKVPool::page_bytes(
        options.page_tokens, kMtpKvLayers,
        static_cast<uint32_t>(mtp.num_key_value_heads),
        static_cast<uint32_t>(mtp.attention_head_dim), options.kv_dtype) *
        static_cast<std::size_t>(pages);

    ExecutorConfig executor_config;
    executor_config.max_batch_rows = mtp_drafter_rows(options);
    executor_config.max_scheduled_requests = 1u;
    executor_config.max_scheduled_output_rows = 1u;
    executor_config.external_hidden_input = true;
    const std::size_t executor = qwen35_executor_reserve_bytes(
        mtp, token_row_intermediate, executor_config, 1u, exl3_linear_workspace) +
        grouped_executor_bytes;

    const state::GdnStatePoolLayout gdn = make_qwen35_gdn_state_layout(config);
    const std::size_t gdn_state =
        static_cast<std::size_t>(gdn.num_gdn_states) *
        (static_cast<std::size_t>(gdn.conv_history_stride) * gdn.conv_history *
             sizeof(bf16_t) +
         static_cast<std::size_t>(gdn.num_v_heads) * gdn.head_k * gdn.head_v *
             sizeof(fp16_t));
    const std::size_t snapshots = 2u * gdn_state;
    std::size_t history = 0u;
    std::size_t compact = 0u;
    if (options.allocate_spec_buffers) {
        history = static_cast<std::size_t>(options.num_drafts) * gdn.num_gdn_states *
            gdn.conv_history_stride * gdn.conv_history * sizeof(bf16_t);
        const std::uint64_t log_rows = static_cast<std::uint64_t>(options.num_drafts) + 1u;
        const std::uint64_t vdim =
            static_cast<std::uint64_t>(gdn.num_v_heads) * gdn.head_v;
        const std::uint64_t qkdim =
            static_cast<std::uint64_t>(gdn.num_v_heads) * gdn.head_k;
        compact = static_cast<std::size_t>(gdn.num_gdn_states) * log_rows *
            (vdim + qkdim + gdn.num_v_heads) * sizeof(float);
    }
    constexpr std::size_t kMargin = 64ull * 1024ull * 1024ull;
    return kv + executor + gdn_state + snapshots + history + compact + kMargin;
}

Result<std::unique_ptr<MtpDrafter>> create_mtp_drafter(
    const MtpDrafterInput& input,
    Executor* target,
    const state::GdnStatePool& target_gdn_pool,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const MtpDrafterConfig& options,
    const std::vector<MtpGdnRankBinding>* extra_ranks,
    TpBatchHook* tp_hook) {

    if (input.config == nullptr || input.embed_tokens == nullptr) {
        return Status::invalid_argument("mtp drafter input is incomplete", __FILE__,
                                        __LINE__);
    }
    if (input.lowered.graph.external_inputs.size() != 2u) {
        return Status::invalid_argument(
            "the drafter graph takes a token id and the previous hidden", __FILE__,
            __LINE__);
    }
    if (options.num_drafts == 0u || options.num_drafts > kMtpMaxDrafts) {
        return Status::invalid_argument("mtp num_drafts is out of range", __FILE__,
                                        __LINE__);
    }
    if (options.max_seq_len == 0u) {
        return Status::invalid_argument("mtp max_seq_len must be positive", __FILE__,
                                        __LINE__);
    }
    if (options.page_tokens == 0u) {
        return Status::invalid_argument("mtp page_tokens must be positive", __FILE__,
                                        __LINE__);
    }
    if (options.max_batch_rows == 0u) {
        return Status::invalid_argument("mtp max_batch_rows must be positive", __FILE__,
                                        __LINE__);
    }

    std::unique_ptr<MtpDrafter> drafter(new MtpDrafter());
    drafter->target_executor = target;
    drafter->config = mtp_executor_config(*input.config);
    drafter->max_seq_len = options.max_seq_len;
    drafter->num_drafts = options.num_drafts;
    drafter->hidden_width = input.hidden_width != 0u
        ? input.hidden_width
        : static_cast<std::uint32_t>(input.config->hidden_size);
    drafter->wide_hidden = input.wide_hidden;
    drafter->target_pool = &target_gdn_pool;
    drafter->target_stream = stream;
    hipGetDevice(&drafter->target_device);
    drafter->tp_hook = tp_hook;

    const uint32_t pages =
        ceil_div_u32(options.max_seq_len + 1u, options.page_tokens) + 1u;
    auto slot_pool = state::SequenceSlotPool::create(arena, 1u, pages);
    if (!slot_pool.ok()) return slot_pool.status();
    drafter->slot_pool = std::make_unique<state::SequenceSlotPool>(slot_pool.release());

    const state::GdnStatePoolLayout gdn_layout = make_qwen35_gdn_state_layout(drafter->config);
    auto gdn_pool = state::GdnStatePool::create(arena, 1u, gdn_layout);
    if (!gdn_pool.ok()) return gdn_pool.status();
    drafter->gdn_pool = std::make_unique<state::GdnStatePool>(gdn_pool.release());

    auto kv_pool = state::PagedKVPool::create(
        arena, pages, options.page_tokens, kMtpKvLayers,
        static_cast<uint32_t>(drafter->config.num_key_value_heads),
        static_cast<uint32_t>(drafter->config.attention_head_dim), options.kv_dtype);
    if (!kv_pool.ok()) return kv_pool.status();
    drafter->kv_pool = std::make_unique<state::PagedKVPool>(kv_pool.release());

    auto sequence = state::create_paged_sequence_state(
        *drafter->slot_pool, *drafter->gdn_pool, *drafter->kv_pool,
        options.max_seq_len, stream);
    if (!sequence.ok()) return sequence.status();
    drafter->sequence =
        std::make_unique<state::PagedSequenceState>(sequence.release());

    ExecutorModelInput model_input;
    model_input.model.config = &drafter->config;
    model_input.model.embed_tokens = input.embed_tokens;
    model_input.lowered = input.lowered;

    ExecutorConfig executor_config;
    executor_config.max_batch_rows = mtp_drafter_rows(options);
    executor_config.max_scheduled_requests = 1u;
    executor_config.max_scheduled_output_rows = 1u;
    executor_config.external_hidden_input = true;
    executor_config.external_hidden_width = drafter->hidden_width;
    executor_config.emit_normed_token_hidden = drafter->wide_hidden;
    executor_config.normed_token_hidden_width = drafter->hidden_width;
    executor_config.rope_profile = options.rope_profile;
    executor_config.backend = runtime::DecodeBackend::Hip;

    auto executor = create_executor(std::move(model_input), *drafter->slot_pool,
                                    *drafter->gdn_pool, *drafter->kv_pool, arena,
                                    executor_config);
    if (!executor.ok()) {
        Status st = state::release_paged_sequence_state(*drafter->sequence, stream);
        (void)st;
        return executor.status();
    }
    drafter->executor = executor.release();

    auto fail_with_sequence = [&](const Status& status) {
        Status st = shutdown_gdn_spec_history(drafter->gdn_history);
        (void)st;
        st = executor_shutdown(drafter->executor);
        (void)st;
        st = state::release_paged_sequence_state(*drafter->sequence, stream);
        (void)st;
        return status;
    };
    {
        const std::uint32_t cache_rows = options.num_drafts + 1u;
        auto cache = arena.allocate_aligned(
            static_cast<std::size_t>(cache_rows) * drafter->hidden_width *
                sizeof(bf16_t),
            256u);
        if (!cache.ok()) return fail_with_sequence(cache.status());
        auto view = cache.release();
        auto tensor = gpu::Tensor::create<bf16_t>(
            view, {cache_rows, static_cast<std::size_t>(drafter->hidden_width)},
            {static_cast<std::size_t>(drafter->hidden_width), 1u});
        if (!tensor.ok()) return fail_with_sequence(tensor.status());
        drafter->hidden_cache = tensor.release();
    }

    if (options.allocate_spec_buffers) {
        Status qsa = allocate_qsa_history(
            target, arena, options.num_drafts + 1u, drafter->qsa_partial_history,
            drafter->qsa_count_history, drafter->qsa_partial_state_stride,
            drafter->qsa_count_state_stride);
        if (!qsa.ok()) return fail_with_sequence(qsa);
        drafter->qsa_history_rows = options.num_drafts + 1u;
        Status ple = allocate_ple_conv_history(
            target, arena, options.num_drafts + 1u, drafter->ple_conv_history,
            drafter->ple_conv_state_stride);
        if (!ple.ok()) return fail_with_sequence(ple);
        drafter->ple_conv_history_rows = options.num_drafts + 1u;
    }

    drafter->gdn_conv_bytes = spec_gdn_conv_bytes(target_gdn_pool);
    drafter->gdn_recurrent_bytes = spec_gdn_recurrent_bytes(target_gdn_pool);
    if (drafter->gdn_conv_bytes != 0u) {
        auto view = arena.allocate_aligned(drafter->gdn_conv_bytes, 256u);
        if (!view.ok()) return fail_with_sequence(view.status());
        drafter->gdn_conv_snapshot = view.release().data();
    }
    if (drafter->gdn_recurrent_bytes != 0u) {
        auto view = arena.allocate_aligned(drafter->gdn_recurrent_bytes, 256u);
        if (!view.ok()) return fail_with_sequence(view.status());
        drafter->gdn_recurrent_snapshot = view.release().data();
    }

    const GdnStatePoolDeviceView pool = target_gdn_pool.device_view();
    if (options.allocate_spec_buffers) {
        auto history = create_gdn_spec_history(arena, target_gdn_pool,
                                               options.num_drafts, false);
        if (!history.ok()) return fail_with_sequence(history.status());
        drafter->gdn_history = history.release();

        const std::uint64_t log_rows = static_cast<std::uint64_t>(options.num_drafts) + 1u;
        const std::uint64_t vdim =
            static_cast<std::uint64_t>(pool.num_v_heads) * pool.head_v;
        const std::uint64_t qkdim =
            static_cast<std::uint64_t>(pool.num_v_heads) * pool.head_k;
        drafter->compact_delta_layer_stride = log_rows * vdim;
        drafter->compact_k_layer_stride = log_rows * qkdim;
        drafter->compact_a_layer_stride = log_rows * pool.num_v_heads;
        const std::size_t delta_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
            drafter->compact_delta_layer_stride * sizeof(float);
        const std::size_t k_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
            drafter->compact_k_layer_stride * sizeof(float);
        const std::size_t a_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
            drafter->compact_a_layer_stride * sizeof(float);
        {
            auto view = arena.allocate_aligned(delta_bytes, 256u);
            if (!view.ok()) return fail_with_sequence(view.status());
            drafter->compact_delta = static_cast<float*>(view.release().data());
        }
        {
            auto view = arena.allocate_aligned(k_bytes, 256u);
            if (!view.ok()) return fail_with_sequence(view.status());
            drafter->compact_k = static_cast<float*>(view.release().data());
        }
        {
            auto view = arena.allocate_aligned(a_bytes, 256u);
            if (!view.ok()) return fail_with_sequence(view.status());
            drafter->compact_a = static_cast<float*>(view.release().data());
        }
    }
    {
        auto view = arena.allocate_aligned(
            sizeof(::shelf::runtime::DeviceRequestDescriptor), 256u);
        if (!view.ok()) return fail_with_sequence(view.status());
        drafter->commit_request =
            static_cast<::shelf::runtime::DeviceRequestDescriptor*>(view.release().data());
    }
    drafter->commit_request_host.row_begin = 0u;
    drafter->commit_request_host.row_count = 0u;
    drafter->gdn_key_heads = static_cast<std::uint32_t>(
        input.config->linear_num_key_heads != 0u ? input.config->linear_num_key_heads : 16u);

    if (extra_ranks != nullptr) {
        for (const MtpGdnRankBinding& binding : *extra_ranks) {
            if (binding.pool == nullptr || binding.arena == nullptr) {
                return fail_with_sequence(Status::invalid_argument(
                    "mtp extra rank binding is incomplete", __FILE__, __LINE__));
            }
            auto scope = shelf::gpu::ScopedDevice::create(binding.device);
            if (!scope.ok()) return fail_with_sequence(scope.status());
            MtpGdnRankState state;
            state.pool = binding.pool;
            state.stream = binding.stream;
            state.device = binding.device;
            state.target = binding.executor;
            if (options.allocate_spec_buffers) {
                Status qsa = allocate_qsa_history(
                    binding.executor, *binding.arena, options.num_drafts + 1u,
                    state.qsa_partial_history, state.qsa_count_history,
                    state.qsa_partial_state_stride, state.qsa_count_state_stride);
                if (!qsa.ok()) return fail_with_sequence(qsa);
                state.qsa_history_rows = options.num_drafts + 1u;
                Status ple = allocate_ple_conv_history(
                    binding.executor, *binding.arena, options.num_drafts + 1u,
                    state.ple_conv_history, state.ple_conv_state_stride);
                if (!ple.ok()) return fail_with_sequence(ple);
                state.ple_conv_history_rows = options.num_drafts + 1u;
            }
            state.conv_bytes = spec_gdn_conv_bytes(*binding.pool);
            state.recurrent_bytes = spec_gdn_recurrent_bytes(*binding.pool);
            if (state.conv_bytes != 0u) {
                auto view = binding.arena->allocate_aligned(state.conv_bytes, 256u);
                if (!view.ok()) return fail_with_sequence(view.status());
                state.conv_snapshot = view.release().data();
            }
            if (state.recurrent_bytes != 0u) {
                auto view = binding.arena->allocate_aligned(state.recurrent_bytes, 256u);
                if (!view.ok()) return fail_with_sequence(view.status());
                state.recurrent_snapshot = view.release().data();
            }
            const auto pool = binding.pool->device_view();
            if (!options.allocate_spec_buffers) {
                auto request = binding.arena->allocate_aligned(
                    sizeof(::shelf::runtime::DeviceRequestDescriptor), 256u);
                if (!request.ok()) return fail_with_sequence(request.status());
                state.commit_request = static_cast<::shelf::runtime::DeviceRequestDescriptor*>(
                    request.release().data());
                state.commit_request_host.row_begin = 0u;
                state.commit_request_host.row_count = 0u;
                state.gdn_key_heads = drafter->gdn_key_heads;
                drafter->extra_ranks.push_back(std::move(state));
                continue;
            }
            auto history = create_gdn_spec_history(*binding.arena, *binding.pool,
                                                   options.num_drafts, false);
            if (!history.ok()) return fail_with_sequence(history.status());
            state.history = history.release();
            const std::uint64_t log_rows = static_cast<std::uint64_t>(options.num_drafts) + 1u;
            const std::uint64_t vdim =
                static_cast<std::uint64_t>(pool.num_v_heads) * pool.head_v;
            const std::uint64_t qkdim =
                static_cast<std::uint64_t>(pool.num_v_heads) * pool.head_k;
            state.compact_delta_layer_stride = log_rows * vdim;
            state.compact_k_layer_stride = log_rows * qkdim;
            state.compact_a_layer_stride = log_rows * pool.num_v_heads;
            const std::size_t delta_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
                state.compact_delta_layer_stride * sizeof(float);
            const std::size_t k_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
                state.compact_k_layer_stride * sizeof(float);
            const std::size_t a_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
                state.compact_a_layer_stride * sizeof(float);
            auto delta = binding.arena->allocate_aligned(delta_bytes, 256u);
            if (!delta.ok()) return fail_with_sequence(delta.status());
            state.compact_delta = static_cast<float*>(delta.release().data());
            auto kb = binding.arena->allocate_aligned(k_bytes, 256u);
            if (!kb.ok()) return fail_with_sequence(kb.status());
            state.compact_k = static_cast<float*>(kb.release().data());
            auto ab = binding.arena->allocate_aligned(a_bytes, 256u);
            if (!ab.ok()) return fail_with_sequence(ab.status());
            state.compact_a = static_cast<float*>(ab.release().data());
            auto request = binding.arena->allocate_aligned(
                sizeof(::shelf::runtime::DeviceRequestDescriptor), 256u);
            if (!request.ok()) return fail_with_sequence(request.status());
            state.commit_request =
                static_cast<::shelf::runtime::DeviceRequestDescriptor*>(request.release().data());
            state.commit_request_host.row_begin = 0u;
            state.commit_request_host.row_count = 0u;
            state.gdn_key_heads = drafter->gdn_key_heads;
            drafter->extra_ranks.push_back(std::move(state));
        }
    }

    drafter->last_tokens.reserve(options.num_drafts + 1u);
    drafter->initialized = true;
    return drafter;
}

Status mtp_drafter_shutdown(MtpDrafter& drafter, hipStream_t stream) noexcept {
    Status first_error = Status::make_ok();
    if (drafter.sequence != nullptr) {
        Status st = state::release_paged_sequence_state(*drafter.sequence,
                                                        stream);
        if (first_error.ok() && !st.ok()) first_error = st;
        drafter.sequence.reset();
    }
    Status st = shutdown_gdn_spec_history(drafter.gdn_history);
    if (first_error.ok() && !st.ok()) first_error = st;
    st = executor_shutdown(drafter.executor);
    if (first_error.ok() && !st.ok()) first_error = st;
    drafter.kv_pool.reset();
    drafter.gdn_pool.reset();
    drafter.slot_pool.reset();
    drafter.hidden_cache = gpu::Tensor{};
    drafter.gdn_conv_snapshot = nullptr;
    drafter.gdn_recurrent_snapshot = nullptr;
    drafter.compact_delta = nullptr;
    drafter.compact_k = nullptr;
    drafter.compact_a = nullptr;
    drafter.commit_request = nullptr;
    drafter.qsa_partial_history = nullptr;
    drafter.qsa_count_history = nullptr;
    drafter.qsa_history_rows = 0u;
    drafter.ple_conv_history = nullptr;
    drafter.ple_conv_state_stride = 0u;
    drafter.ple_conv_history_rows = 0u;
    drafter.initialized = false;
    return first_error;
}

Status mtp_drafter_reset_sequence(MtpDrafter& drafter, hipStream_t stream) noexcept {
    if (!drafter.initialized || drafter.sequence == nullptr) {
        return Status::invalid_state("mtp reset: drafter is not initialized", __FILE__,
                                     __LINE__);
    }
    Status st = state::release_paged_sequence_state(*drafter.sequence, stream);
    if (!st.ok()) return st;
    auto sequence = state::create_paged_sequence_state(
        *drafter.slot_pool, *drafter.gdn_pool, *drafter.kv_pool, drafter.max_seq_len,
        stream);
    if (!sequence.ok()) {
        drafter.sequence.reset();
        return sequence.status();
    }
    drafter.sequence =
        std::make_unique<state::PagedSequenceState>(sequence.release());
    drafter.last_tokens.clear();
    drafter.last_position = 0u;
    drafter.last_hidden_offset = 0u;
    drafter.sample_index = 0u;
    drafter.timing = MtpTiming{};
    return Status::make_ok();
}

Result<MtpPrefillOutput> mtp_spec_prefill(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    const int32_t* prompt_tokens,
    uint32_t prompt_count,
    const SamplingConfig& sampling) {

    if (!drafter.initialized) {
        return Status::invalid_state("mtp_spec_prefill: drafter is not initialized",
                                     __FILE__, __LINE__);
    }
    if (prompt_tokens == nullptr || prompt_count == 0u) {
        return Status::invalid_argument("mtp_spec_prefill: empty prompt", __FILE__,
                                        __LINE__);
    }
    if (prompt_count >= drafter.max_seq_len) {
        return Status::out_of_range("mtp_spec_prefill: prompt leaves no room", __FILE__,
                                    __LINE__);
    }
    if (target_sequence.position != 0u || drafter.sequence->position != 0u) {
        return Status::invalid_state("mtp_spec_prefill: sequence is not empty", __FILE__,
                                     __LINE__);
    }

    const hipStream_t stream = drafter.target_stream;
    const uint32_t chunk_limit =
        target.config.max_batch_rows != 0u ? target.config.max_batch_rows : prompt_count;
    const uint32_t mtp_limit = drafter.executor.config.max_batch_rows;
    drafter.sampling = sampling;
    int32_t pending = -1;
    std::vector<int32_t> chunk_tokens;
    std::vector<int32_t> mtp_tokens;

    for (uint32_t begin = 0; begin < prompt_count; begin += chunk_limit) {
        const uint32_t end = std::min(prompt_count, begin + chunk_limit);
        const uint32_t rows = end - begin;
        const bool final_chunk = end == prompt_count;
        chunk_tokens.assign(prompt_tokens + begin, prompt_tokens + end);
        auto target_output = run_target(
            drafter, target, target_sequence, chunk_tokens, begin, final_chunk,
            final_chunk, 1u, ::shelf::runtime::ExecutionClass::PREFILL, false, sampling,
            drafter.sample_index, nullptr, stream);
        if (!target_output.ok()) return target_output.status();
        if (final_chunk) {
            auto sampled = copy_sampled_token(target_output.value().sampled_tokens, stream);
            if (!sampled.ok()) return sampled.status();
            pending = sampled.value();
            drafter.sample_index += 1u;
            const uint32_t last_row = end - 1u;
            const uint32_t build_end = rows > 1u ? last_row : begin;
            for (uint32_t mtp_begin = begin; mtp_begin < build_end; mtp_begin += mtp_limit) {
                const uint32_t mtp_end = std::min(build_end, mtp_begin + mtp_limit);
                const uint32_t mtp_rows = mtp_end - mtp_begin;
                mtp_tokens.resize(mtp_rows);
                for (uint32_t i = 0u; i < mtp_rows; ++i) {
                    mtp_tokens[i] = prompt_tokens[mtp_begin + i + 1u];
                }
                Status copy = copy_hidden_to_input(
                    drafter, target.normed_token_hidden, mtp_begin - begin, mtp_rows, stream);
                if (!copy.ok()) return copy;
                auto mtp_output = run_mtp_forward(drafter, mtp_tokens, mtp_begin, false,
                                                  stream);
                if (!mtp_output.ok()) return mtp_output.status();
            }
            Status cache = cache_target_hidden(drafter, target, last_row - begin, 1u,
                                               stream);
            if (!cache.ok()) return cache;
            drafter.last_tokens.assign(1u, prompt_tokens[last_row]);
            drafter.last_position = last_row;
            drafter.last_hidden_offset = 0u;
            continue;
        }

        for (uint32_t mtp_begin = begin; mtp_begin < end; mtp_begin += mtp_limit) {
            const uint32_t mtp_end = std::min(end, mtp_begin + mtp_limit);
            const uint32_t mtp_rows = mtp_end - mtp_begin;
            mtp_tokens.resize(mtp_rows);
            for (uint32_t i = 0u; i < mtp_rows; ++i) {
                mtp_tokens[i] = prompt_tokens[mtp_begin + i + 1u];
            }
            Status copy = copy_hidden_to_input(
                drafter, target.normed_token_hidden, mtp_begin - begin, mtp_rows, stream);
            if (!copy.ok()) return copy;
            auto mtp_output = run_mtp_forward(drafter, mtp_tokens, mtp_begin, false,
                                              stream);
            if (!mtp_output.ok()) return mtp_output.status();
        }
    }

    MtpPrefillOutput output;
    output.pending_token = pending;
    return output;
}

namespace {

Result<MtpIterationOutput> mtp_decode_only(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& sequence,
    int32_t pending_token,
    uint32_t remaining_tokens,
    hipStream_t stream) {
    (void)remaining_tokens;
    const uint32_t position = sequence.position;
    std::vector<int32_t> tokens{pending_token};
    auto executed = run_target(
        drafter, target, sequence, tokens, position, true, true, 1u,
        ::shelf::runtime::ExecutionClass::DECODE, false, drafter.sampling,
        drafter.sample_index, nullptr, stream);
    if (!executed.ok()) return executed.status();
    drafter.sample_index += 1u;
    auto sampled = copy_sampled_token(executed.value().sampled_tokens, stream);
    if (!sampled.ok()) return sampled.status();
    Status cache = cache_target_hidden(drafter, target, 0u, 1u, stream);
    if (!cache.ok()) return cache;
    drafter.last_tokens.assign(1u, pending_token);
    drafter.last_position = position;
    drafter.last_hidden_offset = 0u;
    MtpIterationOutput output;
    output.emitted[0] = sampled.value();
    output.emitted_count = 1u;
    output.pending_token = sampled.value();
    output.finished = qwen_hybrid::is_stop_token(target.model.config->stop_tokens,
                                                 sampled.value());
    return output;
}

Result<MtpIterationOutput> accept_verified(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& sequence,
    uint32_t position,
    const int32_t* proposals,
    uint32_t drafts,
    const std::vector<int32_t>& verify_tokens,
    const int32_t* sampled,
    uint32_t remaining_tokens,
    std::vector<MtpGdnRankState>& request_ranks,
    const std::vector<MtpGdnRankState>& log_ranks,
    uint32_t row_base,
    hipStream_t stream) {
    SpecVerifyResult accepted_result;
    Status accepted = spec_greedy_accept(proposals, drafts, sampled, true, accepted_result);
    if (!accepted.ok()) return accepted;
    // Cut at the first stop token before the state commit, so rows past it are not committed.
    const bool stopped = spec_bound_result(accepted_result, remaining_tokens,
        target.model.config->stop_tokens);
    const uint32_t accepted_drafts = accepted_result.num_accepted_drafts;
    MtpIterationOutput output;
    output.finished = stopped;
    output.num_accepted = accepted_drafts;
    output.emitted_count = std::min(
        static_cast<uint32_t>(accepted_result.emitted_tokens.size()), remaining_tokens);
    for (uint32_t i = 0u; i < output.emitted_count; ++i) {
        output.emitted[i] = accepted_result.emitted_tokens[i];
    }
    output.pending_token = accepted_result.correction_valid
        ? accepted_result.correction_token
        : accepted_result.bonus_token;

    const uint32_t page_tokens = sequence.kv_pool()->page_tokens();
    if (page_tokens == 0u) {
        return Status::invalid_state("mtp accept: page_tokens is zero", __FILE__, __LINE__);
    }
    if (accepted_drafts == drafts) {
        const uint32_t committed = drafts + 1u;
        Status cache = cache_target_hidden(drafter, target, row_base, committed, stream);
        if (!cache.ok()) return cache;
        drafter.last_tokens = verify_tokens;
        drafter.last_position = position;
        drafter.last_hidden_offset = 0u;
        const uint32_t required = ceil_div_u32(sequence.position, page_tokens);
        Status rollback = rollback_sequence_append(sequence, required, stream);
        if (!rollback.ok()) return rollback;
        drafter.timing.full_accepts += 1u;
    } else {
        const uint32_t committed = accepted_drafts + 1u;
        const auto restore_started = std::chrono::steady_clock::now();
        for (std::size_t ri = 0u; ri < request_ranks.size(); ++ri) {
            MtpGdnRankState& state = request_ranks[ri];
            auto scope = shelf::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            Status restore = spec_gdn_restore(
                *state.pool, sequence.slot, state.conv_snapshot,
                state.recurrent_snapshot, state.conv_bytes, state.recurrent_bytes,
                state.stream);
            if (!restore.ok()) return restore;
            Status conv = restore_gdn_spec_conv(*state.pool, sequence.slot,
                                                log_ranks[ri].history,
                                                row_base + accepted_drafts, state.stream);
            if (!conv.ok()) return conv;
            Status qsa = restore_rank_qsa(log_ranks[ri], sequence.slot,
                                          row_base + accepted_drafts + 1u);
            if (!qsa.ok()) return qsa;
            Status ple = restore_rank_ple(log_ranks[ri], sequence.slot,
                                          row_base + accepted_drafts + 1u);
            if (!ple.ok()) return ple;
        }
        const auto restore_done = std::chrono::steady_clock::now();
        for (std::size_t ri = 0u; ri < request_ranks.size(); ++ri) {
            MtpGdnRankState& state = request_ranks[ri];
            auto scope = shelf::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            ::shelf::runtime::DeviceRequestDescriptor request = state.commit_request_host;
            request.request_handle = sequence.request_handle();
            request.row_begin = row_base;
            request.row_count = 0u;
            hipError_t upload = hipMemcpyAsync(state.commit_request, &request,
                                               sizeof(request), hipMemcpyHostToDevice,
                                               state.stream);
            if (upload != hipSuccess) {
                return Status::hip_error("mtp commit request upload",
                                         hipGetErrorString(upload), __FILE__, __LINE__);
            }
            GdnCompactLogDeviceView compact{};
            compact.delta = log_ranks[ri].compact_delta;
            compact.k = log_ranks[ri].compact_k;
            compact.a = log_ranks[ri].compact_a;
            compact.delta_layer_stride = log_ranks[ri].compact_delta_layer_stride;
            compact.k_layer_stride = log_ranks[ri].compact_k_layer_stride;
            compact.a_layer_stride = log_ranks[ri].compact_a_layer_stride;
            Status st = commit_gdn_compact_log(
                *state.pool, sequence.slot, compact, state.commit_request, committed,
                state.gdn_key_heads, state.stream);
            if (!st.ok()) return st;
        }
        const auto compact_done = std::chrono::steady_clock::now();
        Status cache = cache_target_hidden(drafter, target, row_base, committed, stream);
        if (!cache.ok()) return cache;
        sequence.position = position + committed;
        const uint32_t required = ceil_div_u32(sequence.position, page_tokens);
        Status rollback = rollback_sequence_append(sequence, required, stream);
        if (!rollback.ok()) return rollback;
        drafter.last_tokens.assign(verify_tokens.begin(),
                                   verify_tokens.begin() + committed);
        drafter.last_position = position;
        drafter.last_hidden_offset = 0u;
        drafter.timing.partial_accepts += 1u;
        drafter.timing.gdn_restores += 1u;
        drafter.timing.gdn_restore_ms +=
            std::chrono::duration<double, std::milli>(restore_done - restore_started).count();
        drafter.timing.compact_replay_ms +=
            std::chrono::duration<double, std::milli>(compact_done - restore_done).count();
    }

    for (uint32_t i = 0u; i < output.emitted_count; ++i) {
        if (qwen_hybrid::is_stop_token(target.model.config->stop_tokens,
                                       output.emitted[i])) {
            output.emitted_count = i + 1u;
            output.finished = true;
            break;
        }
    }
    if (qwen_hybrid::is_stop_token(target.model.config->stop_tokens,
                                   output.pending_token)) {
        output.finished = true;
    }
    return output;
}

}  // namespace

Result<MtpDraftResult> mtp_draft(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    int32_t pending_token,
    uint32_t remaining_tokens) {
    if (!drafter.initialized) {
        return Status::invalid_state("mtp_draft: drafter is not initialized", __FILE__,
                                     __LINE__);
    }
    if (remaining_tokens == 0u) {
        return Status::invalid_argument("mtp_draft: remaining_tokens is zero", __FILE__,
                                        __LINE__);
    }
    if (target_sequence.position != drafter.last_position + drafter.last_tokens.size()) {
        return Status::invalid_state("mtp_draft: target sequence diverged from drafter",
                                     __FILE__, __LINE__);
    }

    const hipStream_t stream = drafter.target_stream;
    const uint32_t position = target_sequence.position;
    const uint32_t room = position < drafter.max_seq_len
        ? drafter.max_seq_len - position
        : 0u;
    uint32_t drafts = std::min(drafter.num_drafts, remaining_tokens - 1u);
    if (room < 2u) drafts = 0u;
    else drafts = std::min(drafts, room - 1u);

    MtpDraftResult result;
    result.position = position;
    result.drafts = drafts;
    if (drafts == 0u) {
        result.decode_only = true;
        return result;
    }

    const uint32_t prefix_rows = target_sequence.position - drafter.last_position;
    if (prefix_rows == 0u || prefix_rows > drafter.last_tokens.size()) {
        return Status::invalid_state("mtp_draft: invalid target prefix", __FILE__,
                                     __LINE__);
    }
    const uint32_t required_rewind = ceil_div_u32(drafter.last_position,
                                                  drafter.sequence->kv_pool()->page_tokens());
    Status rewind = rewind_sequence(*drafter.sequence, drafter.last_position,
                                    required_rewind, stream);
    if (!rewind.ok()) return rewind;

    std::vector<int32_t> first_pass(prefix_rows);
    for (uint32_t i = 0u; i + 1u < prefix_rows; ++i) {
        first_pass[i] = drafter.last_tokens[i + 1u];
    }
    first_pass[prefix_rows - 1u] = pending_token;
    Status copy = copy_hidden_to_input(drafter, drafter.hidden_cache, 0u, prefix_rows,
                                       stream);
    if (!copy.ok()) return copy;
    auto first = run_mtp_forward(drafter, first_pass, drafter.last_position, true,
                                 stream);
    if (!first.ok()) return first.status();
    auto first_sampled = copy_sampled_token(first.value().sampled_tokens, stream);
    if (!first_sampled.ok()) return first_sampled.status();

    result.proposals[0] = first_sampled.value();
    copy = copy_hidden_to_input(drafter, drafter_hidden_output(drafter), 0u, 1u, stream);
    if (!copy.ok()) return copy;
    for (uint32_t i = 1u; i < drafts; ++i) {
        std::vector<int32_t> step_tokens{result.proposals[i - 1u]};
        auto step = run_mtp_forward(drafter, step_tokens, position + i - 1u, true,
                                    stream);
        if (!step.ok()) return step.status();
        auto sampled = copy_sampled_token(step.value().sampled_tokens, stream);
        if (!sampled.ok()) return sampled.status();
        result.proposals[i] = sampled.value();
        copy = copy_hidden_to_input(drafter, drafter_hidden_output(drafter), 0u, 1u,
                                    stream);
        if (!copy.ok()) return copy;
    }

    std::vector<MtpGdnRankState> rank_states = all_rank_states(drafter);
    for (const MtpGdnRankState& state : rank_states) {
        auto scope = shelf::gpu::ScopedDevice::create(state.device);
        if (!scope.ok()) return scope.status();
        Status snapshot = spec_gdn_snapshot(
            *state.pool, target_sequence.slot, state.conv_snapshot,
            state.recurrent_snapshot, state.conv_bytes, state.recurrent_bytes,
            state.stream);
        if (!snapshot.ok()) return snapshot;
    }
    return result;
}

Result<MtpIterationOutput> mtp_spec_step(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    int32_t pending_token,
    uint32_t remaining_tokens) {
    if (qwen_hybrid::is_stop_token(target.model.config->stop_tokens, pending_token)) {
        // The stop token was already emitted by the caller; do not draft past it.
        MtpIterationOutput stopped;
        stopped.pending_token = pending_token;
        stopped.finished = true;
        return stopped;
    }
    const auto started = std::chrono::steady_clock::now();
    auto draft = mtp_draft(drafter, target, target_sequence, pending_token,
                           remaining_tokens);
    if (!draft.ok()) return draft.status();
    const MtpDraftResult& plan = draft.value();
    const hipStream_t stream = drafter.target_stream;
    if (plan.decode_only) {
        auto output = mtp_decode_only(drafter, target, target_sequence, pending_token,
                                      remaining_tokens, stream);
        if (output.ok()) {
            drafter.timing.rounds += 1u;
            drafter.timing.generated_tokens += output.value().emitted_count;
            drafter.timing.verify_rows += 1u;
            drafter.timing.round_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        }
        return output;
    }

    const auto draft_done = std::chrono::steady_clock::now();
    std::vector<MtpGdnRankState> rank_states = all_rank_states(drafter);
    std::vector<int32_t> verify_tokens(plan.drafts + 1u);
    verify_tokens[0] = pending_token;
    for (uint32_t i = 0u; i < plan.drafts; ++i) {
        verify_tokens[i + 1u] = plan.proposals[i];
    }
    std::vector<ExecuteBatchOptions> verify_options =
        build_verify_options(drafter, plan.drafts);
    auto verified = run_target(
        drafter, target, target_sequence, verify_tokens, plan.position, true, true,
        plan.drafts + 1u,
        plan.drafts + 1u == 1u ? ::shelf::runtime::ExecutionClass::DECODE
                               : ::shelf::runtime::ExecutionClass::SPEC_VERIFY,
        true, drafter.sampling, drafter.sample_index, &verify_options, stream);
    if (!verified.ok()) return verified.status();
    drafter.sample_index += static_cast<uint64_t>(plan.drafts) + 1u;

    std::vector<int32_t> sampled(plan.drafts + 1u, -1);
    const hipError_t copy_err = hipMemcpy(
        sampled.data(), verified.value().sampled_tokens.data<int32_t>(),
        sampled.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
    if (copy_err != hipSuccess) {
        return Status::hip_error("mtp verify sample copy", hipGetErrorString(copy_err),
                                 __FILE__, __LINE__);
    }
    if (sampled[0] < 0) {
        return Status::invalid_state("mtp verify produced no sampled token", __FILE__,
                                     __LINE__);
    }
    const auto verify_done = std::chrono::steady_clock::now();
    auto output = accept_verified(
        drafter, target, target_sequence, plan.position, plan.proposals.data(),
        plan.drafts, verify_tokens, sampled.data(), remaining_tokens, rank_states,
        rank_states, 0u, stream);
    if (!output.ok()) return output.status();

    drafter.timing.rounds += 1u;
    drafter.timing.accepted_drafts += output.value().num_accepted;
    drafter.timing.generated_tokens += output.value().emitted_count;
    drafter.timing.verify_rows += plan.drafts + 1u;
    const auto finished_at = std::chrono::steady_clock::now();
    drafter.timing.draft_ms +=
        std::chrono::duration<double, std::milli>(draft_done - started).count();
    drafter.timing.verify_ms +=
        std::chrono::duration<double, std::milli>(verify_done - draft_done).count();
    drafter.timing.round_ms +=
        std::chrono::duration<double, std::milli>(finished_at - started).count();
    return output;
}

std::size_t mtp_spec_batch_reserve_bytes(
    const shelf::qwen_hybrid::QwenHybridConfig& config,
    const MtpDrafterConfig& options,
    std::size_t max_requests,
    std::size_t token_row_intermediate,
    std::size_t grouped_executor_bytes,
    bool exl3_linear_workspace) noexcept {
    if (max_requests == 0u || options.max_seq_len == 0u || options.page_tokens == 0u) {
        return 0u;
    }
    MtpDrafterConfig per_request = options;
    per_request.allocate_spec_buffers = false;
    const std::size_t fixed = mtp_drafter_reserve_bytes(
        config, per_request, token_row_intermediate, grouped_executor_bytes,
        exl3_linear_workspace);

    const state::GdnStatePoolLayout gdn = make_qwen35_gdn_state_layout(config);
    const std::uint64_t rows = options.max_batch_rows;
    const std::size_t history = static_cast<std::size_t>(rows) * gdn.num_gdn_states *
        gdn.conv_history_stride * gdn.conv_history * sizeof(bf16_t);
    const std::uint64_t vdim = static_cast<std::uint64_t>(gdn.num_v_heads) * gdn.head_v;
    const std::uint64_t qkdim = static_cast<std::uint64_t>(gdn.num_v_heads) * gdn.head_k;
    const std::size_t compact = static_cast<std::size_t>(gdn.num_gdn_states) * rows *
        (vdim + qkdim + gdn.num_v_heads) * sizeof(float);
    return fixed * max_requests + history + compact;
}

Result<MtpSpecBatchContext> create_mtp_spec_batch_context(
    Executor& target,
    const MtpDrafterInput& input,
    const state::GdnStatePool& target_gdn_pool,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const MtpDrafterConfig& options,
    uint32_t max_requests,
    const std::vector<MtpGdnRankBinding>* extra_ranks,
    TpBatchHook* tp_hook) {
    if (max_requests == 0u) {
        return Status::invalid_argument("mtp batch context needs a positive slot count",
                                        __FILE__, __LINE__);
    }
    if (options.max_batch_rows < options.num_drafts + 1u) {
        return Status::invalid_argument("mtp batch rows are below one verify round",
                                        __FILE__, __LINE__);
    }
    MtpSpecBatchContext context;
    context.target = &target;
    context.max_rows = options.max_batch_rows;
    context.max_requests = max_requests;
    context.num_drafts = options.num_drafts;

    MtpDrafterConfig per_request = options;
    per_request.allocate_spec_buffers = false;
    context.drafters.reserve(max_requests);
    for (uint32_t i = 0u; i < max_requests; ++i) {
        auto drafter = create_mtp_drafter(input, &target, target_gdn_pool, arena,
                                          stream, per_request, extra_ranks, tp_hook);
        if (!drafter.ok()) {
            (void)shutdown_mtp_spec_batch_context(context);
            return drafter.status();
        }
        context.drafters.push_back(std::move(drafter.value()));
    }

    int rank0_device = 0;
    hipGetDevice(&rank0_device);
    auto make_rank = [&](const state::GdnStatePool& pool, gpu::GpuArena& rank_arena,
                         hipStream_t rank_stream, int device,
                         Executor* executor) -> Result<MtpGdnRankState> {
        MtpGdnRankState state;
        state.pool = &pool;
        state.stream = rank_stream;
        state.device = device;
        state.target = executor;
        Status qsa = allocate_qsa_history(
            executor, rank_arena, options.max_batch_rows, state.qsa_partial_history,
            state.qsa_count_history, state.qsa_partial_state_stride,
            state.qsa_count_state_stride);
        if (!qsa.ok()) return qsa;
        state.qsa_history_rows = options.max_batch_rows;
        Status ple = allocate_ple_conv_history(
            executor, rank_arena, options.max_batch_rows, state.ple_conv_history,
            state.ple_conv_state_stride);
        if (!ple.ok()) return ple;
        state.ple_conv_history_rows = options.max_batch_rows;
        auto history = create_gdn_spec_history(rank_arena, pool, options.max_batch_rows,
                                               false);
        if (!history.ok()) return history.status();
        state.history = history.release();
        const auto view = pool.device_view();
        const std::uint64_t rows = options.max_batch_rows;
        const std::uint64_t vdim = static_cast<std::uint64_t>(view.num_v_heads) * view.head_v;
        const std::uint64_t qkdim = static_cast<std::uint64_t>(view.num_v_heads) * view.head_k;
        state.compact_delta_layer_stride = rows * vdim;
        state.compact_k_layer_stride = rows * qkdim;
        state.compact_a_layer_stride = rows * view.num_v_heads;
        const std::size_t delta_bytes = static_cast<std::size_t>(view.num_gdn_states) *
            state.compact_delta_layer_stride * sizeof(float);
        const std::size_t k_bytes = static_cast<std::size_t>(view.num_gdn_states) *
            state.compact_k_layer_stride * sizeof(float);
        const std::size_t a_bytes = static_cast<std::size_t>(view.num_gdn_states) *
            state.compact_a_layer_stride * sizeof(float);
        auto delta = rank_arena.allocate_aligned(delta_bytes, 256u);
        if (!delta.ok()) return delta.status();
        state.compact_delta = static_cast<float*>(delta.release().data());
        auto kb = rank_arena.allocate_aligned(k_bytes, 256u);
        if (!kb.ok()) return kb.status();
        state.compact_k = static_cast<float*>(kb.release().data());
        auto ab = rank_arena.allocate_aligned(a_bytes, 256u);
        if (!ab.ok()) return ab.status();
        state.compact_a = static_cast<float*>(ab.release().data());
        return state;
    };

    auto rank0 = make_rank(target_gdn_pool, arena, stream, rank0_device, &target);
    if (!rank0.ok()) {
        (void)shutdown_mtp_spec_batch_context(context);
        return rank0.status();
    }
    context.ranks.push_back(std::move(rank0.value()));
    if (extra_ranks != nullptr) {
        for (const MtpGdnRankBinding& binding : *extra_ranks) {
            if (binding.pool == nullptr || binding.arena == nullptr) {
                (void)shutdown_mtp_spec_batch_context(context);
                return Status::invalid_argument("mtp batch rank binding is incomplete",
                                                __FILE__, __LINE__);
            }
            auto state = make_rank(*binding.pool, *binding.arena, binding.stream,
                                   binding.device, binding.executor);
            if (!state.ok()) {
                (void)shutdown_mtp_spec_batch_context(context);
                return state.status();
            }
            context.ranks.push_back(std::move(state.value()));
        }
    }
    context.initialized = true;
    return context;
}

Status shutdown_mtp_spec_batch_context(MtpSpecBatchContext& context) noexcept {
    Status first_error = Status::make_ok();
    for (MtpGdnRankState& rank : context.ranks) {
        Status st = shutdown_gdn_spec_history(rank.history);
        if (first_error.ok() && !st.ok()) first_error = st;
        rank.compact_delta = nullptr;
        rank.compact_k = nullptr;
        rank.compact_a = nullptr;
    }
    context.ranks.clear();
    for (std::size_t i = context.drafters.size(); i-- > 0u;) {
        if (context.drafters[i] == nullptr) continue;
        Status st = mtp_drafter_shutdown(*context.drafters[i], nullptr);
        if (first_error.ok() && !st.ok()) first_error = st;
        context.drafters[i].reset();
    }
    context.initialized = false;
    return first_error;
}

Result<MtpSpecBatchOutput> mtp_spec_step_batch(
    MtpSpecBatchContext& context,
    std::span<const MtpSpecBatchSlot> slots) {
    if (!context.initialized || context.target == nullptr) {
        return Status::invalid_state("mtp batch: context is not initialized", __FILE__,
                                     __LINE__);
    }
    if (slots.empty()) {
        return Status::invalid_argument("mtp batch: no slots", __FILE__, __LINE__);
    }
    if (slots.size() > context.max_requests) {
        return Status::out_of_range("mtp batch: more slots than the context holds",
                                    __FILE__, __LINE__);
    }
    const auto started = std::chrono::steady_clock::now();
    MtpSpecBatchOutput out;
    out.slots.resize(slots.size());
    std::vector<MtpDraftResult> drafts(slots.size());
    std::vector<uint32_t> row_begin(slots.size(), 0u);
    std::vector<char> stopped_slot(slots.size(), 0);
    uint32_t total_rows = 0u;
    for (std::size_t i = 0u; i < slots.size(); ++i) {
        const MtpSpecBatchSlot& slot = slots[i];
        if (slot.drafter == nullptr || slot.sequence == nullptr) {
            return Status::invalid_argument("mtp batch: slot is incomplete", __FILE__,
                                            __LINE__);
        }
        row_begin[i] = total_rows;
        if (qwen_hybrid::is_stop_token(context.target->model.config->stop_tokens,
                                       slot.pending_token)) {
            // The stop token was already emitted by the caller; this slot runs nothing.
            stopped_slot[i] = 1;
            drafts[i].decode_only = true;
            out.slots[i].pending_token = slot.pending_token;
            out.slots[i].finished = true;
            continue;
        }
        auto draft = mtp_draft(*slot.drafter, *context.target, *slot.sequence,
                               slot.pending_token, slot.remaining_tokens);
        if (!draft.ok()) return draft.status();
        drafts[i] = draft.value();
        if (!drafts[i].decode_only) {
            total_rows += drafts[i].drafts + 1u;
        }
    }
    if (total_rows > context.max_rows) {
        return Status::out_of_range("mtp batch: rows exceed the context capacity",
                                    __FILE__, __LINE__);
    }
    const hipStream_t stream = context.ranks.empty() ? nullptr
                                                     : context.ranks.front().stream;
    std::vector<std::size_t> pending;
    pending.reserve(slots.size());
    for (std::size_t i = 0u; i < slots.size(); ++i) {
        if (!drafts[i].decode_only) pending.push_back(i);
    }

    std::vector<std::size_t> chunk;
    std::vector<uint32_t> chunk_row_begin;
    std::vector<int32_t> token_ids;
    std::vector<ScheduledRequest> requests;
    std::vector<int32_t> sampled;
    std::size_t cursor = 0u;
    while (cursor < pending.size()) {
        chunk.clear();
        chunk_row_begin.clear();
        uint32_t chunk_rows = 0u;
        while (cursor < pending.size() &&
               chunk.size() < ::shelf::kernel::kGdnRecurrenceVerifyMaxRequests) {
            const std::size_t i = pending[cursor];
            const uint32_t rows = drafts[i].drafts + 1u;
            if (!chunk.empty() &&
                chunk_rows + rows > ::shelf::kernel::kGdnRecurrenceVerifyBatchRows) {
                break;
            }
            chunk.push_back(i);
            chunk_row_begin.push_back(chunk_rows);
            chunk_rows += rows;
            ++cursor;
        }
        if (chunk.empty()) {
            return Status::invalid_state("mtp batch verify chunk is empty", __FILE__,
                                         __LINE__);
        }
        token_ids.clear();
        token_ids.reserve(chunk_rows);
        requests.clear();
        requests.reserve(chunk.size());
        for (std::size_t c = 0u; c < chunk.size(); ++c) {
            const std::size_t i = chunk[c];
            const MtpSpecBatchSlot& slot = slots[i];
            std::vector<MtpGdnRankState> request_ranks = all_rank_states(*slot.drafter);
            for (MtpGdnRankState& state : request_ranks) {
                auto scope = shelf::gpu::ScopedDevice::create(state.device);
                if (!scope.ok()) return scope.status();
                ::shelf::runtime::DeviceRequestDescriptor request =
                    state.commit_request_host;
                request.request_handle = slot.sequence->request_handle();
                request.row_begin = chunk_row_begin[c];
                request.row_count = 0u;
                if (state.commit_request != nullptr) {
                    const hipError_t err = hipMemcpy(
                        state.commit_request, &request, sizeof(request),
                        hipMemcpyHostToDevice);
                    if (err != hipSuccess) {
                        return Status::hip_error("mtp batch commit descriptor",
                                                 hipGetErrorString(err), __FILE__,
                                                 __LINE__);
                    }
                }
            }
            token_ids.push_back(slot.pending_token);
            for (uint32_t j = 0u; j < drafts[i].drafts; ++j) {
                token_ids.push_back(drafts[i].proposals[j]);
            }
            ScheduledRequest req;
            req.sequence = slot.sequence;
            req.handle = slot.sequence->request_handle();
            req.execution_class = ::shelf::runtime::ExecutionClass::SPEC_VERIFY;
            req.token_begin = chunk_row_begin[c];
            req.num_tokens = drafts[i].drafts + 1u;
            req.prefix_tokens = drafts[i].position;
            req.compute_logits = true;
            req.sample = true;
            req.num_output_rows = drafts[i].drafts + 1u;
            req.sampling = slot.drafter->sampling;
            req.sampling_index = slot.drafter->sample_index;
            requests.push_back(req);
        }

        ScheduledBatch batch;
        batch.token_ids = token_ids.data();
        batch.token_ids_location = shelf::qwen35::TokenIdsLocation::Host;
        batch.requests = requests;
        batch.num_tokens = chunk_rows;
        batch.num_requests = static_cast<uint32_t>(requests.size());
        batch.num_decode_requests = 0u;
        batch.num_verify_requests = static_cast<uint32_t>(requests.size());
        batch.num_prefill_requests = 0u;
        batch.speculative_verify = true;

        std::vector<ExecuteBatchOptions> options(context.ranks.size());
        for (std::size_t r = 0u; r < options.size(); ++r) {
            options[r].gdn_spec_history =
                gdn_spec_history_view(context.ranks[r].history, chunk_rows);
            options[r].gdn_compact.delta = context.ranks[r].compact_delta;
            options[r].gdn_compact.k = context.ranks[r].compact_k;
            options[r].gdn_compact.a = context.ranks[r].compact_a;
            options[r].gdn_compact.delta_layer_stride =
                context.ranks[r].compact_delta_layer_stride;
            options[r].gdn_compact.k_layer_stride = context.ranks[r].compact_k_layer_stride;
            options[r].gdn_compact.a_layer_stride = context.ranks[r].compact_a_layer_stride;
            options[r].qsa_spec_history.partial = context.ranks[r].qsa_partial_history;
            options[r].qsa_spec_history.counts = context.ranks[r].qsa_count_history;
            options[r].qsa_spec_history.rows = context.ranks[r].qsa_history_rows;
            options[r].qsa_spec_history.partial_state_stride =
                context.ranks[r].qsa_partial_state_stride;
            options[r].qsa_spec_history.count_state_stride =
                context.ranks[r].qsa_count_state_stride;
            options[r].ple_conv_history.store = context.ranks[r].ple_conv_history;
            options[r].ple_conv_history.rows = context.ranks[r].ple_conv_history_rows;
            options[r].ple_conv_history.state_stride = context.ranks[r].ple_conv_state_stride;
        }
        MtpDrafter& leader = *slots[chunk.front()].drafter;
        auto executed = leader.tp_hook != nullptr
            ? leader.tp_hook->on_execute(batch, options.data())
            : execute_batch(*context.target, batch, stream, options[0]);
        if (!executed.ok()) return executed.status();
        const BatchExecutionOutput output = executed.release();
        if (output.num_outputs < chunk_rows) {
            return Status::invalid_state("mtp batch verify output rows mismatch", __FILE__,
                                         __LINE__);
        }
        sampled.assign(chunk_rows, -1);
        const hipError_t err = hipMemcpy(
            sampled.data(), output.sampled_tokens.data<int32_t>(),
            static_cast<std::size_t>(chunk_rows) * sizeof(int32_t),
            hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            return Status::hip_error("mtp batch sample copy", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }

        for (std::size_t c = 0u; c < chunk.size(); ++c) {
            const std::size_t i = chunk[c];
            const MtpSpecBatchSlot& slot = slots[i];
            std::vector<MtpGdnRankState> request_ranks = all_rank_states(*slot.drafter);
            std::vector<int32_t> verify_tokens(drafts[i].drafts + 1u);
            verify_tokens[0] = slot.pending_token;
            for (uint32_t j = 0u; j < drafts[i].drafts; ++j) {
                verify_tokens[j + 1u] = drafts[i].proposals[j];
            }
            auto accepted = accept_verified(
                *slot.drafter, *context.target, *slot.sequence, drafts[i].position,
                drafts[i].proposals.data(), drafts[i].drafts, verify_tokens,
                sampled.data() + chunk_row_begin[c], slot.remaining_tokens, request_ranks,
                context.ranks, chunk_row_begin[c], stream);
            if (!accepted.ok()) return accepted.status();
            slot.drafter->sample_index += static_cast<uint64_t>(drafts[i].drafts) + 1u;
            out.slots[i] = accepted.value();
        }
    }

    for (std::size_t i = 0u; i < slots.size(); ++i) {
        if (!drafts[i].decode_only || stopped_slot[i]) continue;
        const MtpSpecBatchSlot& slot = slots[i];
        auto output = mtp_decode_only(*slot.drafter, *context.target, *slot.sequence,
                                      slot.pending_token, slot.remaining_tokens, stream);
        if (!output.ok()) return output.status();
        out.slots[i] = output.value();
    }
    out.total_rows = total_rows;
    out.rounds = 1u;
    out.round_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}

}  // namespace shelf::qwen35::runtime
