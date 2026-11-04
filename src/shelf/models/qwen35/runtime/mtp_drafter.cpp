#include <shelf/models/qwen35/runtime/mtp_drafter.h>

#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/qwen35/runtime/spec_decode.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace shelf::qwen35::runtime {

namespace {

constexpr uint32_t kMtpKvLayers = 1u;

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) noexcept {
    return (value + divisor - 1u) / divisor;
}

Qwen35TextConfig mtp_executor_config(const Qwen35TextConfig& source) {
    Qwen35TextConfig config = source;
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
                            uint32_t rows, hipStream_t stream) {
    if (rows == 0u) return Status::make_ok();
    const std::size_t hidden = drafter.config.hidden_size;
    void* dst = drafter.executor.input_hidden.data<void>();
    if (dst == nullptr) {
        return Status::invalid_state("mtp drafter has no hidden input buffer", __FILE__,
                                     __LINE__);
    }
    const hipError_t err = hipMemcpyAsync(
        dst, source.data<void>(), static_cast<std::size_t>(rows) * hidden * sizeof(bf16_t),
        hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("mtp hidden input copy", hipGetErrorString(err), __FILE__,
                                 __LINE__);
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
                                        hipStream_t stream) {
    (void)drafter;
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
    return execute_batch(target, batch, stream);
}

}  // namespace

std::size_t mtp_drafter_reserve_bytes(const Qwen35TextConfig& config,
                                      const MtpDrafterConfig& options) noexcept {
    if (options.max_seq_len == 0u || options.page_tokens == 0u) {
        return 0u;
    }
    Qwen35TextConfig mtp = mtp_executor_config(config);
    const uint32_t pages =
        ceil_div_u32(options.max_seq_len + 1u, options.page_tokens) + 1u;
    const std::size_t kv = state::PagedKVPool::page_bytes(
        options.page_tokens, kMtpKvLayers,
        static_cast<uint32_t>(mtp.num_key_value_heads),
        static_cast<uint32_t>(mtp.attention_head_dim), options.kv_dtype) *
        static_cast<std::size_t>(pages);

    ExecutorConfig executor_config;
    executor_config.max_batch_rows =
        std::max(options.max_batch_rows, options.num_drafts + 1u);
    executor_config.max_scheduled_requests = 1u;
    executor_config.max_scheduled_output_rows = 1u;
    executor_config.external_hidden_input = true;
    const std::size_t executor = qwen35_executor_reserve_bytes(
        mtp, config.ffn.intermediate_size, executor_config, 1u);

    const state::GdnStatePoolLayout gdn = make_qwen35_gdn_state_layout(config);
    const std::size_t gdn_state =
        static_cast<std::size_t>(gdn.num_gdn_states) *
        (static_cast<std::size_t>(gdn.conv_history_stride) * gdn.conv_history *
             sizeof(bf16_t) +
         static_cast<std::size_t>(gdn.num_v_heads) * gdn.head_k * gdn.head_v *
             sizeof(fp16_t));
    const std::size_t snapshots = 2u * gdn_state;
    constexpr std::size_t kMargin = 64ull * 1024ull * 1024ull;
    return kv + executor + gdn_state + snapshots + kMargin;
}

Result<std::unique_ptr<MtpDrafter>> create_mtp_drafter(
    const Qwen35Model& model,
    const state::GdnStatePool& target_gdn_pool,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const MtpDrafterConfig& options) {

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
    drafter->config = mtp_executor_config(model.text_config());
    drafter->max_seq_len = options.max_seq_len;
    drafter->num_drafts = options.num_drafts;

    auto lower = lower_qwen35_mtp_to_primitives(model.text_config(), model.weights(),
                                                 options.rope_profile);
    if (!lower.ok()) return lower.status();

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

    ExecutorModelInput input;
    input.model.config = &drafter->config;
    input.model.embed_tokens = &model.weights().embed_tokens;
    input.lowered = lower.release();

    ExecutorConfig executor_config;
    executor_config.max_batch_rows =
        std::max(options.max_batch_rows, options.num_drafts + 1u);
    executor_config.max_scheduled_requests = 1u;
    executor_config.max_scheduled_output_rows = 1u;
    executor_config.external_hidden_input = true;
    executor_config.rope_profile = options.rope_profile;
    executor_config.backend = runtime::DecodeBackend::Hip;

    auto executor = create_executor(std::move(input), *drafter->slot_pool,
                                    *drafter->gdn_pool, *drafter->kv_pool, arena,
                                    executor_config);
    if (!executor.ok()) {
        Status st = state::release_paged_sequence_state(*drafter->sequence, stream);
        (void)st;
        return executor.status();
    }
    drafter->executor = executor.release();

    auto fail_with_sequence = [&](const Status& status) {
        Status st = executor_shutdown(drafter->executor);
        (void)st;
        st = state::release_paged_sequence_state(*drafter->sequence, stream);
        (void)st;
        return status;
    };
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
    Status st = executor_shutdown(drafter.executor);
    if (first_error.ok() && !st.ok()) first_error = st;
    drafter.kv_pool.reset();
    drafter.gdn_pool.reset();
    drafter.slot_pool.reset();
    drafter.gdn_conv_snapshot = nullptr;
    drafter.gdn_recurrent_snapshot = nullptr;
    drafter.initialized = false;
    return first_error;
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

    const hipStream_t stream = nullptr;
    const uint32_t chunk_limit =
        target.config.max_batch_rows != 0u ? target.config.max_batch_rows : prompt_count;
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
            drafter.sample_index, stream);
        if (!target_output.ok()) return target_output.status();
        if (final_chunk) {
            auto sampled = copy_sampled_token(target_output.value().sampled_tokens, stream);
            if (!sampled.ok()) return sampled.status();
            pending = sampled.value();
            drafter.sample_index += 1u;
            drafter.last_tokens.assign(prompt_tokens + begin, prompt_tokens + end);
            drafter.last_position = begin;
            continue;
        }

        mtp_tokens.resize(rows);
        for (uint32_t i = 0u; i < rows; ++i) {
            mtp_tokens[i] = prompt_tokens[begin + i + 1u];
        }
        Status copy = copy_hidden_to_input(drafter, target.normed_token_hidden, rows,
                                           stream);
        if (!copy.ok()) return copy;
        auto mtp_output = run_mtp_forward(drafter, mtp_tokens, begin, false, stream);
        if (!mtp_output.ok()) return mtp_output.status();
    }

    MtpPrefillOutput output;
    output.pending_token = pending;
    return output;
}

Result<MtpIterationOutput> mtp_spec_step(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    int32_t pending_token,
    uint32_t remaining_tokens) {

    if (!drafter.initialized) {
        return Status::invalid_state("mtp_spec_step: drafter is not initialized",
                                     __FILE__, __LINE__);
    }
    if (remaining_tokens == 0u) {
        return Status::invalid_argument("mtp_spec_step: remaining_tokens is zero",
                                        __FILE__, __LINE__);
    }
    if (target_sequence.position != drafter.last_position + drafter.last_tokens.size()) {
        return Status::invalid_state("mtp_spec_step: target sequence diverged from drafter",
                                     __FILE__, __LINE__);
    }

    const hipStream_t stream = nullptr;
    const uint32_t position = target_sequence.position;
    const uint32_t room = position < drafter.max_seq_len
        ? drafter.max_seq_len - position
        : 0u;
    uint32_t drafts = std::min(drafter.num_drafts, remaining_tokens - 1u);
    if (room < 2u) drafts = 0u;
    else drafts = std::min(drafts, room - 1u);

    MtpIterationOutput output;
    auto started = std::chrono::steady_clock::now();

    if (drafts == 0u) {
        std::vector<int32_t> tokens{pending_token};
        auto executed = run_target(
            drafter, target, target_sequence, tokens, position, true, true, 1u,
            ::shelf::runtime::ExecutionClass::DECODE, false,
            drafter.sampling, drafter.sample_index, stream);
        if (!executed.ok()) return executed.status();
        drafter.sample_index += 1u;
        auto sampled = copy_sampled_token(executed.value().sampled_tokens, stream);
        if (!sampled.ok()) return sampled.status();
        drafter.last_tokens.assign(1u, pending_token);
        drafter.last_position = position;
        output.emitted[0] = sampled.value();
        output.emitted_count = 1u;
        output.pending_token = sampled.value();
        output.finished =
            qwen_hybrid::is_stop_token(target.model.config->stop_tokens,
                                       sampled.value());
        return output;
    }

    const uint32_t prefix_rows =
        target_sequence.position - drafter.last_position;
    if (prefix_rows == 0u || prefix_rows > drafter.last_tokens.size()) {
        return Status::invalid_state("mtp_spec_step: invalid target prefix", __FILE__,
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
    Status copy = copy_hidden_to_input(drafter, target.normed_token_hidden,
                                       prefix_rows, stream);
    if (!copy.ok()) return copy;
    auto first = run_mtp_forward(drafter, first_pass, drafter.last_position, true,
                                 stream);
    if (!first.ok()) return first.status();
    auto first_sampled = copy_sampled_token(first.value().sampled_tokens, stream);
    if (!first_sampled.ok()) return first_sampled.status();

    std::array<int32_t, kMtpMaxDrafts> proposals{};
    proposals[0] = first_sampled.value();
    copy = copy_hidden_to_input(drafter, drafter.executor.token_hidden, 1u, stream);
    if (!copy.ok()) return copy;

    for (uint32_t i = 1u; i < drafts; ++i) {
        std::vector<int32_t> step_tokens{proposals[i - 1u]};
        auto step = run_mtp_forward(drafter, step_tokens, position + i - 1u, true,
                                    stream);
        if (!step.ok()) return step.status();
        auto sampled = copy_sampled_token(step.value().sampled_tokens, stream);
        if (!sampled.ok()) return sampled.status();
        proposals[i] = sampled.value();
        copy = copy_hidden_to_input(drafter, drafter.executor.token_hidden, 1u, stream);
        if (!copy.ok()) return copy;
    }

    auto draft_started = std::chrono::steady_clock::now();

    Status snapshot = spec_gdn_snapshot(
        *target.gdn_state_pool, target_sequence.slot, drafter.gdn_conv_snapshot,
        drafter.gdn_recurrent_snapshot, drafter.gdn_conv_bytes,
        drafter.gdn_recurrent_bytes, stream);
    if (!snapshot.ok()) return snapshot;

    std::vector<int32_t> verify_tokens(drafts + 1u);
    verify_tokens[0] = pending_token;
    for (uint32_t i = 0u; i < drafts; ++i) verify_tokens[i + 1u] = proposals[i];

    auto verified = run_target(
        drafter, target, target_sequence, verify_tokens, position, true, true,
        drafts + 1u,
        drafts + 1u == 1u ? ::shelf::runtime::ExecutionClass::DECODE
                          : ::shelf::runtime::ExecutionClass::SPEC_VERIFY,
        true, drafter.sampling, drafter.sample_index, stream);
    if (!verified.ok()) return verified.status();
    drafter.sample_index += static_cast<uint64_t>(drafts) + 1u;

    std::vector<int32_t> sampled(drafts + 1u, -1);
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

    SpecVerifyResult accepted_result;
    Status accepted = spec_greedy_accept(proposals.data(), drafts, sampled.data(), true,
                                          accepted_result);
    if (!accepted.ok()) return accepted;
    const uint32_t accepted_drafts = accepted_result.num_accepted_drafts;
    output.num_accepted = accepted_drafts;
    output.emitted_count =
        std::min(static_cast<uint32_t>(accepted_result.emitted_tokens.size()),
                 remaining_tokens);
    for (uint32_t i = 0u; i < output.emitted_count; ++i) {
        output.emitted[i] = accepted_result.emitted_tokens[i];
    }
    output.pending_token = accepted_result.correction_valid
        ? accepted_result.correction_token
        : accepted_result.bonus_token;

    auto verify_done = std::chrono::steady_clock::now();

    const uint32_t page_tokens = target_sequence.kv_pool()->page_tokens();
    if (accepted_drafts == drafts) {
        drafter.last_tokens = verify_tokens;
        drafter.last_position = position;
        const uint32_t required = ceil_div_u32(target_sequence.position, page_tokens);
        Status rollback = rollback_sequence_append(target_sequence, required, stream);
        if (!rollback.ok()) return rollback;
        drafter.timing.full_accepts += 1u;
    } else {
        Status restore = spec_gdn_restore(
            *target.gdn_state_pool, target_sequence.slot, drafter.gdn_conv_snapshot,
            drafter.gdn_recurrent_snapshot, drafter.gdn_conv_bytes,
            drafter.gdn_recurrent_bytes, stream);
        if (!restore.ok()) return restore;
        const uint32_t committed = accepted_drafts + 1u;
        const uint32_t required = ceil_div_u32(position + committed, page_tokens);
        target_sequence.position = position;
        Status rollback = rollback_sequence_append(target_sequence, required, stream);
        if (!rollback.ok()) return rollback;
        std::vector<int32_t> rerun(verify_tokens.begin(),
                                   verify_tokens.begin() + committed);
        for (uint32_t i = 0u; i < committed; ++i) {
            std::vector<int32_t> one{rerun[i]};
            auto rerun_output = run_target(
                drafter, target, target_sequence, one, position + i, false, false, 1u,
                ::shelf::runtime::ExecutionClass::DECODE, false, drafter.sampling,
                drafter.sample_index, stream);
            if (!rerun_output.ok()) return rerun_output.status();
        }
        drafter.last_tokens = rerun;
        drafter.last_position = position;
        drafter.timing.partial_accepts += 1u;
        drafter.timing.reruns += 1u;
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

    drafter.timing.rounds += 1u;
    drafter.timing.accepted_drafts += accepted_drafts;
    drafter.timing.generated_tokens += output.emitted_count;
    drafter.timing.verify_rows += drafts + 1u;
    const auto finished_at = std::chrono::steady_clock::now();
    drafter.timing.draft_ms +=
        std::chrono::duration<double, std::milli>(draft_started - started).count();
    drafter.timing.verify_ms +=
        std::chrono::duration<double, std::milli>(verify_done - draft_started).count();
    drafter.timing.round_ms +=
        std::chrono::duration<double, std::milli>(finished_at - started).count();
    if (accepted_drafts != drafts) {
        drafter.timing.rerun_ms +=
            std::chrono::duration<double, std::milli>(finished_at - verify_done).count();
    }
    return output;
}

}  // namespace shelf::qwen35::runtime
