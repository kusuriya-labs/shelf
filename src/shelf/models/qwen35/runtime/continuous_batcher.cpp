#include <shelf/models/qwen35/runtime/continuous_batcher.h>
#include <shelf/models/qwen35/runtime/prefix_cache.h>
#include <shelf/models/qwen35/runtime/runtime_request.h>
#include <shelf/models/qwen35/runtime/spec_decode.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace shelf {
namespace qwen35 {
namespace runtime {

ContinuousBatcher::ContinuousBatcher(
    Executor& executor,
    SequenceSlotPool& seq_pool,
    GdnStatePool& gdn_pool,
    PagedKVPool& kv_pool,
    ContinuousBatcherConfig config,
    hipStream_t stream)
    : executor_(executor),
      seq_pool_(seq_pool),
      gdn_pool_(gdn_pool),
      kv_pool_(kv_pool),
      config_(config),
      stream_(stream),
      capacity_(kv_pool.num_pages()) {
}

Result<uint64_t> ContinuousBatcher::submit(
    std::vector<int32_t> input_tokens, uint32_t max_new_tokens) {
    return submit(std::move(input_tokens), max_new_tokens, SamplingConfig{},
                  dflash2_available());
}

Result<uint64_t> ContinuousBatcher::submit(
    std::vector<int32_t> input_tokens,
    uint32_t max_new_tokens,
    const SamplingConfig& sampling) {
    return submit(std::move(input_tokens), max_new_tokens, sampling,
                  dflash2_available());
}

Result<uint64_t> ContinuousBatcher::submit(
    std::vector<int32_t> input_tokens,
    uint32_t max_new_tokens,
    const SamplingConfig& sampling,
    bool speculative_decode) {
    return submit(std::move(input_tokens), max_new_tokens, sampling,
                  speculative_decode, ThinkingBudget{});
}

Result<uint64_t> ContinuousBatcher::submit(
    std::vector<int32_t> input_tokens,
    uint32_t max_new_tokens,
    const SamplingConfig& sampling,
    const ThinkingBudget& thinking) {
    return submit(std::move(input_tokens), max_new_tokens, sampling,
                  dflash2_available(), thinking);
}

Result<uint64_t> ContinuousBatcher::submit(
    std::vector<int32_t> input_tokens,
    uint32_t max_new_tokens,
    const SamplingConfig& sampling,
    bool speculative_decode,
    const ThinkingBudget& thinking) {
    if (input_tokens.empty()) {
        return Status::invalid_argument("submit requires at least one input token",
                                        __FILE__, __LINE__);
    }
    const Status sampling_status = validate_sampling_config(sampling);
    if (!sampling_status.ok()) {
        return sampling_status;
    }
    if (static_cast<uint64_t>(input_tokens.size()) + max_new_tokens > config_.max_seq_len) {
        return Status::out_of_range("input_tokens + max_new_tokens exceeds max_seq_len",
                                    __FILE__, __LINE__);
    }
    if (speculative_decode && !dflash2_available()) {
        return Status::invalid_state("speculative decode requires a DFlash2 binding",
                                     __FILE__, __LINE__);
    }
    auto max_kv_tokens = compute_max_kv_tokens(
        static_cast<uint32_t>(input_tokens.size()), max_new_tokens);
    if (!max_kv_tokens.ok()) {
        return max_kv_tokens.status();
    }
    const uint32_t max_kv_pages =
        kv_pages_for_tokens(max_kv_tokens.value(), config_.page_tokens);
    if (max_kv_pages > kv_pool_.num_pages()) {
        return Status::out_of_range(
            "request max_kv_pages exceeds kv pool capacity", __FILE__, __LINE__);
    }
    RuntimeRequest request;
    request.id = next_id_++;
    request.input_tokens = std::move(input_tokens);
    request.max_new_tokens = max_new_tokens;
    request.sampling = sampling;
    request.max_kv_tokens = max_kv_tokens.value();
    request.max_kv_pages = max_kv_pages;
    request.speculative_decode = speculative_decode;
    request.thinking = thinking;

    requests_.push_back(std::move(request));
    request_index_[requests_.back().id] = std::prev(requests_.end());
    return requests_.back().id;
}

Status ContinuousBatcher::set_dflash2(const DFlash2BatcherBinding& binding) {
    if (binding.config == nullptr || binding.executor == nullptr ||
        binding.batch == nullptr || binding.arena == nullptr ||
        binding.sessions == nullptr) {
        return Status::invalid_argument(
            "dflash2 binding requires config / executor / batch / arena / sessions",
            __FILE__, __LINE__);
    }
    if (binding.num_drafts == 0u ||
        binding.num_drafts > binding.config->max_draft_tokens()) {
        return Status::invalid_argument("dflash2 binding num_drafts out of range",
                                        __FILE__, __LINE__);
    }
    if (binding.sessions->capacity() == 0u) {
        return Status::invalid_argument("dflash2 session pool is empty", __FILE__,
                                        __LINE__);
    }
    dflash_ = binding;
    return Status::make_ok();
}

Status ContinuousBatcher::cancel(uint64_t id) {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) {
        return Status::invalid_argument("unknown request id", __FILE__, __LINE__);
    }
    RuntimeRequest& request = *it->second;
    if (request.state == RequestState::Finished) {
        return Status::make_ok();
    }

    return finish_request(request, FinishReason::Cancelled, stream_);
}

Status ContinuousBatcher::retire(uint64_t id) {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) {
        return Status::invalid_argument("unknown request id", __FILE__, __LINE__);
    }
    if (it->second->state != RequestState::Finished) {
        return Status::invalid_state("cannot retire a non-finished request",
                                     __FILE__, __LINE__);
    }
    requests_.erase(it->second);
    request_index_.erase(it);
    return Status::make_ok();
}

Status ContinuousBatcher::cancel_all() {
    Status first_error = Status::make_ok();
    for (auto& request : requests_) {
        if (request.state == RequestState::Queued || request.state == RequestState::Active) {
            auto st = cancel(request.id);
            if (!st.ok() && first_error.ok()) {
                first_error = st;
            }
        }
    }
    return first_error;
}

const RuntimeRequest* ContinuousBatcher::request(uint64_t id) const {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) {
        return nullptr;
    }
    return &*it->second;
}

Status ContinuousBatcher::set_external_request_id(uint64_t id,
                                                  int64_t external_request_id) {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) {
        return Status::invalid_argument("request id not found", __FILE__, __LINE__);
    }
    it->second->external_request_id = external_request_id;
    return Status::make_ok();
}

std::size_t ContinuousBatcher::queued_count() const noexcept {
    std::size_t n = 0;
    for (const auto& r : requests_) {
        if (r.state == RequestState::Queued) ++n;
    }
    return n;
}

std::size_t ContinuousBatcher::active_count() const noexcept {
    std::size_t n = 0;
    for (const auto& r : requests_) {
        if (r.state == RequestState::Active) ++n;
    }
    return n;
}

std::size_t ContinuousBatcher::finished_count() const noexcept {
    std::size_t n = 0;
    for (const auto& r : requests_) {
        if (r.state == RequestState::Finished) ++n;
    }
    return n;
}

bool ContinuousBatcher::has_pending() const noexcept {
    return queued_count() + active_count() > 0;
}

Status ContinuousBatcher::finish_request(
    RuntimeRequest& request, FinishReason reason, hipStream_t stream) {
    request.state = RequestState::Finished;
    request.finish_reason = reason;
    request.pending_decode_token = -1;
    prefix_finish_capture(request, reason, stream);
    Status dflash_release = Status::make_ok();
    if (request.dflash != nullptr) {
        if (request.dflash->ready) {
            const Status decoder_st =
                dflash2_spec_decoder_shutdown(*request.dflash->decoder);
            if (!decoder_st.ok() && dflash_release.ok()) dflash_release = decoder_st;
            request.dflash->decoder.reset();
            request.dflash->ready = false;
        }
        if (request.dflash->pool_index != kDFlash2NoSessionSlot &&
            dflash_.sessions != nullptr) {
            const Status pool_st =
                dflash_.sessions->release(request.dflash->pool_index);
            if (!pool_st.ok() && dflash_release.ok()) dflash_release = pool_st;
        }
        request.dflash->pool_index = kDFlash2NoSessionSlot;
        request.dflash->context = nullptr;
        request.dflash.reset();
    }
    if (!request.sequence.is_allocated()) {
        return dflash_release;
    }
    Status mirror_release = Status::make_ok();
    if (tp_batch_hook_ != nullptr) {
        mirror_release = tp_batch_hook_->on_sequence_released(request.sequence);
    }
    Status first_error = dflash_release;
    auto release_st = release_paged_sequence_state(request.sequence, stream);
    if (!release_st.ok() && first_error.ok()) {
        first_error = release_st;
    }
    if (!mirror_release.ok() && first_error.ok()) {
        first_error = mirror_release;
    }
    auto claim_st = capacity_.release_claim(request.id);
    if (!claim_st.ok() && first_error.ok()) {
        first_error = claim_st;
    }
    return first_error;
}

void ContinuousBatcher::prefix_save_prefill(RuntimeRequest& request) {
    if (prefix_cache_ == nullptr) {
        return;
    }
    const uint32_t position = request.sequence.position;
    if (position == 0 || position % config_.page_tokens != 0) {
        return;
    }
    if (position > request.input_tokens.size()) {
        return;
    }
    const std::vector<uint32_t> tokens(
        request.input_tokens.begin(),
        request.input_tokens.begin() + static_cast<std::ptrdiff_t>(position));
    SpecRingRef ring;
    const SpecRingRef* ring_ptr = nullptr;
    if (request.dflash != nullptr &&
        request.dflash->context != nullptr &&
        request.dflash->context->initialized) {
        ring.k = request.dflash->context->k_ring.data<bf16_t>();
        ring.v = request.dflash->context->v_ring.data<bf16_t>();
        ring.length = request.dflash->context->length;
        ring.next_position = request.dflash->context->next_position;
        ring_ptr = &ring;
    }
    const Status save_st = prefix_cache_->save_input_boundary(
        request.sequence, tokens, position, stream_, ring_ptr);
    if (getenv("SHELF_PFXDBG") != nullptr) {
        if (!save_st.ok()) {
            std::fprintf(stderr, "PFXSAVE_FAIL pos=%u %s\n", position,
                         save_st.message().c_str());
        } else {
            const PrefixCacheStats st2 = prefix_cache_->stats();
            std::fprintf(stderr,
                         "PFXSAVE_OK pos=%u saves=%llu short=%llu\n", position,
                         (unsigned long long)st2.saves,
                         (unsigned long long)st2.skipped_too_short);
        }
    }
}

void ContinuousBatcher::prefix_begin_capture(RuntimeRequest& request) {
    if (prefix_cache_ == nullptr || request.prefix_capture != 0) {
        return;
    }
    auto capture = prefix_cache_->begin_capture(request.sequence);
    if (!capture.ok()) {
        return;
    }
    request.prefix_capture = capture.value();
    request.prefix_boundary_pos = 0;
    const uint32_t position = request.sequence.position;
    if (position > 0 && position % config_.page_tokens == 0) {
        const Status st =
            prefix_cache_->capture_boundary(capture.value(), request.sequence, stream_);
        if (st.ok()) {
            request.prefix_boundary_pos = position;
        }
    }
}

void ContinuousBatcher::prefix_capture_after_step(RuntimeRequest& request) {
    if (prefix_cache_ == nullptr || request.prefix_capture == 0) {
        return;
    }
    const uint32_t position = request.sequence.position;
    if (position == 0 || position % config_.page_tokens != 0) {
        return;
    }
    const Status st =
        prefix_cache_->capture_boundary(request.prefix_capture, request.sequence, stream_);
    if (st.ok()) {
        request.prefix_boundary_pos = position;
    }
}

void ContinuousBatcher::prefix_finish_capture(RuntimeRequest& request, FinishReason reason,
                                              hipStream_t stream) {
    if (prefix_cache_ == nullptr || request.prefix_capture == 0) {
        return;
    }
    const CaptureId capture = request.prefix_capture;
    const uint32_t captured_pos = request.prefix_boundary_pos;
    request.prefix_capture = 0;
    request.prefix_boundary_pos = 0;
    const bool tail_eligible =
        (reason == FinishReason::Eos || reason == FinishReason::MaxNewTokens) &&
        !request.generated.empty() && request.sequence.is_allocated();
    if (!tail_eligible) {
        (void)prefix_cache_->cancel_capture(capture);
        return;
    }
    const uint32_t position = request.sequence.position;
    const uint32_t boundary =
        (position / config_.page_tokens) * config_.page_tokens;
    SpecRingRef ring;
    const SpecRingRef* ring_ptr = nullptr;
    if (request.dflash != nullptr && request.dflash->ready &&
        request.dflash->context != nullptr &&
        request.dflash->context->initialized) {
        ring.k = request.dflash->context->k_ring.data<bf16_t>();
        ring.v = request.dflash->context->v_ring.data<bf16_t>();
        ring.length = request.dflash->context->length;
        ring.next_position = request.dflash->context->next_position;
        ring_ptr = &ring;
    }
    bool saved = false;
    const uint32_t input_len = static_cast<uint32_t>(request.input_tokens.size());
    const bool position_covers_input = position >= input_len;
    const uint32_t committed = position_covers_input ? position - input_len : 0u;
    const bool committed_valid =
        position_covers_input && committed <= request.generated.size();
    if (captured_pos == boundary && committed_valid) {
        std::vector<uint32_t> tokens;
        tokens.reserve(position);
        for (int32_t token : request.input_tokens) {
            tokens.push_back(static_cast<uint32_t>(token));
        }
        for (uint32_t i = 0; i < committed; ++i) {
            tokens.push_back(static_cast<uint32_t>(request.generated[i]));
        }
        const Status st = prefix_cache_->save_output_tail(
            capture, request.sequence, tokens, position, stream, ring_ptr);
        if (st.ok()) {
            saved = true;
        }
        if (getenv("SHELF_PFXDBG") != nullptr) {
            std::fprintf(stderr,
                         "PFXDBG save_output_tail pos=%u boundary=%u committed=%u ring=%d "
                         "ok=%d %s\n",
                         position, boundary, committed,
                         static_cast<int>(ring_ptr != nullptr),
                         static_cast<int>(st.ok()), st.message().c_str());
        }
    }
    if (!saved) {
        (void)prefix_cache_->cancel_capture(capture);
    }
}

KVBankerState ContinuousBatcher::build_kv_banker_state() const {
    KVBankerState state;
    state.available_pages = kv_pool_.num_free_pages();
    for (const auto& request : requests_) {
        if (request.state != RequestState::Active) {
            continue;
        }
        state.processes.push_back({
            request.max_kv_pages,
            static_cast<uint32_t>(request.sequence.block_table.size())});
    }
    return state;
}

KVCapacitySnapshot ContinuousBatcher::kv_capacity_snapshot() const {
    KVCapacitySnapshot snapshot;
    snapshot.total_pages = kv_pool_.num_pages();
    snapshot.physical_used_pages = kv_pool_.num_used_pages();
    snapshot.physical_free_pages = kv_pool_.num_free_pages();
    const KVBankerState state = build_kv_banker_state();
    snapshot.active_requests = static_cast<uint32_t>(state.processes.size());
    uint64_t max_claim = 0;
    uint64_t remaining_need = 0;
    for (const auto& process : state.processes) {
        max_claim += process.max_pages;
        remaining_need += process.max_pages - process.allocated_pages;
    }
    snapshot.max_claim_pages = max_claim;
    snapshot.remaining_need_pages = remaining_need;
    const uint64_t total = snapshot.total_pages;
    snapshot.conservative_headroom_pages = max_claim > total ? 0 : total - max_claim;
    snapshot.overcommit_pages = max_claim > total ? max_claim - total : 0;
    snapshot.banker_safe = is_safe(state);
    return snapshot;
}

Result<StepResult> ContinuousBatcher::step() {
    const bool banker_mode = config_.admission_policy == KVAdmissionPolicy::BankerSafe;
    KVBankerState kv_state = build_kv_banker_state();
    uint32_t dflash_prefilled_requests = 0u;
    uint32_t dflash_prefilled_tokens = 0u;
    if (banker_mode && !is_safe(kv_state)) {
        return Status::invalid_state(
            "banker invariant violated: active KV state is unsafe", __FILE__, __LINE__);
    }

    for (auto& request : requests_) {
        if (request.state != RequestState::Queued) {
            continue;
        }
        if (seq_pool_.num_free_slots() == 0) {
            break;
        }
        const bool kv_admit_ok = banker_mode
            ? can_add_process(kv_state, request.max_kv_pages)
            : capacity_.can_admit(request.max_kv_pages);
        if (!kv_admit_ok) {
            break;
        }
        auto admit_st = admit_request(request, seq_pool_, gdn_pool_, kv_pool_, stream_);
        if (!admit_st.ok()) {
            break;
        }
        if (tp_batch_hook_ != nullptr) {
            auto mirror_st = tp_batch_hook_->on_sequence_created(request.sequence);
            if (!mirror_st.ok()) {
                (void)release_paged_sequence_state(request.sequence, stream_);
                request.state = RequestState::Queued;
                return mirror_st;
            }
        }
        if (getenv("SHELF_PFXDBG") != nullptr) {
            std::fprintf(stderr, "PFXDBG admission spec=%d avail=%d cache=%p\n",
                         static_cast<int>(request.speculative_decode),
                         static_cast<int>(dflash2_available()),
                         static_cast<const void*>(prefix_cache_));
        }
        // The pool may be smaller than the request count; requests that find no
        // free session decode without speculation.
        if (request.speculative_decode && dflash2_available() &&
            dflash_.sessions->num_free() > 0u) {
            const Status open_st = open_dflash2_context(request);
            if (!open_st.ok()) {
                request.error_message = open_st.message();
                (void)capacity_.register_claim(request.id, request.max_kv_pages);
                const Status finish_st =
                    finish_request(request, FinishReason::Error, stream_);
                if (!finish_st.ok()) {
                    return finish_st;
                }
                continue;
            }
        }
        if (prefix_cache_ != nullptr) {
            const std::vector<uint32_t> prefix_tokens(request.input_tokens.begin(),
                                                      request.input_tokens.end());
            const PrefixHit hit = prefix_cache_->lookup(prefix_tokens);
            if (getenv("SHELF_PFXDBG") != nullptr) {
                std::fprintf(stderr,
                             "PFXDBG lookup hit=%d state=%u ring_len=%u ring_next=%u "
                             "in=%zu\n",
                             static_cast<int>(hit.hit()), hit.state_position,
                             hit.ring_length, hit.ring_next_position,
                             prefix_tokens.size());
            }
            const bool ring_ready = request.dflash == nullptr || hit.ring_covers_state();
            if (hit.hit() && ring_ready &&
                hit.state_position < static_cast<uint32_t>(request.input_tokens.size())) {
                SpecRingOut ring_out;
                SpecRingOut* ring_out_ptr = nullptr;
                if (request.dflash != nullptr && request.dflash->context != nullptr &&
                    request.dflash->context->initialized) {
                    ring_out.k = request.dflash->context->k_ring.data<bf16_t>();
                    ring_out.v = request.dflash->context->v_ring.data<bf16_t>();
                    ring_out_ptr = &ring_out;
                }
                const Status restore_st = prefix_cache_->restore(
                    hit.entry_id, request.sequence, stream_, ring_out_ptr);
                if (!restore_st.ok()) {
                    request.error_message = restore_st.message();
                    (void)capacity_.register_claim(request.id, request.max_kv_pages);
                    const Status finish_st =
                        finish_request(request, FinishReason::Error, stream_);
                    if (!finish_st.ok()) {
                        return finish_st;
                    }
                    continue;
                }
                if (ring_out_ptr != nullptr) {
                    if (request.dflash->context != nullptr &&
                        request.dflash->context->initialized) {
                        request.dflash->context->length = ring_out.length;
                        request.dflash->context->next_position = ring_out.next_position;
                    }
                }
                request.restored_tokens = hit.state_position;
                if (tp_batch_hook_ != nullptr) {
                    auto mirror_st = tp_batch_hook_->on_sequence_restored(request.sequence);
                    if (!mirror_st.ok()) {
                        (void)tp_batch_hook_->on_sequence_released(request.sequence);
                        (void)release_paged_sequence_state(request.sequence, stream_);
                        request.state = RequestState::Queued;
                        return mirror_st;
                    }
                }
                prefix_begin_capture(request);
            }
        }
        auto claim_st = capacity_.register_claim(request.id, request.max_kv_pages);
        if (!claim_st.ok()) {
            if (tp_batch_hook_ != nullptr) {
                (void)tp_batch_hook_->on_sequence_released(request.sequence);
            }
            (void)release_paged_sequence_state(request.sequence, stream_);
            request.state = RequestState::Queued;
            return claim_st;
        }
        if (request.dflash != nullptr) {
            const uint32_t prefill_tokens =
                static_cast<uint32_t>(request.input_tokens.size());
            const Status session_st = open_dflash2_session(request);
            if (!session_st.ok()) {
                request.error_message = session_st.message();
                const Status finish_st =
                    finish_request(request, FinishReason::Error, stream_);
                if (!finish_st.ok()) {
                    return finish_st;
                }
                continue;
            }
            dflash_prefilled_tokens += prefill_tokens > request.restored_tokens
                ? prefill_tokens - request.restored_tokens
                : 0u;
            ++dflash_prefilled_requests;
        }
        kv_state.processes.push_back(
            {request.max_kv_pages,
             static_cast<uint32_t>(request.sequence.block_table.size())});
    }

    if (dflash_prefilled_requests > 0u) {
        return StepResult{true, dflash_prefilled_tokens, dflash_prefilled_requests};
    }

    std::vector<RuntimeRequest*> active_requests;
    active_requests.reserve(active_count());
    for (auto& request : requests_) {
        if (request.state != RequestState::Active) {
            continue;
        }
        active_requests.push_back(&request);
    }
    std::vector<const RuntimeRequest*> active(active_requests.begin(),
                                              active_requests.end());

    bool dflash_executed = false;
    uint32_t dflash_tokens = 0;
    uint32_t dflash_requests = 0;
    const Status dflash_st = run_dflash2_round(active_requests, dflash_executed,
                                               dflash_tokens, dflash_requests);
    if (!dflash_st.ok()) {
        return dflash_st;
    }
    kv_state = build_kv_banker_state();

    auto plan_result = schedule_requests(active, config_.max_num_batched_tokens, config_.max_scheduled_requests, config_.page_tokens, &kv_state);
    if (!plan_result.ok()) {
        return plan_result.status();
    }
    SchedulePlan plan = plan_result.release();
    if (plan.empty()) {
        return StepResult{dflash_executed, dflash_tokens, dflash_requests};
    }

    std::unordered_map<uint64_t, RuntimeRequest*> by_handle;
    for (auto& request : requests_) {
        if (!request.sequence.is_allocated()) {
            continue;
        }
        const auto handle = request.sequence.request_handle();
        by_handle[::shelf::runtime::request_handle_key(handle)] = &request;
    }

    auto resolve = [&](const ScheduledRequest& scheduled) -> RuntimeRequest* {
        const uint64_t key = ::shelf::runtime::request_handle_key(scheduled.handle);
        auto it = by_handle.find(key);
        if (it == by_handle.end()) {
            return nullptr;
        }
        RuntimeRequest* request = it->second;
        if (!::shelf::runtime::request_handle_equal(
                request->sequence.request_handle(), scheduled.handle)) {
            return nullptr;
        }
        return request;
    };

    const auto exec_started = std::chrono::steady_clock::now();
    auto exec_result = tp_batch_hook_ != nullptr
                          ? tp_batch_hook_->on_execute(plan.batch)
                          : execute_batch(executor_, plan.batch, stream_);
    const double exec_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - exec_started)
                               .count();
    for (auto& scheduled : plan.scheduled_requests) {
        if (scheduled.execution_class != ::shelf::runtime::ExecutionClass::PREFILL) {
            continue;
        }
        RuntimeRequest* prefill_request = resolve(scheduled);
        if (prefill_request != nullptr) {
            prefill_request->prefill_ms += exec_ms;
            const uint32_t processed =
                scheduled.prefix_tokens + scheduled.num_tokens;
            const uint32_t base = prefill_request->restored_tokens;
            const uint32_t processed_new = processed > base ? processed - base : 0u;
            const uint32_t last_new =
                prefill_request->prefill_log_tokens > base
                    ? prefill_request->prefill_log_tokens - base
                    : 0u;
            const double interval_ms =
                prefill_request->prefill_ms - prefill_request->prefill_log_ms;
            if (interval_ms >= 3000.0 && processed_new > last_new) {
                std::fprintf(
                    stderr,
                    "[req %lld] prefill_tok_s=%.1f tokens=%u interval_s=%.2f\n",
                    static_cast<long long>(prefill_request->external_request_id),
                    static_cast<double>(processed_new - last_new) * 1000.0 /
                        interval_ms,
                    processed_new, interval_ms / 1000.0);
                prefill_request->prefill_log_ms = prefill_request->prefill_ms;
                prefill_request->prefill_log_tokens = processed;
            }
        }
    }
    if (!exec_result.ok()) {
        Status first_error = Status::make_ok();
        for (auto& scheduled : plan.scheduled_requests) {
            if (!scheduled.sequence->is_poisoned()) {
                continue;
            }
            RuntimeRequest* request = resolve(scheduled);
            if (request != nullptr && request->state == RequestState::Active) {
                auto st = finish_request(*request, FinishReason::Error, stream_);
                if (!st.ok() && first_error.ok()) {
                    first_error = st;
                }
            }
        }
        if (!first_error.ok()) {
            return first_error;
        }
        return exec_result.status();
    }

    BatchExecutionOutput output = exec_result.release();

    std::vector<int32_t> host_sampled;
    host_sampled.resize(output.num_outputs);
    if (output.num_outputs > 0) {
        hipError_t copy_err = hipMemcpy(
            host_sampled.data(),
            output.sampled_tokens.data<int32_t>(),
            output.num_outputs * sizeof(int32_t),
            hipMemcpyDeviceToHost);
        if (copy_err != hipSuccess) {
            return Status::hip_error(
                "step hipMemcpy sampled tokens",
                hipGetErrorString(copy_err), __FILE__, __LINE__);
        }
    }

    Status first_error = Status::make_ok();
    uint32_t row = 0;
    for (auto& scheduled : plan.scheduled_requests) {
        RuntimeRequest* request_ptr = resolve(scheduled);
        if (request_ptr == nullptr) {
            return Status::invalid_state("planned sequence not found", __FILE__, __LINE__);
        }
        RuntimeRequest& request = *request_ptr;
        int32_t token = -1;
        if (scheduled.compute_logits) {
            if (row >= output.num_outputs) {
                return Status::invalid_state(
                    "sample output index out of range", __FILE__, __LINE__);
            }
            token = host_sampled[row];
            ++row;
        }
        {
            const bool final_prefill =
                scheduled.execution_class == ::shelf::runtime::ExecutionClass::PREFILL &&
                scheduled.prefix_tokens + scheduled.num_tokens ==
                    static_cast<uint32_t>(request.input_tokens.size());
            if (prefix_cache_ != nullptr &&
                scheduled.execution_class == ::shelf::runtime::ExecutionClass::PREFILL) {
                prefix_save_prefill(request);
                if (final_prefill) {
                    prefix_begin_capture(request);
                }
            }
            if (scheduled.compute_logits) {
                if (scheduled.sample) {
                    if (token < 0) {
                        return Status::invalid_state(
                            "greedy sampled request returned invalid token",
                            __FILE__, __LINE__);
                    }
                    const CommitAction action =
                        commit_sampled_token(request, token, config_.eos_token_ids);
                    if (action == CommitAction::Finished) {
                        auto st = finish_request(request, request.finish_reason, stream_);
                        if (!st.ok() && first_error.ok()) {
                            first_error = st;
                        }
                    }
                } else {
                    if (token != -1) {
                        return Status::invalid_state(
                            "logits-only request returned sampled token", __FILE__, __LINE__);
                    }
                }
            } else if (final_prefill) {
                const CommitAction action = commit_final_prefill_without_sample(request);
                if (action == CommitAction::Finished) {
                    auto st = finish_request(request, request.finish_reason, stream_);
                    if (!st.ok() && first_error.ok()) {
                        first_error = st;
                    }
                }
            }
        }
    }
    if (row != output.num_outputs) {
        return Status::invalid_state(
            "sample output count mismatch", __FILE__, __LINE__);
    }
    if (prefix_cache_ != nullptr) {
        for (auto& request : requests_) {
            if (request.state != RequestState::Active || request.prefix_capture == 0) {
                continue;
            }
            prefix_capture_after_step(request);
        }
    }
    if (!first_error.ok()) {
        return first_error;
    }

    return StepResult{true, plan.batch.num_tokens + dflash_tokens,
                      plan.batch.num_requests + dflash_requests};
}

Status ContinuousBatcher::open_dflash2_context(RuntimeRequest& request) {
    if (request.dflash != nullptr) {
        return Status::make_ok();
    }
    if (dflash_.sessions == nullptr) {
        return Status::invalid_state("dflash2 session pool is not bound", __FILE__, __LINE__);
    }
    auto index_result = dflash_.sessions->acquire();
    if (!index_result.ok()) {
        return index_result.status();
    }
    const uint32_t index = index_result.value();
    DFlash2SessionPool& pool = *dflash_.sessions;
    auto session = std::make_unique<DFlash2RequestSession>();
    session->context = &pool.slot(index).context;
    session->pool_index = index;
    request.dflash = std::move(session);
    return Status::make_ok();
}

Status ContinuousBatcher::open_dflash2_session(RuntimeRequest& request) {
    if (getenv("SHELF_PFXDBG") != nullptr) {
        std::fprintf(stderr, "PFXDBG open_session enter\n");
    }
    if (request.dflash == nullptr) {
        return Status::invalid_state("dflash2 session requires a session", __FILE__,
                                     __LINE__);
    }
    if (request.dflash->context == nullptr) {
        return Status::invalid_state("dflash2 session requires a context", __FILE__,
                                     __LINE__);
    }
    DFlash2RequestSession& session = *request.dflash;
    if (request.max_new_tokens == 0u) {
        request.finish_reason = FinishReason::MaxNewTokens;
        return Status::make_ok();
    }
    DFlash2SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = dflash_.num_drafts;
    decoder_config.eos_tokens = config_.eos_token_ids;
    DFlash2SpecWorkspace* workspace = nullptr;
    if (dflash_.sessions != nullptr) {
        workspace = &dflash_.sessions->slot(session.pool_index).workspace;
    }
    auto decoder_result = create_dflash2_spec_decoder(
        executor_, *dflash_.executor, *session.context,
        request.sequence, gdn_pool_, *dflash_.arena,
        decoder_config, stream_, dflash_.rank_bindings,
        tp_batch_hook_, workspace);
    if (!decoder_result.ok()) {
        return decoder_result.status();
    }
    session.decoder =
        std::make_unique<DFlash2SpecDecoder>(decoder_result.release());
    session.decoder->timing = &session.timing;
    dflash2_spec_decoder_set_sampling(*session.decoder, request.sampling);
    const uint32_t prompt_count =
        static_cast<uint32_t>(request.input_tokens.size());
    const uint32_t page_tokens = config_.page_tokens;
    const uint32_t last_boundary =
        page_tokens == 0u ? prompt_count : (prompt_count / page_tokens) * page_tokens;
    const auto prefill_started = std::chrono::steady_clock::now();
    auto last_prefill_log = prefill_started;
    uint32_t last_prefill_offset = request.restored_tokens;
    auto save_input_boundary = [&](uint32_t offset) {
        const auto now = std::chrono::steady_clock::now();
        const double interval_s = std::chrono::duration<double>(
                                      now - last_prefill_log)
                                      .count();
        if (interval_s >= 3.0 && offset > last_prefill_offset) {
            const uint32_t delta = offset - last_prefill_offset;
            std::fprintf(stderr,
                         "[req %lld] prefill_tok_s=%.1f tokens=%u interval_s=%.2f\n",
                         static_cast<long long>(request.external_request_id),
                         static_cast<double>(delta) / interval_s,
                         offset - request.restored_tokens, interval_s);
            last_prefill_log = now;
            last_prefill_offset = offset;
        }
        if (prefix_cache_ == nullptr || page_tokens == 0u) {
            return;
        }
        // Like the non-speculative prefill: every page-aligned chunk end is a restore point. (Only
        // `last_boundary` used to be saved, which a prompt longer than one chunk never ends on.)
        if (offset == 0u || offset % page_tokens != 0u || offset > prompt_count) {
            return;
        }
        if (request.sequence.position != offset) {
            return;
        }
        const std::vector<uint32_t> tokens(
            request.input_tokens.begin(),
            request.input_tokens.begin() + static_cast<std::ptrdiff_t>(offset));
        SpecRingRef ring;
        const SpecRingRef* ring_ptr = nullptr;
        if (session.context != nullptr && session.context->initialized) {
            ring.k = session.context->k_ring.data<bf16_t>();
            ring.v = session.context->v_ring.data<bf16_t>();
            ring.length = session.context->length;
            ring.next_position = session.context->next_position;
            ring_ptr = &ring;
        }
        const Status st = prefix_cache_->save_input_boundary(
            request.sequence, tokens, offset, stream_, ring_ptr);
        if (getenv("SHELF_PFXDBG") != nullptr) {
            std::fprintf(stderr,
                         "PFXDBG save_input_boundary at=%u ring=%d ok=%d %s\n", offset,
                         static_cast<int>(ring_ptr != nullptr), static_cast<int>(st.ok()),
                         st.message().c_str());
        }
    };
    // A prompt that fits in one chunk has no chunk end to save at, so a separate pass over its page-aligned head
    // captures one (the prefill proper keeps the greedy chunking). Longer prompts are saved by the callback above.
    constexpr uint32_t kMaxShadowPrefixTokens = 1024u;
    if (prefix_cache_ != nullptr && tp_batch_hook_ == nullptr &&
        request.restored_tokens == 0u && last_boundary > 0u && last_boundary < prompt_count &&
        last_boundary <= kMaxShadowPrefixTokens) {
        for (DFlash2GdnRankState& rank : session.decoder->gdn_ranks) {
            auto scope = shelf::gpu::ScopedDevice::create(rank.device);
            if (!scope.ok()) return scope.status();
            Status st = spec_gdn_snapshot(*rank.pool, request.sequence.slot, rank.conv_snapshot,
                                          rank.rec_snapshot, rank.conv_bytes, rank.rec_bytes,
                                          rank.stream);
            if (!st.ok()) return st;
        }
        auto boundary = dflash2_spec_prefill(*session.decoder, request.input_tokens.data(),
                                             last_boundary, 0u, 0u, save_input_boundary);
        if (!boundary.ok()) return boundary.status();
        for (DFlash2GdnRankState& rank : session.decoder->gdn_ranks) {
            auto scope = shelf::gpu::ScopedDevice::create(rank.device);
            if (!scope.ok()) return scope.status();
            Status st = spec_gdn_restore(*rank.pool, request.sequence.slot, rank.conv_snapshot,
                                         rank.rec_snapshot, rank.conv_bytes, rank.rec_bytes,
                                         rank.stream);
            if (!st.ok()) return st;
        }
        request.sequence.position = 0u;
        Status rollback = rollback_sequence_append(request.sequence, 0u, stream_);
        if (!rollback.ok()) return rollback;
        Status reset = dflash2::dflash2_context_reset(*session.context, 0u);
        if (!reset.ok()) return reset;
        session.decoder->token_history.clear();
        session.decoder->sample_index = 0u;
    }
    auto prefill_result = dflash2_spec_prefill(
        *session.decoder, request.input_tokens.data(), prompt_count,
        request.restored_tokens, 0u, save_input_boundary);
    request.prefill_ms += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - prefill_started)
                              .count();
    if (!prefill_result.ok()) {
        (void)dflash2_spec_decoder_shutdown(*session.decoder);
        session.decoder.reset();
        return prefill_result.status();
    }
    session.pending_token = prefill_result.value().pending_token;
    session.ready = true;
    request.generated.push_back(session.pending_token);
    note_generated_token(request, session.pending_token);
    request.generated_tokens = 1u;
    prefix_begin_capture(request);
    if (dflash_.stats != nullptr) {
        DFlash2BatcherStats& stats = *dflash_.stats;
        uint64_t history_bytes = 0;
        for (const DFlash2GdnRankState& rank : session.decoder->gdn_ranks) {
            if (!rank.history.initialized) continue;
            history_bytes += static_cast<uint64_t>(rank.history.rows) *
                             static_cast<uint64_t>(rank.history.conv_state_bytes +
                                                   rank.history.recurrent_state_bytes);
        }
        stats.gdn_history_bytes = history_bytes;
        stats.gdn_snapshot_bytes = session.decoder->gdn_snapshot_bytes;
        stats.gdn_compact_bytes = session.decoder->gdn_compact_bytes;
    }
    if (qwen_hybrid::is_stop_token(config_.eos_token_ids, session.pending_token)) {
        request.finish_reason = FinishReason::Eos;
    }
    return Status::make_ok();
}

Status ContinuousBatcher::run_dflash2_round(std::vector<RuntimeRequest*>& active,
                                            bool& executed, uint32_t& tokens,
                                            uint32_t& requests) {
    if (!dflash2_available()) {
        return Status::make_ok();
    }
    std::vector<DFlash2SpecBatchSlot> slots;
    std::vector<RuntimeRequest*> slot_requests;
    std::vector<RuntimeRequest*> pending_finish;
    for (RuntimeRequest* request : active) {
        if (request->state != RequestState::Active || request->dflash == nullptr) {
            continue;
        }
        if (request->finish_reason != FinishReason::None) {
            pending_finish.push_back(request);
            continue;
        }
        if (request->generated_tokens >= request->max_new_tokens) {
            request->finish_reason = FinishReason::MaxNewTokens;
            pending_finish.push_back(request);
            continue;
        }
        if (!request->dflash->ready) {
            return Status::invalid_state("dflash2 session is not ready", __FILE__,
                                         __LINE__);
        }
        DFlash2SpecBatchSlot slot;
        slot.decoder = request->dflash->decoder.get();
        slot.anchor_token = request->dflash->pending_token;
        slot.num_drafts = dflash_.num_drafts;
        slot.remaining_tokens = request->max_new_tokens - request->generated_tokens;
        slot.page_tokens =
            request->prefix_capture != 0u ? config_.page_tokens : 0u;
        slot.thinking_allowance = thinking_allowance(*request);
        slot.thinking_close_token = request->thinking.close_token;
        slots.push_back(slot);
        slot_requests.push_back(request);
    }

    Status first_error = Status::make_ok();
    if (!pending_finish.empty()) {
        executed = true;
    }
    for (RuntimeRequest* request : pending_finish) {
        const Status st =
            finish_request(*request, request->finish_reason, stream_);
        if (!st.ok() && first_error.ok()) first_error = st;
    }
    if (!first_error.ok()) {
        return first_error;
    }

    if (!slots.empty()) {
        std::vector<DFlash2SpecTiming> timing_before;
        if (dflash_.stats != nullptr) {
            timing_before.reserve(slot_requests.size());
            for (RuntimeRequest* request : slot_requests) {
                timing_before.push_back(request->dflash->timing);
            }
        }
        auto batch_result = dflash2_spec_step_batch(*dflash_.batch, slots);
        if (!batch_result.ok()) {
            for (RuntimeRequest* request : slot_requests) {
                const Status st = finish_request(*request, FinishReason::Error, stream_);
                if (!st.ok() && first_error.ok()) first_error = st;
            }
            if (!first_error.ok()) {
                return first_error;
            }
            return batch_result.status();
        }
        auto results = batch_result.release();
        executed = true;
        requests = static_cast<uint32_t>(slots.size());
        tokens = results.total_rows;
        if (dflash_.stats != nullptr) {
            DFlash2BatcherStats& stats = *dflash_.stats;
            ++stats.rounds;
            stats.round_ms += results.round_ms;
            stats.draft_gpu_ms += results.draft_gpu_ms;
            stats.verify_gpu_ms += results.verify_gpu_ms;
            for (std::size_t i = 0; i < slot_requests.size(); ++i) {
                const DFlash2SpecTiming& now = slot_requests[i]->dflash->timing;
                const DFlash2SpecTiming& before = timing_before[i];
                stats.draft_ms += now.draft_ms - before.draft_ms;
                stats.draft_d2h_ms += now.draft_d2h_ms - before.draft_d2h_ms;
                stats.proposal_wait_ms += now.proposal_wait_ms - before.proposal_wait_ms;
                stats.proposal_copy_ms += now.proposal_copy_ms - before.proposal_copy_ms;
                stats.gdn_snapshot_ms += now.gdn_snapshot_ms - before.gdn_snapshot_ms;
                stats.verify_ms += now.verify_ms - before.verify_ms;
                stats.decision_wait_ms += now.decision_wait_ms - before.decision_wait_ms;
                stats.decision_copy_ms += now.decision_copy_ms - before.decision_copy_ms;
                stats.gdn_restore_ms += now.gdn_restore_ms - before.gdn_restore_ms;
                stats.rerun_ms += now.rerun_ms - before.rerun_ms;
                stats.commit_ms += now.dflash_commit_ms - before.dflash_commit_ms;
                stats.verify_rows += now.verify_rows_total - before.verify_rows_total;
            }
        }
        for (std::size_t i = 0; i < slot_requests.size(); ++i) {
            RuntimeRequest& request = *slot_requests[i];
            const DFlash2SpecIterationOutput& step = results.slots[i];
            for (uint32_t j = 0; j < step.emitted_count; ++j) {
                request.generated.push_back(step.emitted[j]);
                note_generated_token(request, step.emitted[j]);
            }
            request.generated_tokens = static_cast<uint32_t>(request.generated.size());
            request.dflash->pending_token = step.pending_token;
            if (dflash_.stats != nullptr) {
                DFlash2BatcherStats& stats = *dflash_.stats;
                stats.accepted_drafts += step.num_accepted;
                stats.generated_tokens += step.emitted_count;
                if (step.num_accepted >= step.num_drafts) {
                    ++stats.full_accepts;
                } else {
                    ++stats.partial_accepts;
                }
                const uint32_t bucket = std::min<uint32_t>(
                    request.sequence.position / kDFlash2PositionBucketTokens,
                    kDFlash2PositionBuckets - 1u);
                DFlash2PositionStats& position = stats.position[bucket];
                ++position.rounds;
                position.accepted_drafts += step.num_accepted;
                position.generated_tokens += step.emitted_count;
                if (slot_requests.size() == 1u) {
                    position.round_ms += results.round_ms;
                }
            }
            if (step.finished) {
                request.finish_reason = FinishReason::Eos;
            } else if (request.generated_tokens >= request.max_new_tokens) {
                request.finish_reason = FinishReason::MaxNewTokens;
            }
            if (request.finish_reason != FinishReason::None) {
                const Status st =
                    finish_request(request, request.finish_reason, stream_);
                if (!st.ok() && first_error.ok()) first_error = st;
            }
        }
    }

    if (prefix_cache_ != nullptr) {
        for (RuntimeRequest* request : active) {
            if (request->state != RequestState::Active ||
                request->dflash == nullptr || request->prefix_capture == 0) {
                continue;
            }
            prefix_capture_after_step(*request);
        }
    }
    return first_error;
}

}
}
}
