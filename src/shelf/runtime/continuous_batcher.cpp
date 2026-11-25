#include <shelf/runtime/continuous_batcher.h>
#include <shelf/inference/execution/selectors/gdn_conv_selector.h>
#include <shelf/inference/execution/selectors/gdn_recurrence_selector.h>
#include <shelf/runtime/dflash2/dflash2_adaptive_batch.h>
#include <shelf/runtime/prefix_cache.h>
#include <shelf/runtime/runtime_request.h>
#include <shelf/runtime/spec_decode.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/runtime/executor.h>
#include <shelf/kernels/dflash2/feature_concat.h>
#include <shelf/runtime/state_types.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <thread>

namespace shelf {
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

ContinuousBatcher::~ContinuousBatcher() noexcept {
    shutdown_ngram_verify();
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
    if (config_.ngram.enabled) {
        request.ngram.configure(config_.ngram);
        request.ngram.ingest_span(request.input_tokens);
    }

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
    if (binding.adaptive && tp_batch_hook_ != nullptr) {
        return Status::unsupported("dflash2 adaptive requires a single device", __FILE__, __LINE__);
    }
    if (binding.sessions->capacity() == 0u) {
        return Status::invalid_argument("dflash2 session pool is empty", __FILE__,
                                        __LINE__);
    }
    dflash_ = binding;
    return Status::make_ok();
}

bool supports_ngram_history(const GdnStatePoolDeviceView& gdn) {
    if (gdn.recurrent_base == nullptr) return true;
    const uint64_t value_width = static_cast<uint64_t>(gdn.num_v_heads) * gdn.head_v;
    const uint64_t key_pair = 2ull * gdn.head_k;
    if (key_pair == 0u || value_width >= gdn.conv_dim || (gdn.conv_dim - value_width) % key_pair != 0u) {
        return false;
    }
    const uint32_t key_heads = static_cast<uint32_t>((gdn.conv_dim - value_width) / key_pair);
    const auto conv = execution::select_gdn_conv_implementation(
        {ValueDType::BF16, ValueDType::F32, gdn.conv_dim, gdn.conv_history, 1u, 1u, 1u});
    const auto recurrence = execution::select_gdn_recurrence_implementation(
        {1u, 1u, 1u, key_heads, gdn.num_v_heads, gdn.head_k, gdn.head_v});
    return conv == execution::GdnConvImplementation::Optimized &&
           recurrence == execution::GdnRecurrenceImplementation::Optimized;
}

Status ContinuousBatcher::set_ngram_verify(
    gpu::GpuArena* arena,
    const std::vector<NgramVerifyRankBinding>* rank_bindings) {
    shutdown_ngram_verify();
    ngram_verify_.arena = arena;
    ngram_verify_.bindings.clear();
    if (rank_bindings != nullptr) {
        ngram_verify_.bindings = *rank_bindings;
    }
    if (!supports_ngram_history(gdn_pool_.device_view()) ||
        std::any_of(ngram_verify_.bindings.begin(), ngram_verify_.bindings.end(),
            [](const auto& rank) { return rank.pool && !supports_ngram_history(rank.pool->device_view()); })) {
        config_.ngram.enabled = false;
    }
    return Status::make_ok();
}

OnlineNgramStats ContinuousBatcher::ngram_stats() const noexcept {
    OnlineNgramStats total = completed_ngram_stats_;
    for (const RuntimeRequest& request : requests_) {
        if (!request.ngram_stats_committed) {
            total += request.ngram.stats();
        }
    }
    return total;
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

Status ContinuousBatcher::set_token_bitmask(uint64_t id, const uint32_t* mask,
                                             bool waiting) {
    auto it = request_index_.find(id);
    if (it == request_index_.end() || it->second->state == RequestState::Finished) {
        return Status::invalid_argument("request is not active", __FILE__, __LINE__);
    }
    it->second->token_bitmask = mask;
    it->second->waiting_for_token_bitmask = waiting;
    return Status::make_ok();
}

Status ContinuousBatcher::set_structured_dflash(uint64_t id, bool delayed, int32_t close_token) {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) return Status::invalid_argument("request not found", __FILE__, __LINE__);
    it->second->structured_dflash = true;
    it->second->grammar_delayed = delayed;
    it->second->grammar_close_token = close_token;
    return Status::make_ok();
}

Status ContinuousBatcher::activate_grammar(uint64_t id) {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) return Status::invalid_argument("request not found", __FILE__, __LINE__);
    it->second->grammar_delayed = false;
    return Status::make_ok();
}

Status ContinuousBatcher::set_draft_masks(uint64_t id, uint64_t round, const uint32_t* mask,
                                          uint32_t stride, uint32_t valid_prefix) {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) return Status::invalid_argument("request not found", __FILE__, __LINE__);
    RuntimeRequest& request = *it->second;
    if (request.draft_mask_state != DraftMaskState::PendingMask || request.draft_round_id != round ||
        mask == nullptr || stride == 0u || valid_prefix > request.pending_drafts.size())
        return Status::invalid_argument("unexpected draft masks", __FILE__, __LINE__);
    request.draft_bitmask = mask;
    request.draft_mask_stride = stride;
    request.draft_valid_prefix = valid_prefix;
    request.draft_mask_state = DraftMaskState::Ready;
    request.waiting_for_token_bitmask = false;
    return Status::make_ok();
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

Status ContinuousBatcher::set_ignore_eos(uint64_t id, bool ignore_eos) {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) {
        return Status::invalid_argument("request id not found", __FILE__, __LINE__);
    }
    auto& request = *it->second;
    request.ignore_eos = ignore_eos;
    if (request.dflash && request.dflash->decoder)
        request.dflash->decoder->config.eos_tokens = ignore_eos ? qwen_hybrid::StopTokens{} : config_.eos_token_ids;
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
    if (config_.ngram.enabled && !request.ngram_stats_committed) {
        completed_ngram_stats_ += request.ngram.stats();
        request.ngram_stats_committed = true;
    }
    request.state = RequestState::Finished;
    request.finish_reason = reason;
    request.pending_decode_token = -1;
    prefix_finish_capture(request, reason, stream);
    if (prefix_cache_ != nullptr) {
        prefix_cache_->release_reuse_slot(request.id);
    }
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
    const bool slot_mode =
        prefix_cache_->strategy() == PrefixCacheStrategy::Slot;
    if (slot_mode && request.prefix_slot == kInvalidReuseSlot) {
        return;
    }
    const uint32_t position = request.sequence.position;
    if (position == 0 || position % config_.page_tokens != 0) {
        return;
    }
    if (position > request.input_tokens.size()) {
        return;
    }
    if (slot_mode) {
        if (position != request.prefix_save_position) return;
    } else {
        const bool split_point = position == request.prefix_save_position;
        const bool junction = position == request.prefix_junction_position;
        const bool input_end = position == request.input_tokens.size();
        const uint32_t interval = prefix_cache_->save_interval();
        const bool periodic = interval != 0u &&
            position >= request.prefix_last_save_position + interval;
        if (!split_point && !junction && !input_end && !periodic) return;
        if (split_point) {
            const uint32_t input_len =
                static_cast<uint32_t>(request.input_tokens.size());
            const uint32_t tail = input_len > config_.page_tokens
                ? ((input_len - 1u) / config_.page_tokens) * config_.page_tokens
                : 0u;
            request.prefix_save_position = tail > position ? tail : 0u;
        }
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
    const Status save_st = slot_mode
        ? prefix_cache_->save_slot_boundary(request.id, request.sequence, tokens,
                                            position, stream_, ring_ptr)
        : prefix_cache_->save_input_boundary(request.sequence, tokens, position,
                                             stream_, ring_ptr);
    if (save_st.ok()) request.prefix_last_save_position = position;
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
    if (prefix_cache_->strategy() == PrefixCacheStrategy::Slot) {
        return;
    }
    auto capture = prefix_cache_->begin_capture(request.sequence, stream_);
    if (!capture.ok()) {
        return;
    }
    request.prefix_capture = capture.value();
    const uint32_t position = request.sequence.position;
    request.prefix_boundary_pos = position % config_.page_tokens == 0u ? position : 0u;
}

void ContinuousBatcher::prefix_capture_after_step(RuntimeRequest& request) {
    if (prefix_cache_ == nullptr || request.prefix_capture == 0) {
        return;
    }
    if (prefix_cache_->strategy() == PrefixCacheStrategy::Slot) {
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
    if (prefix_cache_->strategy() == PrefixCacheStrategy::Slot) {
        request.prefix_capture = 0;
        request.prefix_boundary_pos = 0;
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
    static const bool bt_wall = std::getenv("SHELF_BATCH_STEPWALL") != nullptr;
    const auto bt_step_t0 = bt_wall ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    timespec bt_cpu_start{}, bt_cpu_end{};
    if (bt_wall) clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bt_cpu_start);
    const bool banker_mode = config_.admission_policy == KVAdmissionPolicy::BankerSafe;
    KVBankerState kv_state = build_kv_banker_state();
    uint32_t dflash_prefilled_requests = 0u;
    uint32_t dflash_prefilled_tokens = 0u;
    if (banker_mode && !is_safe(kv_state)) {
        return Status::invalid_state(
            "banker invariant violated: active KV state is unsafe", __FILE__, __LINE__);
    }

    const bool decode_due = config_.prefill_decode_interval != 0u &&
        prefill_chunks_since_decode_ >= config_.prefill_decode_interval &&
        std::any_of(requests_.begin(), requests_.end(), [](const RuntimeRequest& request) {
            return request_decode_ready(request) ||
                (request.state == RequestState::Active && request.dflash && request.dflash->ready &&
                 request.finish_reason == FinishReason::None && !request.waiting_for_token_bitmask);
        });
    for (auto& request : requests_) {
        if (decode_due || (dflash_prefilled_requests != 0u && config_.prefill_decode_interval != 0u &&
            static_cast<uint64_t>(prefill_chunks_since_decode_) + dflash_prefilled_requests >=
                config_.prefill_decode_interval)) break;
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
            const PrefixHit hit = prefix_cache_->lookup(
                prefix_tokens,
                static_cast<uint32_t>(prefix_tokens.size()) - 1u);
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
                const std::span<const int32_t> restored_prefix(
                    request.input_tokens.data(), request.restored_tokens);
                const Status seed_st = tp_batch_hook_ != nullptr
                    ? tp_batch_hook_->on_prefix_restored(request.sequence,
                                                         restored_prefix)
                    : (executor_.external_prefix_restore
                           ? executor_.external_prefix_restore(request.sequence.slot,
                                                               restored_prefix)
                           : Status::make_ok());
                if (!seed_st.ok()) {
                    request.error_message = seed_st.message();
                    (void)capacity_.register_claim(request.id, request.max_kv_pages);
                    const Status finish_st =
                        finish_request(request, FinishReason::Error, stream_);
                    if (!finish_st.ok()) {
                        return finish_st;
                    }
                    continue;
                }
                prefix_begin_capture(request);
            }
            if (prefix_cache_->strategy() == PrefixCacheStrategy::Slot) {
                const uint32_t input_len =
                    static_cast<uint32_t>(request.input_tokens.size());
                request.prefix_slot =
                    prefix_cache_->reserve_reuse_slot(request.id, hit);
                request.prefix_save_position =
                    request.prefix_slot != kInvalidReuseSlot &&
                            input_len > config_.page_tokens
                        ? ((input_len - 1u) / config_.page_tokens) * config_.page_tokens
                        : 0u;
            } else {
                const uint32_t input_len =
                    static_cast<uint32_t>(request.input_tokens.size());
                const uint32_t tail = input_len > config_.page_tokens
                    ? ((input_len - 1u) / config_.page_tokens) * config_.page_tokens
                    : 0u;
                const uint32_t matched = hit.match_tokens;
                const uint32_t junction =
                    matched > request.restored_tokens && matched < input_len
                        ? matched
                        : 0u;
                request.prefix_junction_position = junction;
                request.prefix_save_position =
                    junction != 0u ? junction : tail;
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
        if (config_.prefill_decode_interval != 0u)
            prefill_chunks_since_decode_ = static_cast<uint32_t>(std::min<uint64_t>(
                config_.prefill_decode_interval, static_cast<uint64_t>(prefill_chunks_since_decode_) +
                dflash_prefilled_requests));
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
    const bool decode_interrupt = prefill_decode_should_interrupt(
        active, config_.prefill_decode_interval, prefill_chunks_since_decode_);
    const auto adaptive_start = dflash_.adaptive ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
    uint64_t adaptive_before_tokens = 0;
    uint64_t adaptive_before_rounds = 0;
    if (dflash_.adaptive && dflash_.stats != nullptr) {
        adaptive_before_rounds = dflash_.stats->adaptive_rounds;
        for (const RuntimeRequest* request : active_requests) {
            adaptive_before_tokens += request->generated_tokens;
        }
    }
    auto record_adaptive_actual = [&] {
        if (!dflash_.adaptive || dflash_.stats == nullptr ||
            dflash_.stats->adaptive_rounds == adaptive_before_rounds) return;
        uint64_t after = 0;
        for (const RuntimeRequest* request : active_requests) after += request->generated_tokens;
        dflash_.stats->adaptive_actual_tokens += static_cast<double>(after - adaptive_before_tokens);
        dflash_.stats->adaptive_actual_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - adaptive_start).count();
    };
    bool dflash_executed = false;
    uint32_t dflash_tokens = 0;
    uint32_t dflash_requests = 0;
    const Status dflash_st = run_dflash2_round(active_requests, dflash_executed,
                                               dflash_tokens, dflash_requests);
    if (!dflash_st.ok()) {
        return dflash_st;
    }
    kv_state = build_kv_banker_state();

    auto plan_result = schedule_requests(active, config_.max_num_batched_tokens,
                                         config_.max_scheduled_requests, config_.page_tokens,
                                         &kv_state, decode_interrupt);
    if (!plan_result.ok()) {
        return plan_result.status();
    }
    SchedulePlan plan = plan_result.release();
    if (config_.prefill_decode_interval != 0u) {
        if (plan.batch.num_decode_requests || dflash_executed)
            prefill_chunks_since_decode_ = 0u;
        else if (plan.batch.num_prefill_requests)
            prefill_chunks_since_decode_ = std::min(config_.prefill_decode_interval,
                                                   prefill_chunks_since_decode_ + plan.batch.num_prefill_requests);
        else if (decode_due)
            prefill_chunks_since_decode_ = 0u;
    }
    if (plan.empty()) {
        record_adaptive_actual();
        return StepResult{dflash_executed, dflash_tokens, dflash_requests};
    }

    std::vector<ExecuteBatchOptions> verify_options;
    if (config_.ngram.enabled && !dflash2_available()) {
        const Status st = plan_ngram_round(plan, active_requests, kv_state);
        if (!st.ok()) return st;
        if (plan.batch.num_verify_requests != 0u) {
            const uint32_t rows = plan.batch.num_tokens;
            const Status init = ensure_ngram_verify_initialized();
            if (!init.ok()) return init;
            for (const auto& rank : ngram_verify_.ranks) {
                ExecuteBatchOptions options;
                options.gdn_spec_history = gdn_spec_history_view(rank.history, rows);
                options.use_verify_graph = plan.batch.num_requests == 1u &&
                    (rows == 4u || rows == 8u || rows == 16u);
                verify_options.push_back(options);
            }
        }
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

    if (dflash_.adaptive) {
        for (auto& scheduled : plan.scheduled_requests) {
            RuntimeRequest* request = resolve(scheduled);
            if (request != nullptr && request->dflash != nullptr &&
                request->pending_decode_token >= 0) {
                scheduled.sampling_index = request->dflash->decoder->sample_index;
            }
        }
    }

    const auto exec_started = std::chrono::steady_clock::now();
    const bool bt_decode = plan.batch.num_tokens == plan.batch.num_requests;
    plan.batch.sampled_ids_only = config_.sampled_ids_only;
    auto exec_result = [&] {
        if (tp_batch_hook_ != nullptr) {
            return verify_options.empty() ? tp_batch_hook_->on_execute(plan.batch)
                                          : tp_batch_hook_->on_execute(plan.batch, verify_options.data());
        }
        if (verify_options.empty()) return execute_batch(executor_, plan.batch, stream_);
        return execute_batch(executor_, plan.batch, stream_, verify_options.front());
    }();
    const auto bt_exec_end = bt_wall ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
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

    auto* host_sampled = static_cast<int32_t*>(executor_.host_sampled_staging);
    if (output.num_outputs > executor_.host_sampled_staging_bytes / sizeof(int32_t)) {
        return Status::invalid_state("sample output exceeds host staging", __FILE__, __LINE__);
    }
    if (output.num_outputs > 0) {
        hipError_t copy_err = hipMemcpy(
            host_sampled,
            output.sampled_tokens.data<int32_t>(),
            output.num_outputs * sizeof(int32_t),
            hipMemcpyDeviceToHost);
        if (copy_err != hipSuccess) {
            return Status::hip_error(
                "step hipMemcpy sampled tokens",
                hipGetErrorString(copy_err), __FILE__, __LINE__);
        }
    }

    const auto bt_sync_end = bt_wall ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
    if (dflash_.adaptive) {
        std::vector<dflash2::DFlash2TargetTapAppendRequest> tap_requests;
        std::vector<DFlash2SpecDecoder*> tap_decoders;
        tap_requests.reserve(plan.scheduled_requests.size());
        tap_decoders.reserve(plan.scheduled_requests.size());
        for (const auto& scheduled : plan.scheduled_requests) {
            RuntimeRequest* request = resolve(scheduled);
            if (request == nullptr || request->dflash == nullptr ||
                scheduled.execution_class != ::shelf::runtime::ExecutionClass::DECODE ||
                request->pending_decode_token < 0) continue;
            DFlash2SpecDecoder& decoder = *request->dflash->decoder;
            dflash2::DFlash2TargetTapAppendRequest tap_request;
            tap_request.context = decoder.context;
            tap_request.position = scheduled.prefix_tokens;
            for (uint32_t t = 0; t < shelf::kernel::kDFlash2TargetTaps; ++t) {
                tap_request.taps[t] = executor_.dflash_target_hidden[t].data<bf16_t>() +
                    static_cast<std::size_t>(scheduled.token_begin) * decoder.hidden_size;
            }
            tap_requests.push_back(tap_request);
            tap_decoders.push_back(&decoder);
        }
        if (!tap_requests.empty()) {
            const Status st = dflash2::dflash2_append_target_taps_batch(
                *dflash_.executor, tap_requests.data(), static_cast<uint32_t>(tap_requests.size()), stream_);
            if (!st.ok()) return st;
            for (DFlash2SpecDecoder* decoder : tap_decoders) ++decoder->sample_index;
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
        if (scheduled.execution_class == ExecutionClass::SPEC_VERIFY) {
            if (scheduled.num_tokens > output.num_outputs - row) {
                return Status::invalid_state("ngram verify output count mismatch", __FILE__, __LINE__);
            }
            const Status st = commit_ngram_verify(request, scheduled,
                plan.token_ids.data() + scheduled.token_begin + 1u, host_sampled + row);
            if (!st.ok()) {
                if (first_error.ok()) first_error = st;
                if (request.state == RequestState::Active) {
                    request.sequence.mark_poisoned();
                    request.error_message = st.message();
                    const Status cleanup = finish_request(request, FinishReason::Error, stream_);
                    if (!cleanup.ok()) return cleanup;
                }
            }
            row += scheduled.num_tokens;
            continue;
        }
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
                        char reason[160];
                        std::snprintf(reason, sizeof(reason),
                            "greedy sampling returned %d for request %lld at position %u (output %u of %u)",
                            token, static_cast<long long>(request.external_request_id), scheduled.prefix_tokens,
                            row - 1u, output.num_outputs);
                        return Status::invalid_state(reason, __FILE__, __LINE__);
                    }
                    const CommitAction action =
                        commit_sampled_token(request, token, config_.eos_token_ids);
                    if (config_.ngram.enabled) {
                        request.ngram.ingest_timed(request.generated.back());
                    }
                    if (dflash_.adaptive && request.dflash != nullptr &&
                        scheduled.execution_class == ::shelf::runtime::ExecutionClass::DECODE &&
                        action == CommitAction::Continue) {
                        DFlash2SpecDecoder& decoder = *request.dflash->decoder;
                        decoder.token_history.push_back(request.generated.back());
                        if (decoder.token_history.size() > 8192u) {
                            decoder.token_history.erase(decoder.token_history.begin(),
                                decoder.token_history.end() - 4096u);
                        }
                        request.dflash->pending_token = request.pending_decode_token;
                        request.pending_decode_token = -1;
                    }
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

    if (bt_wall && bt_decode) {
        const auto bt_end = std::chrono::steady_clock::now();
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bt_cpu_end);
        const double cpu_ms = (bt_cpu_end.tv_sec - bt_cpu_start.tv_sec) * 1000.0 +
                              (bt_cpu_end.tv_nsec - bt_cpu_start.tv_nsec) / 1000000.0;
        auto ms = [](auto x, auto y) {
            return std::chrono::duration<double, std::milli>(y - x).count();
        };
        std::fprintf(stderr, "STEPWALL rows=%u pre=%.3f exec_submit=%.3f sync=%.3f post=%.3f total=%.3f cpu=%.3f\n",
                     plan.batch.num_requests, ms(bt_step_t0, exec_started),
                     ms(exec_started, bt_exec_end), ms(bt_exec_end, bt_sync_end),
                     ms(bt_sync_end, bt_end), ms(bt_step_t0, bt_end), cpu_ms);
    }
    record_adaptive_actual();
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
    request.dflash_used = true;
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
    if (dflash_.adaptive) {
        const uint32_t max_k = dflash2_adaptive_max_k(dflash_.num_drafts);
        session.adaptive.reset(std::min(max_k, kDFlash2AdaptiveInitialK));
        session.adaptive_rows = session.adaptive.current_k + 1u;
    }
    if (request.max_new_tokens == 0u) {
        request.finish_reason = FinishReason::MaxNewTokens;
        return Status::make_ok();
    }
    DFlash2SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = dflash_.num_drafts;
    decoder_config.tree_budget = dflash_.tree_budget;
    decoder_config.eos_tokens =
        request.ignore_eos ? qwen_hybrid::StopTokens{} : config_.eos_token_ids;
    decoder_config.host_token_ids = dflash_.host_token_ids;
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
        if (request.sequence.position != offset) {
            return;
        }
        prefix_save_prefill(request);
    };
    const uint32_t tail_position = dflash2_prefill_tail_position(prompt_count, page_tokens);
    const uint32_t first_checkpoint = request.prefix_save_position;
    const uint32_t second_checkpoint =
        tail_position != first_checkpoint ? tail_position : 0u;
    auto prefill_result = dflash2_spec_prefill(
        *session.decoder, request.input_tokens.data(), prompt_count,
        request.restored_tokens, 0u, save_input_boundary,
        first_checkpoint, second_checkpoint,
        request.grammar_delayed ? nullptr : request.token_bitmask);
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
    if (config_.ngram.enabled) {
        request.ngram.ingest_timed(session.pending_token);
    }
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
    if (!request.ignore_eos &&
        qwen_hybrid::is_stop_token(config_.eos_token_ids, session.pending_token)) {
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
    static const bool host_trace = std::getenv("SHELF_DFLASH2_COST_TRACE") != nullptr;
    static const char* phase_trace_env = std::getenv("SHELF_DFLASH2_BATCH_PHASE_TRACE");
    static const bool phase_trace = phase_trace_env != nullptr && phase_trace_env[0] != '\0' &&
                                    phase_trace_env[0] != '0';
    timespec host_start{}, host_end{};
    const auto host_wall_start = host_trace ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
    if (host_trace) clock_gettime(CLOCK_THREAD_CPUTIME_ID, &host_start);
    uint32_t host_sessions = 0u;
    uint32_t host_rows = 0u;
    if (dflash_.adaptive) {
        select_dflash2_adaptive(active, dflash_, config_, adaptive_round_);
    }
    std::vector<DFlash2SpecBatchSlot> slots;
    std::vector<RuntimeRequest*> slot_requests;
    std::vector<bool> slot_used_ngram;
    std::vector<RuntimeRequest*> pending_finish;
    uint32_t other_decoding_requests = 0u;
    for (RuntimeRequest* request : active) {
        if (request->state != RequestState::Active) continue;
        if (request->dflash == nullptr) {
            if (request->generated_tokens > 0u &&
                request->generated_tokens < request->max_new_tokens &&
                request->finish_reason == FinishReason::None) ++other_decoding_requests;
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
        const uint32_t remaining = request->max_new_tokens - request->generated_tokens;
        const uint32_t seq_room = request->sequence.max_seq_len > request->sequence.position
            ? request->sequence.max_seq_len - request->sequence.position
            : 0u;
        const uint32_t draft_room = (remaining > 1u && seq_room >= 2u)
            ? std::min(remaining - 1u, seq_room - 1u)
            : 0u;
        const bool stochastic = request->sampling.temperature != 0.0f;
        bool adaptive_ngram_rescue = false;
        if (dflash_.adaptive && !request->structured_dflash && request->dflash->adaptive_rows == 0u) {
            if (config_.ngram.enabled && draft_room > 0u) {
                const OnlineNgramProposal prop = request->ngram.propose(dflash_.num_drafts, stochastic);
                if (prop.hit) {
                    request->pending_ngram_proposal = prop;
                    adaptive_ngram_rescue = true;
                }
            }
            if (!adaptive_ngram_rescue) {
                request->pending_decode_token = request->dflash->pending_token;
                ++other_decoding_requests;
                continue;
            }
        }
        if (dflash_.adaptive) request->pending_decode_token = -1;
        DFlash2SpecBatchSlot slot;
        slot.decoder = request->dflash->decoder.get();
        slot.anchor_token = request->dflash->pending_token;
        slot.num_drafts = (dflash_.adaptive && !request->structured_dflash && !adaptive_ngram_rescue)
            ? request->dflash->adaptive_rows - 1u
            : dflash_.num_drafts;
        static const char* const measure_drafts = std::getenv("SHELF_DFLASH2_MEASURE_VERIFY_DRAFTS");
        if (!dflash_.adaptive && measure_drafts != nullptr) {
            char* end = nullptr;
            const unsigned long requested = std::strtoul(measure_drafts, &end, 10);
            if (end != measure_drafts && *end == '\0' && requested > 0u &&
                requested <= dflash_.num_drafts) {
                slot.num_drafts = static_cast<uint32_t>(requested);
            }
        }
        if (!dflash_.adaptive && measure_drafts != nullptr) slot.proposal_drafts = dflash_.num_drafts;
        slot.batch_anchor_only = dflash_.adaptive && !request->structured_dflash &&
            !dflash_.batch->tree_budget_auto && dflash_.batch->tree_budget == 0u;
        slot.remaining_tokens = remaining;
        slot.page_tokens =
            request->prefix_capture != 0u ? config_.page_tokens : 0u;
        if (slot.page_tokens != 0u && dflash2_can_capture_boundary(*slot.decoder))
            slot.capture_boundary = [this, request] { prefix_capture_after_step(*request); };
        slot.thinking_allowance = thinking_allowance(*request);
        slot.thinking_close_token = request->thinking.close_token;
        bool use_ngram_for_slot = false;
        if (request->structured_dflash) {
            slot.disable_tree = true;
            if (request->grammar_delayed) {
                slot.stop_after_token = request->grammar_close_token;
                if (config_.ngram.enabled && draft_room > 0u) {
                    const OnlineNgramProposal prop = request->ngram.propose(slot.num_drafts, stochastic);
                    if (prop.hit) {
                        request->pending_ngram_proposal = prop;
                        slot.precomputed_drafts = request->pending_ngram_proposal.tokens.data();
                        request->ngram.note_drafter_skipped();
                        use_ngram_for_slot = true;
                    }
                }
            } else if (request->draft_mask_state == DraftMaskState::PendingMask) {
                continue;
            } else if (request->draft_mask_state == DraftMaskState::Idle) {
                if (config_.ngram.enabled && draft_room > 0u) {
                    const OnlineNgramProposal prop = request->ngram.propose(slot.num_drafts, stochastic);
                    if (prop.hit) {
                        request->ngram.note_drafter_skipped();
                        request->ngram_round_active = true;
                        request->pending_drafts.assign(prop.tokens.begin(),
                                                       prop.tokens.begin() + slot.num_drafts);
                        ++request->draft_round_id;
                        request->draft_mask_state = DraftMaskState::PendingMask;
                        request->waiting_for_token_bitmask = true;
                        executed = true;
                        continue;
                    }
                }
                request->ngram_round_active = false;
                auto proposed = dflash2_spec_propose_slot(*dflash_.batch, slot);
                if (!proposed.ok()) return proposed.status();
                request->pending_drafts = proposed.release();
                ++request->draft_round_id;
                request->draft_mask_state = DraftMaskState::PendingMask;
                request->waiting_for_token_bitmask = true;
                executed = true;
                continue;
            } else {
                slot.num_drafts = static_cast<uint32_t>(request->pending_drafts.size());
                slot.precomputed_drafts = request->pending_drafts.data();
                slot.token_bitmask = request->draft_bitmask;
                slot.token_bitmask_row_stride_words = request->draft_mask_stride;
                slot.max_accept_drafts = request->draft_valid_prefix;
                use_ngram_for_slot = request->ngram_round_active;
            }
        } else if (config_.ngram.enabled && draft_room > 0u) {
            if (!adaptive_ngram_rescue) {
                request->pending_ngram_proposal = request->ngram.propose(slot.num_drafts, stochastic);
            }
            if (request->pending_ngram_proposal.hit) {
                slot.precomputed_drafts = request->pending_ngram_proposal.tokens.data();
                slot.disable_tree = true;
                request->ngram.note_drafter_skipped();
                use_ngram_for_slot = true;
            }
        }
        slots.push_back(slot);
        slot_requests.push_back(request);
        slot_used_ngram.push_back(use_ngram_for_slot);
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
        static const bool cost_trace = std::getenv("SHELF_DFLASH2_COST_TRACE") != nullptr;
        timespec cpu_start{}, cpu_end{};
        if (cost_trace) clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start);
        const auto cost_start = cost_trace ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        auto batch_result = dflash2_spec_step_batch(*dflash_.batch, slots, other_decoding_requests);
        if (cost_trace) clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end);
        const double cost_wall = cost_trace ? std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cost_start).count() : 0.0;
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
        host_sessions = static_cast<uint32_t>(slots.size());
        host_rows = results.total_rows;
        if (cost_trace) {
            const double cpu_ms = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                                  (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000000.0;
            std::fprintf(stderr, "DFCOST sessions=%zu rows=%u wall=%.3f draft_gpu=%.3f verify_gpu=%.3f cpu=%.3f\n",
                         slots.size(), results.total_rows, cost_wall,
                         results.draft_gpu_ms, results.verify_gpu_ms, cpu_ms);
        }
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
            if (request.structured_dflash && request.draft_mask_state == DraftMaskState::Ready) {
                request.draft_mask_state = DraftMaskState::Idle;
                request.pending_drafts.clear();
                request.draft_bitmask = nullptr;
            }
            const DFlash2SpecIterationOutput& step = results.slots[i];
            if (phase_trace) {
                std::fprintf(stderr, "DFPHASE_REQUEST id=%lld slot=%u accepted=%u emitted=%u drafts=%u\n",
                             static_cast<long long>(request.external_request_id),
                             request.sequence.slot, step.num_accepted, step.emitted_count, step.num_drafts);
            }
            if (dflash_.adaptive && step.num_drafts == request.dflash->adaptive.current_k &&
                !request.structured_dflash) {
                observe_dflash2_adaptive(*request.dflash, step.num_accepted,
                                         static_cast<uint32_t>(slot_requests.size()),
                                         dflash_.num_drafts);
            }
            if (slot_used_ngram[i]) {
                request.ngram.record_outcome(step.num_accepted);
                request.ngram_round_active = false;
            }
            for (uint32_t j = 0; j < step.emitted_count; ++j) {
                request.generated.push_back(step.emitted[j]);
                note_generated_token(request, step.emitted[j]);
                if (config_.ngram.enabled) {
                    request.ngram.ingest_timed(step.emitted[j]);
                }
            }
            request.generated_tokens = static_cast<uint32_t>(request.generated.size());
            request.dflash->pending_token = step.pending_token;
            if (dflash_.stats != nullptr) {
                DFlash2BatcherStats& stats = *dflash_.stats;
                stats.proposed_drafts += step.num_drafts;
                stats.accepted_drafts += step.num_accepted;
                stats.generated_tokens += step.emitted_count;
                if (step.num_drafts != 0u) {
                    if (step.num_accepted == step.num_drafts) ++stats.full_accepts;
                    else ++stats.partial_accepts;
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
            prefix_capture_after_step(request);
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

    if (host_trace && host_sessions != 0u) {
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &host_end);
        const double cpu_ms = (host_end.tv_sec - host_start.tv_sec) * 1000.0 +
                              (host_end.tv_nsec - host_start.tv_nsec) / 1000000.0;
        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - host_wall_start).count();
        std::fprintf(stderr, "DFHOST sessions=%u rows=%u wall=%.3f cpu=%.3f\n",
                     host_sessions, host_rows, wall_ms, cpu_ms);
    }
    return first_error;
}

Status ContinuousBatcher::ensure_ngram_verify_initialized() {
    if (ngram_verify_.initialized) return Status::make_ok();
    const uint32_t drafts = config_.ngram.num_drafts == 0u ? 7u : config_.ngram.num_drafts;
    const uint32_t needed_rows = standalone_ngram_max_rows(executor_.config.max_scheduled_requests, drafts);
    std::vector<NgramVerifyRankBinding> bindings = ngram_verify_.bindings;
    if (bindings.empty()) {
        bindings.push_back(NgramVerifyRankBinding{
            &gdn_pool_, ngram_verify_.arena, stream_, 0});
    }
    ngram_verify_.ranks.reserve(bindings.size());
    for (const NgramVerifyRankBinding& binding : bindings) {
        NgramVerifyRankState rank;
        rank.pool = binding.pool;
        rank.arena = binding.arena != nullptr ? binding.arena : ngram_verify_.arena;
        rank.stream = binding.stream != nullptr ? binding.stream : stream_;
        rank.device = binding.device;
        if (rank.pool != nullptr && rank.pool->num_gdn_states() > 0u &&
            rank.pool->device_view().recurrent_base != nullptr) {
            if (rank.arena == nullptr) {
                shutdown_ngram_verify();
                return Status::invalid_state(
                    "standalone ngram verify requires a bound GPU arena for GDN history",
                    __FILE__, __LINE__);
            }
            auto scope = shelf::gpu::ScopedDevice::create(rank.device);
            if (!scope.ok()) {
                shutdown_ngram_verify();
                return scope.status();
            }
            auto hist = create_gdn_spec_history(*rank.arena, *rank.pool, needed_rows, true);
            if (!hist.ok()) {
                shutdown_ngram_verify();
                return hist.status();
            }
            rank.history = hist.release();
        }
        ngram_verify_.ranks.push_back(std::move(rank));
    }
    ngram_verify_.initialized = true;
    return Status::make_ok();
}

void ContinuousBatcher::shutdown_ngram_verify() noexcept {
    for (NgramVerifyRankState& rank : ngram_verify_.ranks) {
        if (rank.history.initialized) {
            (void)shutdown_gdn_spec_history(rank.history);
        }
    }
    ngram_verify_.ranks.clear();
    ngram_verify_.initialized = false;
}

Status ContinuousBatcher::plan_ngram_round(SchedulePlan& plan,
                                         const std::vector<RuntimeRequest*>& active,
                                         KVBankerState& kv) {
    const uint32_t count = plan.batch.num_requests;
    const uint32_t configured_k = config_.ngram.num_drafts != 0u ? config_.ngram.num_drafts : 7u;
    if (count >= kNgramDecodeOnlyBatchSize || plan.batch.num_decode_requests != count ||
        active.size() != count ||
        executor_.config.external_hidden_input || executor_.uses_qsa_state ||
        executor_.uses_ple_conv_state || configured_k > OnlineNgramProposal::kMaxDraftTokens) {
        return Status::make_ok();
    }
    const uint32_t budget = std::min({standalone_ngram_max_rows(count, configured_k),
        config_.max_num_batched_tokens, executor_.config.max_scheduled_output_rows});
    if (budget <= count) return Status::make_ok();

    const auto least_generated = std::min_element(active.begin(), active.end(),
        [](const RuntimeRequest* a, const RuntimeRequest* b) {
            return a->generated_tokens < b->generated_tokens;
        });
    const uint32_t minimum = (*least_generated)->generated_tokens;
    uint32_t extra = budget - count;
    for (uint32_t i = 0; i < count; ++i) {
        RuntimeRequest& req = *active[i];
        const uint32_t position = req.sequence.position;
        const uint32_t room = std::min({req.sequence.max_seq_len - position,
            req.max_new_tokens - req.generated_tokens, thinking_allowance(req)});
        uint32_t k = room > 1u ? std::min({configured_k, extra, room - 1u}) : 0u;
        if (count == 1u && k != configured_k) k = 0u;
        req.pending_ngram_proposal.hit = false;
        if (k == 0u || req.token_bitmask != nullptr || req.dflash != nullptr ||
            (count >= 3u && req.generated_tokens - minimum > configured_k)) continue;
        const auto blocks = kv.processes[i].allocated_pages;
        const auto requested = additional_kv_pages(position, blocks, k + 1u, config_.page_tokens);
        if (requested != 0u && !can_grant_pages(kv, i, requested)) continue;
        auto prop = req.ngram.propose(k, req.sampling.temperature != 0.0f);
        if (!prop.hit) continue;
        if (count > 1u) prop.count = prop.valid_drafts;
        extra -= prop.count;
        const auto pages = additional_kv_pages(position,
            kv.processes[i].allocated_pages, prop.count + 1u, config_.page_tokens);
        kv.processes[i].allocated_pages += pages;
        kv.available_pages -= pages;
        req.pending_ngram_proposal = prop;
    }
    if (extra == budget - count) return Status::make_ok();

    SchedulePlan mixed;
    for (auto* req : active) {
        if (req->pending_ngram_proposal.hit) continue;
        auto st = mixed.append_decode(&req->sequence,
            req->pending_decode_token, req->sequence.position, true, true,
            req->sampling, req->generated_tokens);
        if (!st.ok()) return st;
        mixed.scheduled_requests.back().token_bitmask = req->token_bitmask;
    }
    for (auto* req : active) {
        const auto& prop = req->pending_ngram_proposal;
        if (!prop.hit) continue;
        std::array<int32_t, OnlineNgramProposal::kMaxDraftTokens + 1u> tokens;
        tokens[0] = req->pending_decode_token;
        std::copy_n(prop.tokens.begin(), prop.count, tokens.begin() + 1);
        const auto rows = prop.count + 1u;
        auto st = mixed.append_verify(&req->sequence, std::span(tokens.data(), rows),
                                      req->sequence.position, rows, req->sampling,
                                      req->generated_tokens);
        if (!st.ok()) return st;
    }
    plan = std::move(mixed);
    plan.finalize();
    return Status::make_ok();
}

Status ContinuousBatcher::commit_ngram_verify(RuntimeRequest& req, const ScheduledRequest& scheduled,
                                             const int32_t* drafts, const int32_t* sampled) {
    const uint32_t k = scheduled.num_tokens - 1u;
    SpecVerifyResult verify;
    auto st = spec_greedy_accept(drafts, k, sampled, true, verify);
    if (!st.ok()) return st;
    spec_bound_result(verify, req.max_new_tokens - req.generated_tokens, config_.eos_token_ids);
    const uint32_t accepted = verify.num_accepted_drafts;
    if (accepted < k) {
        for (const auto& rank : ngram_verify_.ranks) {
            if (!rank.history.initialized) continue;
            auto scope = shelf::gpu::ScopedDevice::create(rank.device);
            if (!scope.ok()) return scope.status();
            st = restore_gdn_spec_history(*rank.pool, req.sequence.slot, rank.history,
                                          scheduled.token_begin + accepted, rank.stream);
            if (!st.ok()) return st;
        }
        req.sequence.position = scheduled.prefix_tokens + verify.num_committed_target_tokens;
        const uint32_t pages = kv_pages_for_tokens(req.sequence.position, config_.page_tokens);
        st = rollback_sequence_append(req.sequence, pages, stream_);
        if (!st.ok()) return st;
    }
    req.ngram.record_outcome(accepted);
    for (int32_t token : verify.emitted_tokens) {
        const auto action = commit_sampled_token(req, token, config_.eos_token_ids);
        req.ngram.ingest_timed(req.generated.back());
        if (action == CommitAction::Finished) {
            return finish_request(req, req.finish_reason, stream_);
        }
    }
    return Status::make_ok();
}

}
}
