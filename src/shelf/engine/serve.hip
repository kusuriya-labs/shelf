#include <shelf/engine/serve.h>

#include "compute_session_impl.h"

#include <shelf/models/loaded_model.h>
#include <shelf/runtime/dflash2/dflash2_session_pool.h>
#include <shelf/runtime/mtp/mtp_drafter.h>
#include <shelf/runtime/continuous_batcher.h>
#include <shelf/runtime/dflash2/context_state.h>
#include <shelf/runtime/dflash2/executor.h>
#include <shelf/runtime/dflash2/tp_proposal.h>
#include <shelf/runtime/dflash2/weights.h>
#include <shelf/runtime/decode_backend.h>
#include <shelf/runtime/executor.h>
#include <shelf/runtime/prefix_cache.h>
#include <shelf/runtime/runtime_request.h>
#include <shelf/runtime/state_types.h>
#include <shelf/runtime/tp/tensor_parallel.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace shelf::engine {
namespace {

constexpr std::size_t kRecentTerminalLimit = 256;
constexpr std::size_t kMtpSnapshotLimit = 16u;

struct ServeGeneration {
    int64_t external_request_id = 0;
    uint64_t internal_request_id = 0;
    std::size_t emitted_tokens = 0;
    uint32_t max_new_tokens = 0;
    std::chrono::steady_clock::time_point submitted_at;
    std::chrono::steady_clock::time_point first_token_at;
    std::chrono::steady_clock::time_point last_log_at;
    std::size_t last_log_tokens = 0;
    bool has_first_token = false;
    std::shared_ptr<uint32_t> token_bitmask;
    std::shared_ptr<uint32_t> draft_bitmask;
    std::size_t draft_bitmask_capacity = 0;
    uint64_t emitted_draft_round = 0;
    bool waiting_for_mask = false;
    bool grammar_delayed = false;
    int32_t grammar_close_token = -1;
};

struct MtpPrefillLog {
    int64_t external_id = 0;
    uint32_t last_tokens = 0;
    double last_ms = 0.0;
};

void mtp_prefill_progress(void* context, uint32_t processed_tokens, double elapsed_ms) {
    auto* log = static_cast<MtpPrefillLog*>(context);
    const double interval_s = (elapsed_ms - log->last_ms) / 1000.0;
    if (interval_s < 3.0 || processed_tokens <= log->last_tokens) return;
    const uint32_t delta = processed_tokens - log->last_tokens;
    std::fprintf(stderr,
                 "[req %lld] prefill_tok_s=%.1f tokens=%u interval_s=%.2f\n",
                 static_cast<long long>(log->external_id),
                 static_cast<double>(delta) / interval_s, processed_tokens, interval_s);
    log->last_ms = elapsed_ms;
    log->last_tokens = processed_tokens;
}

struct MtpPrefillHooksContext {
    const ServePrefillHooks* hooks = nullptr;
    MtpPrefillLog log;
    bool cancelled = false;
};

void mtp_prefill_report(void* context, uint32_t processed_tokens, double elapsed_ms) {
    auto* hooks = static_cast<MtpPrefillHooksContext*>(context);
    mtp_prefill_progress(&hooks->log, processed_tokens, elapsed_ms);
    if (hooks->hooks->report) hooks->hooks->report(processed_tokens, elapsed_ms);
}

bool mtp_prefill_should_stop(void* context) {
    auto* hooks = static_cast<MtpPrefillHooksContext*>(context);
    const bool stop = hooks->hooks->should_stop ? hooks->hooks->should_stop() : false;
    if (stop) hooks->cancelled = true;
    return stop;
}

struct MtpRequestTiming {
    std::chrono::steady_clock::time_point submitted_at;
    std::chrono::steady_clock::time_point first_token_at;
    std::chrono::steady_clock::time_point last_log_at;
    std::size_t last_log_tokens = 0;
    bool has_first_token = false;
};

void timing_submitted(MtpRequestTiming& timing) {
    timing.submitted_at = std::chrono::steady_clock::now();
}

void timing_first_token(MtpRequestTiming& timing) {
    timing.first_token_at = std::chrono::steady_clock::now();
    timing.last_log_at = timing.first_token_at;
    timing.last_log_tokens = 0;
    timing.has_first_token = true;
}

void log_decode_interval(int64_t external_id, MtpRequestTiming& timing,
                         std::size_t emitted_tokens) {
    if (!timing.has_first_token) return;
    const auto now = std::chrono::steady_clock::now();
    const double interval_s =
        std::chrono::duration<double>(now - timing.last_log_at).count();
    if (interval_s < 3.0 || emitted_tokens <= timing.last_log_tokens) return;
    const std::size_t delta = emitted_tokens - timing.last_log_tokens;
    std::fprintf(stderr,
                 "[req %lld] decode_tok_s=%.1f tokens=%zu interval_s=%.2f\n",
                 static_cast<long long>(external_id),
                 static_cast<double>(delta) / interval_s, emitted_tokens, interval_s);
    timing.last_log_at = now;
    timing.last_log_tokens = emitted_tokens;
}

void log_request_summary(int64_t external_id, const MtpRequestTiming& timing,
                         std::size_t prompt, std::size_t restored,
                         std::size_t generated, double prefill_ms) {
    const auto finished_at = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(
                                finished_at - timing.submitted_at)
                                .count();
    const double ttft_ms =
        timing.has_first_token
            ? std::chrono::duration<double, std::milli>(
                  timing.first_token_at - timing.submitted_at)
                  .count()
            : total_ms;
    const double decode_ms =
        timing.has_first_token
            ? std::chrono::duration<double, std::milli>(
                  finished_at - timing.first_token_at)
                  .count()
            : 0.0;
    const std::size_t prefill = prompt - std::min(restored, prompt);
    const std::size_t decode_tokens = generated > 0u ? generated - 1u : 0u;
    const double prefill_tok_s =
        prefill_ms > 0.0 ? static_cast<double>(prefill) * 1000.0 / prefill_ms : 0.0;
    const double decode_tok_s =
        decode_ms > 0.0 && decode_tokens > 0u
            ? static_cast<double>(decode_tokens) * 1000.0 / decode_ms
            : 0.0;
    std::fprintf(
        stderr,
        "[req %lld] prompt=%zu restored=%zu prefill=%zu gen=%zu "
        "prefill_ms=%.1f prefill_tok_s=%.1f ttft_ms=%.1f decode_ms=%.1f "
        "decode_tok_s=%.1f\n",
        static_cast<long long>(external_id), prompt, restored, prefill, generated,
        prefill_ms, prefill_tok_s, ttft_ms, decode_ms, decode_tok_s);
}

struct MtpRequest {
    int64_t external_id = 0;
    std::unique_ptr<shelf::state::PagedSequenceState> sequence;
    uint32_t drafter_slot = 0u;
    int32_t pending_token = -1;
    uint32_t max_new_tokens = 0u;
    uint32_t generated = 0u;
    uint32_t restored_tokens = 0u;
    std::size_t prompt_tokens = 0u;
    std::vector<int32_t> generated_ids;
    std::vector<int32_t> input_tokens;
    runtime::CaptureId prefix_capture = 0u;
    uint32_t tail_position = 0u;
    shelf::runtime::mtp::MtpDraftSnapshot tail_snapshot;
    MtpRequestTiming timing;
    double prefill_ms = 0.0;
    bool ignore_eos = false;
};

struct MtpServeContext {
    bool enabled = false;
    uint32_t drafts = 4;
    shelf::runtime::mtp::MtpSpecBatchContext batch;
    std::vector<std::unique_ptr<MtpRequest>> active;
    std::vector<uint32_t> free_slots;
    uint64_t rounds = 0u;
    uint64_t accepted_drafts = 0u;
    uint64_t emitted_tokens = 0u;
    uint64_t verify_chunks = 0u;
    uint64_t draft_forwards = 0u;
    uint32_t max_active_requests = 0u;
    uint32_t max_draft_forwards_per_round = 0u;
    uint32_t max_verify_chunks_per_round = 0u;
    double round_ms = 0.0;
    std::deque<uint64_t> snapshot_order;
    std::unordered_map<uint64_t, shelf::runtime::mtp::MtpDraftSnapshot> snapshots;
};

struct MtpSnapshotSaveContext {
    ServeController* controller = nullptr;
    MtpServeContext* mtp = nullptr;
    shelf::runtime::mtp::MtpDrafter* drafter = nullptr;
    const std::vector<int32_t>* input_ids = nullptr;
    shelf::state::PagedSequenceState* target_sequence = nullptr;
    uint32_t save_interval = 0u;
    uint32_t last_save_position = 0u;
    uint32_t tail_position = 0u;
};

struct MtpBoundaryContext {
    ServeController* controller = nullptr;
    MtpRequest* request = nullptr;
};

}  // namespace

struct ServeController::Impl {
    Impl(ComputeSession& compute_session)
        : session(compute_session), state(*compute_session.impl_) {}

    ComputeSession& session;
    ComputeSession::Impl& state;
    ServeConfig config;

    bool dflash_executor_ready = false;
    bool dflash_batch_ready = false;
    bool dflash_sessions_ready = false;
    runtime::dflash2::DFlash2Weights dflash_weights{};
    std::unique_ptr<runtime::dflash2::DFlash2TpProposal> dflash_tp_proposal;
    runtime::dflash2::DFlash2Executor dflash_executor{};
    shelf::runtime::DFlash2SpecBatchContext dflash_batch{};
    shelf::runtime::DFlash2SessionPool dflash_sessions{};
    shelf::runtime::DFlash2BatcherStats dflash_stats{};

    std::map<int64_t, ServeGeneration> inflight;
    std::deque<int64_t> recent_terminal_ids;
    uint32_t max_inflight = 1;
    bool exit_requested = false;
    MtpServeContext mtp;

    void push_event(std::vector<ServeEvent>& events, ServeEvent event) {
        events.push_back(std::move(event));
    }

    void push_token(std::vector<ServeEvent>& events, int64_t request_id,
                    int32_t token_id, bool need_mask = true) {
        ServeEvent event;
        event.kind = ServeEvent::Kind::Token;
        event.request_id = request_id;
        event.token_id = token_id;
        event.need_mask = need_mask;
        push_event(events, std::move(event));
    }

    void push_terminal(int64_t external_request_id) {
        recent_terminal_ids.push_back(external_request_id);
        while (recent_terminal_ids.size() > kRecentTerminalLimit) {
            recent_terminal_ids.pop_front();
        }
    }

    bool is_recent_terminal(int64_t external_request_id) const {
        return std::find(recent_terminal_ids.begin(), recent_terminal_ids.end(),
                         external_request_id) != recent_terminal_ids.end();
    }

    void store_snapshot(shelf::runtime::mtp::MtpDraftSnapshot&& snapshot) {
        if (snapshot.key == 0u) return;
        auto existing = mtp.snapshots.find(snapshot.key);
        if (existing != mtp.snapshots.end()) {
            existing->second = std::move(snapshot);
            return;
        }
        const uint64_t key = snapshot.key;
        mtp.snapshots.emplace(key, std::move(snapshot));
        mtp.snapshot_order.push_back(key);
        while (mtp.snapshot_order.size() > kMtpSnapshotLimit) {
            const uint64_t old = mtp.snapshot_order.front();
            mtp.snapshot_order.pop_front();
            mtp.snapshots.erase(old);
        }
    }

    void emit_tokens(std::vector<ServeEvent>& events, ServeGeneration& generation) {
        const RequestView request = session.request(generation.internal_request_id);
        if (!request.found) return;
        while (generation.emitted_tokens < request.generated_size) {
            if (!generation.has_first_token) {
                generation.first_token_at = std::chrono::steady_clock::now();
                generation.last_log_at = generation.first_token_at;
                generation.has_first_token = true;
            }
            const int32_t token = request.generated[generation.emitted_tokens];
            const bool structured_spec = generation.token_bitmask && request.dflash_used &&
                                         request.structured_dflash;
            push_token(events, generation.external_request_id, token, !structured_spec);
            ++generation.emitted_tokens;
            if (generation.grammar_delayed && token == generation.grammar_close_token) {
                generation.grammar_delayed = false;
                if (structured_spec)
                    (void)session.activate_grammar(generation.internal_request_id);
            }
            if (!structured_spec && generation.token_bitmask && !generation.grammar_delayed &&
                request.state != runtime::RequestState::Finished) {
                generation.waiting_for_mask = true;
                (void)session.set_token_bitmask(generation.internal_request_id,
                                                generation.token_bitmask.get(), true);
            }
        }
        if (!generation.has_first_token) return;
        const auto now = std::chrono::steady_clock::now();
        const double interval_s =
            std::chrono::duration<double>(now - generation.last_log_at).count();
        if (interval_s < 3.0 || generation.emitted_tokens <= generation.last_log_tokens) {
            return;
        }
        const std::size_t delta = generation.emitted_tokens - generation.last_log_tokens;
        std::fprintf(stderr,
                     "[req %lld] decode_tok_s=%.1f tokens=%zu interval_s=%.2f\n",
                     static_cast<long long>(generation.external_request_id),
                     static_cast<double>(delta) / interval_s, generation.emitted_tokens,
                     interval_s);
        generation.last_log_at = now;
        generation.last_log_tokens = generation.emitted_tokens;
    }

    void emit_done(std::vector<ServeEvent>& events, ServeGeneration& generation) {
        const RequestView request = session.request(generation.internal_request_id);
        ServeEvent done;
        done.kind = ServeEvent::Kind::Done;
        done.request_id = generation.external_request_id;
        if (request.found) {
            done.generated_ids.assign(request.generated, request.generated + request.generated_size);
            done.finish_reason = finish_reason_name(request.finish_reason);
            done.prompt_tokens = request.input_tokens_size;
            done.restored_tokens = request.restored_tokens;
            done.prefill_tokens = request.input_tokens_size - request.restored_tokens;
            done.cache_checkpoint_tokens = 0;
            if (request.error_message != nullptr && request.error_message[0] != '\0') {
                done.error_message = request.error_message;
            }
        } else {
            done.finish_reason = "error";
        }
        if (request.found) {
            const auto finished_at = std::chrono::steady_clock::now();
            const double total_ms = std::chrono::duration<double, std::milli>(
                                        finished_at - generation.submitted_at)
                                        .count();
            const double ttft_ms =
                generation.has_first_token
                    ? std::chrono::duration<double, std::milli>(
                          generation.first_token_at - generation.submitted_at)
                          .count()
                    : total_ms;
            const double decode_ms =
                generation.has_first_token
                    ? std::chrono::duration<double, std::milli>(
                          finished_at - generation.first_token_at)
                          .count()
                    : 0.0;
            const std::size_t prompt = request.input_tokens_size;
            const std::size_t restored =
                std::min<std::size_t>(request.restored_tokens, prompt);
            const std::size_t prefill = prompt - restored;
            const std::size_t generated = request.generated_size;
            const double prefill_tok_s =
                request.prefill_ms > 0.0
                    ? static_cast<double>(prefill) * 1000.0 / request.prefill_ms
                    : 0.0;
            const double decode_tok_s =
                decode_ms > 0.0 ? static_cast<double>(generated) * 1000.0 / decode_ms : 0.0;
            std::fprintf(
                stderr,
                "[req %lld] prompt=%zu restored=%zu prefill=%zu gen=%zu "
                "prefill_ms=%.1f prefill_tok_s=%.1f ttft_ms=%.1f decode_ms=%.1f "
                "decode_tok_s=%.1f\n",
                static_cast<long long>(generation.external_request_id), prompt, restored,
                prefill, generated, request.prefill_ms, prefill_tok_s, ttft_ms, decode_ms,
                decode_tok_s);
        }
        push_event(events, std::move(done));
        push_terminal(generation.external_request_id);
    }

    void retire(std::vector<ServeEvent>& events, ServeGeneration& generation) {
        (void)events;
        auto retire_status = session.retire(generation.internal_request_id);
        if (!retire_status.ok()) {
            fprintf(stderr, "retire request %lld failed: %s\n",
                    static_cast<long long>(generation.external_request_id),
                    retire_status.message().c_str());
        }
        generation.draft_bitmask.reset();
        inflight.erase(generation.external_request_id);
    }

    void fail_all(std::vector<ServeEvent>& events, const std::string& message) {
        auto cleanup = session.cancel_all();
        if (!cleanup.ok()) {
            fprintf(stderr, "fail-safe cancel_all failed: %s\n", cleanup.message().c_str());
        }
        std::vector<int64_t> ids;
        ids.reserve(inflight.size());
        for (const auto& entry : inflight) ids.push_back(entry.first);
        for (int64_t id : ids) {
            ServeEvent error;
            error.kind = ServeEvent::Kind::Error;
            error.request_id = id;
            error.message = message;
            push_event(events, std::move(error));
            push_terminal(id);
        }
        for (int64_t id : ids) {
            auto it = inflight.find(id);
            if (it == inflight.end()) continue;
            auto retire_status = session.retire(it->second.internal_request_id);
            if (!retire_status.ok()) {
                fprintf(stderr, "retire after failure failed: %s\n",
                        retire_status.message().c_str());
            }
            it->second.draft_bitmask.reset();
            inflight.erase(it);
        }
    }

    void cancel_all_inflight(std::vector<ServeEvent>& events) {
        auto cancel_status = session.cancel_all();
        if (!cancel_status.ok()) {
            fprintf(stderr, "cancel_all failed: %s\n", cancel_status.message().c_str());
        }
        std::vector<int64_t> ids;
        ids.reserve(inflight.size());
        for (const auto& entry : inflight) ids.push_back(entry.first);
        for (int64_t id : ids) {
            auto it = inflight.find(id);
            if (it == inflight.end()) continue;
            emit_tokens(events, it->second);
            emit_done(events, it->second);
            retire(events, it->second);
        }
    }

    void step_batcher(std::vector<ServeEvent>& events) {
        if (inflight.empty()) return;
        auto step_result = session.step();
        if (!step_result.ok()) {
            fail_all(events, step_result.status().message());
            return;
        }
        if (!step_result.value().executed) {
            if (std::all_of(inflight.begin(), inflight.end(),
                            [](const auto& entry) { return entry.second.waiting_for_mask; })) {
                return;
            }
            std::string detail =
                "decode stuck: pending request remains but step did not execute";
            for (const auto& entry : inflight) {
                const RequestView request = session.request(entry.second.internal_request_id);
                if (!request.found) continue;
                if (request.state == runtime::RequestState::Finished) {
                    if (request.error_message != nullptr && request.error_message[0] != '\0') {
                        detail = request.error_message;
                        break;
                    }
                    detail = "request finished with reason " +
                             std::string(finish_reason_name(request.finish_reason));
                    break;
                }
            }
            fail_all(events, detail);
            return;
        }
        if (std::getenv("SHELF_BATCH_TRACE") != nullptr) {
            fprintf(stderr, "SHELF_BATCH_STEP requests=%u tokens=%u arena_used=%zu\n",
                    step_result.value().num_requests, step_result.value().num_tokens,
                    state.arena().used());
        }
        std::vector<int64_t> finished;
        for (auto& entry : inflight) {
            emit_tokens(events, entry.second);
            const RequestView request = session.request(entry.second.internal_request_id);
            if (request.found &&
                request.draft_mask_state == runtime::DraftMaskState::PendingMask &&
                request.draft_round_id > entry.second.emitted_draft_round) {
                ServeEvent draft;
                draft.kind = ServeEvent::Kind::Draft;
                draft.request_id = entry.first;
                draft.round_id = request.draft_round_id;
                draft.draft_tokens.assign(request.pending_drafts,
                                          request.pending_drafts + request.pending_drafts_size);
                push_event(events, std::move(draft));
                entry.second.emitted_draft_round = request.draft_round_id;
                entry.second.waiting_for_mask = true;
            }
            if (request.found && request.state == runtime::RequestState::Finished) {
                finished.push_back(entry.first);
            }
        }
        for (int64_t id : finished) {
            auto it = inflight.find(id);
            if (it == inflight.end()) continue;
            emit_done(events, it->second);
            retire(events, it->second);
        }
    }

    bool cancel_one(int64_t external_request_id, std::vector<ServeEvent>& events) {
        auto it = inflight.find(external_request_id);
        if (it == inflight.end()) return false;
        ServeGeneration& generation = it->second;
        const shelf::Status cancel_status = session.cancel(generation.internal_request_id);
        if (!cancel_status.ok()) {
            fail_all(events, cancel_status.message());
            return true;
        }
        emit_tokens(events, generation);
        const RequestView request = session.request(generation.internal_request_id);
        if (!request.found || request.state != runtime::RequestState::Finished) {
            fail_all(events, "cancel did not finish the request");
            return true;
        }
        emit_done(events, generation);
        retire(events, generation);
        return true;
    }

    void mtp_emit_done(std::vector<ServeEvent>& events, MtpRequest& request,
                       const char* reason) {
        const bool tail_eligible = std::strcmp(reason, "cancelled") != 0;
        mtp_finish_prefix(request, tail_eligible);
        log_request_summary(request.external_id, request.timing, request.prompt_tokens,
                            request.restored_tokens, request.generated, request.prefill_ms);
        ServeEvent done;
        done.kind = ServeEvent::Kind::Done;
        done.request_id = request.external_id;
        done.generated_ids = request.generated_ids;
        done.finish_reason = reason;
        done.prompt_tokens = request.prompt_tokens;
        done.restored_tokens = request.restored_tokens;
        done.prefill_tokens = request.prompt_tokens - request.restored_tokens;
        done.cache_checkpoint_tokens = 0;
        push_event(events, std::move(done));
        push_terminal(request.external_id);
    }

    void mtp_finish_prefix(MtpRequest& request, bool tail_eligible) {
        auto* cache = state.prefix_cache_.has_value() ? &state.prefix_cache_.value() : nullptr;
        if (cache == nullptr || request.prefix_capture == 0u) return;
        const runtime::CaptureId capture = request.prefix_capture;
        request.prefix_capture = 0u;
        const uint32_t position = request.tail_position;
        const bool usable = tail_eligible && position > 0u &&
                            !request.tail_snapshot.ring_k.empty() &&
                            position <= request.sequence->position;
        if (usable) {
            std::vector<uint32_t> tokens;
            tokens.reserve(position);
            for (std::size_t i = 0u;
                 i < request.input_tokens.size() && tokens.size() < position; ++i) {
                tokens.push_back(static_cast<uint32_t>(request.input_tokens[i]));
            }
            for (std::size_t i = 0u;
                 tokens.size() < position && i < request.generated_ids.size(); ++i) {
                tokens.push_back(static_cast<uint32_t>(request.generated_ids[i]));
            }
            if (tokens.size() == position) {
                const shelf::Status saved = cache->save_output_tail(
                    capture, *request.sequence, tokens, position, state.stream(), nullptr);
                if (saved.ok()) {
                    request.tail_snapshot.key =
                        shelf::runtime::mtp::mtp_prefix_key(tokens, position);
                    store_snapshot(std::move(request.tail_snapshot));
                    if (std::getenv("SHELF_PFXDBG") != nullptr) {
                        std::fprintf(stderr, "PFXDBG mtp save_output_tail pos=%u ok\n",
                                     position);
                    }
                    return;
                }
                if (std::getenv("SHELF_PFXDBG") != nullptr) {
                    std::fprintf(stderr, "PFXDBG mtp save_output_tail pos=%u: %s\n",
                                 position, saved.message().c_str());
                }
            }
        }
        (void)cache->cancel_capture(capture);
    }

    void mtp_release(MtpServeContext& context, std::size_t index) {
        MtpRequest& request = *context.active[index];
        if (request.prefix_capture != 0u) {
            if (state.prefix_cache_.has_value())
                (void)state.prefix_cache_->cancel_capture(request.prefix_capture);
            request.prefix_capture = 0u;
        }
        auto* tp = state.tp_coordinator_;
        if (tp != nullptr) {
            (void)tp->on_sequence_released(*request.sequence);
        }
        (void)shelf::state::release_paged_sequence_state(*request.sequence, state.stream());
        context.free_slots.push_back(request.drafter_slot);
        context.active[index] = std::move(context.active.back());
        context.active.pop_back();
    }

    shelf::Status mtp_save_boundary(void* opaque, uint32_t position, hipStream_t stream) {
        auto* context = static_cast<MtpSnapshotSaveContext*>(opaque);
        if (context == nullptr || context->controller == nullptr ||
            context->drafter == nullptr || context->target_sequence == nullptr ||
            context->input_ids == nullptr) {
            return shelf::Status::make_ok();
        }
        auto* cache = state.prefix_cache_.has_value() ? &state.prefix_cache_.value() : nullptr;
        if (cache == nullptr || position == 0u || context->drafter->ring_pages == 0u) {
            return shelf::Status::make_ok();
        }
        const uint32_t page_tokens = state.config_.page_tokens;
        if (page_tokens == 0u || position % page_tokens != 0u)
            return shelf::Status::make_ok();
        const uint32_t input_len = static_cast<uint32_t>(context->input_ids->size());
        if (position > input_len) return shelf::Status::make_ok();
        const bool first_boundary = context->last_save_position == 0u;
        const bool input_end = position == input_len;
        const bool tail = context->tail_position != 0u &&
                          position == context->tail_position;
        const bool periodic = context->save_interval != 0u &&
                              position >= context->last_save_position + context->save_interval;
        if (!first_boundary && !input_end && !tail && !periodic)
            return shelf::Status::make_ok();
        std::vector<uint32_t> tokens(position);
        for (uint32_t i = 0u; i < position; ++i)
            tokens[i] = static_cast<uint32_t>((*context->input_ids)[i]);
        const uint64_t saves_before = cache->stats().saves;
        const shelf::Status saved = cache->save_input_boundary(
            *context->target_sequence, tokens, position, stream, nullptr);
        if (!saved.ok()) {
            if (std::getenv("SHELF_PFXDBG") != nullptr) {
                std::fprintf(stderr, "PFXDBG mtp save %u: %s\n", position,
                             saved.message().c_str());
            }
            return shelf::Status::make_ok();
        }
        if (cache->stats().saves == saves_before) return shelf::Status::make_ok();
        context->last_save_position = position;
        auto snapshot = shelf::runtime::mtp::mtp_save_draft_snapshot(
            *context->drafter, position, stream);
        if (snapshot.ok()) {
            snapshot.value().key =
                shelf::runtime::mtp::mtp_prefix_key(tokens, position);
            context->controller->impl_->store_snapshot(std::move(snapshot.value()));
        }
        if (std::getenv("SHELF_PFXDBG") != nullptr) {
            std::fprintf(stderr, "PFXDBG mtp save %u ok\n", position);
        }
        return shelf::Status::make_ok();
    }

    static void mtp_capture_boundary(void* context, uint32_t position,
                                     shelf::runtime::mtp::MtpDraftSnapshot& snapshot) {
        auto* boundary = static_cast<MtpBoundaryContext*>(context);
        if (boundary == nullptr || boundary->controller == nullptr ||
            boundary->request == nullptr)
            return;
        auto& self = *boundary->controller->impl_;
        MtpRequest& request = *boundary->request;
        auto* cache = self.state.prefix_cache_.has_value()
                          ? &self.state.prefix_cache_.value()
                          : nullptr;
        if (cache == nullptr || request.prefix_capture == 0u) return;
        const shelf::Status st = cache->capture_boundary(
            request.prefix_capture, *request.sequence, self.state.stream());
        if (!st.ok()) return;
        request.tail_position = position;
        request.tail_snapshot = std::move(snapshot);
    }

    void mtp_step(std::vector<ServeEvent>& events) {
        if (mtp.active.empty()) return;
        std::vector<shelf::runtime::mtp::MtpSpecBatchSlot> slots;
        std::vector<std::size_t> indices;
        slots.reserve(mtp.active.size());
        indices.reserve(mtp.active.size());
        const bool capture_boundaries = state.prefix_cache_.has_value();
        for (std::size_t i = 0u; i < mtp.active.size(); ++i) {
            MtpRequest& request = *mtp.active[i];
            if (request.generated >= request.max_new_tokens) continue;
            shelf::runtime::mtp::MtpSpecBatchSlot slot;
            slot.drafter = mtp.batch.drafters[request.drafter_slot].get();
            slot.sequence = request.sequence.get();
            slot.pending_token = request.pending_token;
            slot.num_drafts = mtp.drafts;
            slot.remaining_tokens = request.max_new_tokens - request.generated;
            slot.page_tokens = capture_boundaries ? state.config_.page_tokens : 0u;
            slot.ignore_eos = request.ignore_eos;
            slots.push_back(slot);
            indices.push_back(i);
        }
        if (slots.empty()) {
            for (std::size_t i = mtp.active.size(); i-- > 0u;) {
                mtp_emit_done(events, *mtp.active[i], "length");
                mtp_release(mtp, i);
            }
            return;
        }
        std::vector<shelf::runtime::mtp::MtpBoundaryCapture> boundaries(slots.size());
        std::vector<MtpBoundaryContext> boundary_contexts(slots.size());
        for (std::size_t j = 0u; j < slots.size(); ++j) {
            MtpRequest& request = *mtp.active[indices[j]];
            if (request.prefix_capture == 0u) continue;
            boundary_contexts[j].controller = this_controller;
            boundary_contexts[j].request = &request;
            boundaries[j].context = &boundary_contexts[j];
            boundaries[j].capture = mtp_capture_boundary;
            slots[j].boundary_capture = &boundaries[j];
        }
        auto stepped = shelf::runtime::mtp::mtp_spec_step_batch(mtp.batch, slots);
        if (!stepped.ok()) {
            const std::string message = "mtp batch step failed: " + stepped.status().message();
            for (std::size_t i = 0u; i < mtp.active.size(); ++i) {
                ServeEvent error;
                error.kind = ServeEvent::Kind::Error;
                error.request_id = mtp.active[i]->external_id;
                error.message = message;
                push_event(events, std::move(error));
                push_terminal(mtp.active[i]->external_id);
            }
            for (std::size_t i = mtp.active.size(); i-- > 0u;) mtp_release(mtp, i);
            return;
        }
        const auto& result = stepped.value();
        mtp.rounds += 1u;
        mtp.round_ms += result.round_ms;
        mtp.verify_chunks += result.verify_chunks;
        mtp.draft_forwards += result.draft_forwards;
        if (result.draft_forwards > mtp.max_draft_forwards_per_round)
            mtp.max_draft_forwards_per_round = result.draft_forwards;
        if (result.verify_chunks > mtp.max_verify_chunks_per_round)
            mtp.max_verify_chunks_per_round = result.verify_chunks;
        if (static_cast<uint32_t>(slots.size()) > mtp.max_active_requests)
            mtp.max_active_requests = static_cast<uint32_t>(slots.size());
        std::vector<bool> finished(slots.size(), false);
        std::vector<const char*> reasons(slots.size(), "length");
        for (std::size_t j = 0u; j < slots.size(); ++j) {
            MtpRequest& request = *mtp.active[indices[j]];
            const auto& out = result.slots[j];
            for (uint32_t t = 0u; t < out.emitted_count; ++t) {
                push_token(events, request.external_id, out.emitted[t]);
                request.generated_ids.push_back(out.emitted[t]);
                request.generated += 1u;
            }
            request.pending_token = out.pending_token;
            mtp.accepted_drafts += out.num_accepted;
            mtp.emitted_tokens += out.emitted_count;
            log_decode_interval(request.external_id, request.timing, request.generated);
            const bool length_done = request.generated >= request.max_new_tokens;
            finished[j] = out.finished || length_done;
            reasons[j] = out.finished ? "stop" : "length";
        }
        for (std::size_t j = slots.size(); j-- > 0u;) {
            if (!finished[j]) continue;
            MtpRequest& request = *mtp.active[indices[j]];
            mtp_emit_done(events, request, reasons[j]);
            request.max_new_tokens = request.generated;
            mtp_release(mtp, indices[j]);
        }
        if (mtp.active.empty() && !exit_requested) {
            fprintf(stderr,
                    "MTP_BATCH rounds=%llu accepted=%llu emitted=%llu chunks=%llu "
                    "draft_forwards=%llu max_draft_per_round=%u max_verify_per_round=%u "
                    "max_active=%u round_ms=%.3f shared_draft=%u\n",
                    static_cast<unsigned long long>(mtp.rounds),
                    static_cast<unsigned long long>(mtp.accepted_drafts),
                    static_cast<unsigned long long>(mtp.emitted_tokens),
                    static_cast<unsigned long long>(mtp.verify_chunks),
                    static_cast<unsigned long long>(mtp.draft_forwards),
                    mtp.max_draft_forwards_per_round,
                    mtp.max_verify_chunks_per_round,
                    mtp.max_active_requests,
                    mtp.rounds > 0u ? mtp.round_ms / static_cast<double>(mtp.rounds) : 0.0,
                    static_cast<unsigned>(mtp.batch.shared_draft != nullptr));
        }
    }

    shelf::Result<ServeAdmitResult> mtp_admit(const ServeRequestSpec& request,
                                               const ServePrefillHooks& hooks,
                                               std::vector<ServeEvent>& events) {
        ServeAdmitResult outcome;
        const int64_t external_request_id = request.request_id;
        MtpRequestTiming timing;
        timing_submitted(timing);
        if (inflight.count(external_request_id) ||
            inflight.size() + mtp.active.size() >= max_inflight) {
            outcome.message = "duplicate request_id or compute concurrency limit reached";
            return outcome;
        }
        for (const auto& entry : mtp.active) {
            if (entry->external_id == external_request_id) {
                outcome.message = "duplicate or concurrent request_id";
                return outcome;
            }
        }
        if (is_recent_terminal(external_request_id)) {
            outcome.message = "duplicate or concurrent request_id";
            return outcome;
        }
        if (mtp.free_slots.empty()) {
            outcome.code = "busy";
            outcome.message = "mtp slots are exhausted";
            return outcome;
        }
        if (static_cast<uint64_t>(request.input_ids.size()) + request.max_new_tokens >
            static_cast<uint64_t>(state.config_.max_seq_len)) {
            outcome.message = "input_ids + max_new_tokens exceeds max_seq_len";
            return outcome;
        }

        auto sequence_result = shelf::state::create_paged_sequence_state(
            state.slot_pool(), state.gdn_pool(), state.kv_pool(), state.config_.max_seq_len,
            state.stream());
        if (!sequence_result.ok()) {
            outcome.code = "internal";
            outcome.message = "sequence create failed: " + sequence_result.status().message();
            return outcome;
        }
        auto sequence = std::make_unique<shelf::state::PagedSequenceState>(
            sequence_result.release());
        auto* tp = state.tp_coordinator_;
        if (tp != nullptr) {
            const shelf::Status mirror = tp->on_sequence_created(*sequence);
            if (!mirror.ok()) {
                outcome.code = "internal";
                outcome.message = "tp mirror create failed: " + mirror.message();
                (void)shelf::state::release_paged_sequence_state(*sequence, state.stream());
                return outcome;
            }
        }
        uint32_t restored_tokens = 0u;
        shelf::runtime::mtp::MtpDraftSnapshot restored_snapshot;
        bool has_snapshot = false;
        if (state.prefix_cache_.has_value()) {
            auto* cache = &state.prefix_cache_.value();
            std::vector<uint32_t> prefix(request.input_ids.size());
            for (std::size_t i = 0u; i < request.input_ids.size(); ++i)
                prefix[i] = static_cast<uint32_t>(request.input_ids[i]);
            const runtime::PrefixHit hit =
                cache->lookup(prefix, static_cast<uint32_t>(prefix.size()) - 1u);
            if (std::getenv("SHELF_PFXDBG") != nullptr) {
                std::fprintf(stderr,
                             "PFXDBG mtp lookup hit=%d state=%u in=%zu snapshots=%zu\n",
                             static_cast<int>(hit.hit()), hit.state_position, prefix.size(),
                             mtp.snapshots.size());
            }
            if (hit.hit() && hit.state_position < prefix.size()) {
                const uint64_t key =
                    shelf::runtime::mtp::mtp_prefix_key(prefix, hit.state_position);
                auto snapshot = mtp.snapshots.find(key);
                if (snapshot != mtp.snapshots.end()) {
                    const shelf::Status restored =
                        cache->restore(hit.entry_id, *sequence, state.stream());
                    if (restored.ok()) {
                        const std::span<const int32_t> restored_prefix(
                            request.input_ids.data(), hit.state_position);
                        if (tp != nullptr) {
                            auto mirror = tp->on_sequence_restored(*sequence);
                            if (mirror.ok())
                                mirror = tp->on_prefix_restored(*sequence, restored_prefix);
                            if (!mirror.ok()) {
                                outcome.code = "internal";
                                outcome.message = "tp prefix restore failed: " + mirror.message();
                                (void)tp->on_sequence_released(*sequence);
                                (void)shelf::state::release_paged_sequence_state(
                                    *sequence, state.stream());
                                return outcome;
                            }
                        } else if (state.executor().external_prefix_restore) {
                            const shelf::Status seed = state.executor().external_prefix_restore(
                                sequence->slot, restored_prefix);
                            if (!seed.ok()) {
                                outcome.code = "internal";
                                outcome.message = "prefix history restore failed: " +
                                                  seed.message();
                                (void)shelf::state::release_paged_sequence_state(
                                    *sequence, state.stream());
                                return outcome;
                            }
                        }
                        restored_snapshot = snapshot->second;
                        has_snapshot = true;
                        restored_tokens = hit.state_position;
                    } else {
                        outcome.code = "internal";
                        outcome.message = "prefix restore failed: " + restored.message();
                        if (tp != nullptr) (void)tp->on_sequence_released(*sequence);
                        (void)shelf::state::release_paged_sequence_state(*sequence,
                                                                         state.stream());
                        return outcome;
                    }
                }
            }
        }

        const uint32_t slot = mtp.free_slots.back();
        auto& drafter = *mtp.batch.drafters[slot];
        shelf::Status reset =
            shelf::runtime::mtp::mtp_drafter_reset_sequence(drafter, state.stream());
        if (!reset.ok()) {
            outcome.code = "internal";
            outcome.message = "mtp reset failed: " + reset.message();
            if (tp != nullptr) (void)tp->on_sequence_released(*sequence);
            (void)shelf::state::release_paged_sequence_state(*sequence, state.stream());
            return outcome;
        }
        if (has_snapshot) {
            const shelf::Status restored = shelf::runtime::mtp::mtp_restore_draft_snapshot(
                drafter, restored_snapshot, state.stream());
            if (!restored.ok()) {
                outcome.code = "internal";
                outcome.message = "mtp snapshot restore failed: " + restored.message();
                mtp.free_slots.push_back(slot);
                if (tp != nullptr) (void)tp->on_sequence_released(*sequence);
                (void)shelf::state::release_paged_sequence_state(*sequence, state.stream());
                return outcome;
            }
            if (std::getenv("SHELF_PFXDBG") != nullptr) {
                std::fprintf(stderr,
                             "PFXDBG mtp restored prefix=%u target=%u mtp=%u last=%u\n",
                             restored_tokens, sequence->position, drafter.sequence->position,
                             drafter.last_position);
            }
        }
        mtp.free_slots.pop_back();
        drafter.sampling = request.sampling;
        MtpPrefillHooksContext hooks_context;
        hooks_context.hooks = &hooks;
        hooks_context.log.external_id = external_request_id;
        shelf::runtime::mtp::MtpPrefillProgress prefill_progress;
        prefill_progress.context = &hooks_context;
        prefill_progress.report = mtp_prefill_report;
        prefill_progress.should_stop = mtp_prefill_should_stop;
        prefill_progress.stop_context = &hooks_context;
        const uint32_t prompt_len = static_cast<uint32_t>(request.input_ids.size());
        const uint32_t page_tokens = state.config_.page_tokens;
        const uint32_t tail_position =
            state.prefix_cache_.has_value() && page_tokens != 0u && prompt_len > page_tokens
                ? (prompt_len - 1u) / page_tokens * page_tokens
                : 0u;
        MtpSnapshotSaveContext snapshot_save;
        snapshot_save.controller = this_controller;
        snapshot_save.mtp = &mtp;
        snapshot_save.drafter = &drafter;
        snapshot_save.input_ids = &request.input_ids;
        snapshot_save.target_sequence = sequence.get();
        snapshot_save.save_interval = state.config_.prefix_cache_save_interval;
        snapshot_save.tail_position = tail_position;
        shelf::runtime::mtp::MtpPrefillBoundary boundary;
        boundary.context = &snapshot_save;
        boundary.save = [](void* opaque, uint32_t position, hipStream_t stream) {
            auto* context = static_cast<MtpSnapshotSaveContext*>(opaque);
            return context->controller->impl_->mtp_save_boundary(context, position, stream);
        };
        const auto prefill_started = std::chrono::steady_clock::now();
        auto prefill = shelf::runtime::mtp::mtp_spec_prefill(
            drafter, state.executor(), *sequence, request.input_ids.data(), prompt_len,
            request.sampling, &prefill_progress, &boundary, restored_tokens, tail_position);
        const double prefill_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - prefill_started)
                                      .count();
        const bool cancelled = hooks_context.cancelled;
        if (!prefill.ok()) {
            if (cancelled) {
                MtpRequest cancelled_request;
                cancelled_request.external_id = external_request_id;
                cancelled_request.prompt_tokens = request.input_ids.size();
                cancelled_request.timing = timing;
                cancelled_request.prefill_ms = prefill_ms;
                mtp_emit_done(events, cancelled_request, "cancelled");
                outcome.accepted = true;
                outcome.cancelled = true;
            } else {
                outcome.code = "internal";
                outcome.message = "mtp prefill failed: " + prefill.status().message();
            }
            mtp.free_slots.push_back(slot);
            if (tp != nullptr) (void)tp->on_sequence_released(*sequence);
            (void)shelf::state::release_paged_sequence_state(*sequence, state.stream());
            return outcome;
        }

        auto admitted = std::make_unique<MtpRequest>();
        admitted->external_id = external_request_id;
        admitted->sequence = std::move(sequence);
        admitted->drafter_slot = slot;
        admitted->pending_token = prefill.value().pending_token;
        admitted->max_new_tokens = request.max_new_tokens;
        admitted->restored_tokens = restored_tokens;
        admitted->prompt_tokens = request.input_ids.size();
        admitted->input_tokens = request.input_ids;
        admitted->generated = 1u;
        admitted->generated_ids.push_back(prefill.value().pending_token);
        admitted->timing = timing;
        admitted->prefill_ms = prefill_ms;
        admitted->ignore_eos = request.ignore_eos;
        timing_first_token(admitted->timing);
        push_token(events, external_request_id, prefill.value().pending_token);
        const bool stop = !admitted->ignore_eos && shelf::qwen_hybrid::is_stop_token(
            state.model().hybrid_config().stop_tokens, prefill.value().pending_token);
        if (stop || admitted->generated >= admitted->max_new_tokens) {
            mtp_emit_done(events, *admitted, stop ? "stop" : "length");
            if (tp != nullptr) (void)tp->on_sequence_released(*admitted->sequence);
            (void)shelf::state::release_paged_sequence_state(*admitted->sequence,
                                                             state.stream());
            mtp.free_slots.push_back(slot);
            outcome.accepted = true;
            return outcome;
        }
        if (state.prefix_cache_.has_value()) {
            auto capture = state.prefix_cache_->begin_capture(*admitted->sequence, state.stream());
            if (capture.ok()) admitted->prefix_capture = capture.value();
        }
        mtp.active.push_back(std::move(admitted));
        outcome.accepted = true;
        return outcome;
    }

    shelf::Result<ServeAdmitResult> admit(const ServeRequestSpec& request,
                                           const ServePrefillHooks& hooks,
                                           std::vector<ServeEvent>& events) {
        if (mtp.enabled && !request.grammar) {
            return mtp_admit(request, hooks, events);
        }
        ServeAdmitResult outcome;
        const int64_t external_request_id = request.request_id;
        if (inflight.count(external_request_id) > 0 ||
            is_recent_terminal(external_request_id)) {
            outcome.message = "duplicate request_id";
            return outcome;
        }
        for (const auto& active : mtp.active) {
            if (active->external_id == external_request_id) {
                outcome.message = "duplicate request_id";
                return outcome;
            }
        }
        if (inflight.size() + mtp.active.size() >= max_inflight) {
            outcome.message = "compute concurrency limit reached";
            return outcome;
        }
        if (static_cast<uint64_t>(request.input_ids.size()) + request.max_new_tokens >
            static_cast<uint64_t>(state.config_.max_seq_len)) {
            outcome.message = "input_ids + max_new_tokens exceeds max_seq_len";
            return outcome;
        }

        ServeGeneration generation;
        generation.external_request_id = external_request_id;
        generation.max_new_tokens = request.max_new_tokens;
        generation.submitted_at = std::chrono::steady_clock::now();
        generation.grammar_delayed = request.grammar_delayed;
        generation.grammar_close_token = request.grammar_close_token;
        if (request.has_token_bitmask) {
            const std::size_t bytes = request.token_bitmask.size() * sizeof(uint32_t);
            uint32_t* device = nullptr;
            if (hipMalloc(reinterpret_cast<void**>(&device), bytes) != hipSuccess) {
                outcome.code = "internal";
                outcome.message = "token mask allocation failed";
                return outcome;
            }
            generation.token_bitmask =
                std::shared_ptr<uint32_t>(device, [](uint32_t* p) { (void)hipFree(p); });
            if (hipMemcpy(generation.token_bitmask.get(), request.token_bitmask.data(), bytes,
                          hipMemcpyHostToDevice) != hipSuccess) {
                outcome.code = "internal";
                outcome.message = "token mask upload failed";
                return outcome;
            }
        }

        shelf::Result<uint64_t> submit_result = request.grammar
            ? (config.dflash_enabled
                   ? session.submit_structured_dflash(request.input_ids,
                                                      request.max_new_tokens,
                                                      request.sampling, request.thinking)
                   : session.submit_unconstrained_decode(request.input_ids,
                                                         request.max_new_tokens,
                                                         request.sampling, request.thinking))
            : session.submit(request.input_ids, request.max_new_tokens, request.sampling,
                             request.thinking);
        if (!submit_result.ok()) {
            outcome.code = status_error_code(submit_result.status().code());
            outcome.message = submit_result.status().message();
            return outcome;
        }
        generation.internal_request_id = submit_result.release();
        if (request.ignore_eos) {
            const shelf::Status ignore_eos_status =
                session.set_ignore_eos(generation.internal_request_id, true);
            if (!ignore_eos_status.ok()) {
                (void)session.cancel(generation.internal_request_id);
                (void)session.retire(generation.internal_request_id);
                outcome.code = "internal";
                outcome.message = ignore_eos_status.message();
                return outcome;
            }
        }
        if (request.grammar && config.dflash_enabled) {
            auto st = session.set_structured_dflash(generation.internal_request_id,
                                                    generation.grammar_delayed,
                                                    generation.grammar_close_token);
            if (!st.ok()) {
                (void)session.cancel(generation.internal_request_id);
                (void)session.retire(generation.internal_request_id);
                outcome.message = st.message();
                return outcome;
            }
        }
        if (request.grammar && !generation.grammar_delayed) {
            auto st = session.set_token_bitmask(generation.internal_request_id,
                                                generation.token_bitmask.get(), false);
            if (!st.ok()) {
                (void)session.cancel(generation.internal_request_id);
                (void)session.retire(generation.internal_request_id);
                outcome.message = st.message();
                return outcome;
            }
        }
        const shelf::Status external_status = session.set_external_request_id(
            generation.internal_request_id, external_request_id);
        if (!external_status.ok()) {
            (void)session.cancel(generation.internal_request_id);
            outcome.code = "internal";
            outcome.message = external_status.message();
            return outcome;
        }
        inflight.emplace(external_request_id, std::move(generation));
        outcome.accepted = true;
        return outcome;
    }

    static const char* finish_reason_name(runtime::FinishReason reason) {
        switch (reason) {
            case runtime::FinishReason::Eos: return "stop";
            case runtime::FinishReason::MaxNewTokens: return "length";
            case runtime::FinishReason::Cancelled: return "cancelled";
            case runtime::FinishReason::Error: return "error";
            case runtime::FinishReason::None: return "none";
        }
        return "none";
    }

    static const char* status_error_code(shelf::Status::Code code) {
        switch (code) {
            case shelf::Status::Code::invalid_argument:
            case shelf::Status::Code::out_of_range:
            case shelf::Status::Code::unsupported:
                return "invalid_argument";
            default:
                return "internal";
        }
    }

    ServeController* this_controller = nullptr;
};

ServeController::~ServeController() {
    if (impl_ == nullptr) return;
    Impl& self = *impl_;
    for (std::size_t i = self.mtp.active.size(); i-- > 0u;) {
        self.mtp_release(self.mtp, i);
    }
    if (self.mtp.batch.initialized) {
        const shelf::Status mtp_st =
            shelf::runtime::mtp::shutdown_mtp_spec_batch_context(self.mtp.batch);
        if (!mtp_st.ok()) {
            fprintf(stderr, "mtp batch shutdown failed: %s\n", mtp_st.message().c_str());
        }
    }
    if (self.dflash_sessions_ready) {
        const shelf::Status sessions_st = self.dflash_sessions.shutdown();
        self.dflash_sessions_ready = false;
        if (!sessions_st.ok()) {
            fprintf(stderr, "dflash2 session pool shutdown failed: %s\n",
                    sessions_st.message().c_str());
        }
    }
    if (self.dflash_batch_ready) {
        const shelf::Status batch_st =
            runtime::shutdown_dflash2_spec_batch_context(self.dflash_batch);
        self.dflash_batch_ready = false;
        if (!batch_st.ok()) {
            fprintf(stderr, "dflash2 batch context shutdown failed: %s\n",
                    batch_st.message().c_str());
        }
    }
    if (self.dflash_executor_ready) {
        const shelf::Status draft_st =
            runtime::dflash2::dflash2_executor_shutdown(self.dflash_executor);
        self.dflash_executor_ready = false;
        if (!draft_st.ok()) {
            fprintf(stderr, "dflash2 executor shutdown failed: %s\n",
                    draft_st.message().c_str());
        }
    }
}

Result<std::unique_ptr<ServeController>> ServeController::create(
    ComputeSession& session, const ServeConfig& config) {
    std::unique_ptr<ServeController> controller(new ServeController());
    controller->impl_.reset(new Impl(session));
    Impl& self = *controller->impl_;
    self.this_controller = controller.get();
    self.config = config;
    self.max_inflight = session.config().max_concurrent_requests;

    auto* tp = self.state.tp_coordinator_;
    std::vector<runtime::DFlash2GdnRankBinding> rank_bindings;
    if (tp != nullptr) {
        rank_bindings.reserve(static_cast<std::size_t>(tp->tp_size()));
        for (int r = 0; r < tp->tp_size(); ++r) {
            auto& rank = tp->rank(static_cast<std::size_t>(r));
            rank_bindings.push_back(runtime::DFlash2GdnRankBinding{
                &rank.gdn_pool(), &rank.arena(), rank.stream(), rank.device_id(),
                &rank.executor().host_model_state, &rank.executor()});
        }
    }
    const auto* rank_bindings_ptr = rank_bindings.empty() ? nullptr : &rank_bindings;

    const auto* tp_rank0_model =
        tp != nullptr && tp->tp_size() > 0 ? tp->rank(0).rank_model() : nullptr;
    const bool target_lm_head_sharded =
        self.state.config_.tp > 1 ||
        (tp_rank0_model != nullptr && tp_rank0_model->lm_head() != nullptr &&
         tp_rank0_model->lm_head()->partition.has_value());
    const shelf::weights::MatrixWeight* private_lm_head = nullptr;
    const shelf::weights::MatrixWeight* private_embed = nullptr;
    if (target_lm_head_sharded && tp != nullptr && tp->tp_size() > 0) {
        auto& rank0 = tp->rank(0);
        const auto* rank_model = rank0.rank_model();
        private_lm_head = rank_model != nullptr ? rank_model->replicated_lm_head() : nullptr;
        private_embed = rank_model != nullptr ? rank_model->replicated_embed_tokens() : nullptr;
        if (private_lm_head == nullptr) {
            private_lm_head = &rank0.private_lm_head();
            private_embed = &rank0.private_embed_tokens();
        }
    }

    if (config.dflash_enabled) {
        const uint32_t session_rows = runtime::dflash2_session_rows(
            config.dflash_config.block_size, config.dflash_drafts,
            self.state.config_.dflash2_tree_budget);
        if (self.state.config_.dflash2_sessions() >
            runtime::kDFlash2SpecMaxVerifyRows / session_rows) {
            return Status::unsupported(
                "dflash2 serve: sessions exceed verify row cap", __FILE__, __LINE__);
        }
        const std::size_t weights_before = self.state.arena().used();
        auto weights_result = runtime::dflash2::load_dflash2_weights(
            self.state.config_.dflash2_model_dir, config.dflash_config,
            self.state.arena(), self.state.stream(), {});
        fprintf(stderr, "DFLASH2_WEIGHT_ARENA_BYTES=%llu\n",
                static_cast<unsigned long long>(self.state.arena().used() - weights_before));
        if (!weights_result.ok()) return weights_result.status();
        self.dflash_weights = weights_result.release();

        const auto& text_config = self.state.model().hybrid_config();
        const uint32_t batch_max_rows = std::min<uint32_t>(
            self.state.config_.dflash2_sessions() * session_rows,
            runtime::kDFlash2SpecMaxVerifyRows);
        std::vector<const shelf::weights::MatrixWeight*> head_shards;
        uint32_t head_vocab_offset = 0;
        uint32_t head_local_vocab = 0;
        shelf::kernel::EmbeddingShardTable embed_config_shards{};
        if (target_lm_head_sharded) {
            bool sharded = tp != nullptr;
            for (int r = 0; sharded && r < tp->tp_size(); ++r) {
                const auto* head =
                    tp->rank(static_cast<std::size_t>(r)).rank_model()->lm_head();
                if (head == nullptr || !head->partition.has_value()) {
                    sharded = false;
                    break;
                }
                head_shards.push_back(head);
            }
            if (sharded) {
                std::vector<uint32_t> offsets;
                for (const auto* head : head_shards) {
                    offsets.push_back(static_cast<uint32_t>(head->partition->index) *
                                      static_cast<uint32_t>(head->rows));
                }
                runtime::dflash2::DFlash2TpProposalConfig proposal_config;
                proposal_config.coordinator = tp;
                proposal_config.heads = head_shards;
                proposal_config.vocab_offsets = offsets;
                proposal_config.local_vocab = static_cast<uint32_t>(head_shards[0]->rows);
                proposal_config.top_k = static_cast<uint32_t>(config.dflash_config.selector_top_k);
                proposal_config.hidden_stride =
                    static_cast<uint32_t>(config.dflash_config.hidden_size);
                proposal_config.max_rows = std::max<uint32_t>(
                    static_cast<uint32_t>(config.dflash_config.block_size), batch_max_rows);
                auto proposal_result =
                    runtime::dflash2::DFlash2TpProposal::create(proposal_config);
                if (!proposal_result.ok()) return proposal_result.status();
                self.dflash_tp_proposal =
                    std::make_unique<runtime::dflash2::DFlash2TpProposal>(
                        proposal_result.release());
                head_vocab_offset = offsets[0];
                head_local_vocab = proposal_config.local_vocab;
                shelf::kernel::EmbeddingShardTable embed_shards{};
                bool embed_sharded = tp->tp_size() <=
                                     static_cast<int>(shelf::kernel::kEmbeddingMaxShards);
                for (int r = 0; embed_sharded && r < tp->tp_size(); ++r) {
                    const auto* embed = tp->rank(static_cast<std::size_t>(r))
                                            .rank_model()
                                            ->embed_tokens();
                    if (embed == nullptr || !embed->partition.has_value() ||
                        !embed->host_resident) {
                        embed_sharded = false;
                        break;
                    }
                    embed_shards.refs[static_cast<std::size_t>(r)] =
                        shelf::kernel::EmbeddingShardRef{
                            static_cast<const shelf::bf16_t*>(embed->host_payload),
                            static_cast<uint32_t>(embed->partition->index) * embed->rows,
                            embed->rows};
                }
                if (embed_sharded && tp->tp_size() > 1) {
                    embed_shards.count = static_cast<uint32_t>(tp->tp_size());
                    embed_config_shards = embed_shards;
                }
            } else {
                return Status::unsupported(
                    "dflash2 tp: target lm_head is not split across ranks, "
                    "a full private copy is required",
                    __FILE__, __LINE__);
            }
        }
        runtime::dflash2::DFlash2ExecutorConfig draft_config;
        draft_config.coarse_head_bytes = self.state.config_.dflash2_coarse_bytes;
        draft_config.init_stream = self.state.stream();
        draft_config.max_rows = static_cast<uint32_t>(config.dflash_config.block_size);
        draft_config.max_batch_rows = batch_max_rows;
        draft_config.max_context_rows =
            static_cast<uint32_t>(config.dflash_config.sliding_window);
        draft_config.max_prefill_feature_rows = self.state.config_.dflash2_prefill_rows;
        draft_config.proposal_distributor = self.dflash_tp_proposal.get();
        draft_config.vocab_offset = head_vocab_offset;
        draft_config.local_vocab = head_local_vocab;
        draft_config.target_embed_shards = embed_config_shards;
        const shelf::weights::MatrixWeight* draft_embed =
            &self.state.model().embed_tokens();
        if (target_lm_head_sharded) {
            draft_embed = self.dflash_tp_proposal != nullptr
                ? tp->rank(0).rank_model()->embed_tokens()
                : private_embed;
        }
        draft_config.target_lm_head =
            self.dflash_tp_proposal != nullptr
                ? head_shards[0]
                : (target_lm_head_sharded ? private_lm_head
                                          : &self.state.model().lm_head());
        draft_config.target_embed_tokens = draft_embed;
        auto draft_result = runtime::dflash2::create_dflash2_executor(
            config.dflash_config, self.dflash_weights, self.state.arena(), draft_config);
        if (!draft_result.ok()) return draft_result.status();
        self.dflash_executor = draft_result.release();
        self.dflash_executor_ready = true;
        {
            std::vector<runtime::DFlash2GdnRankBinding> bindings;
            if (rank_bindings_ptr != nullptr) {
                bindings = rank_bindings;
            } else {
                runtime::DFlash2GdnRankBinding binding;
                binding.pool = &self.state.gdn_pool();
                binding.arena = &self.state.arena();
                binding.stream = self.state.stream();
                binding.device = self.state.config_.devices[0];
                binding.model_state = &self.state.executor().host_model_state;
                binding.executor = &self.state.executor();
                bindings.push_back(binding);
            }
            auto batch_result = runtime::create_dflash2_spec_batch_context(
                bindings, batch_max_rows, self.state.config_.dflash2_tree_budget);
            if (!batch_result.ok()) return batch_result.status();
            self.dflash_batch = batch_result.release();
            self.dflash_batch.tree_budget_auto = self.state.config_.dflash2_tree_budget_auto;
            self.dflash_batch_ready = true;
        }
        {
            runtime::DFlash2SpecDecoderConfig session_config;
            session_config.num_drafts = config.dflash_drafts;
            session_config.tree_budget = self.state.config_.dflash2_tree_budget;
            session_config.eos_tokens = text_config.stop_tokens;
            auto sessions_result = runtime::DFlash2SessionPool::create(
                self.state.executor(), &self.dflash_executor, self.state.gdn_pool(),
                self.state.arena(), session_config, self.state.config_.dflash2_sessions(),
                self.state.stream(), rank_bindings_ptr);
            if (!sessions_result.ok()) return sessions_result.status();
            self.dflash_sessions = sessions_result.release();
            self.dflash_sessions_ready = true;
        }
        runtime::DFlash2BatcherBinding batcher_binding;
        batcher_binding.config = &self.config.dflash_config;
        batcher_binding.executor = &self.dflash_executor;
        batcher_binding.batch = &self.dflash_batch;
        batcher_binding.arena = &self.state.arena();
        batcher_binding.sessions = &self.dflash_sessions;
        batcher_binding.num_drafts = config.dflash_drafts;
        batcher_binding.tree_budget = self.state.config_.dflash2_tree_budget;
        batcher_binding.adaptive = config.dflash_adaptive;
        batcher_binding.host_token_ids = self.state.config_.dflash2_host_token_ids;
        batcher_binding.stats = &self.dflash_stats;
        batcher_binding.rank_bindings = rank_bindings_ptr;
        const shelf::Status bind_st = self.state.batcher_->set_dflash2(batcher_binding);
        if (!bind_st.ok()) return bind_st;
        fprintf(stderr, "DFLASH2_ENABLED=1\n");
        fprintf(stderr, "DFLASH2_DRAFTS=%u\n", config.dflash_drafts);
        fprintf(stderr, "DFLASH2_BLOCK_SIZE=%zu\n", config.dflash_config.block_size);
        fprintf(stderr, "DFLASH2_SESSION_POOL_SLOTS=%u\n", self.dflash_sessions.capacity());
        fprintf(stderr, "DFLASH2_SESSION_POOL_BYTES=%llu\n",
                static_cast<unsigned long long>(self.dflash_sessions.reserved_bytes()));
    }

    if (self.state.config_.mtp) {
        self.mtp.enabled = true;
        self.mtp.drafts =
            self.state.config_.mtp_drafts == 0u ? 4u : self.state.config_.mtp_drafts;
        shelf::runtime::mtp::MtpDrafterConfig mtp_config;
        mtp_config.num_drafts = self.mtp.drafts;
        mtp_config.window_tokens = static_cast<uint32_t>(self.state.config_.mtp_window);
        mtp_config.sink_tokens = static_cast<uint32_t>(self.state.config_.mtp_sink);
        mtp_config.max_seq_len = self.state.config_.max_seq_len;
        mtp_config.page_tokens = self.state.config_.page_tokens;
        mtp_config.max_batch_rows = std::min(
            self.state.executor().config.max_batch_rows,
            shelf::runtime::mtp::kMtpMaxPrefillRows);
        mtp_config.kv_dtype =
            self.state.config_.kv_cache_dtype == shelf::kernel::KVCacheDType::OCTAVE
                ? shelf::kernel::KVCacheDType::BF16 : self.state.config_.kv_cache_dtype;
        if (self.state.config_.rope_profile.has_value()) {
            mtp_config.rope_profile = *self.state.config_.rope_profile;
        }
        const shelf::weights::MatrixWeight* lm_head_override = nullptr;
        if (private_lm_head != nullptr && private_lm_head->rows != 0u) {
            lm_head_override = private_lm_head;
        }
        std::vector<shelf::runtime::mtp::MtpDrafterInput> rank_inputs;
        shelf::runtime::mtp::MtpDrafterInput leader_input;
        if (tp != nullptr && tp->tp_size() > 0 && tp->rank(0).rank_model() != nullptr &&
            tp->rank(0).rank_model()->mtp_uses_per_rank_input()) {
            const int rank_count = tp->tp_size();
            rank_inputs.reserve(static_cast<std::size_t>(rank_count));
            for (int r = 0; r < rank_count; ++r) {
                auto& rank = tp->rank(static_cast<std::size_t>(r));
                auto per_rank = mtp_config;
                per_rank.local_kv_heads = rank.context().local_key_value_heads;
                auto input = rank.rank_model()->mtp_drafter_input(per_rank);
                if (!input.ok()) return input.status();
                rank_inputs.push_back(std::move(input.value()));
            }
            mtp_config.local_kv_heads = tp->rank(0).context().local_key_value_heads;
        } else {
            auto input = self.state.model().mtp_drafter_input(mtp_config, lm_head_override);
            if (!input.ok()) return input.status();
            leader_input = std::move(input.value());
        }
        const shelf::runtime::mtp::MtpDrafterInput& drafter_input =
            rank_inputs.empty() ? leader_input : rank_inputs.front();
        std::vector<shelf::runtime::mtp::MtpGdnRankBinding> extra_ranks;
        if (tp != nullptr) {
            for (int r = 1; r < tp->tp_size(); ++r) {
                auto& rank = tp->rank(static_cast<std::size_t>(r));
                shelf::runtime::mtp::MtpGdnRankBinding binding;
                binding.pool = &rank.gdn_pool();
                binding.arena = &rank.arena();
                binding.stream = rank.stream();
                binding.device = rank.device_id();
                binding.executor = &rank.executor();
                binding.local_kv_heads = rank.context().local_key_value_heads;
                binding.tp_rank = static_cast<uint32_t>(r);
                if (r < static_cast<int>(rank_inputs.size()))
                    binding.mtp_input = &rank_inputs[static_cast<std::size_t>(r)];
                extra_ranks.push_back(binding);
            }
        }
        const uint32_t verify_rows = self.mtp.drafts + 1u;
        uint32_t max_slots = self.state.config_.max_concurrent_requests;
        if (max_slots == 0u) max_slots = 1u;
        if (mtp_config.max_batch_rows / verify_rows < max_slots) {
            max_slots = mtp_config.max_batch_rows / verify_rows;
            if (max_slots == 0u) max_slots = 1u;
        }
        auto batch = shelf::runtime::mtp::create_mtp_spec_batch_context(
            self.state.executor(), drafter_input, self.state.gdn_pool(),
            self.state.arena(), self.state.stream(), mtp_config, max_slots,
            extra_ranks.empty() ? nullptr : &extra_ranks, tp);
        if (!batch.ok()) return batch.status();
        self.mtp.batch = batch.release();
        for (uint32_t i = 0u; i < max_slots; ++i) {
            self.mtp.free_slots.push_back(max_slots - 1u - i);
        }
        fprintf(stderr, "MTP_SERVE=batch\n");
        fprintf(stderr, "MTP_SLOTS=%u\n", max_slots);
        fprintf(stderr, "MTP_DRAFTS=%u\n", self.mtp.drafts);
    }
    return controller;
}

bool ServeController::mtp_enabled() const noexcept { return impl_->mtp.enabled; }
bool ServeController::dflash_enabled() const noexcept { return impl_->config.dflash_enabled; }
bool ServeController::has_inflight() const noexcept { return !impl_->inflight.empty(); }
bool ServeController::has_mtp_active() const noexcept { return !impl_->mtp.active.empty(); }
bool ServeController::has_request(int64_t request_id) const noexcept {
    if (impl_->inflight.count(request_id) != 0) return true;
    for (const auto& active : impl_->mtp.active) {
        if (active->external_id == request_id) return true;
    }
    return false;
}
bool ServeController::busy() const noexcept {
    return !impl_->inflight.empty() || !impl_->mtp.active.empty();
}
bool ServeController::all_waiting_for_mask() const noexcept {
    return std::all_of(impl_->inflight.begin(), impl_->inflight.end(),
                       [](const auto& entry) { return entry.second.waiting_for_mask; });
}
bool ServeController::is_recent_terminal(int64_t request_id) const noexcept {
    return impl_->is_recent_terminal(request_id);
}
bool ServeController::exit_requested() const noexcept { return impl_->exit_requested; }
uint32_t ServeController::max_inflight() const noexcept { return impl_->max_inflight; }
uint32_t ServeController::vocab_size() const noexcept {
    return static_cast<uint32_t>(impl_->state.model().hybrid_config().vocab_size);
}

Result<ServeAdmitResult> ServeController::admit(const ServeRequestSpec& request,
                                                const ServePrefillHooks& hooks,
                                                std::vector<ServeEvent>& events) {
    return impl_->admit(request, hooks, events);
}

void ServeController::step(std::vector<ServeEvent>& events) {
    if (impl_->mtp.enabled && !impl_->mtp.active.empty()) impl_->mtp_step(events);
    if (!impl_->inflight.empty()) impl_->step_batcher(events);
}

bool ServeController::cancel(int64_t external_request_id,
                             std::vector<ServeEvent>& events) {
    if (impl_->inflight.count(external_request_id) > 0) {
        return impl_->cancel_one(external_request_id, events);
    }
    if (impl_->mtp.enabled) {
        for (std::size_t i = 0u; i < impl_->mtp.active.size(); ++i) {
            if (impl_->mtp.active[i]->external_id != external_request_id) continue;
            impl_->mtp_emit_done(events, *impl_->mtp.active[i], "cancelled");
            impl_->mtp_release(impl_->mtp, i);
            return true;
        }
    }
    return false;
}

void ServeController::cancel_all_inflight(std::vector<ServeEvent>& events) {
    impl_->cancel_all_inflight(events);
}

void ServeController::abandon_inflight(int64_t external_request_id,
                                       std::vector<ServeEvent>& events) {
    auto it = impl_->inflight.find(external_request_id);
    if (it == impl_->inflight.end()) return;
    (void)impl_->session.cancel(it->second.internal_request_id);
    impl_->push_terminal(external_request_id);
    impl_->retire(events, it->second);
}

void ServeController::request_shutdown(std::vector<ServeEvent>& events) {
    impl_->cancel_all_inflight(events);
    impl_->exit_requested = true;
}

Status ServeController::apply_token_mask(int64_t external_request_id,
                                         const std::vector<uint32_t>& words) {
    auto it = impl_->inflight.find(external_request_id);
    if (it == impl_->inflight.end()) {
        return Status::invalid_argument("unknown request_id", __FILE__, __LINE__);
    }
    if (!it->second.waiting_for_mask || words.empty()) {
        return Status::invalid_argument("unexpected or invalid bitmask", __FILE__, __LINE__);
    }
    const std::size_t bytes = words.size() * sizeof(uint32_t);
    if (it->second.token_bitmask == nullptr) {
        uint32_t* device = nullptr;
        if (hipMalloc(reinterpret_cast<void**>(&device), bytes) != hipSuccess) {
            return Status::hip_error("hipMalloc", "token mask allocation failed", __FILE__,
                                     __LINE__);
        }
        it->second.token_bitmask =
            std::shared_ptr<uint32_t>(device, [](uint32_t* p) { (void)hipFree(p); });
    }
    if (hipMemcpy(it->second.token_bitmask.get(), words.data(), bytes,
                  hipMemcpyHostToDevice) != hipSuccess) {
        return Status::hip_error("hipMemcpy", "token mask upload failed", __FILE__, __LINE__);
    }
    auto st = impl_->session.set_token_bitmask(it->second.internal_request_id,
                                               it->second.token_bitmask.get(), false);
    if (!st.ok()) return st;
    it->second.waiting_for_mask = false;
    return Status::make_ok();
}

Status ServeController::draft_mask_request(int64_t external_request_id, uint64_t round,
                                           bool& pending, std::size_t& draft_count,
                                           std::size_t& words_per_row) const {
    pending = false;
    draft_count = 0;
    words_per_row = 0;
    auto it = impl_->inflight.find(external_request_id);
    if (it == impl_->inflight.end()) return Status::make_ok();
    const RequestView active = impl_->session.request(it->second.internal_request_id);
    if (!active.found || active.state == runtime::RequestState::Finished ||
        active.draft_mask_state != runtime::DraftMaskState::PendingMask ||
        round != active.draft_round_id) {
        return Status::make_ok();
    }
    pending = true;
    draft_count = active.pending_drafts_size;
    words_per_row = (impl_->session.model_info().vocab_size + 31u) / 32u;
    return Status::make_ok();
}

Status ServeController::apply_draft_masks(int64_t external_request_id, uint64_t round,
                                          const uint32_t* masks, std::size_t words_per_row,
                                          uint32_t valid_prefix) {
    auto it = impl_->inflight.find(external_request_id);
    if (it == impl_->inflight.end()) {
        return Status::invalid_argument("unexpected or invalid bitmask", __FILE__,
                                       __LINE__);
    }
    const RequestView active = impl_->session.request(it->second.internal_request_id);
    if (!active.found || active.draft_mask_state != runtime::DraftMaskState::PendingMask ||
        round != active.draft_round_id) {
        return Status::make_ok();
    }
    const std::size_t rows = active.pending_drafts_size + 1u;
    const std::size_t bytes = rows * words_per_row * sizeof(uint32_t);
    if (it->second.draft_bitmask_capacity < bytes) {
        uint32_t* device = nullptr;
        if (hipMalloc(reinterpret_cast<void**>(&device), bytes) != hipSuccess) {
            return Status::hip_error("hipMalloc", "draft mask allocation failed", __FILE__,
                                     __LINE__);
        }
        it->second.draft_bitmask = std::shared_ptr<uint32_t>(device, [](uint32_t* p) {
            const hipError_t err = hipFree(p);
            if (err != hipSuccess)
                std::fprintf(stderr, "draft mask cleanup failed: %s\n",
                             hipGetErrorString(err));
        });
        it->second.draft_bitmask_capacity = bytes;
    }
    if (hipMemcpy(it->second.draft_bitmask.get(), masks, bytes, hipMemcpyHostToDevice) !=
        hipSuccess) {
        return Status::hip_error("hipMemcpy", "draft mask upload failed", __FILE__,
                                 __LINE__);
    }
    auto st = impl_->session.set_draft_masks(
        it->second.internal_request_id, round, it->second.draft_bitmask.get(),
        static_cast<uint32_t>(words_per_row), valid_prefix);
    if (!st.ok()) return st;
    it->second.waiting_for_mask = false;
    return Status::make_ok();
}

ServeStats ServeController::stats() const {
    ServeStats out;
    out.prefix_enabled = impl_->session.prefix_cache_active();
    out.page_tokens = impl_->state.config_.page_tokens;
    if (impl_->state.prefix_cache_.has_value()) {
        const auto& cache = impl_->state.prefix_cache_.value();
        const runtime::PrefixCacheStats prefix_stats = cache.stats();
        out.reuse_slots = cache.reuse_slot_count();
        out.entries = prefix_stats.entries;
        out.saves = prefix_stats.saves;
        out.hits = prefix_stats.hits;
        out.misses = prefix_stats.misses;
        out.restores = prefix_stats.restores;
        out.evictions = prefix_stats.entries_evicted;
    }
    const runtime::KVCapacitySnapshot snapshot = impl_->session.kv_capacity_snapshot();
    out.total_pages = snapshot.total_pages;
    out.used_pages = snapshot.physical_used_pages;
    out.free_pages = snapshot.physical_free_pages;
    out.active_requests = snapshot.active_requests;
    out.queued_requests = impl_->session.queued_request_count();
    out.banker_safe = snapshot.banker_safe;
    out.overcommit_pages = snapshot.overcommit_pages;
    return out;
}

void ServeController::write_final_stats() {
    const char* serve_stats_env = std::getenv("SHELF_DFLASH2_SERVE_STATS");
    if (!impl_->config.dflash_enabled ||
        !(impl_->config.dflash_adaptive ||
          (serve_stats_env != nullptr && serve_stats_env[0] != '\0' &&
           serve_stats_env[0] != '0'))) {
        return;
    }
    const auto& t = impl_->dflash_stats;
    const float rounds = static_cast<float>(t.rounds);
    const double lane_rounds = static_cast<double>(t.full_accepts) + t.partial_accepts;
    const double mean = lane_rounds > 0.0 ? t.accepted_drafts / lane_rounds : 0.0;
    const double full_rate = lane_rounds > 0.0 ? t.full_accepts / lane_rounds : 0.0;
    const double acceptance = t.proposed_drafts ? static_cast<double>(t.accepted_drafts) / t.proposed_drafts : 0.0;
    fprintf(stderr, "DFLASH2_PROPOSED_DRAFTS=%llu\n", static_cast<unsigned long long>(t.proposed_drafts));
    fprintf(stderr, "DFLASH2_ACCEPTANCE_RATE=%.6f\n", acceptance);
    const float emitted =
        t.rounds > 0u ? static_cast<float>(t.generated_tokens) / rounds : 0.0f;
    if (impl_->config.dflash_adaptive && t.adaptive_rounds != 0u) {
        fprintf(stderr, "DFLASH2_ADAPTIVE rounds=%llu actual_tps=%.3f\n",
                static_cast<unsigned long long>(t.adaptive_rounds),
                t.adaptive_actual_ms > 0.0
                    ? 1000.0 * t.adaptive_actual_tokens / t.adaptive_actual_ms : 0.0);
        for (std::size_t k = 0; k < t.adaptive_k_counts.size(); ++k) {
            fprintf(stderr, "DFLASH2_ADAPTIVE_K k=%zu count=%llu\n", k,
                    static_cast<unsigned long long>(t.adaptive_k_counts[k]));
        }
    }
    fprintf(stderr, "DFLASH2_GDN_MODE=compact\n");
    fprintf(stderr, "DFLASH2_GDN_HISTORY_BYTES=%llu\n",
            static_cast<unsigned long long>(t.gdn_history_bytes));
    fprintf(stderr, "DFLASH2_GDN_SNAPSHOT_BYTES=%llu\n",
            static_cast<unsigned long long>(t.gdn_snapshot_bytes));
    fprintf(stderr, "DFLASH2_GDN_COMPACT_BYTES=%llu\n",
            static_cast<unsigned long long>(t.gdn_compact_bytes));
    fprintf(stderr, "DFLASH2_ROUNDS=%u\n", t.rounds);
    fprintf(stderr, "DFLASH2_ACCEPTED_DRAFTS=%u\n", t.accepted_drafts);
    fprintf(stderr, "DFLASH2_MEAN_ACCEPTED=%.3f\n", mean);
    fprintf(stderr, "DFLASH2_FULL_ACCEPT_RATE=%.3f\n", full_rate);
    fprintf(stderr, "DFLASH2_EMITTED_PER_ROUND=%.3f\n", emitted);
    fprintf(stderr, "DFLASH2_GENERATED_TOKENS=%u\n", t.generated_tokens);
    fprintf(stderr, "DFLASH2_ROUND_MS=%.3f\n", t.round_ms);
    if (t.rounds > 0u) {
        const double count = static_cast<double>(t.rounds);
        fprintf(stderr,
                "DFLASH2_STAGE round=%.3f draft=%.3f draft_gpu=%.3f draft_d2h=%.3f "
                "proposal_wait=%.3f proposal_copy=%.3f gdn_snapshot=%.3f verify=%.3f "
                "verify_gpu=%.3f decision_wait=%.3f decision_copy=%.3f "
                "gdn_restore=%.3f rerun=%.3f commit=%.3f\n",
                t.round_ms / count, t.draft_ms / count, t.draft_gpu_ms / count,
                t.draft_d2h_ms / count, t.proposal_wait_ms / count,
                t.proposal_copy_ms / count, t.gdn_snapshot_ms / count,
                t.verify_ms / count, t.verify_gpu_ms / count, t.decision_wait_ms / count,
                t.decision_copy_ms / count, t.gdn_restore_ms / count, t.rerun_ms / count,
                t.commit_ms / count);
        fprintf(stderr, "DFLASH2_VERIFY_ROWS_PER_ROUND=%.3f\n",
                static_cast<double>(t.verify_rows) / count);
    }
    for (uint32_t b = 0; b < runtime::kDFlash2PositionBuckets; ++b) {
        const runtime::DFlash2PositionStats& p = t.position[b];
        const uint32_t lo = b * runtime::kDFlash2PositionBucketTokens;
        const uint32_t hi = (b + 1u) * runtime::kDFlash2PositionBucketTokens;
        const float bucket_rounds = static_cast<float>(p.rounds);
        const float bucket_mean =
            p.rounds > 0u ? static_cast<float>(p.accepted_drafts) / bucket_rounds : 0.0f;
        const float bucket_emitted =
            p.rounds > 0u ? static_cast<float>(p.generated_tokens) / bucket_rounds : 0.0f;
        const double bucket_round_ms =
            p.rounds > 0u ? p.round_ms / static_cast<double>(p.rounds) : 0.0;
        fprintf(stderr,
                "DFLASH2_POSITION=%u-%u ROUNDS=%u ACCEPTED=%u MEAN_ACCEPTED=%.3f "
                "EMITTED_PER_ROUND=%.3f ROUND_MS=%.3f\n",
                lo, hi, p.rounds, p.accepted_drafts, bucket_mean, bucket_emitted,
                bucket_round_ms);
    }
    fflush(stderr);
}

}  // namespace shelf::engine
