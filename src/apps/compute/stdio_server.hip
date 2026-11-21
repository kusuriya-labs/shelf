#include "stdio_server.h"
#include "private_target_weights.h"

#include "compute_common.h"

#include <shelf/runtime/dflash2/context_state.h>
#include <shelf/runtime/decode_backend.h>
#include <shelf/models/qwen35/runtime/mtp_drafter.h>
#include <shelf/runtime/runtime_request.h>
#include <shelf/runtime/state_types.h>
#include <shelf/core/memory/size_format.h>
#include <nlohmann/json.hpp>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <poll.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace shelf::app {
namespace {

void emit_json(const nlohmann::json& value) {
    const std::string line = value.dump();
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void emit_error(const nlohmann::json& request_id, const std::string& message) {
    nlohmann::json ev;
    ev["event"] = "error";
    if (!request_id.is_null()) ev["request_id"] = request_id;
    ev["message"] = message;
    emit_json(ev);
}

void emit_error_with_code(
    const nlohmann::json& request_id, const std::string& code,
    const std::string& message) {
    nlohmann::json ev;
    ev["event"] = "error";
    if (!request_id.is_null()) ev["request_id"] = request_id;
    if (!code.empty()) ev["code"] = code;
    ev["message"] = message;
    emit_json(ev);
}

const char* status_error_code(shelf::Status::Code code) {
    switch (code) {
        case shelf::Status::Code::invalid_argument:
        case shelf::Status::Code::out_of_range:
        case shelf::Status::Code::unsupported:
            return "invalid_argument";
        default:
            return "internal";
    }
}

void emit_token(int64_t request_id, int32_t token_id, bool need_mask = true) {
    nlohmann::json event = {{"event", "token"}, {"request_id", request_id}, {"token_id", token_id}};
    if (!need_mask) event["need_mask"] = false;
    emit_json(event);
}

constexpr std::size_t kRecentTerminalLimit = 256;

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
    size_t draft_bitmask_capacity = 0;
    uint64_t emitted_draft_round = 0;
    bool waiting_for_mask = false;
    bool grammar_delayed = false;
    int32_t grammar_close_token = -1;
};

// Strict RFC 4648 decoding; reject malformed or non-canonical input.
bool decode_mask(const std::string& encoded, std::vector<uint8_t>& bytes) {
    if (encoded.empty() || encoded.size() % 4 != 0) return false;
    auto digit = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    for (size_t i = 0; i < encoded.size(); i += 4) {
        int a = digit(encoded[i]), b = digit(encoded[i + 1]);
        int c = encoded[i + 2] == '=' ? -2 : digit(encoded[i + 2]);
        int d = encoded[i + 3] == '=' ? -2 : digit(encoded[i + 3]);
        if (a < 0 || b < 0 || c == -1 || d == -1 ||
            (c == -2 && d != -2) ||
            ((c == -2 || d == -2) && i + 4 != encoded.size()) ||
            (c == -2 && (b & 15)) || (d == -2 && c >= 0 && (c & 3))) return false;
        bytes.push_back(static_cast<uint8_t>((a << 2) | (b >> 4)));
        if (c >= 0) bytes.push_back(static_cast<uint8_t>((b << 4) | (c >> 2)));
        if (d >= 0) bytes.push_back(static_cast<uint8_t>((c << 6) | d));
    }
    return true;
}

bool upload_mask(shelf::app::Qwen35ComputeRuntime& runtime,
                 ServeGeneration& generation, const nlohmann::json& value) {
    if (!value.is_string()) return false;
    std::vector<uint8_t> bytes;
    const size_t expected = ((runtime.model().hybrid_config().vocab_size + 31) / 32) * 4;
    if (!decode_mask(value.get<std::string>(), bytes) || bytes.size() != expected ||
        std::none_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b != 0; })) return false;
    if (!generation.token_bitmask) {
        uint32_t* device = nullptr;
        if (hipMalloc(reinterpret_cast<void**>(&device), expected) != hipSuccess) return false;
        generation.token_bitmask = std::shared_ptr<uint32_t>(device, [](uint32_t* p) { (void)hipFree(p); });
    }
    return hipMemcpy(generation.token_bitmask.get(), bytes.data(), expected,
                     hipMemcpyHostToDevice) == hipSuccess;
}

struct MtpPrefillLog {
    int64_t external_id = 0;
    uint32_t last_tokens = 0;
    double last_ms = 0.0;
};

void serve_mtp_prefill_progress(void* context, uint32_t processed_tokens,
                                double elapsed_ms) {
    auto* log = static_cast<MtpPrefillLog*>(context);
    const double interval_s = (elapsed_ms - log->last_ms) / 1000.0;
    if (interval_s < 3.0 || processed_tokens <= log->last_tokens) return;
    const uint32_t delta = processed_tokens - log->last_tokens;
    std::fprintf(stderr,
                 "[req %lld] prefill_tok_s=%.1f tokens=%u interval_s=%.2f\n",
                 static_cast<long long>(log->external_id),
                 static_cast<double>(delta) / interval_s, processed_tokens,
                 interval_s);
    log->last_ms = elapsed_ms;
    log->last_tokens = processed_tokens;
}

struct MtpRequestTiming {
    std::chrono::steady_clock::time_point submitted_at;
    std::chrono::steady_clock::time_point first_token_at;
    std::chrono::steady_clock::time_point last_log_at;
    std::size_t last_log_tokens = 0;
    bool has_first_token = false;
};

void mtp_timing_submitted(MtpRequestTiming& timing) {
    timing.submitted_at = std::chrono::steady_clock::now();
}

void mtp_timing_first_token(MtpRequestTiming& timing) {
    timing.first_token_at = std::chrono::steady_clock::now();
    timing.last_log_at = timing.first_token_at;
    timing.last_log_tokens = 0;
    timing.has_first_token = true;
}

void mtp_log_decode_interval(int64_t external_id, MtpRequestTiming& timing,
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

void mtp_log_request_summary(int64_t external_id, const MtpRequestTiming& timing,
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
    MtpRequestTiming timing;
    double prefill_ms = 0.0;
};

struct MtpServeContext {
    bool enabled = false;
    uint32_t drafts = 4;
    shelf::qwen35::runtime::MtpSpecBatchContext batch;
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
    std::unordered_map<uint64_t, shelf::qwen35::runtime::MtpDraftSnapshot> snapshots;
};

struct ServeState {
    std::map<int64_t, ServeGeneration> inflight;
    std::deque<int64_t> recent_terminal_ids;
    uint32_t max_inflight = 1;
    bool exit_requested = false;
    MtpServeContext mtp;
};

struct JsonlInput {
    std::string buffer;
    bool eof = false;
};

void jsonl_poll_and_read(JsonlInput& input, bool block) {
    if (input.eof) return;
    struct pollfd descriptor;
    descriptor.fd = STDIN_FILENO;
    descriptor.events = POLLIN;
    descriptor.revents = 0;
    const int timeout_ms = block ? -1 : 0;
    const int ready = ::poll(&descriptor, 1, timeout_ms);
    if (ready <= 0) return;
    if ((descriptor.revents & (POLLIN | POLLHUP | POLLERR)) == 0) return;
    char chunk[4096];
    const ssize_t count = ::read(STDIN_FILENO, chunk, sizeof(chunk));
    if (count > 0) {
        input.buffer.append(chunk, static_cast<std::size_t>(count));
    } else if (count == 0) {
        input.eof = true;
    } else if (errno != EAGAIN && errno != EINTR) {
        input.eof = true;
    }
}

bool jsonl_take_line(JsonlInput& input, std::string& line) {
    const std::size_t newline = input.buffer.find('\n');
    if (newline == std::string::npos) return false;
    line = input.buffer.substr(0, newline);
    input.buffer.erase(0, newline + 1);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

// MTP prefill runs the whole prompt before the serve loop can read stdin. Poll it between
// chunks and consume a matching cancel line so the prompt does not have to finish first.
struct MtpPrefillCancel {
    JsonlInput* input = nullptr;
    int64_t external_id = 0;
    bool cancelled = false;
};

bool serve_mtp_prefill_should_stop(void* context) {
    auto* cancel = static_cast<MtpPrefillCancel*>(context);
    if (cancel->input == nullptr) return false;
    jsonl_poll_and_read(*cancel->input, false);
    const std::string& buffer = cancel->input->buffer;
    std::size_t line_begin = 0u;
    while (true) {
        const std::size_t newline = buffer.find('\n', line_begin);
        if (newline == std::string::npos) break;
        std::string line = buffer.substr(line_begin, newline - line_begin);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        bool matches = false;
        if (!line.empty() && line.front() == '{') {
            try {
                const nlohmann::json op = nlohmann::json::parse(line);
                matches = op.value("op", std::string{}) == "cancel" &&
                          op.value("request_id", int64_t{0}) == cancel->external_id;
            } catch (const nlohmann::json::exception&) {
            }
        }
        if (matches) {
            cancel->input->buffer.erase(line_begin, newline + 1u - line_begin);
            cancel->cancelled = true;
            return true;
        }
        line_begin = newline + 1u;
    }
    return false;
}

void serve_emit_tokens(shelf::app::Qwen35ComputeRuntime& runtime,
                       ServeGeneration& generation) {
    const auto* request = runtime.request(generation.internal_request_id);
    if (request == nullptr) return;
    while (generation.emitted_tokens < request->generated.size()) {
        if (!generation.has_first_token) {
            generation.first_token_at = std::chrono::steady_clock::now();
            generation.last_log_at = generation.first_token_at;
            generation.has_first_token = true;
        }
        const int32_t token = request->generated[generation.emitted_tokens];
        const bool structured_spec = generation.token_bitmask && request->dflash_used &&
                                     request->structured_dflash;
        emit_token(generation.external_request_id, token, !structured_spec);
        ++generation.emitted_tokens;
        if (generation.grammar_delayed && token == generation.grammar_close_token) {
            generation.grammar_delayed = false;
            if (structured_spec) (void)runtime.activate_grammar(generation.internal_request_id);
        }
        if (!structured_spec && generation.token_bitmask && !generation.grammar_delayed &&
            request->state != shelf::qwen35::runtime::RequestState::Finished) {
            generation.waiting_for_mask = true;
            (void)runtime.set_token_bitmask(generation.internal_request_id,
                                            generation.token_bitmask.get(), true);
        }
    }
    if (!generation.has_first_token) return;
    const auto now = std::chrono::steady_clock::now();
    const double interval_s = std::chrono::duration<double>(
                                  now - generation.last_log_at)
                                  .count();
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

void serve_push_terminal(ServeState& state, int64_t external_request_id) {
    state.recent_terminal_ids.push_back(external_request_id);
    while (state.recent_terminal_ids.size() > kRecentTerminalLimit) {
        state.recent_terminal_ids.pop_front();
    }
}

bool serve_is_recent_terminal(const ServeState& state, int64_t external_request_id) {
    return std::find(state.recent_terminal_ids.begin(),
                     state.recent_terminal_ids.end(),
                     external_request_id) != state.recent_terminal_ids.end();
}

void serve_emit_done(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                     const ServeGeneration& generation) {
    const auto* request = runtime.request(generation.internal_request_id);
    nlohmann::json done;
    done["event"] = "done";
    done["request_id"] = generation.external_request_id;
    if (request != nullptr) {
        done["generated_ids"] = request->generated;
        done["finish_reason"] = finish_reason_name(request->finish_reason);
        done["prompt_tokens"] = request->input_tokens.size();
        done["restored_tokens"] = request->restored_tokens;
        done["prefill_tokens"] =
            request->input_tokens.size() - request->restored_tokens;
        done["cache_checkpoint_tokens"] = 0;
        if (!request->error_message.empty()) {
            done["error_message"] = request->error_message;
        }
    } else {
        done["generated_ids"] = std::vector<int32_t>{};
        done["finish_reason"] = "error";
    }
    if (request != nullptr) {
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
        const std::size_t prompt = request->input_tokens.size();
        const std::size_t restored =
            std::min<std::size_t>(request->restored_tokens, prompt);
        const std::size_t prefill = prompt - restored;
        const std::size_t generated = request->generated.size();
        const double prefill_tok_s =
            request->prefill_ms > 0.0
                ? static_cast<double>(prefill) * 1000.0 / request->prefill_ms
                : 0.0;
        const double decode_tok_s =
            decode_ms > 0.0 ? static_cast<double>(generated) * 1000.0 / decode_ms
                            : 0.0;
        std::fprintf(
            stderr,
            "[req %lld] prompt=%zu restored=%zu prefill=%zu gen=%zu "
            "prefill_ms=%.1f prefill_tok_s=%.1f ttft_ms=%.1f decode_ms=%.1f "
            "decode_tok_s=%.1f\n",
            static_cast<long long>(generation.external_request_id), prompt, restored,
            prefill, generated, request->prefill_ms, prefill_tok_s, ttft_ms,
            decode_ms, decode_tok_s);
    }
    emit_json(done);
    serve_push_terminal(state, generation.external_request_id);
}

void serve_retire(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                  ServeGeneration& generation) {
    auto retire_status = runtime.retire(generation.internal_request_id);
    if (!retire_status.ok()) {
        fprintf(stderr, "retire request %lld failed: %s\n",
                static_cast<long long>(generation.external_request_id),
                retire_status.message().c_str());
    }
    generation.draft_bitmask.reset();
    state.inflight.erase(generation.external_request_id);
}

void serve_fail_all(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                    const std::string& message) {
    auto cleanup = runtime.cancel_all();
    if (!cleanup.ok()) {
        fprintf(stderr, "fail-safe cancel_all failed: %s\n", cleanup.message().c_str());
    }
    std::vector<int64_t> ids;
    ids.reserve(state.inflight.size());
    for (const auto& entry : state.inflight) {
        ids.push_back(entry.first);
    }
    for (int64_t id : ids) {
        emit_error(nlohmann::json(id), message);
        serve_push_terminal(state, id);
    }
    for (int64_t id : ids) {
        auto it = state.inflight.find(id);
        if (it == state.inflight.end()) continue;
        ServeGeneration& generation = it->second;
        auto retire_status = runtime.retire(generation.internal_request_id);
        if (!retire_status.ok()) {
            fprintf(stderr, "retire after failure failed: %s\n",
                    retire_status.message().c_str());
        }
        generation.draft_bitmask.reset();
        state.inflight.erase(it);
    }
}

void serve_cancel_all_inflight(shelf::app::Qwen35ComputeRuntime& runtime,
                               ServeState& state) {
    auto cancel_status = runtime.cancel_all();
    if (!cancel_status.ok()) {
        fprintf(stderr, "cancel_all failed: %s\n", cancel_status.message().c_str());
    }
    std::vector<int64_t> ids;
    ids.reserve(state.inflight.size());
    for (const auto& entry : state.inflight) {
        ids.push_back(entry.first);
    }
    for (int64_t id : ids) {
        auto it = state.inflight.find(id);
        if (it == state.inflight.end()) continue;
        ServeGeneration& generation = it->second;
        serve_emit_tokens(runtime, generation);
        serve_emit_done(runtime, state, generation);
        serve_retire(runtime, state, generation);
    }
}

void serve_step(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state) {
    if (state.inflight.empty()) return;
    auto step_result = runtime.step();
    if (!step_result.ok()) {
        serve_fail_all(runtime, state, step_result.status().message());
        return;
    }
    if (!step_result.value().executed) {
        if (std::all_of(state.inflight.begin(), state.inflight.end(),
                        [](const auto& entry) { return entry.second.waiting_for_mask; })) return;
        std::string detail = "decode stuck: pending request remains but step did not execute";
        for (const auto& entry : state.inflight) {
            const auto* request = runtime.request(entry.second.internal_request_id);
            if (request == nullptr) continue;
            if (request->state == shelf::qwen35::runtime::RequestState::Finished) {
                if (!request->error_message.empty()) {
                    detail = request->error_message;
                    break;
                }
                detail = "request finished with reason " +
                         std::string(finish_reason_name(request->finish_reason));
                break;
            }
        }
        serve_fail_all(runtime, state, detail);
        return;
    }
    if (std::getenv("SHELF_BATCH_TRACE") != nullptr) {
        fprintf(stderr, "SHELF_BATCH_STEP requests=%u tokens=%u arena_used=%zu\n",
                step_result.value().num_requests, step_result.value().num_tokens,
                runtime.arena().used());
    }
    std::vector<int64_t> finished;
    for (auto& entry : state.inflight) {
        serve_emit_tokens(runtime, entry.second);
        const auto* request = runtime.request(entry.second.internal_request_id);
        if (request != nullptr && request->draft_mask_state == shelf::qwen35::runtime::DraftMaskState::PendingMask &&
            request->draft_round_id > entry.second.emitted_draft_round) {
            emit_json({{"event", "draft"}, {"request_id", entry.first},
                       {"round_id", request->draft_round_id}, {"draft_tokens", request->pending_drafts}});
            entry.second.emitted_draft_round = request->draft_round_id;
            entry.second.waiting_for_mask = true;
        }
        if (request != nullptr &&
            request->state == shelf::qwen35::runtime::RequestState::Finished) {
            finished.push_back(entry.first);
        }
    }
    for (int64_t id : finished) {
        auto it = state.inflight.find(id);
        if (it == state.inflight.end()) continue;
        serve_emit_done(runtime, state, it->second);
        serve_retire(runtime, state, it->second);
    }
}

bool serve_cancel_one(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                      int64_t external_request_id) {
    auto it = state.inflight.find(external_request_id);
    if (it == state.inflight.end()) return false;
    ServeGeneration& generation = it->second;
    const shelf::Status cancel_status = runtime.cancel(generation.internal_request_id);
    if (!cancel_status.ok()) {
        serve_fail_all(runtime, state, cancel_status.message());
        return true;
    }
    serve_emit_tokens(runtime, generation);
    const auto* request = runtime.request(generation.internal_request_id);
    if (request == nullptr ||
        request->state != shelf::qwen35::runtime::RequestState::Finished) {
        serve_fail_all(runtime, state, "cancel did not finish the request");
        return true;
    }
    serve_emit_done(runtime, state, generation);
    serve_retire(runtime, state, generation);
    return true;
}

bool serve_parse_generate(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                          DFlashServeContext& dflash,
                          const nlohmann::json& request) {
    if (!request.contains("request_id") || !request["request_id"].is_number_integer()) {
        emit_error(nullptr, "missing or invalid request_id");
        return false;
    }
    const nlohmann::json request_id = request["request_id"];
    const int64_t external_request_id = request_id.get<int64_t>();
    if (state.inflight.count(external_request_id) > 0 ||
        serve_is_recent_terminal(state, external_request_id)) {
        emit_error(request_id, "duplicate request_id");
        return false;
    }
    for (const auto& active : state.mtp.active) {
        if (active->external_id == external_request_id) {
            emit_error(request_id, "duplicate request_id");
            return false;
        }
    }
    if (state.inflight.size() + state.mtp.active.size() >= state.max_inflight) {
        emit_error(request_id, "compute concurrency limit reached");
        return false;
    }

    for (const auto& entry : request.items()) {
        const std::string& key = entry.key();
        if (key != "op" && key != "request_id" && key != "input_ids" &&
            key != "max_new_tokens" && key != "temperature" && key != "top_p" &&
            key != "min_p" && key != "top_k" && key != "seed" &&
            key != "reasoning_budget_tokens" && key != "reasoning_close_token" &&
            key != "prefix_cache_checkpoint_position" && key != "grammar" &&
            key != "bitmask" && key != "grammar_after_reasoning") {
            emit_error_with_code(request_id, "invalid_argument",
                                 "unknown request field: " + key);
            return false;
        }
    }

    if (!request.contains("input_ids") || !request["input_ids"].is_array() ||
        request["input_ids"].empty()) {
        emit_error(request_id, "input_ids must be a non-empty array");
        return false;
    }
    std::vector<int32_t> input_ids;
    input_ids.reserve(request["input_ids"].size());
    bool ids_ok = true;
    for (const auto& value : request["input_ids"]) {
        if (!value.is_number_integer()) { ids_ok = false; break; }
        input_ids.push_back(value.get<int32_t>());
    }
    if (!ids_ok) {
        emit_error(request_id, "input_ids must contain integers");
        return false;
    }

    if (!request.contains("max_new_tokens") || !request["max_new_tokens"].is_number_integer() ||
        request["max_new_tokens"].get<long long>() <= 0) {
        emit_error(request_id, "max_new_tokens must be a positive integer");
        return false;
    }
    const uint32_t max_new_tokens =
        static_cast<uint32_t>(request["max_new_tokens"].get<long long>());

    if (request.contains("prefix_cache_checkpoint_position")) {
        if (!request["prefix_cache_checkpoint_position"].is_number_integer()) {
            emit_error_with_code(
                request_id, "invalid_argument",
                "prefix_cache_checkpoint_position must be an integer");
            return false;
        }
        const long long checkpoint_value =
            request["prefix_cache_checkpoint_position"].get<long long>();
        if (checkpoint_value != 0) {
            emit_error_with_code(
                request_id, "invalid_argument",
                "prefix_cache_checkpoint_position must be 0; prefix reuse is "
                "resolved by the compute-side prefix cache");
            return false;
        }
    }

    if (static_cast<uint64_t>(input_ids.size()) + max_new_tokens >
        static_cast<uint64_t>(runtime.config().max_seq_len)) {
        emit_error(request_id, "input_ids + max_new_tokens exceeds max_seq_len");
        return false;
    }

    shelf::qwen35::runtime::SamplingConfig sampling;
    if (request.contains("temperature")) {
        if (!request["temperature"].is_number()) {
            emit_error(request_id, "temperature must be a number");
            return false;
        }
        sampling.temperature = request["temperature"].get<float>();
    }
    if (request.contains("top_p")) {
        if (!request["top_p"].is_number()) {
            emit_error(request_id, "top_p must be a number");
            return false;
        }
        sampling.top_p = request["top_p"].get<float>();
    }
    if (request.contains("min_p")) {
        if (!request["min_p"].is_number()) {
            emit_error(request_id, "min_p must be a number");
            return false;
        }
        sampling.min_p = request["min_p"].get<float>();
    }
    if (request.contains("top_k")) {
        if (!request["top_k"].is_number_integer()) {
            emit_error(request_id, "top_k must be an integer");
            return false;
        }
        sampling.top_k = static_cast<uint32_t>(request["top_k"].get<long long>());
    }
    if (request.contains("seed")) {
        if (!request["seed"].is_number_integer()) {
            emit_error(request_id, "seed must be an integer");
            return false;
        }
        sampling.seed = static_cast<uint64_t>(request["seed"].get<long long>());
    }
    const shelf::Status sampling_status = shelf::qwen35::runtime::validate_sampling_config(sampling);
    if (!sampling_status.ok()) {
        emit_error(request_id, sampling_status.message());
        return false;
    }

    shelf::qwen35::runtime::ThinkingBudget thinking;
    if (request.contains("reasoning_budget_tokens")) {
        if (!request["reasoning_budget_tokens"].is_number_integer() ||
            request["reasoning_budget_tokens"].get<long long>() < 0) {
            emit_error_with_code(request_id, "invalid_argument",
                                 "reasoning_budget_tokens must be a non-negative integer");
            return false;
        }
        thinking.tokens =
            static_cast<uint32_t>(request["reasoning_budget_tokens"].get<long long>());
    }
    if (request.contains("reasoning_close_token")) {
        if (!request["reasoning_close_token"].is_number_integer()) {
            emit_error_with_code(request_id, "invalid_argument",
                                 "reasoning_close_token must be an integer");
            return false;
        }
        thinking.close_token =
            static_cast<int32_t>(request["reasoning_close_token"].get<long long>());
    }
    if (thinking.tokens > 0 && thinking.close_token < 0) {
        emit_error_with_code(request_id, "invalid_argument",
                             "reasoning_close_token is required with "
                             "reasoning_budget_tokens");
        return false;
    }
    if (thinking.tokens > max_new_tokens) {
        emit_error_with_code(request_id, "invalid_argument",
                             "reasoning_budget_tokens exceeds max_new_tokens");
        return false;
    }

    ServeGeneration generation;
    generation.external_request_id = external_request_id;
    generation.emitted_tokens = 0;
    generation.max_new_tokens = max_new_tokens;
    generation.submitted_at = std::chrono::steady_clock::now();

    if (request.contains("grammar") && !request["grammar"].is_boolean()) {
        emit_error_with_code(request_id, "invalid_argument", "grammar must be a boolean");
        return false;
    }
    const bool grammar = request.value("grammar", false);
    if (request.contains("grammar_after_reasoning") && !request["grammar_after_reasoning"].is_boolean()) {
        emit_error_with_code(request_id, "invalid_argument", "grammar_after_reasoning must be a boolean");
        return false;
    }
    generation.grammar_delayed = request.value("grammar_after_reasoning", false);
    if (generation.grammar_delayed && (!grammar || thinking.close_token < 0)) {
        emit_error_with_code(request_id, "invalid_argument", "delayed grammar requires a close token");
        return false;
    }
    if (grammar && thinking.tokens > 0 && !generation.grammar_delayed) {
        emit_error_with_code(request_id, "invalid_argument", "grammar with reasoning requires delayed grammar");
        return false;
    }
    generation.grammar_close_token = thinking.close_token;
    if ((grammar && (!request.contains("bitmask") ||
                     !upload_mask(runtime, generation, request["bitmask"]))) ||
        (!grammar && request.contains("bitmask"))) {
        emit_error_with_code(request_id, "invalid_argument", "invalid initial bitmask");
        return false;
    }

    shelf::Result<uint64_t> submit_result = grammar
        ? (dflash.enabled ? runtime.submit_structured_dflash(std::move(input_ids), max_new_tokens, sampling, thinking)
                          : runtime.submit_unconstrained_decode(std::move(input_ids), max_new_tokens, sampling, thinking))
        : runtime.submit(std::move(input_ids), max_new_tokens, sampling, thinking);
    if (!submit_result.ok()) {
        emit_error_with_code(request_id,
                             status_error_code(submit_result.status().code()),
                             submit_result.status().message());
        return false;
    }
    generation.internal_request_id = submit_result.release();
    if (grammar && dflash.enabled) {
        auto st = runtime.set_structured_dflash(generation.internal_request_id,
                                                generation.grammar_delayed, generation.grammar_close_token);
        if (!st.ok()) {
            (void)runtime.cancel(generation.internal_request_id);
            (void)runtime.retire(generation.internal_request_id);
            emit_error(request_id, st.message());
            return false;
        }
    }
    if (grammar && !generation.grammar_delayed) {
        auto st = runtime.set_token_bitmask(generation.internal_request_id,
                                            generation.token_bitmask.get(), false);
        if (!st.ok()) {
            (void)runtime.cancel(generation.internal_request_id);
            (void)runtime.retire(generation.internal_request_id);
            emit_error(request_id, st.message());
            return false;
        }
    }
    const shelf::Status external_status = runtime.set_external_request_id(
        generation.internal_request_id, external_request_id);
    if (!external_status.ok()) {
        (void)runtime.cancel(generation.internal_request_id);
        emit_error_with_code(request_id, "internal", external_status.message());
        return false;
    }
    state.inflight.emplace(external_request_id, std::move(generation));
    return true;
}

void serve_mtp_emit_done(ServeState& state, MtpRequest& request, const char* reason) {
    mtp_log_request_summary(request.external_id, request.timing, request.prompt_tokens,
                            request.restored_tokens, request.generated,
                            request.prefill_ms);

    nlohmann::json done;
    done["event"] = "done";
    done["request_id"] = request.external_id;
    done["generated_ids"] = request.generated_ids;
    done["finish_reason"] = reason;
    done["prompt_tokens"] = request.prompt_tokens;
    done["restored_tokens"] = request.restored_tokens;
    done["prefill_tokens"] = request.prompt_tokens - request.restored_tokens;
    done["cache_checkpoint_tokens"] = 0;
    emit_json(done);
    serve_push_terminal(state, request.external_id);
}

void serve_mtp_release(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                       MtpServeContext& mtp, std::size_t index) {
    MtpRequest& request = *mtp.active[index];
    auto* tp = runtime.tp_coordinator();
    if (tp != nullptr) {
        (void)tp->on_sequence_released(*request.sequence);
    }
    (void)shelf::state::release_paged_sequence_state(*request.sequence, runtime.stream());
    mtp.free_slots.push_back(request.drafter_slot);
    mtp.active[index] = std::move(mtp.active.back());
    mtp.active.pop_back();
}

constexpr std::size_t kMtpSnapshotLimit = 16u;

void mtp_store_snapshot(MtpServeContext& mtp,
                        shelf::qwen35::runtime::MtpDraftSnapshot&& snapshot) {
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

struct MtpSnapshotSaveContext {
    shelf::app::Qwen35ComputeRuntime* runtime = nullptr;
    MtpServeContext* mtp = nullptr;
    shelf::qwen35::runtime::MtpDrafter* drafter = nullptr;
    const std::vector<int32_t>* input_ids = nullptr;
    shelf::state::PagedSequenceState* target_sequence = nullptr;
    uint32_t save_interval = 0u;
    bool saved_first = false;
};

shelf::Status serve_mtp_save_boundary(void* opaque, uint32_t position,
                                      hipStream_t stream) {
    auto* context = static_cast<MtpSnapshotSaveContext*>(opaque);
    if (context == nullptr || context->runtime == nullptr ||
        context->drafter == nullptr || context->target_sequence == nullptr ||
        context->input_ids == nullptr) {
        return shelf::Status::make_ok();
    }
    auto* cache = context->runtime->prefix_cache();
    if (cache == nullptr || position == 0u ||
        context->drafter->ring_pages == 0u) {
        return shelf::Status::make_ok();
    }
    const uint32_t page_tokens = context->runtime->config().page_tokens;
    if (page_tokens == 0u || position % page_tokens != 0u)
        return shelf::Status::make_ok();
    if (!context->saved_first && context->save_interval != 0u &&
        position % context->save_interval != 0u) {
        return shelf::Status::make_ok();
    }
    if (position > context->input_ids->size())
        return shelf::Status::make_ok();
    context->saved_first = true;
    std::vector<uint32_t> tokens(position);
    for (uint32_t i = 0u; i < position; ++i)
        tokens[i] = static_cast<uint32_t>((*context->input_ids)[i]);
    const shelf::Status saved = cache->save_input_boundary(
        *context->target_sequence, tokens, position, stream, nullptr);
    if (!saved.ok()) {
        if (std::getenv("SHELF_PFXDBG") != nullptr) {
            std::fprintf(stderr, "PFXDBG mtp save %u: %s\n", position,
                         saved.message().c_str());
        }
        return shelf::Status::make_ok();
    }
    auto snapshot = shelf::qwen35::runtime::mtp_save_draft_snapshot(
        *context->drafter, position, stream);
    if (snapshot.ok()) {
        snapshot.value().key =
            shelf::qwen35::runtime::mtp_prefix_key(tokens, position);
        mtp_store_snapshot(*context->mtp, std::move(snapshot.value()));
    }
    if (std::getenv("SHELF_PFXDBG") != nullptr) {
        std::fprintf(stderr, "PFXDBG mtp save %u ok\n", position);
    }
    return shelf::Status::make_ok();
}

bool serve_mtp_admit(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                     MtpServeContext& mtp, JsonlInput& input,
                     const nlohmann::json& request) {
    if (!request.contains("request_id") || !request["request_id"].is_number_integer()) {
        emit_error(nullptr, "missing or invalid request_id");
        return false;
    }
    const nlohmann::json request_id = request["request_id"];
    const int64_t external_request_id = request_id.get<int64_t>();
    MtpRequestTiming timing;
    mtp_timing_submitted(timing);
    if (state.inflight.count(external_request_id) ||
        state.inflight.size() + mtp.active.size() >= state.max_inflight) {
        emit_error(request_id, "duplicate request_id or compute concurrency limit reached");
        return false;
    }
    for (const auto& entry : mtp.active) {
        if (entry->external_id == external_request_id) {
            emit_error(request_id, "duplicate or concurrent request_id");
            return false;
        }
    }
    if (serve_is_recent_terminal(state, external_request_id)) {
        emit_error(request_id, "duplicate or concurrent request_id");
        return false;
    }
    for (const auto& entry : request.items()) {
        const std::string& key = entry.key();
        if (key != "op" && key != "request_id" && key != "input_ids" &&
            key != "max_new_tokens" && key != "temperature" && key != "top_p" &&
            key != "min_p" && key != "top_k" && key != "seed" &&
            key != "reasoning_budget_tokens" && key != "reasoning_close_token" &&
            key != "prefix_cache_checkpoint_position") {
            emit_error_with_code(request_id, "invalid_argument",
                                 "unknown request field: " + key);
            return false;
        }
    }
    if (!request.contains("input_ids") || !request["input_ids"].is_array() ||
        request["input_ids"].empty()) {
        emit_error(request_id, "input_ids must be a non-empty array");
        return false;
    }
    std::vector<int32_t> input_ids;
    input_ids.reserve(request["input_ids"].size());
    for (const auto& value : request["input_ids"]) {
        if (!value.is_number_integer()) {
            emit_error(request_id, "input_ids must contain integers");
            return false;
        }
        input_ids.push_back(value.get<int32_t>());
    }
    if (!request.contains("max_new_tokens") ||
        !request["max_new_tokens"].is_number_integer() ||
        request["max_new_tokens"].get<long long>() <= 0) {
        emit_error(request_id, "max_new_tokens must be a positive integer");
        return false;
    }
    const uint32_t max_new_tokens =
        static_cast<uint32_t>(request["max_new_tokens"].get<long long>());
    if (static_cast<uint64_t>(input_ids.size()) + max_new_tokens >
        static_cast<uint64_t>(runtime.config().max_seq_len)) {
        emit_error(request_id, "input_ids + max_new_tokens exceeds max_seq_len");
        return false;
    }
    if (mtp.free_slots.empty()) {
        emit_error_with_code(request_id, "busy", "mtp slots are exhausted");
        return false;
    }
    shelf::qwen35::runtime::SamplingConfig sampling;
    if (request.contains("temperature")) {
        if (!request["temperature"].is_number()) {
            emit_error(request_id, "temperature must be a number");
            return false;
        }
        sampling.temperature = request["temperature"].get<float>();
    }
    if (request.contains("top_p")) {
        if (!request["top_p"].is_number()) {
            emit_error(request_id, "top_p must be a number");
            return false;
        }
        sampling.top_p = request["top_p"].get<float>();
    }
    if (request.contains("min_p")) {
        if (!request["min_p"].is_number()) {
            emit_error(request_id, "min_p must be a number");
            return false;
        }
        sampling.min_p = request["min_p"].get<float>();
    }
    if (request.contains("top_k")) {
        if (!request["top_k"].is_number_integer()) {
            emit_error(request_id, "top_k must be an integer");
            return false;
        }
        sampling.top_k = static_cast<uint32_t>(request["top_k"].get<long long>());
    }
    if (request.contains("seed")) {
        if (!request["seed"].is_number_integer()) {
            emit_error(request_id, "seed must be an integer");
            return false;
        }
        sampling.seed = static_cast<uint64_t>(request["seed"].get<long long>());
    }
    const shelf::Status sampling_status =
        shelf::qwen35::runtime::validate_sampling_config(sampling);
    if (!sampling_status.ok()) {
        emit_error(request_id, sampling_status.message());
        return false;
    }

    auto sequence_result = shelf::state::create_paged_sequence_state(
        runtime.slot_pool(), runtime.gdn_pool(), runtime.kv_pool(),
        runtime.config().max_seq_len, runtime.stream());
    if (!sequence_result.ok()) {
        emit_error_with_code(request_id, "internal",
                             "sequence create failed: " +
                                 sequence_result.status().message());
        return false;
    }
    auto sequence = std::make_unique<shelf::state::PagedSequenceState>(
        sequence_result.release());
    auto* tp = runtime.tp_coordinator();
    if (tp != nullptr) {
        const shelf::Status mirror = tp->on_sequence_created(*sequence);
        if (!mirror.ok()) {
            emit_error_with_code(request_id, "internal",
                                 "tp mirror create failed: " + mirror.message());
            (void)shelf::state::release_paged_sequence_state(*sequence, runtime.stream());
            return false;
        }
    }
    uint32_t restored_tokens = 0u;
    shelf::qwen35::runtime::MtpDraftSnapshot restored_snapshot;
    bool has_snapshot = false;
    if (auto* cache = runtime.prefix_cache(); cache != nullptr) {
        std::vector<uint32_t> prefix(input_ids.size());
        for (std::size_t i = 0u; i < input_ids.size(); ++i)
            prefix[i] = static_cast<uint32_t>(input_ids[i]);
        const shelf::runtime::PrefixHit hit =
            cache->lookup(prefix, static_cast<uint32_t>(prefix.size()) - 1u);
        if (std::getenv("SHELF_PFXDBG") != nullptr) {
            std::fprintf(stderr,
                         "PFXDBG mtp lookup hit=%d state=%u in=%zu snapshots=%zu\n",
                         static_cast<int>(hit.hit()), hit.state_position, prefix.size(),
                         mtp.snapshots.size());
        }
        if (hit.hit() && hit.state_position < prefix.size()) {
            const uint64_t key =
                shelf::qwen35::runtime::mtp_prefix_key(prefix, hit.state_position);
            auto snapshot = mtp.snapshots.find(key);
            if (snapshot != mtp.snapshots.end()) {
                const shelf::Status restored = cache->restore(
                    hit.entry_id, *sequence, runtime.stream());
                if (restored.ok()) {
                    if (tp != nullptr) {
                        auto mirror = tp->on_sequence_restored(*sequence);
                        const std::span<const int32_t> restored_prefix(
                            input_ids.data(), hit.state_position);
                        if (mirror.ok())
                            mirror = tp->on_prefix_restored(*sequence, restored_prefix);
                        if (!mirror.ok()) {
                            emit_error_with_code(request_id, "internal",
                                                 "tp prefix restore failed: " +
                                                     mirror.message());
                            (void)tp->on_sequence_released(*sequence);
                            (void)shelf::state::release_paged_sequence_state(
                                *sequence, runtime.stream());
                            return false;
                        }
                    }
                    restored_snapshot = snapshot->second;
                    has_snapshot = true;
                    restored_tokens = hit.state_position;
                } else {
                    emit_error_with_code(request_id, "internal",
                                         "prefix restore failed: " +
                                             restored.message());
                    if (tp != nullptr) (void)tp->on_sequence_released(*sequence);
                    (void)shelf::state::release_paged_sequence_state(
                        *sequence, runtime.stream());
                    return false;
                }
            }
        }
    }

    const uint32_t slot = mtp.free_slots.back();
    auto& drafter = *mtp.batch.drafters[slot];
    shelf::Status reset =
        shelf::qwen35::runtime::mtp_drafter_reset_sequence(drafter, runtime.stream());
    if (!reset.ok()) {
        emit_error_with_code(request_id, "internal",
                             "mtp reset failed: " + reset.message());
        if (tp != nullptr) (void)tp->on_sequence_released(*sequence);
        (void)shelf::state::release_paged_sequence_state(*sequence, runtime.stream());
        return false;
    }
    if (has_snapshot) {
        const shelf::Status restored = shelf::qwen35::runtime::mtp_restore_draft_snapshot(
            drafter, restored_snapshot, runtime.stream());
        if (!restored.ok()) {
            emit_error_with_code(request_id, "internal",
                                 "mtp snapshot restore failed: " + restored.message());
            mtp.free_slots.push_back(slot);
            if (tp != nullptr) (void)tp->on_sequence_released(*sequence);
            (void)shelf::state::release_paged_sequence_state(*sequence, runtime.stream());
            return false;
        }
        if (std::getenv("SHELF_PFXDBG") != nullptr) {
            std::fprintf(stderr,
                         "PFXDBG mtp restored prefix=%u target=%u mtp=%u last=%u\n",
                         restored_tokens, sequence->position,
                         drafter.sequence->position, drafter.last_position);
        }
    }
    mtp.free_slots.pop_back();
    drafter.sampling = sampling;
    MtpPrefillLog prefill_log;
    prefill_log.external_id = external_request_id;
    MtpPrefillCancel prefill_cancel;
    prefill_cancel.input = &input;
    prefill_cancel.external_id = external_request_id;
    shelf::qwen35::runtime::MtpPrefillProgress prefill_progress;
    prefill_progress.context = &prefill_log;
    prefill_progress.report = serve_mtp_prefill_progress;
    prefill_progress.should_stop = serve_mtp_prefill_should_stop;
    prefill_progress.stop_context = &prefill_cancel;
    MtpSnapshotSaveContext snapshot_save;
    snapshot_save.runtime = &runtime;
    snapshot_save.mtp = &mtp;
    snapshot_save.drafter = &drafter;
    snapshot_save.input_ids = &input_ids;
    snapshot_save.target_sequence = sequence.get();
    snapshot_save.save_interval = runtime.config().prefix_cache_save_interval;
    shelf::qwen35::runtime::MtpPrefillBoundary boundary;
    boundary.context = &snapshot_save;
    boundary.save = serve_mtp_save_boundary;
    const auto prefill_started = std::chrono::steady_clock::now();
    auto prefill = shelf::qwen35::runtime::mtp_spec_prefill(
        drafter, runtime.executor(), *sequence, input_ids.data(),
        static_cast<uint32_t>(input_ids.size()), sampling, &prefill_progress, &boundary,
        restored_tokens);
    const double prefill_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - prefill_started)
                                  .count();
    if (!prefill.ok()) {
        if (prefill_cancel.cancelled) {
            MtpRequest cancelled;
            cancelled.external_id = external_request_id;
            cancelled.prompt_tokens = input_ids.size();
            cancelled.timing = timing;
            cancelled.prefill_ms = prefill_ms;
            serve_mtp_emit_done(state, cancelled, "cancelled");
        } else {
            emit_error_with_code(request_id, "internal",
                                 "mtp prefill failed: " + prefill.status().message());
        }
        mtp.free_slots.push_back(slot);
        if (tp != nullptr) (void)tp->on_sequence_released(*sequence);
        (void)shelf::state::release_paged_sequence_state(*sequence, runtime.stream());
        return prefill_cancel.cancelled;
    }

    auto admitted = std::make_unique<MtpRequest>();
    admitted->external_id = external_request_id;
    admitted->sequence = std::move(sequence);
    admitted->drafter_slot = slot;
    admitted->pending_token = prefill.value().pending_token;
    admitted->max_new_tokens = max_new_tokens;
    admitted->restored_tokens = restored_tokens;
    admitted->prompt_tokens = input_ids.size();
    admitted->generated = 1u;
    admitted->generated_ids.push_back(prefill.value().pending_token);
    admitted->timing = timing;
    admitted->prefill_ms = prefill_ms;
    mtp_timing_first_token(admitted->timing);
    emit_token(external_request_id, prefill.value().pending_token);
    const bool stop = shelf::qwen_hybrid::is_stop_token(
        runtime.model().hybrid_config().stop_tokens, prefill.value().pending_token);
    if (stop || admitted->generated >= admitted->max_new_tokens) {
        serve_mtp_emit_done(state, *admitted, stop ? "stop" : "length");
        if (tp != nullptr) (void)tp->on_sequence_released(*admitted->sequence);
        (void)shelf::state::release_paged_sequence_state(*admitted->sequence,
                                                         runtime.stream());
        mtp.free_slots.push_back(slot);
        return true;
    }
    mtp.active.push_back(std::move(admitted));
    return true;
}

void serve_mtp_step(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                    MtpServeContext& mtp) {
    if (mtp.active.empty()) return;
    std::vector<shelf::qwen35::runtime::MtpSpecBatchSlot> slots;
    std::vector<std::size_t> indices;
    slots.reserve(mtp.active.size());
    indices.reserve(mtp.active.size());
    for (std::size_t i = 0u; i < mtp.active.size(); ++i) {
        MtpRequest& request = *mtp.active[i];
        if (request.generated >= request.max_new_tokens) continue;
        shelf::qwen35::runtime::MtpSpecBatchSlot slot;
        slot.drafter = mtp.batch.drafters[request.drafter_slot].get();
        slot.sequence = request.sequence.get();
        slot.pending_token = request.pending_token;
        slot.num_drafts = mtp.drafts;
        slot.remaining_tokens = request.max_new_tokens - request.generated;
        slot.page_tokens = runtime.config().page_tokens;
        slots.push_back(slot);
        indices.push_back(i);
    }
    if (slots.empty()) {
        for (std::size_t i = mtp.active.size(); i-- > 0u;) {
            MtpRequest& request = *mtp.active[i];
            serve_mtp_emit_done(state, request, "length");
            serve_mtp_release(runtime, state, mtp, i);
        }
        return;
    }
    auto stepped = shelf::qwen35::runtime::mtp_spec_step_batch(mtp.batch, slots);
    if (!stepped.ok()) {
        const std::string message = "mtp batch step failed: " + stepped.status().message();
        for (std::size_t i = 0u; i < mtp.active.size(); ++i) {
            emit_error(nlohmann::json(mtp.active[i]->external_id), message);
            serve_push_terminal(state, mtp.active[i]->external_id);
        }
        for (std::size_t i = mtp.active.size(); i-- > 0u;) {
            serve_mtp_release(runtime, state, mtp, i);
        }
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
            emit_token(request.external_id, out.emitted[t]);
            request.generated_ids.push_back(out.emitted[t]);
            request.generated += 1u;
        }
        request.pending_token = out.pending_token;
        mtp.accepted_drafts += out.num_accepted;
        mtp.emitted_tokens += out.emitted_count;
        mtp_log_decode_interval(request.external_id, request.timing,
                                request.generated);
        const bool length_done = request.generated >= request.max_new_tokens;
        finished[j] = out.finished || length_done;
        reasons[j] = out.finished ? "stop" : "length";
    }
    for (std::size_t j = slots.size(); j-- > 0u;) {
        if (!finished[j]) continue;
        MtpRequest& request = *mtp.active[indices[j]];
        serve_mtp_emit_done(state, request, reasons[j]);
        request.max_new_tokens = request.generated;
        serve_mtp_release(runtime, state, mtp, indices[j]);
    }
    if (mtp.active.empty() && !state.exit_requested) {
        fprintf(stderr,
                "MTP_BATCH rounds=%llu accepted=%llu emitted=%llu chunks=%llu "
                "draft_forwards=%llu max_draft_per_round=%u max_verify_per_round=%u "
                "max_active=%u round_ms=%.3f\n",
                static_cast<unsigned long long>(mtp.rounds),
                static_cast<unsigned long long>(mtp.accepted_drafts),
                static_cast<unsigned long long>(mtp.emitted_tokens),
                static_cast<unsigned long long>(mtp.verify_chunks),
                static_cast<unsigned long long>(mtp.draft_forwards),
                mtp.max_draft_forwards_per_round,
                mtp.max_verify_chunks_per_round,
                mtp.max_active_requests,
                mtp.rounds > 0u ? mtp.round_ms / static_cast<double>(mtp.rounds) : 0.0);
    }
}

void serve_handle_command(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                          DFlashServeContext& dflash, JsonlInput& input,
                          const std::string& line) {
    nlohmann::json request;
    try {
        request = nlohmann::json::parse(line);
    } catch (const std::exception&) {
        emit_error(nullptr, "malformed json");
        return;
    }
    if (!request.is_object()) {
        emit_error(nullptr, "request must be a json object");
        return;
    }

    const std::string op = request.value("op", std::string());
    if (op == "ping") {
        emit_json(nlohmann::json{
            {"event", "pong"},
            {"protocol_version", 1},
            {"capabilities", {{"prefix_cache", runtime.prefix_cache_active()}}},
        });
        return;
    }
    if (op == "stats") {
        const qwen35::runtime::KVCapacitySnapshot snapshot =
            runtime.kv_capacity_snapshot();
        const uint64_t page_tokens = runtime.config().page_tokens;
        nlohmann::json prefix = {
            {"enabled", runtime.prefix_cache_active()},
            {"reuse_slots", 0u},
            {"entries", 0u},
            {"saves", 0u},
            {"hits", 0u},
            {"misses", 0u},
            {"restores", 0u},
            {"evictions", 0u},
        };
        if (runtime.prefix_cache() != nullptr) {
            const qwen35::runtime::PrefixCacheStats prefix_stats =
                runtime.prefix_cache()->stats();
            prefix["reuse_slots"] = runtime.prefix_cache()->reuse_slot_count();
            prefix["entries"] = prefix_stats.entries;
            prefix["saves"] = prefix_stats.saves;
            prefix["hits"] = prefix_stats.hits;
            prefix["misses"] = prefix_stats.misses;
            prefix["restores"] = prefix_stats.restores;
            prefix["evictions"] = prefix_stats.entries_evicted;
        }
        emit_json(nlohmann::json{
            {"event", "stats"},
            {"prefix", std::move(prefix)},
            {"kv",
             {
                 {"page_tokens", page_tokens},
                 {"total_pages", snapshot.total_pages},
                 {"used_pages", snapshot.physical_used_pages},
                 {"free_pages", snapshot.physical_free_pages},
                 {"capacity_tokens",
                  static_cast<uint64_t>(snapshot.total_pages) * page_tokens},
                 {"free_tokens",
                  static_cast<uint64_t>(snapshot.physical_free_pages) * page_tokens},
                 {"active_requests", snapshot.active_requests},
                 {"queued_requests", runtime.queued_request_count()},
                 {"banker_safe", snapshot.banker_safe},
                 {"overcommit_pages", snapshot.overcommit_pages},
             }},
        });
        return;
    }
    if (op == "shutdown") {
        serve_cancel_all_inflight(runtime, state);
        emit_json(nlohmann::json{{"event", "shutdown"}});
        state.exit_requested = true;
        return;
    }
    if (op == "cancel") {
        if (!request.contains("request_id") || !request["request_id"].is_number_integer()) {
            emit_error(nullptr, "missing or invalid request_id");
            return;
        }
        const nlohmann::json request_id = request["request_id"];
        const int64_t id = request_id.get<int64_t>();
        if (state.inflight.count(id) > 0) {
            serve_cancel_one(runtime, state, id);
            return;
        }
        if (state.mtp.enabled) {
            for (std::size_t i = 0u; i < state.mtp.active.size(); ++i) {
                if (state.mtp.active[i]->external_id != id) continue;
                serve_mtp_emit_done(state, *state.mtp.active[i], "cancelled");
                serve_mtp_release(runtime, state, state.mtp, i);
                return;
            }
            if (serve_is_recent_terminal(state, id)) return;
            emit_error(request_id, "unknown or mismatched request_id");
            return;
        }
        if (state.inflight.count(id) > 0) {
            serve_cancel_one(runtime, state, id);
        } else if (serve_is_recent_terminal(state, id)) {
            return;
        } else {
            emit_error(request_id, "unknown or mismatched request_id");
        }
        return;
    }
    if (op == "draft_masks") {
        if (!request.contains("request_id") || !request["request_id"].is_number_integer()) {
            emit_error(nullptr, "missing or invalid request_id");
            return;
        }
        const int64_t id = request["request_id"].get<int64_t>();
        auto it = state.inflight.find(id);
        if (it == state.inflight.end()) return;
        if (!request.contains("round_id") || !request["round_id"].is_number_unsigned()) {
            emit_error_with_code(request["request_id"], "invalid_argument", "invalid draft round id");
            (void)runtime.cancel(it->second.internal_request_id);
            serve_push_terminal(state, id);
            serve_retire(runtime, state, it->second);
            return;
        }
        const auto* active = runtime.request(it->second.internal_request_id);
        const uint64_t round = request["round_id"].get<uint64_t>();
        if (active == nullptr || active->state == shelf::qwen35::runtime::RequestState::Finished ||
            active->draft_mask_state != shelf::qwen35::runtime::DraftMaskState::PendingMask ||
            round != active->draft_round_id) return;
        const size_t words = (runtime.model().hybrid_config().vocab_size + 31u) / 32u;
        const size_t count = active->pending_drafts.size();
        bool valid = request.contains("valid_prefix") && request["valid_prefix"].is_number_unsigned() &&
                     request.contains("bitmasks") && request["bitmasks"].is_array();
        uint64_t prefix = valid ? request["valid_prefix"].get<uint64_t>() : 0;
        valid = valid && prefix <= count && request["bitmasks"].size() == prefix + 1u;
        std::vector<uint32_t> masks((count + 1u) * words, 0xffffffffu);
        if (valid) {
            for (size_t row = 0; row <= prefix; ++row) {
                std::vector<uint8_t> bytes;
                const auto& encoded = request["bitmasks"][row];
                if (!encoded.is_string() || !decode_mask(encoded.get<std::string>(), bytes) ||
                    bytes.size() != words * sizeof(uint32_t) ||
                    std::none_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b != 0; })) {
                    valid = false;
                    break;
                }
                std::memcpy(masks.data() + row * words, bytes.data(), bytes.size());
            }
        }
        const size_t bytes = masks.size() * sizeof(uint32_t);
        if (valid && it->second.draft_bitmask_capacity < bytes) {
            uint32_t* device = nullptr;
            valid = hipMalloc(reinterpret_cast<void**>(&device), bytes) == hipSuccess;
            if (valid) {
                it->second.draft_bitmask = std::shared_ptr<uint32_t>(device, [](uint32_t* p) {
                    const hipError_t err = hipFree(p);
                    if (err != hipSuccess) std::fprintf(stderr, "draft mask cleanup failed: %s\n", hipGetErrorString(err));
                });
                it->second.draft_bitmask_capacity = bytes;
            }
        }
        uint32_t* device = it->second.draft_bitmask.get();
        if (valid) valid = hipMemcpy(device, masks.data(), bytes, hipMemcpyHostToDevice) == hipSuccess;
        if (valid) {
            auto st = runtime.set_draft_masks(it->second.internal_request_id, round, device,
                                              static_cast<uint32_t>(words), static_cast<uint32_t>(prefix));
            valid = st.ok();
        }
        if (!valid) {
            emit_error_with_code(request["request_id"], "invalid_argument", "invalid draft masks");
            (void)runtime.cancel(it->second.internal_request_id);
            serve_push_terminal(state, id);
            serve_retire(runtime, state, it->second);
            return;
        }
        it->second.waiting_for_mask = false;
        return;
    }
    if (op == "mask") {
        if (!request.contains("request_id") || !request["request_id"].is_number_integer()) {
            emit_error(nullptr, "missing or invalid request_id");
            return;
        }
        const int64_t id = request["request_id"].get<int64_t>();
        auto it = state.inflight.find(id);
        if (it == state.inflight.end() && serve_is_recent_terminal(state, id)) return;
        if (it == state.inflight.end() || !it->second.waiting_for_mask ||
            !request.contains("bitmask") || !upload_mask(runtime, it->second, request["bitmask"])) {
            emit_error_with_code(request["request_id"], "invalid_argument", "unexpected or invalid bitmask");
            if (it != state.inflight.end()) {
                (void)runtime.cancel(it->second.internal_request_id);
                serve_push_terminal(state, id);
                serve_retire(runtime, state, it->second);
            }
            return;
        }
        auto st = runtime.set_token_bitmask(it->second.internal_request_id,
                                            it->second.token_bitmask.get(), false);
        if (!st.ok()) { emit_error(request["request_id"], st.message()); return; }
        it->second.waiting_for_mask = false;
        return;
    }
    if (op == "generate") {
        if (state.mtp.enabled && !request.contains("grammar")) {
            serve_mtp_admit(runtime, state, state.mtp, input, request);
        } else {
            serve_parse_generate(runtime, state, dflash, request);
        }
        return;
    }
    emit_error(request.contains("request_id") ? request["request_id"] : nlohmann::json(),
               "unknown op");
}

}  // namespace

std::vector<shelf::qwen35::runtime::DFlash2GdnRankBinding> build_serve_rank_bindings(
    shelf::app::Qwen35ComputeRuntime& runtime) {
    std::vector<shelf::qwen35::runtime::DFlash2GdnRankBinding> bindings;
    shelf::qwen35::runtime::TpCoordinator* tp = runtime.tp_coordinator();
    if (tp == nullptr) return bindings;
    bindings.reserve(static_cast<std::size_t>(tp->tp_size()));
    for (int r = 0; r < tp->tp_size(); ++r) {
        auto& rank = tp->rank(static_cast<std::size_t>(r));
        bindings.push_back(shelf::qwen35::runtime::DFlash2GdnRankBinding{
            &rank.gdn_pool(), &rank.arena(), rank.stream(), rank.device_id(),
            &rank.executor().host_model_state, &rank.executor()});
    }
    return bindings;
}

int run_serve_stdio(shelf::app::Qwen35ComputeRuntime& runtime,
                    DFlashServeContext& dflash) {
    const std::vector<shelf::qwen35::runtime::DFlash2GdnRankBinding> serve_rank_bindings =
        build_serve_rank_bindings(runtime);
    const std::vector<shelf::qwen35::runtime::DFlash2GdnRankBinding>* serve_rank_ptr =
        serve_rank_bindings.empty() ? nullptr : &serve_rank_bindings;
    const shelf::weights::MatrixWeight* private_lm_head = nullptr;
    const shelf::weights::MatrixWeight* private_embed = nullptr;
    if (runtime.config().tp > 1) {
        auto* tp = runtime.tp_coordinator();
        if (tp != nullptr && tp->tp_size() > 0) {
            auto& rank0 = tp->rank(0);
            const auto* q4 = rank0.rank_model() != nullptr
                                 ? rank0.rank_model()->qwen4exp_model()
                                 : nullptr;
            if (q4 != nullptr) {
                private_lm_head = &q4->weights().lm_head;
                private_embed = &q4->weights().embed_tokens;
            } else {
                private_lm_head = &rank0.private_lm_head();
                private_embed = &rank0.private_embed_tokens();
            }
        }
    }
    if (dflash.enabled) {
        const uint32_t session_rows =
            shelf::qwen35::runtime::dflash2_session_rows(dflash.config.block_size, dflash.drafts,
                                                         runtime.config().dflash2_tree_budget);
        if (runtime.config().dflash2_sessions() >
            shelf::qwen35::runtime::kDFlash2SpecMaxVerifyRows / session_rows) {
            fprintf(stderr,
                    "dflash2 serve: dflash2_sessions %u exceeds verify row cap %u "
                    "for %u rows per session\n",
                    runtime.config().dflash2_sessions(),
                    shelf::qwen35::runtime::kDFlash2SpecMaxVerifyRows, session_rows);
            return 1;
        }
        const std::size_t weights_before = runtime.arena().used();
        auto weights_result = runtime.load_dflash2_weights(dflash.config);
        fprintf(stderr, "DFLASH2_WEIGHT_ARENA_BYTES=%llu\n",
                static_cast<unsigned long long>(runtime.arena().used() - weights_before));
        if (!weights_result.ok()) {
            fprintf(stderr, "dflash2 weight load failed: %s\n",
                    weights_result.status().message().c_str());
            return 1;
        }
        dflash.weights = weights_result.release();

        const auto& text_config = runtime.model().hybrid_config();
        const uint32_t batch_max_rows = std::min<uint32_t>(
            runtime.config().dflash2_sessions() * session_rows,
            shelf::qwen35::runtime::kDFlash2SpecMaxVerifyRows);
        const bool serve_tp_sharded = runtime.config().tp > 1;
        std::vector<const shelf::weights::MatrixWeight*> head_shards;
        uint32_t head_vocab_offset = 0;
        uint32_t head_local_vocab = 0;
        shelf::kernel::EmbeddingShardTable embed_config_shards{};
        if (serve_tp_sharded) {
            auto* tp = runtime.tp_coordinator();
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
                ::shelf::runtime::dflash2::DFlash2TpProposalConfig proposal_config;
                proposal_config.coordinator = tp;
                proposal_config.heads = head_shards;
                proposal_config.vocab_offsets = offsets;
                proposal_config.local_vocab = static_cast<uint32_t>(head_shards[0]->rows);
                proposal_config.top_k = static_cast<uint32_t>(dflash.config.selector_top_k);
                proposal_config.hidden_stride =
                    static_cast<uint32_t>(dflash.config.hidden_size);
                proposal_config.max_rows = std::max<uint32_t>(
                    static_cast<uint32_t>(dflash.config.block_size), batch_max_rows);
                auto proposal_result =
                    ::shelf::runtime::dflash2::DFlash2TpProposal::create(proposal_config);
                if (!proposal_result.ok()) {
                    fprintf(stderr, "dflash2 tp proposal create failed: %s\n",
                            proposal_result.status().message().c_str());
                    return 1;
                }
                dflash.tp_proposal =
                    std::make_unique<::shelf::runtime::dflash2::DFlash2TpProposal>(
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
                fprintf(stderr,
                        "dflash2 tp: target lm_head is not split across ranks, "
                        "a full private copy is required\n");
                return 1;
            }
        }
        ::shelf::runtime::dflash2::DFlash2ExecutorConfig draft_config;
        draft_config.max_rows = static_cast<uint32_t>(dflash.config.block_size);
        draft_config.max_batch_rows = batch_max_rows;
        draft_config.max_context_rows = static_cast<uint32_t>(dflash.config.sliding_window);
        draft_config.max_prefill_feature_rows = runtime.config().dflash2_prefill_rows;
        draft_config.proposal_distributor = dflash.tp_proposal.get();
        draft_config.vocab_offset = head_vocab_offset;
        draft_config.local_vocab = head_local_vocab;
        draft_config.target_embed_shards = embed_config_shards;
        const shelf::weights::MatrixWeight* draft_embed = &runtime.model().embed_tokens();
        if (serve_tp_sharded) {
            draft_embed = dflash.tp_proposal != nullptr
                ? runtime.tp_coordinator()->rank(0).rank_model()->embed_tokens()
                : private_embed;
        }
        draft_config.target_lm_head =
            dflash.tp_proposal != nullptr
                ? head_shards[0]
                : (serve_tp_sharded ? private_lm_head : &runtime.model().lm_head());
        draft_config.target_embed_tokens = draft_embed;
        auto draft_result = ::shelf::runtime::dflash2::create_dflash2_executor(
            dflash.config, dflash.weights, runtime.arena(), draft_config);
        if (!draft_result.ok()) {
            fprintf(stderr, "dflash2 executor create failed: %s\n",
                    draft_result.status().message().c_str());
            return 1;
        }
        dflash.executor = draft_result.release();
        dflash.executor_ready = true;
        {
            std::vector<shelf::qwen35::runtime::DFlash2GdnRankBinding> bindings;
            if (serve_rank_ptr != nullptr) {
                bindings = serve_rank_bindings;
            } else {
                shelf::qwen35::runtime::DFlash2GdnRankBinding binding;
                binding.pool = &runtime.gdn_pool();
                binding.arena = &runtime.arena();
                binding.stream = runtime.stream();
                binding.device = runtime.config().devices[0];
                binding.model_state = &runtime.executor().host_model_state;
                binding.executor = &runtime.executor();
                bindings.push_back(binding);
            }
            auto batch_result =
                shelf::qwen35::runtime::create_dflash2_spec_batch_context(
                    bindings, batch_max_rows, runtime.config().dflash2_tree_budget);
            if (!batch_result.ok()) {
                fprintf(stderr, "dflash2 batch context create failed: %s\n",
                        batch_result.status().message().c_str());
                return 1;
            }
            dflash.batch = batch_result.release();
            dflash.batch.tree_budget_auto = runtime.config().dflash2_tree_budget_auto;
            dflash.batch_ready = true;
        }
        {
            shelf::qwen35::runtime::DFlash2SpecDecoderConfig session_config;
            session_config.num_drafts = dflash.drafts;
            session_config.tree_budget = runtime.config().dflash2_tree_budget;
            session_config.eos_tokens = text_config.stop_tokens;
            auto sessions_result = shelf::qwen35::runtime::DFlash2SessionPool::create(
                runtime.executor(), &dflash.executor, runtime.gdn_pool(),
                runtime.arena(), session_config,
                runtime.config().dflash2_sessions(), runtime.stream(), serve_rank_ptr);
            if (!sessions_result.ok()) {
                fprintf(stderr, "dflash2 session pool create failed: %s\n",
                        sessions_result.status().message().c_str());
                return 1;
            }
            dflash.sessions = sessions_result.release();
            dflash.sessions_ready = true;
        }
        shelf::qwen35::runtime::DFlash2BatcherBinding batcher_binding;
        batcher_binding.config = &dflash.config;
        batcher_binding.executor = &dflash.executor;
        batcher_binding.batch = &dflash.batch;
        batcher_binding.arena = &runtime.arena();
        batcher_binding.sessions = &dflash.sessions;
        batcher_binding.num_drafts = dflash.drafts;
        batcher_binding.tree_budget = runtime.config().dflash2_tree_budget;
        batcher_binding.adaptive = dflash.adaptive;
        batcher_binding.host_token_ids = runtime.dflash2_host_token_ids();
        batcher_binding.stats = &dflash.stats;
        batcher_binding.rank_bindings = serve_rank_ptr;
        const shelf::Status bind_st = runtime.set_dflash2_batcher(batcher_binding);
        if (!bind_st.ok()) {
            fprintf(stderr, "dflash2 batcher binding failed: %s\n",
                    bind_st.message().c_str());
            return 1;
        }
        fprintf(stderr, "DFLASH2_ENABLED=1\n");
        fprintf(stderr, "DFLASH2_DRAFTS=%u\n", dflash.drafts);
        fprintf(stderr, "DFLASH2_BLOCK_SIZE=%zu\n", dflash.config.block_size);
        fprintf(stderr, "DFLASH2_SESSION_POOL_SLOTS=%u\n", dflash.sessions.capacity());
        fprintf(stderr, "DFLASH2_SESSION_POOL_BYTES=%llu\n",
                static_cast<unsigned long long>(dflash.sessions.reserved_bytes()));
    }
    ServeState state;
    if (runtime.config().mtp) {
        state.mtp.enabled = true;
        state.mtp.drafts = runtime.config().mtp_drafts == 0u ? 4u : runtime.config().mtp_drafts;
        shelf::qwen35::runtime::MtpDrafterConfig mtp_config;
        mtp_config.num_drafts = state.mtp.drafts;
        mtp_config.window_tokens = static_cast<uint32_t>(runtime.config().mtp_window);
        mtp_config.sink_tokens = static_cast<uint32_t>(runtime.config().mtp_sink);
        mtp_config.max_seq_len = runtime.config().max_seq_len;
        mtp_config.page_tokens = runtime.config().page_tokens;
        mtp_config.max_batch_rows = std::min(
            runtime.executor().config.max_batch_rows,
            shelf::qwen35::runtime::kMtpMaxPrefillRows);
        mtp_config.kv_dtype =
            runtime.config().kv_cache_dtype == shelf::kernel::KVCacheDType::OCTAVE
                ? shelf::kernel::KVCacheDType::BF16 : runtime.config().kv_cache_dtype;
        if (runtime.config().rope_profile.has_value()) {
            mtp_config.rope_profile = *runtime.config().rope_profile;
        }
        const shelf::weights::MatrixWeight* lm_head_override = nullptr;
        if (private_lm_head != nullptr && private_lm_head->rows != 0u) {
            lm_head_override = private_lm_head;
        }
        auto* tp = runtime.tp_coordinator();
        std::vector<shelf::qwen35::runtime::MtpDrafterInput> rank_inputs;
        shelf::qwen35::runtime::MtpDrafterInput leader_input;
        if (tp != nullptr && runtime.model().qwen4exp_model() != nullptr) {
            const int rank_count = tp->tp_size();
            rank_inputs.reserve(static_cast<std::size_t>(rank_count));
            for (int r = 0; r < rank_count; ++r) {
                auto& rank = tp->rank(static_cast<std::size_t>(r));
                auto per_rank = mtp_config;
                per_rank.local_kv_heads = rank.context().local_key_value_heads;
                auto input = rank.rank_model()->mtp_drafter_input(per_rank);
                if (!input.ok()) {
                    fprintf(stderr, "mtp rank %d drafter input failed: %s\n", r,
                            input.status().message().c_str());
                    return 1;
                }
                rank_inputs.push_back(std::move(input.value()));
            }
            mtp_config.local_kv_heads = tp->rank(0).context().local_key_value_heads;
        } else {
            auto input = runtime.model().mtp_drafter_input(mtp_config, lm_head_override);
            if (!input.ok()) {
                fprintf(stderr, "mtp drafter input failed: %s\n",
                        input.status().message().c_str());
                return 1;
            }
            leader_input = std::move(input.value());
        }
        const shelf::qwen35::runtime::MtpDrafterInput& drafter_input =
            rank_inputs.empty() ? leader_input : rank_inputs.front();
        std::vector<shelf::qwen35::runtime::MtpGdnRankBinding> extra_ranks;
        if (tp != nullptr) {
            for (int r = 1; r < tp->tp_size(); ++r) {
                auto& rank = tp->rank(static_cast<std::size_t>(r));
                shelf::qwen35::runtime::MtpGdnRankBinding binding;
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
        const uint32_t verify_rows = state.mtp.drafts + 1u;
        uint32_t max_slots = runtime.config().max_concurrent_requests;
        if (max_slots == 0u) max_slots = 1u;
        if (mtp_config.max_batch_rows / verify_rows < max_slots) {
            max_slots = mtp_config.max_batch_rows / verify_rows;
            if (max_slots == 0u) max_slots = 1u;
        }
        auto batch = shelf::qwen35::runtime::create_mtp_spec_batch_context(
            runtime.executor(), drafter_input, runtime.gdn_pool(),
            runtime.arena(), runtime.stream(), mtp_config, max_slots,
            extra_ranks.empty() ? nullptr : &extra_ranks, tp);
        if (!batch.ok()) {
            fprintf(stderr, "mtp batch context create failed: %s\n",
                    batch.status().message().c_str());
            return 1;
        }
        state.mtp.batch = batch.release();
        for (uint32_t i = 0u; i < max_slots; ++i) {
            state.mtp.free_slots.push_back(max_slots - 1u - i);
        }
        fprintf(stderr, "MTP_SERVE=batch\n");
        fprintf(stderr, "MTP_SLOTS=%u\n", max_slots);
        fprintf(stderr, "MTP_DRAFTS=%u\n", state.mtp.drafts);
    }
    fprintf(stderr, "SHELF_SERVE_READY\n");
    fflush(stderr);
    JsonlInput input;
    state.max_inflight = runtime.config().max_concurrent_requests;
    while (!state.exit_requested) {
        const bool mtp_idle = !state.mtp.enabled || state.mtp.active.empty();
        const bool all_waiting = std::all_of(state.inflight.begin(), state.inflight.end(),
            [](const auto& entry) { return entry.second.waiting_for_mask; });
        jsonl_poll_and_read(input, all_waiting && mtp_idle);
        std::string line;
        while (jsonl_take_line(input, line)) {
            if (line.empty()) continue;
            serve_handle_command(runtime, state, dflash, input, line);
            if (state.exit_requested) break;
        }
        if (state.exit_requested) break;
        if (state.mtp.enabled && !state.mtp.active.empty()) {
            serve_mtp_step(runtime, state, state.mtp);
        }
        if (!state.inflight.empty()) {
            if (input.eof) {
                serve_cancel_all_inflight(runtime, state);
            } else {
                serve_step(runtime, state);
            }
        } else if (input.eof && mtp_idle) {
            break;
        }
    }
    if (!state.inflight.empty()) {
        serve_cancel_all_inflight(runtime, state);
    }

    const char* serve_stats_env = std::getenv("SHELF_DFLASH2_SERVE_STATS");
    if (dflash.enabled &&
        (dflash.adaptive || (serve_stats_env != nullptr && serve_stats_env[0] != '\0' &&
                             serve_stats_env[0] != '0'))) {
        const shelf::qwen35::runtime::DFlash2BatcherStats& t = dflash.stats;
        const float rounds = static_cast<float>(t.rounds);
        const float mean =
            t.rounds > 0u ? static_cast<float>(t.accepted_drafts) / rounds : 0.0f;
        const float full_rate =
            t.rounds > 0u ? static_cast<float>(t.full_accepts) / rounds : 0.0f;
        const float emitted =
            t.rounds > 0u ? static_cast<float>(t.generated_tokens) / rounds : 0.0f;
        if (dflash.adaptive && t.adaptive_rounds != 0u) {
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
            const double rounds = static_cast<double>(t.rounds);
            fprintf(stderr,
                    "DFLASH2_STAGE round=%.3f draft=%.3f draft_gpu=%.3f draft_d2h=%.3f "
                    "proposal_wait=%.3f proposal_copy=%.3f gdn_snapshot=%.3f verify=%.3f "
                    "verify_gpu=%.3f decision_wait=%.3f decision_copy=%.3f "
                    "gdn_restore=%.3f rerun=%.3f commit=%.3f\n",
                    t.round_ms / rounds, t.draft_ms / rounds, t.draft_gpu_ms / rounds,
                    t.draft_d2h_ms / rounds, t.proposal_wait_ms / rounds,
                    t.proposal_copy_ms / rounds, t.gdn_snapshot_ms / rounds,
                    t.verify_ms / rounds, t.verify_gpu_ms / rounds,
                    t.decision_wait_ms / rounds, t.decision_copy_ms / rounds,
                    t.gdn_restore_ms / rounds, t.rerun_ms / rounds, t.commit_ms / rounds);
            fprintf(stderr, "DFLASH2_VERIFY_ROWS_PER_ROUND=%.3f\n",
                    static_cast<double>(t.verify_rows) / rounds);
        }
        for (uint32_t b = 0; b < shelf::qwen35::runtime::kDFlash2PositionBuckets; ++b) {
            const shelf::qwen35::runtime::DFlash2PositionStats& p = t.position[b];
            const uint32_t lo = b * shelf::qwen35::runtime::kDFlash2PositionBucketTokens;
            const uint32_t hi =
                (b + 1u) * shelf::qwen35::runtime::kDFlash2PositionBucketTokens;
            const float bucket_rounds = static_cast<float>(p.rounds);
            const float bucket_mean =
                p.rounds > 0u ? static_cast<float>(p.accepted_drafts) / bucket_rounds
                              : 0.0f;
            const float bucket_emitted =
                p.rounds > 0u ? static_cast<float>(p.generated_tokens) / bucket_rounds
                              : 0.0f;
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

    if (dflash.sessions_ready) {
        const shelf::Status sessions_st = dflash.sessions.shutdown();
        dflash.sessions_ready = false;
        if (!sessions_st.ok()) {
            fprintf(stderr, "dflash2 session pool shutdown failed: %s\n",
                    sessions_st.message().c_str());
        }
    }

    if (dflash.batch_ready) {
        const shelf::Status batch_st =
            shelf::qwen35::runtime::shutdown_dflash2_spec_batch_context(dflash.batch);
        dflash.batch_ready = false;
        if (!batch_st.ok()) {
            fprintf(stderr, "dflash2 batch context shutdown failed: %s\n",
                    batch_st.message().c_str());
        }
    }

    if (dflash.executor_ready) {
        const shelf::Status draft_st =
            ::shelf::runtime::dflash2::dflash2_executor_shutdown(dflash.executor);
        dflash.executor_ready = false;
        if (!draft_st.ok()) {
            fprintf(stderr, "dflash2 executor shutdown failed: %s\n",
                    draft_st.message().c_str());
        }
    }

    for (std::size_t i = state.mtp.active.size(); i-- > 0u;) {
        serve_mtp_release(runtime, state, state.mtp, i);
    }
    if (state.mtp.batch.initialized) {
        const shelf::Status mtp_st =
            shelf::qwen35::runtime::shutdown_mtp_spec_batch_context(state.mtp.batch);
        if (!mtp_st.ok()) {
            fprintf(stderr, "mtp batch shutdown failed: %s\n",
                    mtp_st.message().c_str());
        }
    }

    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

}  // namespace shelf::app
