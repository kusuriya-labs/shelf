#include "stdio_server.h"

#include <shelf/engine/serve.h>
#include <shelf/runtime/runtime_request.h>

#include <nlohmann/json.hpp>

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

using shelf::engine::ServeEvent;

void emit_json(const nlohmann::json& value) {
    const std::string line = value.dump();
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void emit_error(const nlohmann::json& request_id, const std::string& message) {
    nlohmann::json event;
    event["event"] = "error";
    if (!request_id.is_null()) event["request_id"] = request_id;
    event["message"] = message;
    emit_json(event);
}

void emit_error_with_code(const nlohmann::json& request_id, const std::string& code,
                          const std::string& message) {
    nlohmann::json event;
    event["event"] = "error";
    if (!request_id.is_null()) event["request_id"] = request_id;
    if (!code.empty()) event["code"] = code;
    event["message"] = message;
    emit_json(event);
}

void emit_token(int64_t request_id, int32_t token_id, bool need_mask = true) {
    nlohmann::json event = {
        {"event", "token"}, {"request_id", request_id}, {"token_id", token_id}};
    if (!need_mask) event["need_mask"] = false;
    emit_json(event);
}

void emit_events(const std::vector<ServeEvent>& events) {
    for (const auto& event : events) {
        switch (event.kind) {
            case ServeEvent::Kind::Token:
                emit_token(event.request_id, event.token_id, event.need_mask);
                break;
            case ServeEvent::Kind::Error:
                emit_error_with_code(nlohmann::json(event.request_id), event.code,
                                     event.message);
                break;
            case ServeEvent::Kind::Draft:
                emit_json({{"event", "draft"},
                           {"request_id", event.request_id},
                           {"round_id", event.round_id},
                           {"draft_tokens", event.draft_tokens}});
                break;
            case ServeEvent::Kind::Done: {
                nlohmann::json done;
                done["event"] = "done";
                done["request_id"] = event.request_id;
                done["generated_ids"] = event.generated_ids;
                done["finish_reason"] = event.finish_reason;
                done["prompt_tokens"] = event.prompt_tokens;
                done["restored_tokens"] = event.restored_tokens;
                done["prefill_tokens"] = event.prefill_tokens;
                done["cache_checkpoint_tokens"] = event.cache_checkpoint_tokens;
                const auto& stats = event.dflash_stats;
                if (stats.final_k != 0) {
                    done["speculative_stats"] = {
                        {"final_k", stats.final_k}, {"k_switches", stats.k_switches},
                        {"adaptive_samples", stats.adaptive_samples}, {"draft_rounds", stats.draft_rounds},
                        {"proposed_drafts", stats.proposed_drafts}, {"accepted_drafts", stats.accepted_drafts},
                        {"verify_rows", stats.verify_rows}};
                    if (stats.adaptive_samples != 0) done["speculative_stats"]["acc_ema"] = stats.acc_ema;
                }
                if (!event.error_message.empty()) done["error_message"] = event.error_message;
                emit_json(done);
                break;
            }
        }
    }
}

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
    bytes.clear();
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

bool decode_mask_words(const std::string& encoded, std::size_t expected_bytes,
                       std::vector<uint32_t>& words) {
    std::vector<uint8_t> bytes;
    if (!decode_mask(encoded, bytes) || bytes.size() != expected_bytes ||
        std::none_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b != 0; })) {
        return false;
    }
    words.resize(bytes.size() / sizeof(uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return true;
}

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

struct MtpPrefillCancel {
    JsonlInput* input = nullptr;
    int64_t external_id = 0;
    bool cancelled = false;
};

bool mtp_prefill_should_stop(void* opaque) {
    auto* cancel = static_cast<MtpPrefillCancel*>(opaque);
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

bool parse_sampling(const nlohmann::json& request, const nlohmann::json& request_id,
                    shelf::runtime::SamplingConfig& sampling) {
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
    const shelf::Status status = shelf::runtime::validate_sampling_config(sampling);
    if (!status.ok()) {
        emit_error(request_id, status.message());
        return false;
    }
    return true;
}

bool parse_thinking(const nlohmann::json& request, const nlohmann::json& request_id,
                    uint32_t max_new_tokens, shelf::runtime::ThinkingBudget& thinking) {
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
    return true;
}

bool parse_generate(shelf::engine::ServeController& serve,
                    shelf::engine::ComputeSession& runtime, JsonlInput& input,
                    const nlohmann::json& request) {
    if (!request.contains("request_id") || !request["request_id"].is_number_integer()) {
        emit_error(nullptr, "missing or invalid request_id");
        return false;
    }
    const nlohmann::json request_id = request["request_id"];
    const int64_t external_request_id = request_id.get<int64_t>();
    const bool mtp_path = serve.mtp_enabled() && !request.contains("grammar");
    if (serve.has_request(external_request_id) ||
        serve.is_recent_terminal(external_request_id)) {
        emit_error(request_id, "duplicate request_id");
        return false;
    }

    for (const auto& entry : request.items()) {
        const std::string& key = entry.key();
        if (mtp_path) {
            if (key != "op" && key != "request_id" && key != "input_ids" &&
                key != "max_new_tokens" && key != "temperature" && key != "top_p" &&
                key != "min_p" && key != "top_k" && key != "seed" &&
                key != "ignore_eos" &&
                key != "reasoning_budget_tokens" && key != "reasoning_close_token" &&
                key != "prefix_cache_checkpoint_position") {
                emit_error_with_code(request_id, "invalid_argument",
                                     "unknown request field: " + key);
                return false;
            }
        } else if (key != "op" && key != "request_id" && key != "input_ids" &&
                   key != "max_new_tokens" && key != "temperature" && key != "top_p" &&
                   key != "min_p" && key != "top_k" && key != "seed" &&
                   key != "ignore_eos" &&
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
        if (!value.is_number_integer()) {
            ids_ok = false;
            break;
        }
        input_ids.push_back(value.get<int32_t>());
    }
    if (!ids_ok) {
        emit_error(request_id, "input_ids must contain integers");
        return false;
    }

    if (!request.contains("max_new_tokens") ||
        !request["max_new_tokens"].is_number_integer() ||
        request["max_new_tokens"].get<long long>() <= 0) {
        emit_error(request_id, "max_new_tokens must be a positive integer");
        return false;
    }
    const uint32_t max_new_tokens =
        static_cast<uint32_t>(request["max_new_tokens"].get<long long>());

    if (request.contains("ignore_eos") && !request["ignore_eos"].is_boolean()) {
        emit_error_with_code(request_id, "invalid_argument",
                             "ignore_eos must be a boolean");
        return false;
    }

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

    shelf::engine::ServeRequestSpec spec;
    spec.request_id = external_request_id;
    spec.input_ids = std::move(input_ids);
    spec.max_new_tokens = max_new_tokens;
    spec.ignore_eos = request.value("ignore_eos", false);
    if (!parse_sampling(request, request_id, spec.sampling)) return false;
    if (!parse_thinking(request, request_id, max_new_tokens, spec.thinking)) return false;

    if (!mtp_path) {
        if (request.contains("grammar") && !request["grammar"].is_boolean()) {
            emit_error_with_code(request_id, "invalid_argument", "grammar must be a boolean");
            return false;
        }
        spec.grammar = request.value("grammar", false);
        if (request.contains("grammar_after_reasoning") &&
            !request["grammar_after_reasoning"].is_boolean()) {
            emit_error_with_code(request_id, "invalid_argument",
                                 "grammar_after_reasoning must be a boolean");
            return false;
        }
        spec.grammar_delayed = request.value("grammar_after_reasoning", false);
        if (spec.grammar_delayed && (!spec.grammar || spec.thinking.close_token < 0)) {
            emit_error_with_code(request_id, "invalid_argument",
                                 "delayed grammar requires a close token");
            return false;
        }
        if (spec.grammar && spec.thinking.tokens > 0 && !spec.grammar_delayed) {
            emit_error_with_code(request_id, "invalid_argument",
                                 "grammar with reasoning requires delayed grammar");
            return false;
        }
        spec.grammar_close_token = spec.thinking.close_token;
        if (spec.grammar) {
            if (!request.contains("bitmask") || !request["bitmask"].is_string()) {
                emit_error_with_code(request_id, "invalid_argument",
                                     "invalid initial bitmask");
                return false;
            }
            const std::size_t expected =
                ((runtime.model_info().vocab_size + 31u) / 32u) * sizeof(uint32_t);
            if (!decode_mask_words(request["bitmask"].get<std::string>(), expected,
                                   spec.token_bitmask)) {
                emit_error_with_code(request_id, "invalid_argument",
                                     "invalid initial bitmask");
                return false;
            }
            spec.has_token_bitmask = true;
        } else if (request.contains("bitmask")) {
            emit_error_with_code(request_id, "invalid_argument", "invalid initial bitmask");
            return false;
        }
    }

    MtpPrefillCancel prefill_cancel;
    prefill_cancel.input = &input;
    prefill_cancel.external_id = external_request_id;
    shelf::engine::ServePrefillHooks hooks;
    hooks.should_stop = [&prefill_cancel]() { return mtp_prefill_should_stop(&prefill_cancel); };
    std::vector<ServeEvent> events;
    auto admitted = serve.admit(spec, hooks, events);
    if (!admitted.ok()) {
        emit_error_with_code(request_id, "internal", admitted.status().message());
        return false;
    }
    if (!admitted.value().accepted) {
        emit_error_with_code(request_id, admitted.value().code, admitted.value().message);
        return false;
    }
    emit_events(events);
    return true;
}

}  // namespace

int run_serve_stdio(shelf::engine::ComputeSession& runtime, DFlashServeContext& dflash) {
    shelf::engine::ServeConfig config;
    config.dflash_enabled = dflash.enabled;
    config.dflash_adaptive = dflash.adaptive;
    config.dflash_config = dflash.config;
    config.dflash_model_dir = dflash.model_dir;
    config.dflash_drafts = dflash.drafts;
    auto controller_result = shelf::engine::ServeController::create(runtime, config);
    if (!controller_result.ok()) {
        fprintf(stderr, "serve controller create failed: %s\n",
                controller_result.status().message().c_str());
        return 1;
    }
    std::unique_ptr<shelf::engine::ServeController> serve = controller_result.release();

    fprintf(stderr, "SHELF_SERVE_READY\n");
    fflush(stderr);
    JsonlInput input;
    bool exit_requested = false;
    while (!exit_requested) {
        const bool mtp_idle = !serve->mtp_enabled() || !serve->has_mtp_active();
        const bool all_waiting = serve->all_waiting_for_mask();
        jsonl_poll_and_read(input, all_waiting && mtp_idle);
        std::string line;
        while (jsonl_take_line(input, line)) {
            if (line.empty()) continue;
            nlohmann::json request;
            try {
                request = nlohmann::json::parse(line);
            } catch (const std::exception&) {
                emit_error(nullptr, "malformed json");
                continue;
            }
            if (!request.is_object()) {
                emit_error(nullptr, "request must be a json object");
                continue;
            }
            const std::string op = request.value("op", std::string());
            if (op == "ping") {
                emit_json(nlohmann::json{
                    {"event", "pong"},
                    {"protocol_version", 1},
                    {"capabilities", {{"prefix_cache", runtime.prefix_cache_active()}}},
                });
                continue;
            }
            if (op == "stats") {
                const shelf::engine::ServeStats stats = serve->stats();
                nlohmann::json prefix = {
                    {"enabled", stats.prefix_enabled},
                    {"reuse_slots", stats.reuse_slots},
                    {"entries", stats.entries},
                    {"saves", stats.saves},
                    {"hits", stats.hits},
                    {"misses", stats.misses},
                    {"restores", stats.restores},
                    {"evictions", stats.evictions},
                };
                emit_json(nlohmann::json{
                    {"event", "stats"},
                    {"prefix", std::move(prefix)},
                    {"kv",
                     {
                         {"page_tokens", stats.page_tokens},
                         {"total_pages", stats.total_pages},
                         {"used_pages", stats.used_pages},
                         {"free_pages", stats.free_pages},
                         {"capacity_tokens", stats.total_pages * stats.page_tokens},
                         {"free_tokens", stats.free_pages * stats.page_tokens},
                         {"active_requests", stats.active_requests},
                         {"queued_requests", stats.queued_requests},
                         {"banker_safe", stats.banker_safe},
                         {"overcommit_pages", stats.overcommit_pages},
                     }},
                });
                continue;
            }
            if (op == "shutdown") {
                std::vector<ServeEvent> events;
                serve->request_shutdown(events);
                emit_events(events);
                emit_json(nlohmann::json{{"event", "shutdown"}});
                exit_requested = true;
                break;
            }
            if (op == "cancel") {
                if (!request.contains("request_id") ||
                    !request["request_id"].is_number_integer()) {
                    emit_error(nullptr, "missing or invalid request_id");
                    continue;
                }
                const nlohmann::json request_id = request["request_id"];
                const int64_t id = request_id.get<int64_t>();
                std::vector<ServeEvent> events;
                if (serve->cancel(id, events)) {
                    emit_events(events);
                    continue;
                }
                if (serve->is_recent_terminal(id)) continue;
                emit_error(request_id, "unknown or mismatched request_id");
                continue;
            }
            if (op == "draft_masks") {
                if (!request.contains("request_id") ||
                    !request["request_id"].is_number_integer()) {
                    emit_error(nullptr, "missing or invalid request_id");
                    continue;
                }
                const nlohmann::json request_id = request["request_id"];
                const int64_t id = request_id.get<int64_t>();
                if (!request.contains("round_id") ||
                    !request["round_id"].is_number_unsigned()) {
                    emit_error_with_code(request_id, "invalid_argument",
                                         "invalid draft round id");
                    std::vector<ServeEvent> events;
                    serve->abandon_inflight(id, events);
                    emit_events(events);
                    continue;
                }
                const uint64_t round = request["round_id"].get<uint64_t>();
                bool pending = false;
                std::size_t count = 0;
                std::size_t words = 0;
                const shelf::Status shape = serve->draft_mask_request(id, round, pending,
                                                                      count, words);
                if (!shape.ok()) {
                    emit_error_with_code(request_id, "invalid_argument",
                                         "invalid draft masks");
                    continue;
                }
                if (!pending) continue;
                bool valid = request.contains("valid_prefix") &&
                             request["valid_prefix"].is_number_unsigned() &&
                             request.contains("bitmasks") &&
                             request["bitmasks"].is_array();
                uint64_t prefix = valid ? request["valid_prefix"].get<uint64_t>() : 0;
                valid = valid && prefix <= count && request["bitmasks"].size() == prefix + 1u;
                std::vector<uint32_t> masks((count + 1u) * words, 0xffffffffu);
                if (valid) {
                    for (size_t row = 0; row <= prefix; ++row) {
                        std::vector<uint32_t> row_words;
                        const auto& encoded = request["bitmasks"][row];
                        if (!encoded.is_string() ||
                            !decode_mask_words(encoded.get<std::string>(),
                                               words * sizeof(uint32_t), row_words)) {
                            valid = false;
                            break;
                        }
                        std::memcpy(masks.data() + row * words, row_words.data(),
                                    row_words.size() * sizeof(uint32_t));
                    }
                }
                if (!valid) {
                    emit_error_with_code(request_id, "invalid_argument",
                                         "invalid draft masks");
                    std::vector<ServeEvent> events;
                    serve->abandon_inflight(id, events);
                    emit_events(events);
                    continue;
                }
                const shelf::Status applied = serve->apply_draft_masks(
                    id, round, masks.data(), words, static_cast<uint32_t>(prefix));
                if (!applied.ok()) {
                    emit_error_with_code(request_id, "invalid_argument",
                                         "invalid draft masks");
                    std::vector<ServeEvent> events;
                    serve->abandon_inflight(id, events);
                    emit_events(events);
                }
                continue;
            }
            if (op == "mask") {
                if (!request.contains("request_id") ||
                    !request["request_id"].is_number_integer()) {
                    emit_error(nullptr, "missing or invalid request_id");
                    continue;
                }
                const nlohmann::json request_id = request["request_id"];
                const int64_t id = request_id.get<int64_t>();
                if (!request.contains("bitmask") || !request["bitmask"].is_string()) {
                    emit_error_with_code(request_id, "invalid_argument",
                                         "unexpected or invalid bitmask");
                    std::vector<ServeEvent> events;
                    serve->abandon_inflight(id, events);
                    emit_events(events);
                    continue;
                }
                const std::size_t expected =
                    ((runtime.model_info().vocab_size + 31u) / 32u) * sizeof(uint32_t);
                std::vector<uint32_t> words;
                if (!decode_mask_words(request["bitmask"].get<std::string>(), expected,
                                       words)) {
                    emit_error_with_code(request_id, "invalid_argument",
                                         "unexpected or invalid bitmask");
                    std::vector<ServeEvent> events;
                    serve->abandon_inflight(id, events);
                    emit_events(events);
                    continue;
                }
                const shelf::Status applied = serve->apply_token_mask(id, words);
                if (!applied.ok()) {
                    emit_error_with_code(request_id, "invalid_argument",
                                         "unexpected or invalid bitmask");
                    std::vector<ServeEvent> events;
                    serve->abandon_inflight(id, events);
                    emit_events(events);
                }
                continue;
            }
            if (op == "generate") {
                parse_generate(*serve, runtime, input, request);
                continue;
            }
            emit_error(request.contains("request_id") ? request["request_id"]
                                                      : nlohmann::json(),
                       "unknown op");
        }
        if (exit_requested) break;
        if ((serve->mtp_enabled() && serve->has_mtp_active()) || serve->has_inflight()) {
            std::vector<ServeEvent> events;
            serve->step(events);
            emit_events(events);
        } else if (input.eof) {
            break;
        }
    }
    if (serve->has_inflight()) {
        std::vector<ServeEvent> events;
        serve->cancel_all_inflight(events);
        emit_events(events);
    }
    serve->write_final_stats();
    serve.reset();
    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

}  // namespace shelf::app
