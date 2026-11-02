#include "stdio_server.h"
#include "private_target_weights.h"

#include "compute_common.h"

#include <shelf/models/qwen35/dflash2/context_state.h>
#include <shelf/models/qwen35/runtime/decode_backend.h>
#include <shelf/models/qwen35/runtime/runtime_request.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/core/memory/size_format.h>
#include <nlohmann/json.hpp>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
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

void emit_token(int64_t request_id, int32_t token_id) {
    emit_json(nlohmann::json{
        {"event", "token"}, {"request_id", request_id}, {"token_id", token_id}});
}

constexpr std::size_t kRecentTerminalLimit = 256;

struct ServeGeneration {
    int64_t external_request_id = 0;
    uint64_t internal_request_id = 0;
    std::size_t emitted_tokens = 0;
    uint32_t max_new_tokens = 0;
};

struct ServeState {
    std::map<int64_t, ServeGeneration> inflight;
    std::deque<int64_t> recent_terminal_ids;
    uint32_t max_inflight = 1;
    bool exit_requested = false;
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

void serve_emit_tokens(shelf::app::Qwen35ComputeRuntime& runtime,
                       ServeGeneration& generation) {
    const auto* request = runtime.request(generation.internal_request_id);
    if (request == nullptr) return;
    while (generation.emitted_tokens < request->generated.size()) {
        emit_token(generation.external_request_id,
                   request->generated[generation.emitted_tokens]);
        ++generation.emitted_tokens;
    }
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
    if (state.inflight.size() >= state.max_inflight) {
        emit_error(request_id, "compute concurrency limit reached");
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

    shelf::Result<uint64_t> submit_result =
        runtime.submit(std::move(input_ids), max_new_tokens, sampling, thinking);
    if (!submit_result.ok()) {
        emit_error_with_code(request_id,
                             status_error_code(submit_result.status().code()),
                             submit_result.status().message());
        return false;
    }
    generation.internal_request_id = submit_result.release();
    state.inflight.emplace(external_request_id, std::move(generation));
    return true;
}

void serve_handle_command(shelf::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                          DFlashServeContext& dflash, const std::string& line) {
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
        emit_json(nlohmann::json{
            {"event", "stats"},
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
        } else if (serve_is_recent_terminal(state, id)) {
            return;
        } else {
            emit_error(request_id, "unknown or mismatched request_id");
        }
        return;
    }
    if (op == "generate") {
        serve_parse_generate(runtime, state, dflash, request);
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
    shelf::weights::MatrixWeight* private_lm_head = nullptr;
    shelf::weights::MatrixWeight* private_embed = nullptr;
    if (runtime.config().tp > 1) {
        auto* tp = runtime.tp_coordinator();
        if (tp != nullptr && tp->tp_size() > 0) {
            auto& rank0 = tp->rank(0);
            private_lm_head = &rank0.private_lm_head();
            private_embed = &rank0.private_embed_tokens();
        }
    }
    if (dflash.enabled) {
        if (runtime.config().max_concurrent_requests >
            shelf::qwen35::runtime::kDFlash2SpecMaxVerifyRows /
                static_cast<uint32_t>(dflash.config.block_size)) {
            fprintf(stderr,
                    "dflash2 serve: max_concurrent_requests %u exceeds verify row cap %u "
                    "for block_size %zu\n",
                    runtime.config().max_concurrent_requests,
                    shelf::qwen35::runtime::kDFlash2SpecMaxVerifyRows,
                    dflash.config.block_size);
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

        const auto& text_config = runtime.model().text_config();
        const uint32_t batch_max_rows = std::min<uint32_t>(
            runtime.config().max_concurrent_requests *
                static_cast<uint32_t>(dflash.config.block_size),
            shelf::qwen35::runtime::kDFlash2SpecMaxVerifyRows);
        const bool serve_tp_sharded = runtime.config().tp > 1;
        shelf::qwen35::dflash2::DFlash2ExecutorConfig draft_config;
        draft_config.max_rows = static_cast<uint32_t>(dflash.config.block_size);
        draft_config.max_batch_rows = batch_max_rows;
        draft_config.max_context_rows = static_cast<uint32_t>(dflash.config.sliding_window);
        draft_config.target_lm_head = serve_tp_sharded ? private_lm_head
                                       : &runtime.model().weights().lm_head;
        draft_config.target_embed_tokens =
            serve_tp_sharded ? private_embed
                             : &runtime.model().weights().embed_tokens;
        auto draft_result = shelf::qwen35::dflash2::create_dflash2_executor(
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
                bindings.push_back(binding);
            }
            auto batch_result =
                shelf::qwen35::runtime::create_dflash2_spec_batch_context(
                    bindings, batch_max_rows);
            if (!batch_result.ok()) {
                fprintf(stderr, "dflash2 batch context create failed: %s\n",
                        batch_result.status().message().c_str());
                return 1;
            }
            dflash.batch = batch_result.release();
            dflash.batch_ready = true;
        }
        {
            shelf::qwen35::runtime::DFlash2SpecDecoderConfig session_config;
            session_config.num_drafts = dflash.drafts;
            session_config.eos_tokens = text_config.stop_tokens;
            auto sessions_result = shelf::qwen35::runtime::DFlash2SessionPool::create(
                runtime.executor(), &dflash.executor, runtime.gdn_pool(),
                runtime.arena(), session_config,
                runtime.config().max_concurrent_requests, runtime.stream(), serve_rank_ptr);
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
    fprintf(stderr, "SHELF_SERVE_READY\n");
    fflush(stderr);
    JsonlInput input;
    ServeState state;
    state.max_inflight = runtime.config().max_concurrent_requests;
    while (!state.exit_requested) {
        jsonl_poll_and_read(input, state.inflight.empty());
        std::string line;
        while (jsonl_take_line(input, line)) {
            if (line.empty()) continue;
            serve_handle_command(runtime, state, dflash, line);
            if (state.exit_requested) break;
        }
        if (state.exit_requested) break;
        if (!state.inflight.empty()) {
            if (input.eof) {
                serve_cancel_all_inflight(runtime, state);
            } else {
                serve_step(runtime, state);
            }
        } else if (input.eof) {
            break;
        }
    }
    if (!state.inflight.empty()) {
        serve_cancel_all_inflight(runtime, state);
    }

    const char* serve_stats_env = std::getenv("SHELF_DFLASH2_SERVE_STATS");
    if (dflash.enabled &&
        serve_stats_env != nullptr && serve_stats_env[0] != '\0' &&
        serve_stats_env[0] != '0') {
        const shelf::qwen35::runtime::DFlash2BatcherStats& t = dflash.stats;
        const float rounds = static_cast<float>(t.rounds);
        const float mean =
            t.rounds > 0u ? static_cast<float>(t.accepted_drafts) / rounds : 0.0f;
        const float full_rate =
            t.rounds > 0u ? static_cast<float>(t.full_accepts) / rounds : 0.0f;
        const float emitted =
            t.rounds > 0u ? static_cast<float>(t.generated_tokens) / rounds : 0.0f;
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
            shelf::qwen35::dflash2::dflash2_executor_shutdown(dflash.executor);
        dflash.executor_ready = false;
        if (!draft_st.ok()) {
            fprintf(stderr, "dflash2 executor shutdown failed: %s\n",
                    draft_st.message().c_str());
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
