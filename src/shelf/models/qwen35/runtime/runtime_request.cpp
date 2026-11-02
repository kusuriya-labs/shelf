#include <shelf/models/qwen35/runtime/runtime_request.h>
#include <shelf/models/qwen35/state/state_types.h>

namespace ps {
namespace qwen35 {
namespace runtime {

Status admit_request(
    RuntimeRequest& request,
    SequenceSlotPool& seq_pool,
    GdnStatePool& gdn_pool,
    PagedKVPool& kv_pool,
    hipStream_t stream) {
    if (request.state != RequestState::Queued) {
        return Status::invalid_state("admit_request requires Queued state", __FILE__, __LINE__);
    }
    if (request.sequence.is_allocated()) {
        return Status::invalid_state("admit_request on allocated sequence", __FILE__, __LINE__);
    }
    if (request.max_kv_tokens == 0) {
        return Status::invalid_argument(
            "request has no KV capacity metadata", __FILE__, __LINE__);
    }
    auto created = create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, request.max_kv_tokens, stream);
    if (!created.ok()) {
        return created.status();
    }
    runtime_install_sequence_state(request.sequence, created.release());
    request.state = RequestState::Active;
    return Status::make_ok();
}

namespace {

void mark_finished(RuntimeRequest& request, FinishReason reason) {
    request.state = RequestState::Finished;
    request.finish_reason = reason;
    request.pending_decode_token = -1;
}

}

bool thinking_budget_enabled(const RuntimeRequest& request) noexcept {
    return request.thinking.tokens != 0 && request.thinking.close_token >= 0;
}

uint32_t thinking_allowance(const RuntimeRequest& request) noexcept {
    if (!thinking_budget_enabled(request) || request.thinking_closed) {
        return kDFlash2NoThinkingAllowance;
    }
    return request.generated_tokens >= request.thinking.tokens
               ? 0u
               : request.thinking.tokens - request.generated_tokens;
}

void note_generated_token(RuntimeRequest& request, int32_t token) noexcept {
    if (!request.thinking_closed && request.thinking.close_token >= 0 &&
        token == request.thinking.close_token) {
        request.thinking_closed = true;
    }
}

int32_t enforce_thinking_budget(RuntimeRequest& request, int32_t token) noexcept {
    note_generated_token(request, token);
    if (request.thinking_closed) return token;
    if (!thinking_budget_enabled(request)) return token;
    if (request.generated_tokens >= request.thinking.tokens) {
        token = request.thinking.close_token;
        request.thinking_closed = true;
    }
    return token;
}

CommitAction commit_sampled_token(
    RuntimeRequest& request,
    int32_t token,
    const StopTokens& stop_tokens) {
    token = enforce_thinking_budget(request, token);
    ++request.generated_tokens;
    request.generated.push_back(token);
    if (is_stop_token(stop_tokens, token)) {
        mark_finished(request, FinishReason::Eos);
        return CommitAction::Finished;
    }
    if (request.generated_tokens >= request.max_new_tokens) {
        mark_finished(request, FinishReason::MaxNewTokens);
        return CommitAction::Finished;
    }
    request.pending_decode_token = token;
    return CommitAction::Continue;
}

CommitAction commit_final_prefill_without_sample(RuntimeRequest& request) {
    if (request.generated_tokens >= request.max_new_tokens) {
        mark_finished(request, FinishReason::MaxNewTokens);
        return CommitAction::Finished;
    }
    return CommitAction::Continue;
}

Status record_sampled_token(RuntimeRequest& request, int32_t token) {
    if (token < 0) {
        return Status::invalid_argument("invalid sampled token", __FILE__, __LINE__);
    }
    ++request.generated_tokens;
    request.generated.push_back(token);
    return Status::make_ok();
}

}
}
}
