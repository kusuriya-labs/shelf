#include <shelf/models/qwen35/runtime/token_budget_scheduler.h>
#include <cstdio>
#include <vector>

using namespace shelf::qwen35::runtime;

int main() {
    RuntimeRequest waiting, free, prefill;
    for (auto* r : {&waiting, &free, &prefill}) {
        r->state = RequestState::Active;
        r->sequence.max_seq_len = 32;
    }
    waiting.input_tokens = {1}; waiting.sequence.position = 1;
    waiting.pending_decode_token = 1; waiting.waiting_for_token_bitmask = true;
    free.input_tokens = {1}; free.sequence.position = 1; free.pending_decode_token = 2;
    prefill.input_tokens = {1, 2}; prefill.max_new_tokens = 4;
    uint32_t mask = 1; prefill.token_bitmask = &mask;
    std::vector<const RuntimeRequest*> active{&waiting, &free, &prefill};
    auto plan = schedule_requests(active, 16, 0, 16, nullptr);
    if (!plan.ok() || plan.value().scheduled_requests.size() != 2 ||
        plan.value().scheduled_requests[0].sequence != &free.sequence ||
        plan.value().scheduled_requests[0].token_bitmask != nullptr ||
        plan.value().scheduled_requests[1].sequence != &prefill.sequence ||
        plan.value().scheduled_requests[1].token_bitmask != &mask) return 1;
    waiting.waiting_for_token_bitmask = false;
    waiting.token_bitmask = &mask;
    plan = schedule_requests(active, 16, 0, 16, nullptr);
    if (!plan.ok() || plan.value().scheduled_requests.size() != 3 ||
        plan.value().scheduled_requests[0].token_bitmask != &mask) return 2;

    // A verify request has one independent mask per output row.
    shelf::state::PagedSequenceState seq;
    seq.slot = 1; seq.generation = 1; seq.max_seq_len = 32;
    int32_t tokens[] = {1, 2, 3};
    shelf::scheduling::ScheduledRequest verify;
    verify.sequence = &seq; verify.handle = seq.request_handle();
    verify.execution_class = shelf::runtime::ExecutionClass::SPEC_VERIFY;
    verify.num_tokens = 3; verify.num_output_rows = 3;
    verify.compute_logits = true; verify.sample = true;
    verify.token_bitmask = &mask; verify.token_bitmask_row_stride = 2;
    shelf::scheduling::ScheduledBatch batch;
    batch.token_ids = tokens; batch.requests = std::span(&verify, 1);
    batch.num_tokens = 3; batch.num_requests = 1; batch.num_verify_requests = 1;
    batch.speculative_verify = true;
    if (!shelf::scheduling::validate_scheduled_batch(batch, 0, 0).ok()) return 3;
    verify.token_bitmask_row_stride = 0;
    if (shelf::scheduling::validate_scheduled_batch(batch, 0, 0).ok()) return 4;
    verify.token_bitmask = nullptr; verify.token_bitmask_row_stride = 2;
    if (shelf::scheduling::validate_scheduled_batch(batch, 0, 0).ok()) return 5;
    std::puts("token mask scheduling passed");
}
