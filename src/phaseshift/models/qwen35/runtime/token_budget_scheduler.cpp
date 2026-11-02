#include <phaseshift/models/qwen35/runtime/token_budget_scheduler.h>

#include <algorithm>
#include <span>

namespace ps {
namespace qwen35 {
namespace runtime {

Result<SchedulePlan> schedule_requests(
    std::span<const RuntimeRequest*> active,
    uint32_t max_scheduled_tokens,
    uint32_t max_scheduled_requests,
    uint32_t page_tokens,
    KVBankerState* kv) {
    SchedulePlan plan;

    auto budget_left = [&]() -> uint32_t {
        return max_scheduled_tokens - static_cast<uint32_t>(plan.token_ids.size());
    };
    auto request_cap_reached = [&]() -> bool {
        return max_scheduled_requests != 0 &&
               static_cast<uint32_t>(plan.scheduled_requests.size()) >=
                   max_scheduled_requests;
    };
    auto grant_allowed = [&](std::size_t proc_idx, uint32_t pages) -> bool {
        if (kv == nullptr || pages == 0) {
            return true;
        }
        if (proc_idx >= kv->processes.size()) {
            return false;
        }
        return max_safe_grant_pages(*kv, proc_idx, pages) >= pages;
    };
    auto commit_grant = [&](std::size_t proc_idx, uint32_t pages) {
        if (kv == nullptr || pages == 0) {
            return;
        }
        if (proc_idx >= kv->processes.size()) {
            return;
        }
        kv->available_pages -= pages;
        kv->processes[proc_idx].allocated_pages += pages;
    };
    std::size_t decode_proc = 0;
    for (const RuntimeRequest* rp : active) {
        if (rp->state != RequestState::Active) continue;
        const std::size_t proc_idx = decode_proc++;
        if (rp->sequence.position < rp->input_tokens.size()) continue;
        if (rp->pending_decode_token < 0) continue;
        if (budget_left() < 1) break;
        if (request_cap_reached()) break;
        const uint32_t blocks = static_cast<uint32_t>(rp->sequence.block_table.size());
        const uint32_t delta =
            additional_kv_pages(rp->sequence.position, blocks, 1, page_tokens);
        if (!grant_allowed(proc_idx, delta)) {
            continue;
        }
        auto append_st = plan.append_decode(
            const_cast<PagedSequenceState*>(&rp->sequence),
            rp->pending_decode_token,
            rp->sequence.position,
            true,
            true,
            rp->sampling,
            rp->generated_tokens);
        if (!append_st.ok()) {
            return append_st;
        }
        commit_grant(proc_idx, delta);
    }

    std::size_t prefill_proc = 0;
    for (const RuntimeRequest* rp : active) {
        if (rp->state != RequestState::Active) continue;
        const std::size_t proc_idx = prefill_proc++;
        const uint32_t position = rp->sequence.position;
        if (position >= rp->input_tokens.size()) continue;
        if (budget_left() < 1) break;
        if (request_cap_reached()) break;
        const uint32_t remaining = static_cast<uint32_t>(rp->input_tokens.size()) - position;
        uint32_t take = remaining < budget_left() ? remaining : budget_left();
        const uint32_t end = position + take;
        if (end > rp->sequence.max_seq_len) {
            take = rp->sequence.max_seq_len - position;
            if (take == 0) continue;
        }
        const uint32_t blocks = static_cast<uint32_t>(rp->sequence.block_table.size());
        if (kv != nullptr) {
            const uint32_t requested_delta =
                additional_kv_pages(position, blocks, take, page_tokens);
            if (requested_delta > 0) {
                if (proc_idx >= kv->processes.size()) {
                    continue;
                }
                const uint32_t safe_pages =
                    max_safe_grant_pages(*kv, proc_idx, requested_delta);
                if (safe_pages == 0) {
                    continue;
                }
                const uint64_t max_end =
                    static_cast<uint64_t>(blocks + safe_pages) * page_tokens;
                if (max_end <= position) {
                    continue;
                }
                const uint64_t kv_take = max_end - position;
                if (static_cast<uint64_t>(take) > kv_take) {
                    take = static_cast<uint32_t>(kv_take);
                }
            }
        }
        const uint32_t final_end = position + take;
        const bool final_chunk = final_end == static_cast<uint32_t>(rp->input_tokens.size());
        const bool sample = final_chunk && rp->max_new_tokens > 0;
        const std::span<const int32_t> tokens(
            rp->input_tokens.data() + position, take);
        auto append_st = plan.append_prefill(
            const_cast<PagedSequenceState*>(&rp->sequence),
            tokens,
            position,
            sample,
            sample,
            rp->sampling,
            rp->generated_tokens);
        if (!append_st.ok()) {
            return append_st;
        }
        commit_grant(proc_idx,
                     additional_kv_pages(position, blocks, take, page_tokens));
    }

    plan.finalize();

    return plan;
}

}
}
}
