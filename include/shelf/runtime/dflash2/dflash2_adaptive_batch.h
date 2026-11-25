#pragma once

#include <shelf/runtime/continuous_batcher.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <vector>

namespace shelf::runtime {

inline uint32_t dflash2_policy_max_sessions() {
    static const uint32_t value = [] {
        const char* text = std::getenv("SHELF_DFLASH2_POLICY_MAX_SESSIONS");
        if (text == nullptr || *text == '\0' || *text == '-') return 32u;
        char* end = nullptr;
        const unsigned long n = std::strtoul(text, &end, 10);
        return *end == '\0' && n <= std::numeric_limits<uint32_t>::max()
                   ? static_cast<uint32_t>(n) : 32u;
    }();
    return value;
}

inline bool dflash2_can_capture_boundary(const DFlash2SpecDecoder& decoder) {
    if (decoder.tp_hook || decoder.gdn_ranks.size() != 1u) return false;
    const auto& pools = decoder.gdn_ranks.front().spec_pools;
    return !pools.qsa && !pools.ple;
}

inline uint32_t select_dflash2_adaptive(std::span<RuntimeRequest*> active,
                                       const DFlash2BatcherBinding& binding,
                                       const ContinuousBatcherConfig& config,
                                       uint64_t& round) {
    ++round;
    std::vector<RuntimeRequest*> candidates;
    uint32_t ordinary = 0u;
    for (RuntimeRequest* request : active) {
        if (request->state != RequestState::Active ||
            request->finish_reason != FinishReason::None ||
            request->generated_tokens >= request->max_new_tokens ||
            request->waiting_for_token_bitmask) continue;
        if (request->dflash == nullptr || !request->dflash->ready) {
            if (request->pending_decode_token >= 0) ++ordinary;
            continue;
        }
        candidates.push_back(request);
    }
    const uint32_t batch_size = ordinary + static_cast<uint32_t>(candidates.size());
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const RuntimeRequest* a, const RuntimeRequest* b) {
                         const bool sampled_a = a->dflash->adaptive.rounds != 0u;
                         const bool sampled_b = b->dflash->adaptive.rounds != 0u;
                         if (sampled_a != sampled_b) return sampled_a;
                         if (a->dflash->adaptive.acc_ema != b->dflash->adaptive.acc_ema)
                             return a->dflash->adaptive.acc_ema > b->dflash->adaptive.acc_ema;
                         return a->id < b->id;
                     });
    const uint32_t cap = dflash2_policy_max_sessions();
    uint32_t admitted = 0u;
    for (RuntimeRequest* request : candidates) {
        DFlash2RequestSession& session = *request->dflash;
        if (binding.num_drafts == 0u || admitted >= cap || ordinary + admitted >= cap) {
            session.adaptive_rows = 0u;
            continue;
        }
        const uint32_t k = session.adaptive.apply_batch_limit(batch_size);
        const uint32_t remaining = request->max_new_tokens - request->generated_tokens;
        const uint32_t position = request->sequence.position;
        const uint32_t room = request->sequence.max_seq_len > position
                                  ? request->sequence.max_seq_len - position : 0u;
        uint32_t limit = remaining <= 1u || room < 2u ? 0u
            : std::min(k, std::min(remaining - 1u, room - 1u));
        const uint32_t allowance = thinking_allowance(*request);
        if (allowance != kDFlash2NoThinkingAllowance) limit = std::min(limit, allowance);
        if (request->prefix_capture != 0u && config.page_tokens != 0u && limit > 0u &&
            (!session.decoder || !dflash2_can_capture_boundary(*session.decoder))) {
            limit = std::min(limit, config.page_tokens - position % config.page_tokens - 1u);
        }
        session.adaptive_rows = room != 0u ? limit + 1u : 0u;
        if (session.adaptive_rows != 0u) ++admitted;
        if (binding.stats != nullptr) {
            auto& counts = binding.stats->adaptive_k_counts;
            if (counts.size() < static_cast<std::size_t>(binding.num_drafts) + 1u)
                counts.resize(static_cast<std::size_t>(binding.num_drafts) + 1u);
            ++counts[session.adaptive_rows ? session.adaptive_rows - 1u : 0u];
        }
    }
    if (binding.stats != nullptr && !candidates.empty()) ++binding.stats->adaptive_rounds;
    return batch_size;
}

}
