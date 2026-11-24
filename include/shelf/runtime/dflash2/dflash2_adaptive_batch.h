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

inline uint32_t dflash2_adaptive_max_k(uint32_t configured_drafts) {
    return std::min(configured_drafts, kDFlash2AdaptiveMaxK);
}

inline void select_dflash2_adaptive(std::span<RuntimeRequest*> active,
                                    const DFlash2BatcherBinding& binding,
                                    const ContinuousBatcherConfig& config,
                                    uint64_t& round) {
    ++round;
    const uint32_t max_k = dflash2_adaptive_max_k(binding.num_drafts);
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
    const uint32_t capped_k = candidates.size() > 16u ? std::min(max_k, 4u) : max_k;
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const RuntimeRequest* a, const RuntimeRequest* b) {
                         if (a->dflash->adaptive.has_sample != b->dflash->adaptive.has_sample)
                             return a->dflash->adaptive.has_sample;
                         if (a->dflash->adaptive.acc_ema != b->dflash->adaptive.acc_ema)
                             return a->dflash->adaptive.acc_ema > b->dflash->adaptive.acc_ema;
                         return a->id < b->id;
                     });
    const uint32_t cap = dflash2_policy_max_sessions();
    uint32_t admitted = 0u;
    for (RuntimeRequest* request : candidates) {
        DFlash2RequestSession& session = *request->dflash;
        if (max_k == 0u || admitted >= cap || ordinary + admitted >= cap) {
            session.adaptive_rows = 0u;
            continue;
        }
        if (session.adaptive.current_k > capped_k) {
            session.adaptive.current_k = capped_k;
            session.adaptive.rounds_since_switch = 0;
        }
        if (session.adaptive.current_k == 0u) {
            session.adaptive.current_k = std::min(capped_k, kDFlash2AdaptiveInitialK);
        }
        const uint32_t k = session.adaptive.current_k;
        const uint32_t remaining = request->max_new_tokens - request->generated_tokens;
        const uint32_t position = request->sequence.position;
        const uint32_t room = request->sequence.max_seq_len > position
                                  ? request->sequence.max_seq_len - position : 0u;
        uint32_t limit = remaining <= 1u || room < 2u ? 0u
            : std::min(k, std::min(remaining - 1u, room - 1u));
        const uint32_t allowance = thinking_allowance(*request);
        if (allowance != kDFlash2NoThinkingAllowance) limit = std::min(limit, allowance);
        if (request->prefix_capture != 0u && config.page_tokens != 0u && limit > 0u) {
            limit = std::min(limit, config.page_tokens - position % config.page_tokens - 1u);
        }
        session.adaptive_rows = limit ? limit + 1u : 0u;
        if (session.adaptive_rows != 0u) ++admitted;
        if (binding.stats != nullptr) {
            auto& counts = binding.stats->adaptive_k_counts;
            if (counts.size() < static_cast<std::size_t>(max_k) + 1u)
                counts.resize(static_cast<std::size_t>(max_k) + 1u);
            ++counts[session.adaptive_rows ? session.adaptive_rows - 1u : 0u];
        }
    }
    if (binding.stats != nullptr && !candidates.empty()) ++binding.stats->adaptive_rounds;
}

inline uint32_t observe_dflash2_adaptive(DFlash2RequestSession& session,
                                         uint32_t accepted, uint32_t batch_size,
                                         uint32_t configured_drafts) {
    const uint32_t max_k = dflash2_adaptive_max_k(configured_drafts);
    const uint32_t capped_k = batch_size > 16u ? std::min(max_k, 4u) : max_k;
    return session.adaptive.observe(static_cast<double>(accepted), batch_size, capped_k);
}

}  // namespace shelf::runtime
