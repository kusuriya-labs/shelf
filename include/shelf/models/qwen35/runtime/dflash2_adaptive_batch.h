#pragma once

#include <shelf/models/qwen35/runtime/continuous_batcher.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <vector>

namespace shelf::qwen35::runtime {

inline uint32_t dflash2_policy_max_sessions() {
    static const uint32_t value = [] {
        const char* text = std::getenv("SHELF_DFLASH2_POLICY_MAX_SESSIONS");
        if (text == nullptr || *text == '\0' || *text == '-') return 12u;
        char* end = nullptr;
        const unsigned long n = std::strtoul(text, &end, 10);
        return *end == '\0' && n <= std::numeric_limits<uint32_t>::max()
                   ? static_cast<uint32_t>(n) : 12u;
    }();
    return value;
}

inline void select_dflash2_adaptive(std::span<RuntimeRequest*> active,
                                     const DFlash2BatcherBinding& binding,
                                     const ContinuousBatcherConfig& config,
                                     uint64_t& round) {
    constexpr double threshold = 0.6;
    constexpr uint32_t probe_interval = 64;
    std::vector<RuntimeRequest*> candidates;
    uint32_t eligible = 0;
    uint32_t normal = 0;
    RuntimeRequest* probe = nullptr;
    ++round;
    for (RuntimeRequest* request : active) {
        if (request->state != RequestState::Active ||
            request->finish_reason != FinishReason::None ||
            request->generated_tokens >= request->max_new_tokens ||
            request->waiting_for_token_bitmask) continue;
        if (request->dflash == nullptr || !request->dflash->ready) {
            if (request->pending_decode_token >= 0) ++normal;
            continue;
        }
        auto& session = *request->dflash;
        const bool enabled = session.adaptive_accepted_ema >= threshold;
        if (enabled) session.adaptive_probe_age = 0;
        else if (session.adaptive_probe_age < probe_interval) ++session.adaptive_probe_age;
        if (!enabled && session.adaptive_probe_age >= probe_interval && probe == nullptr)
            probe = request;
        session.adaptive_rows = enabled ? binding.num_drafts + 1u : 0u;
        if (enabled) ++eligible;
        candidates.push_back(request);
    }
    if (probe != nullptr && eligible < dflash2_policy_max_sessions()) {
        probe->dflash->adaptive_rows = binding.num_drafts + 1u;
        ++eligible;
    }
    const bool too_many = eligible + normal > dflash2_policy_max_sessions();
    for (RuntimeRequest* request : candidates) {
        auto& session = *request->dflash;
        if (too_many) session.adaptive_rows = 0;
        if (session.adaptive_rows != 0u) {
            const uint32_t remaining = request->max_new_tokens - request->generated_tokens;
            const uint32_t position = request->sequence.position;
            const uint32_t room = request->sequence.max_seq_len > position
                                      ? request->sequence.max_seq_len - position : 0u;
            uint32_t limit = remaining <= 1u || room < 2u ? 0u
                : std::min(binding.num_drafts, std::min(remaining - 1u, room - 1u));
            const uint32_t allowance = thinking_allowance(*request);
            if (allowance != kDFlash2NoThinkingAllowance) limit = std::min(limit, allowance);
            if (request->prefix_capture != 0u && config.page_tokens != 0u && limit > 0u) {
                limit = std::min(limit, config.page_tokens - position % config.page_tokens - 1u);
            }
            session.adaptive_rows = limit ? limit + 1u : 0u;
            if (session.adaptive_rows != 0u && session.adaptive_accepted_ema < threshold)
                session.adaptive_probe_age = 0;
        }
        if (binding.stats != nullptr) {
            auto& counts = binding.stats->adaptive_k_counts;
            if (counts.empty()) counts.resize(binding.num_drafts + 1u);
            ++counts[session.adaptive_rows ? session.adaptive_rows - 1u : 0u];
        }
    }
    if (binding.stats != nullptr && !candidates.empty()) ++binding.stats->adaptive_rounds;
}
}
