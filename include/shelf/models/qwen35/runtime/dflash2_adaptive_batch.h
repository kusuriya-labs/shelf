#pragma once

#include <shelf/models/qwen35/runtime/continuous_batcher.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace shelf::qwen35::runtime {

inline void select_dflash2_adaptive(std::span<RuntimeRequest*> active,
                                     const DFlash2BatcherBinding& binding,
                                     const ContinuousBatcherConfig& config,
                                     uint64_t& round) {
    std::vector<RuntimeRequest*> candidates;
    std::vector<adaptive::Acceptance> acceptance;
    std::vector<uint32_t> previous;
    std::vector<uint32_t> disabled;
    uint32_t batch = 0;
    for (RuntimeRequest* request : active) {
        if (request->state != RequestState::Active ||
            request->finish_reason != FinishReason::None ||
            request->generated_tokens >= request->max_new_tokens ||
            request->waiting_for_token_bitmask) continue;
        if (request->dflash != nullptr && request->dflash->ready) {
            ++batch;
            candidates.push_back(request);
            uint32_t limit = binding.num_drafts;
            const uint32_t remaining = request->max_new_tokens - request->generated_tokens;
            const uint32_t position = request->sequence.position;
            const uint32_t room = request->sequence.max_seq_len > position
                                      ? request->sequence.max_seq_len - position : 0u;
            if (remaining <= 1u || room < 2u) limit = 0u;
            else limit = std::min(limit, std::min(remaining - 1u, room - 1u));
            const uint32_t allowance = thinking_allowance(*request);
            if (allowance != kDFlash2NoThinkingAllowance) limit = std::min(limit, allowance);
            if (request->prefix_capture != 0u && config.page_tokens != 0u && limit > 0u) {
                const uint32_t to_boundary = config.page_tokens - position % config.page_tokens;
                limit = std::min(limit, to_boundary - 1u);
            }
            acceptance.push_back(request->dflash->acceptance);
            acceptance.back().conditional.resize(limit);
            previous.push_back(request->dflash->adaptive_was_probe
                                   ? request->dflash->adaptive_pre_probe_rows
                                   : request->dflash->adaptive_rows);
            disabled.push_back(limit == binding.num_drafts
                                   ? request->dflash->adaptive_probe_age : 0u);
            request->dflash->adaptive_was_probe = false;
        } else if (request->pending_decode_token >= 0) {
            ++batch;
        }
    }
    if (!candidates.empty()) {
        const uint32_t probe = adaptive::pick_probe(disabled, ++round, batch <= 4u ? 16u : 64u);
        const auto cost = [&binding, batch](uint32_t sessions, uint32_t rows) {
            return binding.adaptive_cost->milliseconds(batch, sessions, rows);
        };
        const auto decision = adaptive::choose(
            acceptance, batch, binding.num_drafts, cost, previous, 0.03, 8u,
            probe, binding.num_drafts + 1u);
        if (decision.predicted > 0.0) {
            uint32_t selected_sessions = 0;
            uint32_t selected_rows = 0;
            for (uint32_t rows : decision.rows) {
                selected_sessions += rows != 0u;
                selected_rows += rows;
            }
            if (binding.stats != nullptr) {
                auto& stats = *binding.stats;
                ++stats.adaptive_rounds;
                const double predicted_ms = cost(selected_sessions, selected_rows);
                stats.adaptive_predicted_ms += predicted_ms;
                stats.adaptive_predicted_tokens += decision.predicted * predicted_ms;
                if (stats.adaptive_k_counts.empty()) {
                    stats.adaptive_k_counts.resize(binding.num_drafts + 1u);
                }
                for (uint32_t rows : decision.rows) {
                    ++stats.adaptive_k_counts[rows ? rows - 1u : 0u];
                }
                stats.adaptive_k_counts[0] += batch - candidates.size();
            }
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                auto& session = *candidates[i]->dflash;
                session.adaptive_pre_probe_rows = previous[i];
                session.adaptive_rows = decision.rows[i];
                session.adaptive_was_probe = i == probe;
                if (decision.rows[i] < binding.num_drafts + 1u) ++session.adaptive_probe_age;
                else session.adaptive_probe_age = 0;
            }
        } else {
            for (RuntimeRequest* request : candidates) {
                request->dflash->adaptive_rows = 0;
                ++request->dflash->adaptive_probe_age;
            }
        }
    } else if (batch != 0u && binding.stats != nullptr) {
        const double predicted_ms = binding.adaptive_cost->milliseconds(batch, 0u, 0u);
        if (std::isfinite(predicted_ms) && predicted_ms > 0.0) {
            auto& stats = *binding.stats;
            ++stats.adaptive_rounds;
            stats.adaptive_predicted_ms += predicted_ms;
            stats.adaptive_predicted_tokens += batch;
            if (stats.adaptive_k_counts.empty()) {
                stats.adaptive_k_counts.resize(binding.num_drafts + 1u);
            }
            stats.adaptive_k_counts[0] += batch;
        }
    }

}

}
