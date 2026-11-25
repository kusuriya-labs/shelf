#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace shelf::runtime {

constexpr uint32_t kDFlash2AdaptiveInitialK = 5u;
constexpr uint32_t kDFlash2AdaptiveSwitchInterval = 8u;
constexpr double kDFlash2AdaptiveEmaAlpha = 0.25;

struct DFlash2AdaptiveController {
    uint32_t configured_drafts;
    uint32_t current_k;
    uint32_t min_switch_interval;
    double ema_alpha;
    double acc_ema = 0.0;
    uint64_t rounds = 0;
    uint64_t rounds_since_switch = 0;
    uint64_t switches = 0;

    explicit DFlash2AdaptiveController(uint32_t drafts = kDFlash2AdaptiveInitialK,
                                       uint32_t interval = kDFlash2AdaptiveSwitchInterval,
                                       double alpha = kDFlash2AdaptiveEmaAlpha)
        : configured_drafts(drafts), current_k(std::min(drafts, kDFlash2AdaptiveInitialK)),
          min_switch_interval(interval), ema_alpha(alpha) {
        if (interval == 0u || !std::isfinite(alpha) || alpha <= 0.0 || alpha > 1.0) {
            throw std::invalid_argument("DFlash2 adaptive K interval or EMA alpha is invalid");
        }
    }

    uint32_t ceiling(uint32_t batch_size) const {
        return batch_size > 16u ? std::min(configured_drafts, 4u) : configured_drafts;
    }

    uint32_t apply_batch_limit(uint32_t batch_size) {
        set_k(std::min(current_k, ceiling(batch_size)));
        return current_k;
    }

    void observe(uint32_t accepted, uint32_t verified, bool used_ngram, uint32_t batch_size) {
        if (used_ngram || verified == 0u || verified != current_k || accepted > verified ||
            batch_size == 0u) return;
        const double rate = static_cast<double>(accepted) / verified;
        acc_ema = rounds_since_switch == 0u ? rate : acc_ema + ema_alpha * (rate - acc_ema);
        ++rounds;
        if (++rounds_since_switch < min_switch_interval) return;

        const bool large_batch = batch_size > 16u;
        const double promote = current_k == 2u ? 0.45 : (large_batch ? 0.65 : 0.50);
        const double demote = large_batch ? 0.45 : 0.30;
        if (current_k < ceiling(batch_size) && acc_ema >= promote) {
            set_k(current_k + 1u);
        } else if (current_k > 2u && acc_ema * current_k < demote * (current_k - 1u)) {
            set_k(current_k - 1u);
        }
    }

private:
    void set_k(uint32_t k) {
        if (current_k == k) return;
        current_k = k;
        rounds_since_switch = 0;
        ++switches;
    }
};

}
