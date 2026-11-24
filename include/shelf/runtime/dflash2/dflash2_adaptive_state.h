#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace shelf::runtime {

constexpr uint32_t kDFlash2AdaptiveMaxK = 5u;
constexpr uint32_t kDFlash2AdaptiveInitialK = 4u;
constexpr uint32_t kDFlash2AdaptiveSwitchInterval = 8u;
constexpr double kDFlash2AdaptiveEmaAlpha = 0.25;

struct DFlash2AdaptiveController {
    uint32_t current_k = kDFlash2AdaptiveInitialK;
    uint32_t min_switch_interval = kDFlash2AdaptiveSwitchInterval;
    double ema_alpha = kDFlash2AdaptiveEmaAlpha;
    double acc_ema = 0.0;
    bool has_sample = false;
    uint64_t rounds = 0;
    uint64_t rounds_since_switch = 0;
    uint64_t switches = 0;

    DFlash2AdaptiveController() = default;

    explicit DFlash2AdaptiveController(uint32_t initial_k,
                                       uint32_t interval = kDFlash2AdaptiveSwitchInterval,
                                       double alpha = kDFlash2AdaptiveEmaAlpha)
        : current_k(initial_k), min_switch_interval(interval), ema_alpha(alpha) {
        if (interval == 0u || !std::isfinite(alpha) || alpha <= 0.0 || alpha > 1.0) {
            throw std::invalid_argument("DFlash2 adaptive K interval or EMA alpha is invalid");
        }
    }

    void reset(uint32_t initial_k) {
        current_k = initial_k;
        acc_ema = 0.0;
        has_sample = false;
        rounds = 0;
        rounds_since_switch = 0;
        switches = 0;
    }

    static uint32_t select_k(uint32_t current, double ema, uint32_t batch_size, uint32_t max_k) {
        uint32_t candidate = 0u;
        if (batch_size > 16u) {
            const double thr23 = current <= 2u ? 1.95 : 1.65;
            const double thr34 = current <= 3u ? 3.35 : 3.05;
            candidate = ema < thr23 ? 2u : (ema < thr34 ? 3u : 4u);
        } else {
            const double thr23 = current <= 2u ? 1.25 : 1.05;
            const double thr34 = current <= 3u ? 1.95 : 1.65;
            const double thr45 = current <= 4u ? 3.35 : 3.05;
            candidate = ema < thr23 ? 2u
                      : (ema < thr34 ? 3u : (ema < thr45 ? 4u : 5u));
        }
        return std::min(candidate, max_k);
    }

    uint32_t observe(double accepted_per_lane, uint32_t batch_size,
                     uint32_t max_k = kDFlash2AdaptiveMaxK) {
        if (batch_size == 0u || !std::isfinite(accepted_per_lane) || accepted_per_lane < 0.0) {
            return current_k;
        }
        if (!has_sample) {
            acc_ema = accepted_per_lane;
            has_sample = true;
        } else {
            acc_ema = (1.0 - ema_alpha) * acc_ema + ema_alpha * accepted_per_lane;
        }
        ++rounds;
        ++rounds_since_switch;
        if (rounds_since_switch < min_switch_interval) {
            return current_k;
        }
        const uint32_t next = select_k(current_k, acc_ema, batch_size, max_k);
        if (next != current_k) {
            current_k = next;
            ++switches;
            rounds_since_switch = 0;
        }
        return current_k;
    }
};

}
