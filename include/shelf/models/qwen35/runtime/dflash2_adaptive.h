#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace shelf::qwen35::runtime::adaptive {
struct Acceptance {
    std::vector<double> conditional;
    explicit Acceptance(uint32_t count = 0, double prior = 0.8) : conditional(count, prior) {}
    void observe(uint32_t tried, uint32_t accepted, double alpha = 0.15) {
        tried = std::min<uint32_t>(tried, conditional.size());
        for (uint32_t i = 0; i < tried && i <= accepted; ++i) {
            conditional[i] += alpha * ((i < accepted ? 1.0 : 0.0) - conditional[i]);
        }
    }
    double gain(uint32_t count) const {
        double survival = 1.0, result = 0.0;
        for (uint32_t i = 0; i < std::min<uint32_t>(count, conditional.size()); ++i) {
            survival *= conditional[i];
            result += survival;
        }
        return result;
    }
};
inline uint32_t pick_probe(std::span<const uint32_t> disabled_rounds,
                           uint64_t round, uint32_t interval) {
    if (interval == 0 || round % interval != 0) return UINT32_MAX;
    uint32_t index = UINT32_MAX;
    uint32_t oldest = interval - 1;
    for (uint32_t i = 0; i < disabled_rounds.size(); ++i) {
        if (disabled_rounds[i] > oldest) {
            oldest = disabled_rounds[i];
            index = i;
        }
    }
    return index;
}
struct CostPoint {
    uint32_t rows;
    double milliseconds;
};
struct CostModel {
    std::vector<double> baseline_ms;
    std::vector<std::vector<CostPoint>> speculation_ms;

    double milliseconds(uint32_t batch, uint32_t sessions, uint32_t rows) const {
        const double missing = std::numeric_limits<double>::infinity();
        if (sessions > batch || batch - sessions >= baseline_ms.size()) return missing;
        const double normal = baseline_ms[batch - sessions];
        if (!std::isfinite(normal) || normal < 0.0 ||
            (batch != sessions && normal == 0.0)) return missing;
        if (sessions == 0) return rows == 0 ? normal : missing;
        if (sessions >= speculation_ms.size()) return missing;
        const auto& points = speculation_ms[sessions];
        if (points.empty() || rows < points.front().rows || rows > points.back().rows) {
            return missing;
        }
        auto upper = std::lower_bound(points.begin(), points.end(), rows,
            [](const CostPoint& point, uint32_t value) { return point.rows < value; });
        if (upper == points.end()) return missing;
        if (upper == points.begin() || upper->rows == rows) return normal + upper->milliseconds;
        const auto& lower = *(upper - 1);
        const double weight = static_cast<double>(rows - lower.rows) / (upper->rows - lower.rows);
        return normal + lower.milliseconds + weight * (upper->milliseconds - lower.milliseconds);
    }
};
struct Decision {
    std::vector<uint32_t> rows;
    double predicted = 0.0;
};
using Cost = std::function<double(uint32_t, uint32_t)>;
inline Decision choose_rows(std::span<const std::vector<double>> gains_by_row, uint32_t batch_size, uint32_t row_cap,
                const Cost& cost, std::span<const uint32_t> previous, double margin,
                uint32_t max_sessions = UINT32_MAX,
                uint32_t probe_index = UINT32_MAX, uint32_t probe_rows = 0) {
    const uint32_t n = gains_by_row.size();
    const uint32_t max_active = std::min(n, max_sessions);
    const uint32_t max_rows = max_active * row_cap;
    const uint32_t width = max_rows + 1;
    const uint32_t states = (max_active + 1) * width;
    const double invalid = -std::numeric_limits<double>::infinity();
    std::vector<double> dp(states, invalid), next(states, invalid);
    std::vector<std::vector<uint32_t>> choices(n, std::vector<uint32_t>(states, UINT32_MAX));
    dp[0] = 0;
    for (uint32_t i = 0; i < n; ++i) {
        std::fill(next.begin(), next.end(), invalid);
        const uint32_t limit = gains_by_row[i].empty() ? 0 :
            std::min<uint32_t>(row_cap, gains_by_row[i].size() - 1);
        for (uint32_t s = 0; s <= std::min(i, max_active); ++s) for (uint32_t r = 2 * s; r <= s * row_cap; ++r) {
            uint32_t from = s * width + r;
            if (!std::isfinite(dp[from])) continue;
            for (uint32_t rows = 0; rows <= limit; ++rows) {
                if (rows == 1 || (i == probe_index && rows != probe_rows)) continue;
                uint32_t ns = s + (rows != 0), nr = r + rows;
                if (ns > max_active) continue;
                if (nr > max_rows) break;
                uint32_t to = ns * width + nr;
                double gain = dp[from] + (rows ? gains_by_row[i][rows] : 0.0);
                if (gain > next[to]) { next[to] = gain; choices[i][to] = rows; }
            }
        }
        dp.swap(next);
    }
    Decision result; result.rows.assign(n, 0);
    uint32_t best_state = 0;
    for (uint32_t s = 0; s <= max_active; ++s) for (uint32_t r = 0; r <= max_rows; ++r) {
        uint32_t state = s * width + r;
        if (!std::isfinite(dp[state])) continue;
        double ms = cost(s, r);
        if (!(ms > 0.0) || !std::isfinite(ms)) continue;
        double score = (batch_size + dp[state]) / ms;
        if (score > result.predicted) { result.predicted = score; best_state = state; }
    }
    for (uint32_t i = n; i > 0; --i) {
        uint32_t rows = choices[i-1][best_state];
        if (rows == UINT32_MAX) break;
        result.rows[i-1] = rows;
        if (rows) best_state -= width + rows;
    }
    if (previous.size() == n) {
        uint32_t s = 0, rows = 0;
        double gain = 0;
        bool valid = true;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t r = previous[i];
            if (i == probe_index && r != probe_rows) { valid = false; break; }
            if (r == 1 || (r > 0 && (r > row_cap || r >= gains_by_row[i].size()))) { valid = false; break; }
            if (r) { ++s; rows += r; gain += gains_by_row[i][r]; }
        }
        double ms = valid && s <= max_active ? cost(s, rows) : 0;
        double score = ms > 0 ? (batch_size + gain) / ms : 0;
        if (score >= result.predicted * (1.0 - margin) && score > 0) {
            result.rows.assign(previous.begin(), previous.end());
            result.predicted = score;
        }
    }
    return result;
}
inline Decision choose(std::span<const Acceptance> accept, uint32_t batch_size, uint32_t cap,
                       const Cost& cost, std::span<const uint32_t> previous, double margin,
                       uint32_t max_sessions = UINT32_MAX,
                       uint32_t probe_index = UINT32_MAX, uint32_t probe_rows = 0) {
    std::vector<std::vector<double>> gains;
    gains.reserve(accept.size());
    for (const auto& a : accept) {
        const uint32_t limit = std::min<uint32_t>(cap, a.conditional.size());
        std::vector<double> rows(limit + 2, 0.0);
        for (uint32_t k = 1; k <= limit; ++k) rows[k + 1] = a.gain(k);
        gains.push_back(std::move(rows));
    }
    return choose_rows(gains, batch_size, cap + 1, cost, previous, margin, max_sessions,
                       probe_index, probe_rows);
}
}
