#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace shelf::app {

constexpr double kArenaExternalOverheadGiB = 0.3;
constexpr double kGiBBytes = 1024.0 * 1024.0 * 1024.0;

inline bool arena_budget_bytes(double gib, std::size_t& bytes) {
    if (!std::isfinite(gib) || gib == 0.0 || (gib > 0.0 && gib <= kArenaExternalOverheadGiB)) return false;
    const double adjusted = gib > 0.0 ? gib - kArenaExternalOverheadGiB : -gib + kArenaExternalOverheadGiB;
    const double raw = adjusted * kGiBBytes;
    if (raw < 1.0 || raw >= static_cast<double>(std::numeric_limits<std::size_t>::max())) return false;
    bytes = static_cast<std::size_t>(raw);
    return true;
}

inline bool arena_total_minus(std::size_t total, std::size_t margin, std::size_t& capacity) {
    if (margin >= total) return false;
    capacity = total - margin;
    return true;
}

}
