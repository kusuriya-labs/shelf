#include <shelf/runtime/pcie_budget.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace shelf::runtime {
namespace {

constexpr double kBytesPerGigabyte = 1e9;
constexpr double kDefaultFraction = 0.25;
constexpr double kDefaultGbps = 25.0;

double env_double(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || !std::isfinite(parsed)) return fallback;
    return parsed;
}

}  // namespace

PcieBudget::PcieBudget(double fraction, double gbps)
    : fraction_(std::clamp(fraction, 0.05, 1.0)), gbps_(gbps > 0.0 ? gbps : kDefaultGbps) {
    rate_ = fraction_ * gbps_ * kBytesPerGigabyte;
    burst_ = rate_ * 0.002;
    tokens_ = burst_;
}

PcieBudget PcieBudget::from_env() {
    return PcieBudget(env_double("SHELF_PCIE_FRAC", kDefaultFraction),
                      env_double("SHELF_PCIE_GBPS", kDefaultGbps));
}

void PcieBudget::refill_locked() const {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(now - last_).count();
    last_ = now;
    tokens_ = std::min(burst_, tokens_ + elapsed * rate_);
}

bool PcieBudget::try_acquire(std::uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    refill_locked();
    if (static_cast<double>(bytes) > tokens_) {
        ++deferred_;
        return false;
    }
    tokens_ -= static_cast<double>(bytes);
    granted_ += bytes;
    return true;
}

std::uint64_t PcieBudget::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    refill_locked();
    return static_cast<std::uint64_t>(tokens_);
}

std::uint64_t PcieBudget::bytes_for(std::uint64_t milliseconds) const {
    return static_cast<std::uint64_t>(rate_ * static_cast<double>(milliseconds) * 1e-3);
}

std::uint64_t PcieBudget::granted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return granted_;
}

std::uint64_t PcieBudget::deferred() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return deferred_;
}

}  // namespace shelf::runtime
