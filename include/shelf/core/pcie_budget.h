#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace shelf {

// Byte-rate ceiling for the host<->device transfers the runtime schedules itself (expert
// prefetch, cache swaps). Keeping a share of the link free stops those transfers from
// monopolising PCIe and stalling the rest of the pipeline.
//
// `SHELF_PCIE_FRAC` is the share (0.05 .. 1.0, default 0.25) and `SHELF_PCIE_GBPS` the
// assumed link bandwidth (default 25 GB/s, the measured value on R9700).
class PcieBudget {
public:
    PcieBudget(double fraction, double gbps)
        : fraction_(std::clamp(fraction, 0.05, 1.0)),
          gbps_(gbps > 0.0 ? gbps : 25.0) {
        rate_ = fraction_ * gbps_ * 1e9;
        burst_ = rate_ * 0.002;
        tokens_ = burst_;
    }

    PcieBudget(const PcieBudget& other) : fraction_(other.fraction_), gbps_(other.gbps_) {
        rate_ = fraction_ * gbps_ * 1e9;
        burst_ = rate_ * 0.002;
        tokens_ = burst_;
    }

    PcieBudget& operator=(const PcieBudget& other) {
        if (this != &other) {
            std::lock_guard<std::mutex> lock(mutex_);
            fraction_ = other.fraction_;
            gbps_ = other.gbps_;
            rate_ = other.rate_;
            burst_ = other.burst_;
            tokens_ = other.tokens_;
            last_ = std::chrono::steady_clock::now();
        }
        return *this;
    }

    static PcieBudget from_env() {
        return PcieBudget(env_double("SHELF_PCIE_FRAC", 0.25),
                          env_double("SHELF_PCIE_GBPS", 25.0));
    }

    // Consumes `bytes` if the bucket holds them; the caller defers the copy otherwise.
    bool try_acquire(std::uint64_t bytes) {
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

    std::uint64_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        refill_locked();
        return static_cast<std::uint64_t>(tokens_);
    }

    std::uint64_t bytes_for(std::uint64_t milliseconds) const {
        return static_cast<std::uint64_t>(rate_ * static_cast<double>(milliseconds) * 1e-3);
    }

    double fraction() const noexcept { return fraction_; }
    double gbps() const noexcept { return gbps_; }

    std::uint64_t granted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return granted_;
    }

    std::uint64_t deferred() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return deferred_;
    }

private:
    static double env_double(const char* name, double fallback) {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') return fallback;
        char* end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || !std::isfinite(parsed)) return fallback;
        return parsed;
    }

    void refill_locked() const {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_).count();
        last_ = now;
        tokens_ = std::min(burst_, tokens_ + elapsed * rate_);
    }

    double fraction_ = 0.25;
    double gbps_ = 25.0;
    double rate_ = 0.0;
    double burst_ = 0.0;
    mutable double tokens_ = 0.0;
    mutable std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
    mutable std::mutex mutex_;
    mutable std::uint64_t granted_ = 0;
    mutable std::uint64_t deferred_ = 0;
};

}  // namespace shelf
