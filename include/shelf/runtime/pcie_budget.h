#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>

namespace shelf::runtime {

// Byte-rate ceiling for host<->device transfers that the runtime schedules itself
// (expert prefetch, cache swaps). The budget keeps a configured share of the PCIe
// bandwidth free so that transfers do not monopolise the link and stall the rest of
// the pipeline.
//
// `SHELF_PCIE_FRAC` is the share (0.05 .. 1.0, default 0.25) and `SHELF_PCIE_GBPS` the
// assumed link bandwidth for the running device (default 25 GB/s, the measured value on
// R9700). Both are read once per instance.
class PcieBudget {
public:
    static PcieBudget from_env();

    PcieBudget(double fraction, double gbps);

    // Consumes `bytes` if the bucket holds them; the caller defers the copy otherwise.
    bool try_acquire(std::uint64_t bytes);

    // Largest transfer the bucket can admit right now.
    std::uint64_t available() const;

    // Bytes admitted at the configured share over `milliseconds`.
    std::uint64_t bytes_for(std::uint64_t milliseconds) const;

    double fraction() const noexcept { return fraction_; }
    double gbps() const noexcept { return gbps_; }
    std::uint64_t granted() const;
    std::uint64_t deferred() const;

private:
    void refill_locked() const;

    double fraction_ = 0.25;
    double gbps_ = 25.0;
    double rate_ = 0.0;
    double burst_ = 0.0;
    mutable double tokens_ = 0.0;
    mutable std::chrono::steady_clock::time_point last_ =
        std::chrono::steady_clock::now();
    mutable std::mutex mutex_;
    mutable std::uint64_t granted_ = 0;
    mutable std::uint64_t deferred_ = 0;
};

}  // namespace shelf::runtime
