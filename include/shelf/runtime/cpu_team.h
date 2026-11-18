#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace shelf::runtime {

// A fixed team of CPU threads that runs one parallel loop at a time. The model uses it to
// compute offloaded experts while the GPU runs the resident ones, so the loops are short and
// never queue: the caller blocks in `parallel_for` until every member finished.
//
// The size defaults to half the physical cores (logical / 4) so that the GPU submission
// thread and the rest of the runtime keep their own cores. `SHELF_CPU_EXPERT_THREADS`
// overrides it.
class CpuTeam {
public:
    using Body = std::function<void(std::uint32_t, std::uint32_t)>;

    static std::unique_ptr<CpuTeam> create(std::uint32_t threads);
    static std::uint32_t default_threads();

    CpuTeam(const CpuTeam&) = delete;
    CpuTeam& operator=(const CpuTeam&) = delete;
    ~CpuTeam();

    std::uint32_t size() const noexcept { return workers_; }

    // Calls `body(begin, end)` once per member; the caller is member 0 and runs its slice on
    // its own thread, so a team of one degrades to a serial loop.
    void parallel_for(std::uint32_t count, const Body& body);

private:
    explicit CpuTeam(std::uint32_t threads);
    void run_chunk(std::uint32_t index, std::uint32_t count, const Body& body) const;
    void worker(std::uint32_t index);

    std::uint32_t workers_ = 1;
    std::uint32_t chunk_ = 0;
    std::uint32_t count_ = 0;
    std::uint32_t pending_ = 0;
    std::uint64_t generation_ = 0;
    const Body* body_ = nullptr;
    bool stop_ = false;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::vector<std::thread> threads_;
};

}  // namespace shelf::runtime
