#include <shelf/runtime/cpu_team.h>

#include <algorithm>
#include <cstdlib>

namespace shelf::runtime {
namespace {

std::uint32_t env_threads() {
    const char* value = std::getenv("SHELF_CPU_EXPERT_THREADS");
    if (value == nullptr || *value == '\0') return 0u;
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 0u;
}

}  // namespace

std::uint32_t CpuTeam::default_threads() {
    const std::uint32_t configured = env_threads();
    if (configured != 0u) return std::min<std::uint32_t>(configured, 64u);
    const std::uint32_t logical = std::thread::hardware_concurrency();
    if (logical <= 1u) return 1u;
    return std::max<std::uint32_t>(1u, logical / 4u);
}

std::unique_ptr<CpuTeam> CpuTeam::create(std::uint32_t threads) {
    const std::uint32_t size = threads == 0u ? default_threads() : threads;
    return std::unique_ptr<CpuTeam>(new CpuTeam(std::max<std::uint32_t>(size, 1u)));
}

CpuTeam::CpuTeam(std::uint32_t threads) : workers_(threads) {
    threads_.reserve(workers_ > 0u ? workers_ - 1u : 0u);
    for (std::uint32_t index = 1u; index < workers_; ++index) {
        threads_.emplace_back([this, index] { worker(index); });
    }
}

CpuTeam::~CpuTeam() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (auto& thread : threads_) {
        if (thread.joinable()) thread.join();
    }
}

void CpuTeam::run_chunk(std::uint32_t index, std::uint32_t count, const Body& body) const {
    const std::uint32_t begin = index * chunk_;
    if (begin >= count) return;
    const std::uint32_t end = std::min(count, begin + chunk_);
    body(begin, end);
}

void CpuTeam::parallel_for(std::uint32_t count, const Body& body) {
    if (count == 0u || workers_ == 1u) {
        if (count != 0u) body(0u, count);
        return;
    }
    chunk_ = (count + workers_ - 1u) / workers_;
    count_ = count;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        body_ = &body;
        pending_ = workers_ - 1u;
        ++generation_;
    }
    wake_.notify_all();
    run_chunk(0u, count, body);
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0u; });
    body_ = nullptr;
}

void CpuTeam::worker(std::uint32_t index) {
    std::uint64_t seen = 0;
    for (;;) {
        const Body* body = nullptr;
        std::uint32_t count = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this, seen] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            body = body_;
            count = count_;
        }
        run_chunk(index, count, *body);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pending_ > 0u && --pending_ == 0u) done_.notify_one();
        }
    }
}

}  // namespace shelf::runtime
