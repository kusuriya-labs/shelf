#pragma once

#include <shelf/core/status.h>
#include <shelf/models/qwen4exp/weights/qwen4exp_weights.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace shelf {
namespace qwen4exp {

// Frequency-driven residency for routed experts.
//
// The loader leaves `vram_experts` experts in VRAM and `host_experts` in pinned host
// memory. This cache swaps their contents between the two pools between steps. The
// grouped GEMM uses the slot map (>= 0 VRAM slot, -1 - slot host slot); the host plane
// is read over PCIe for misses, so the cache never runs a CPU matmul.
class MoeExpertCache {
public:
    static std::unique_ptr<MoeExpertCache> create(const Qwen4ExpRankWeights& weights);

    MoeExpertCache(const MoeExpertCache&) = delete;
    MoeExpertCache& operator=(const MoeExpertCache&) = delete;

    bool enabled() const noexcept { return weights_ != nullptr; }

    // Reads the routing observed in the last batch, updates the frequency table, and
    // executes the next slice of VRAM<->host moves. Runs after the batch completed and
    // before the next one is submitted.
    Status step(hipStream_t stream);

    std::uint64_t routes() const noexcept { return routes_; }
    std::uint64_t hits() const noexcept { return hits_; }

private:
    explicit MoeExpertCache(const Qwen4ExpRankWeights& weights);

    struct Move {
        std::uint32_t layer = 0;
        std::uint32_t host_slot = 0;
        std::uint32_t vram_slot = 0;
        std::uint64_t score = 0;
    };

    Status swap(std::uint32_t layer, std::uint32_t host_slot, std::uint32_t vram_slot,
                hipStream_t stream);
    void plan();
    void log_oracle() const;

    const Qwen4ExpRankWeights* weights_ = nullptr;
    std::uint32_t layers_ = 0;
    std::uint32_t vram_experts_ = 0;
    std::uint32_t host_experts_ = 0;
    std::uint32_t experts_ = 0;
    std::uint32_t budget_ = 64;
    std::uint32_t burst_ = 512;
    std::uint32_t min_gain_ = 150;
    std::uint32_t rebalance_steps_ = 4;
    std::uint32_t decay_at_ = 4096;
    std::uint32_t log_interval_ = 64;
    std::uint64_t rebalance_routes_ = 0;
    std::uint64_t next_rebalance_routes_ = 0;
    std::uint64_t last_plan_step_ = 0;
    bool oracle_ = false;

    std::vector<Move> pending_;
    std::size_t pending_head_ = 0;

    std::vector<std::uint32_t> freq_acc_;    // [layers][experts]
    std::vector<std::uint32_t> step_freq_;   // [layers][experts]
    std::vector<std::int32_t> map_host_;     // [layers][experts]
    std::vector<std::int32_t> vram_expert_;  // [layers][vram_experts]
    std::vector<std::int32_t> host_expert_;  // [layers][host_experts]

    std::uint64_t step_ = 0;
    std::uint64_t routes_ = 0;
    std::uint64_t hits_ = 0;
    std::uint64_t swaps_ = 0;
};

}  // namespace qwen4exp
}  // namespace shelf
