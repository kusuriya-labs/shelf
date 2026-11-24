#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace shelf::runtime {

struct OnlineNgramConfig {
    bool enabled = false;
    uint32_t num_drafts = 0u;
    uint32_t min_match = 0u;
    uint32_t max_match = 32u;
    uint32_t cooldown_threshold = 2u;
    uint32_t cooldown_rounds = 16u;
    bool stats = false;
};

struct OnlineNgramStats {
    uint64_t rounds = 0u;
    uint64_t hits = 0u;
    uint64_t accepted_drafts = 0u;
    uint64_t proposed_drafts = 0u;
    uint64_t dflash_skipped = 0u;
    uint64_t cooldown_skips = 0u;
    double cpu_us_total = 0.0;

    double mean_accepted() const noexcept {
        return hits == 0u ? 0.0
                          : static_cast<double>(accepted_drafts) / static_cast<double>(hits);
    }

    double cpu_us_per_round() const noexcept {
        return rounds == 0u ? 0.0 : cpu_us_total / static_cast<double>(rounds);
    }

    OnlineNgramStats& operator+=(const OnlineNgramStats& other) noexcept {
        rounds += other.rounds;
        hits += other.hits;
        accepted_drafts += other.accepted_drafts;
        proposed_drafts += other.proposed_drafts;
        dflash_skipped += other.dflash_skipped;
        cooldown_skips += other.cooldown_skips;
        cpu_us_total += other.cpu_us_total;
        return *this;
    }
};

struct OnlineNgramProposal {
    static constexpr uint32_t kMaxDraftTokens = 64u;

    bool hit = false;
    uint32_t match_len = 0u;
    uint32_t valid_drafts = 0u;
    uint32_t count = 0u;
    std::array<int32_t, kMaxDraftTokens> tokens{};
};

class OnlineNgramMatcher {
public:
    static constexpr uint32_t kWindowCapacity = 32768u;
    static constexpr uint32_t kWindowMask = kWindowCapacity - 1u;
    static constexpr uint32_t kBucketCount = 16384u;
    static constexpr uint32_t kBucketMask = kBucketCount - 1u;
    static constexpr uint32_t kBucketWays = 4u;

    OnlineNgramMatcher() = default;
    explicit OnlineNgramMatcher(const OnlineNgramConfig& config);

    void configure(const OnlineNgramConfig& config);
    void reset() noexcept;

    void ingest(int32_t token_id);
    void ingest_span(std::span<const int32_t> tokens);
    void ingest_timed(int32_t token_id);
    void ingest_span_timed(std::span<const int32_t> tokens);

    OnlineNgramProposal propose(uint32_t num_drafts, bool stochastic = false);

    void record_outcome(uint32_t accepted_drafts) noexcept;
    void note_drafter_skipped() noexcept { ++stats_.dflash_skipped; }

    const OnlineNgramConfig& config() const noexcept { return config_; }
    const OnlineNgramStats& stats() const noexcept { return stats_; }
    uint64_t total_tokens() const noexcept { return total_tokens_; }
    uint32_t cooldown_remaining() const noexcept { return cooldown_remaining_; }

    static uint32_t bucket_index_2(int32_t a, int32_t b) noexcept;
    static uint32_t bucket_index_4(int32_t a, int32_t b, int32_t c, int32_t d) noexcept;

private:
    void ensure_storage();
    void push_token_unlocked(int32_t token_id) noexcept;
    uint32_t effective_min_match(bool stochastic) const noexcept;
    uint32_t match_backward_len(uint64_t end_pos, uint64_t cand_pos, uint64_t oldest_valid,
                                uint32_t max_len) const noexcept;

    OnlineNgramConfig config_{};
    OnlineNgramStats stats_{};
    std::vector<int32_t> ring_{};
    std::vector<uint32_t> buckets2_{};
    std::vector<uint32_t> buckets4_{};
    uint64_t total_tokens_ = 0u;
    uint32_t consecutive_zero_accepts_ = 0u;
    uint32_t cooldown_remaining_ = 0u;
    bool initialized_ = false;
};

}
