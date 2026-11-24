#include "shelf/runtime/online_ngram.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(cond) check_impl(static_cast<bool>(cond), #cond, __LINE__)

using shelf::runtime::OnlineNgramConfig;
using shelf::runtime::OnlineNgramMatcher;
using shelf::runtime::OnlineNgramProposal;

void test_boundaries_and_min_match() {
    OnlineNgramConfig cfg{};
    cfg.enabled = true;
    OnlineNgramMatcher matcher(cfg);

    CHECK(!matcher.propose(7u, false).hit);

    matcher.ingest(10);
    CHECK(!matcher.propose(7u, false).hit);

    matcher.ingest_span(std::vector<int32_t>{10, 20, 30, 40, 50, 10, 20, 30});
    CHECK(!matcher.propose(7u, false).hit);
    CHECK(!matcher.propose(0u, false).hit);

    matcher.ingest(40);
    const OnlineNgramProposal greedy_hit = matcher.propose(7u, false);
    CHECK(greedy_hit.hit);
    CHECK(greedy_hit.match_len == 4u);
    CHECK(greedy_hit.count == 7u);
    CHECK(greedy_hit.valid_drafts == 5u);
    CHECK(greedy_hit.tokens[0] == 50);
    CHECK(greedy_hit.tokens[1] == 10);
    CHECK(greedy_hit.tokens[2] == 20);
    CHECK(greedy_hit.tokens[3] == 30);
    CHECK(greedy_hit.tokens[4] == 40);
    CHECK(greedy_hit.tokens[5] == 40);
    CHECK(greedy_hit.tokens[6] == 40);

    const OnlineNgramProposal stochastic_miss = matcher.propose(7u, true);
    CHECK(!stochastic_miss.hit);

    matcher.ingest_span(std::vector<int32_t>{50, 99, 10, 20, 30, 40, 50, 99});
    const OnlineNgramProposal stochastic_hit = matcher.propose(4u, true);
    CHECK(stochastic_hit.hit);
    CHECK(stochastic_hit.match_len == 6u);
    CHECK(stochastic_hit.count == 4u);
    CHECK(stochastic_hit.valid_drafts == 4u);
    CHECK(stochastic_hit.tokens[0] == 10);
    CHECK(stochastic_hit.tokens[1] == 20);
    CHECK(stochastic_hit.tokens[2] == 30);
    CHECK(stochastic_hit.tokens[3] == 40);
}

void test_longest_and_newest_priority() {
    OnlineNgramConfig cfg{};
    cfg.enabled = true;
    cfg.min_match = 4u;
    cfg.max_match = 32u;
    OnlineNgramMatcher matcher(cfg);

    const std::vector<int32_t> seq = {
        1, 2, 3, 4, 5, 6, 7, 8, 111, 112, 113,
        90, 91,
        1, 2, 3, 4, 5, 6, 7, 8, 211, 212, 213,
        92, 93,
        1, 2, 3, 4, 5, 6, 7, 8
    };
    matcher.ingest_span(seq);
    const OnlineNgramProposal equal_len = matcher.propose(3u, false);
    CHECK(equal_len.hit);
    CHECK(equal_len.match_len == 8u);
    CHECK(equal_len.tokens[0] == 211);
    CHECK(equal_len.tokens[1] == 212);
    CHECK(equal_len.tokens[2] == 213);

    OnlineNgramMatcher longest_matcher(cfg);
    const std::vector<int32_t> seq2 = {
        10, 20, 30, 40, 50, 60, 301, 302,
        77, 78,
        88, 89, 30, 40, 50, 60, 401, 402,
        99, 10, 20, 30, 40, 50, 60
    };
    longest_matcher.ingest_span(seq2);
    const OnlineNgramProposal longest = longest_matcher.propose(2u, false);
    CHECK(longest.hit);
    CHECK(longest.match_len == 6u);
    CHECK(longest.tokens[0] == 301);
    CHECK(longest.tokens[1] == 302);
}

void test_max_match_and_window_wraparound() {
    OnlineNgramConfig cfg{};
    cfg.enabled = true;
    cfg.min_match = 4u;
    cfg.max_match = 10u;
    OnlineNgramMatcher matcher(cfg);

    std::vector<int32_t> pattern(16);
    for (int32_t i = 0; i < 16; ++i) pattern[i] = 1000 + i;
    matcher.ingest_span(pattern);
    matcher.ingest(7777);
    matcher.ingest_span(pattern);
    const OnlineNgramProposal capped = matcher.propose(2u, false);
    CHECK(capped.hit);
    CHECK(capped.match_len == 10u);
    CHECK(capped.tokens[0] == 7777);

    cfg.max_match = 32u;
    OnlineNgramMatcher ring_matcher(cfg);
    ring_matcher.ingest_span(std::vector<int32_t>{5001, 5002, 5003, 5004, 5005, 5006});
    for (uint32_t i = 0u; i < OnlineNgramMatcher::kWindowCapacity; ++i) {
        ring_matcher.ingest(100000 + static_cast<int32_t>(i));
    }
    ring_matcher.ingest_span(std::vector<int32_t>{5001, 5002, 5003, 5004});
    CHECK(!ring_matcher.propose(4u, false).hit);
}

void test_hash_collisions() {
    OnlineNgramConfig cfg{};
    cfg.enabled = true;
    cfg.min_match = 4u;
    OnlineNgramMatcher matcher(cfg);

    const uint32_t target_bucket4 = OnlineNgramMatcher::bucket_index_4(1, 2, 3, 4);
    int32_t collider_a = 10;
    int32_t collider_b = 20;
    bool found = false;
    for (int32_t x = 100; x < 200000 && !found; ++x) {
        if (OnlineNgramMatcher::bucket_index_4(collider_a, collider_b, x, 4) == target_bucket4) {
            matcher.ingest_span(std::vector<int32_t>{collider_a, collider_b, x, 4, 888, 889});
            found = true;
        }
    }
    CHECK(found);

    matcher.ingest_span(std::vector<int32_t>{1, 2, 3, 4});
    CHECK(!matcher.propose(2u, false).hit);

    matcher.ingest_span(std::vector<int32_t>{901, 902, 1, 2, 3, 4});
    const OnlineNgramProposal real_hit = matcher.propose(2u, false);
    CHECK(real_hit.hit);
    CHECK(real_hit.tokens[0] == 901);
    CHECK(real_hit.tokens[1] == 902);
}

void test_adaptive_cooldown() {
    OnlineNgramConfig cfg{};
    cfg.enabled = true;
    cfg.min_match = 4u;
    cfg.cooldown_threshold = 2u;
    cfg.cooldown_rounds = 16u;
    OnlineNgramMatcher matcher(cfg);

    matcher.ingest_span(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4});

    CHECK(matcher.propose(3u, false).hit);
    matcher.record_outcome(0u);
    CHECK(matcher.cooldown_remaining() == 0u);

    CHECK(matcher.propose(3u, false).hit);
    matcher.record_outcome(2u);
    CHECK(matcher.cooldown_remaining() == 0u);

    CHECK(matcher.propose(3u, false).hit);
    matcher.record_outcome(0u);
    CHECK(matcher.cooldown_remaining() == 0u);
    CHECK(matcher.propose(3u, false).hit);
    matcher.record_outcome(0u);
    CHECK(matcher.cooldown_remaining() == 16u);

    for (uint32_t r = 0u; r < 16u; ++r) {
        const OnlineNgramProposal during = matcher.propose(3u, false);
        CHECK(!during.hit);
        matcher.ingest(100 + static_cast<int32_t>(r));
    }
    CHECK(matcher.cooldown_remaining() == 0u);
    matcher.ingest_span(std::vector<int32_t>{1, 2, 3, 4});
    const OnlineNgramProposal recovered = matcher.propose(3u, false);
    CHECK(recovered.hit);
    CHECK(recovered.tokens[0] == 100);
    CHECK(recovered.tokens[1] == 101);
    CHECK(matcher.stats().cooldown_skips == 16u);
}

void test_microbenchmark() {
    OnlineNgramConfig cfg{};
    cfg.enabled = true;
    cfg.min_match = 4u;
    cfg.max_match = 32u;
    OnlineNgramMatcher matcher(cfg);

    std::vector<int32_t> seed(16384u);
    for (std::size_t i = 0; i < seed.size(); ++i) {
        seed[i] = static_cast<int32_t>(i % 257u);
    }
    matcher.ingest_span(seed);

    constexpr uint32_t kIters = 4096u;
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t hit_count = 0u;
    for (uint32_t i = 0u; i < kIters; ++i) {
        matcher.ingest_timed(static_cast<int32_t>(i % 257u));
        const OnlineNgramProposal prop = matcher.propose(7u, false);
        if (prop.hit) {
            ++hit_count;
            matcher.record_outcome(1u);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double avg_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(kIters);
    std::printf("online_ngram microbench: avg_us=%.4f hits=%llu stats_cpu_per_round=%.4f\n",
                avg_us, static_cast<unsigned long long>(hit_count),
                matcher.stats().cpu_us_per_round());
    CHECK(hit_count > 0u);
    CHECK(avg_us < 10.0);
}

void test_reset_reuse_and_stats_merge() {
    OnlineNgramConfig cfg{};
    cfg.enabled = true;
    cfg.min_match = 2u;
    cfg.max_match = 8u;
    OnlineNgramMatcher matcher(cfg);
    matcher.ingest_span_timed(std::vector<int32_t>{41, 42, 43, 44, 41, 42});
    const OnlineNgramProposal before_reset = matcher.propose(2u, false);
    CHECK(before_reset.hit);
    CHECK(before_reset.tokens[0] == 43);
    CHECK(before_reset.tokens[1] == 44);
    matcher.note_drafter_skipped();
    matcher.record_outcome(2u);
    CHECK(matcher.stats().dflash_skipped == 1u);
    CHECK(matcher.stats().accepted_drafts == 2u);

    shelf::runtime::OnlineNgramStats accum = matcher.stats();
    matcher.reset();
    CHECK(matcher.total_tokens() == 0u);
    CHECK(matcher.cooldown_remaining() == 0u);
    CHECK(matcher.stats().rounds == 0u);
    matcher.ingest_span(std::vector<int32_t>{41, 42});
    CHECK(!matcher.propose(2u, false).hit);

    accum += matcher.stats();
    CHECK(accum.rounds == 2u);
    CHECK(accum.hits == 1u);
    CHECK(accum.accepted_drafts == 2u);
    CHECK(accum.dflash_skipped == 1u);
}

}

int main() {
    test_boundaries_and_min_match();
    test_longest_and_newest_priority();
    test_max_match_and_window_wraparound();
    test_hash_collisions();
    test_adaptive_cooldown();
    test_microbenchmark();
    test_reset_reuse_and_stats_merge();
    return g_failures == 0 ? 0 : 1;
}
