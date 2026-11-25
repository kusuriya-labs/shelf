#include <shelf/runtime/online_ngram.h>

#include <shelf/kernels/optimized/linear/config.h>

#include <algorithm>
#include <chrono>

namespace shelf::runtime {
namespace {

uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31u);
}

void push_bucket_entry(uint32_t* bucket, uint32_t pos_plus_one) noexcept {
    bucket[3] = bucket[2];
    bucket[2] = bucket[1];
    bucket[1] = bucket[0];
    bucket[0] = pos_plus_one;
}

}

uint32_t standalone_ngram_max_rows(uint32_t requests, uint32_t drafts) noexcept {
    const uint64_t width = static_cast<uint64_t>(drafts) + 1u;
    const uint64_t rows = requests >= kNgramDecodeOnlyBatchSize
        ? std::max<uint64_t>(requests, width) : requests * width;
    return static_cast<uint32_t>(std::min<uint64_t>(kernel::kBf16GemmExactRowsKernelMax, rows));
}

OnlineNgramMatcher::OnlineNgramMatcher(const OnlineNgramConfig& config) : config_(config) {
    ensure_storage();
}

void OnlineNgramMatcher::configure(const OnlineNgramConfig& config) {
    config_ = config;
    ensure_storage();
}

void OnlineNgramMatcher::ensure_storage() {
    if (initialized_) return;
    ring_.assign(kWindowCapacity, 0);
    buckets2_.assign(static_cast<std::size_t>(kBucketCount) * kBucketWays, 0u);
    buckets4_.assign(static_cast<std::size_t>(kBucketCount) * kBucketWays, 0u);
    initialized_ = true;
}

void OnlineNgramMatcher::reset() noexcept {
    if (initialized_) {
        std::fill(buckets2_.begin(), buckets2_.end(), 0u);
        std::fill(buckets4_.begin(), buckets4_.end(), 0u);
    }
    total_tokens_ = 0u;
    consecutive_zero_accepts_ = 0u;
    cooldown_remaining_ = 0u;
    stats_ = {};
}

uint32_t OnlineNgramMatcher::bucket_index_2(int32_t a, int32_t b) noexcept {
    const uint64_t packed =
        (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32u) |
        static_cast<uint64_t>(static_cast<uint32_t>(b));
    return static_cast<uint32_t>(splitmix64(packed) & kBucketMask);
}

uint32_t OnlineNgramMatcher::bucket_index_4(int32_t a, int32_t b, int32_t c, int32_t d) noexcept {
    const uint64_t h0 = splitmix64(
        (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32u) |
        static_cast<uint64_t>(static_cast<uint32_t>(b)));
    const uint64_t h1 = splitmix64(
        (static_cast<uint64_t>(static_cast<uint32_t>(c)) << 32u) |
        static_cast<uint64_t>(static_cast<uint32_t>(d)));
    return static_cast<uint32_t>(splitmix64(h0 ^ (h1 + 0x9e3779b97f4a7c15ULL)) & kBucketMask);
}

void OnlineNgramMatcher::push_token_unlocked(int32_t token_id) noexcept {
    const uint64_t pos = total_tokens_;
    ring_[pos & kWindowMask] = token_id;
    total_tokens_ = pos + 1u;

    const uint32_t encoded_pos = static_cast<uint32_t>(pos + 1u);
    if (pos >= 1u) {
        const int32_t t0 = ring_[(pos - 1u) & kWindowMask];
        const uint32_t idx2 = bucket_index_2(t0, token_id);
        push_bucket_entry(&buckets2_[static_cast<std::size_t>(idx2) * kBucketWays], encoded_pos);
    }
    if (pos >= 3u) {
        const int32_t t0 = ring_[(pos - 3u) & kWindowMask];
        const int32_t t1 = ring_[(pos - 2u) & kWindowMask];
        const int32_t t2 = ring_[(pos - 1u) & kWindowMask];
        const uint32_t idx4 = bucket_index_4(t0, t1, t2, token_id);
        push_bucket_entry(&buckets4_[static_cast<std::size_t>(idx4) * kBucketWays], encoded_pos);
    }
}

void OnlineNgramMatcher::ingest(int32_t token_id) {
    ensure_storage();
    push_token_unlocked(token_id);
}

void OnlineNgramMatcher::ingest_span(std::span<const int32_t> tokens) {
    if (tokens.empty()) return;
    ensure_storage();
    for (const int32_t tok : tokens) {
        push_token_unlocked(tok);
    }
}

void OnlineNgramMatcher::ingest_timed(int32_t token_id) {
    const auto t0 = std::chrono::steady_clock::now();
    ingest(token_id);
    const auto t1 = std::chrono::steady_clock::now();
    stats_.cpu_us_total += std::chrono::duration<double, std::micro>(t1 - t0).count();
}

void OnlineNgramMatcher::ingest_span_timed(std::span<const int32_t> tokens) {
    if (tokens.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    ingest_span(tokens);
    const auto t1 = std::chrono::steady_clock::now();
    stats_.cpu_us_total += std::chrono::duration<double, std::micro>(t1 - t0).count();
}

uint32_t OnlineNgramMatcher::effective_min_match(bool stochastic) const noexcept {
    if (config_.min_match != 0u) return std::max(2u, config_.min_match);
    return stochastic ? 6u : 4u;
}

uint32_t OnlineNgramMatcher::match_backward_len(uint64_t end_pos, uint64_t cand_pos,
                                                uint64_t oldest_valid,
                                                uint32_t max_len) const noexcept {
    const uint64_t avail_back = cand_pos - oldest_valid + 1u;
    const uint32_t limit = static_cast<uint32_t>(std::min<uint64_t>(max_len, avail_back));
    uint32_t len = 0u;
    while (len < limit) {
        if (ring_[(end_pos - len) & kWindowMask] != ring_[(cand_pos - len) & kWindowMask]) {
            break;
        }
        ++len;
    }
    return len;
}

OnlineNgramProposal OnlineNgramMatcher::propose(uint32_t num_drafts, bool stochastic) {
    OnlineNgramProposal out{};
    if (!config_.enabled || num_drafts == 0u ||
        num_drafts > OnlineNgramProposal::kMaxDraftTokens) {
        return out;
    }

    const auto start_tp = std::chrono::steady_clock::now();
    ++stats_.rounds;

    if (cooldown_remaining_ > 0u) {
        --cooldown_remaining_;
        ++stats_.cooldown_skips;
        const auto end_tp = std::chrono::steady_clock::now();
        stats_.cpu_us_total += std::chrono::duration<double, std::micro>(end_tp - start_tp).count();
        return out;
    }

    ensure_storage();
    const uint32_t min_match = effective_min_match(stochastic);
    const uint32_t max_match = std::max(min_match, config_.max_match);
    if (total_tokens_ < static_cast<uint64_t>(min_match) + 1u) {
        const auto end_tp = std::chrono::steady_clock::now();
        stats_.cpu_us_total += std::chrono::duration<double, std::micro>(end_tp - start_tp).count();
        return out;
    }

    const uint64_t end_pos = total_tokens_ - 1u;
    const uint64_t oldest_valid =
        total_tokens_ > kWindowCapacity ? (total_tokens_ - kWindowCapacity) : 0u;

    uint32_t best_len = 0u;
    uint64_t best_cand = 0u;

    auto eval_bucket = [&](const uint32_t* bucket) {
        for (uint32_t w = 0u; w < kBucketWays; ++w) {
            const uint32_t encoded = bucket[w];
            if (encoded == 0u) break;
            const uint64_t cand_pos = static_cast<uint64_t>(encoded) - 1u;
            if (cand_pos >= end_pos || cand_pos < oldest_valid) continue;
            const uint32_t mlen = match_backward_len(end_pos, cand_pos, oldest_valid, max_match);
            if (mlen >= min_match) {
                if (mlen > best_len || (mlen == best_len && cand_pos > best_cand)) {
                    best_len = mlen;
                    best_cand = cand_pos;
                }
            }
        }
    };

    if (end_pos >= 3u) {
        const uint32_t idx4 = bucket_index_4(
            ring_[(end_pos - 3u) & kWindowMask], ring_[(end_pos - 2u) & kWindowMask],
            ring_[(end_pos - 1u) & kWindowMask], ring_[end_pos & kWindowMask]);
        eval_bucket(&buckets4_[static_cast<std::size_t>(idx4) * kBucketWays]);
    }
    if (end_pos >= 1u) {
        const uint32_t idx2 = bucket_index_2(
            ring_[(end_pos - 1u) & kWindowMask], ring_[end_pos & kWindowMask]);
        eval_bucket(&buckets2_[static_cast<std::size_t>(idx2) * kBucketWays]);
    }

    if (best_len >= min_match) {
        const uint64_t avail = end_pos - best_cand;
        const uint32_t valid = static_cast<uint32_t>(std::min<uint64_t>(num_drafts, avail));
        if (valid > 0u) {
            out.hit = true;
            out.match_len = best_len;
            out.valid_drafts = valid;
            out.count = num_drafts;
            for (uint32_t j = 0u; j < valid; ++j) {
                out.tokens[j] = ring_[(best_cand + 1u + j) & kWindowMask];
            }
            const int32_t pad_tok = out.tokens[valid - 1u];
            for (uint32_t j = valid; j < num_drafts; ++j) {
                out.tokens[j] = pad_tok;
            }
            ++stats_.hits;
            stats_.proposed_drafts += num_drafts;
        }
    }

    const auto end_tp = std::chrono::steady_clock::now();
    stats_.cpu_us_total += std::chrono::duration<double, std::micro>(end_tp - start_tp).count();
    return out;
}

void OnlineNgramMatcher::record_outcome(uint32_t accepted_drafts) noexcept {
    stats_.accepted_drafts += accepted_drafts;
    if (accepted_drafts == 0u) {
        ++consecutive_zero_accepts_;
        if (config_.cooldown_threshold > 0u &&
            consecutive_zero_accepts_ >= config_.cooldown_threshold) {
            cooldown_remaining_ = config_.cooldown_rounds;
            consecutive_zero_accepts_ = 0u;
        }
    } else {
        consecutive_zero_accepts_ = 0u;
    }
}

}
