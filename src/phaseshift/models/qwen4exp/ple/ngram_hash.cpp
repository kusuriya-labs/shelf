#include <phaseshift/models/qwen4exp/ple/ngram_hash.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ps::qwen4exp::ple {
namespace {

constexpr std::uint64_t kMask64 = 0xFFFFFFFFFFFFFFFFull;
constexpr std::uint64_t kMaxInt63 = 0x7FFFFFFFFFFFFFFFull;
constexpr std::uint64_t kGoldenGamma = 0x9E3779B97F4A7C15ull;
constexpr std::uint32_t kMaxLayerSlot = 4095;

bool is_prime(std::uint64_t n) {
    if (n < 2ull) return false;
    if (n % 2ull == 0ull) return n == 2ull;
    for (std::uint64_t d = 3ull; d * d <= n; d += 2ull) {
        if (n % d == 0ull) return false;
    }
    return true;
}

Status validate_params(const NgramHashParams& params) {
    if (params.unigram_vocab_size == 0ull) {
        return Status::invalid_argument("unigram_vocab_size must be positive", __FILE__, __LINE__);
    }
    if (params.ple_layer_slot > kMaxLayerSlot) {
        return Status::out_of_range("ple_layer_slot too large", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

std::uint64_t splitmix64(std::uint64_t x) {
    x += kGoldenGamma;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

Status build_ngram_head_sizes(const NgramHashParams& params, std::int64_t* sizes) {
    if (sizes == nullptr) {
        return Status::invalid_argument("sizes must not be null", __FILE__, __LINE__);
    }
    const Status v = validate_params(params);
    if (!v.ok()) return v;

    const std::uint64_t first =
        static_cast<std::uint64_t>(params.ple_layer_slot) * kNgramHeads;
    const std::uint64_t need = first + kNgramHeads;
    std::vector<std::uint64_t> primes;
    primes.reserve(need);
    std::uint64_t cand = kNgramPrimeBase + 1ull;
    while (primes.size() < need) {
        if (is_prime(cand)) primes.push_back(cand);
        if (cand > kMask64 - 1024ull) {
            return Status::overflow("prime search overflow", __FILE__, __LINE__);
        }
        ++cand;
    }
    for (std::uint32_t h = 0; h < kNgramHeads; ++h) {
        sizes[h] = static_cast<std::int64_t>(primes[first + h]);
    }
    return Status::make_ok();
}

Status build_ngram_head_offsets(const std::int64_t* sizes, std::int64_t* offsets) {
    if (sizes == nullptr || offsets == nullptr) {
        return Status::invalid_argument("sizes and offsets must not be null", __FILE__, __LINE__);
    }
    std::int64_t acc = 0;
    for (std::uint32_t h = 0; h < kNgramHeads; ++h) {
        if (sizes[h] <= 0) {
            return Status::invalid_argument("vocab size must be positive", __FILE__, __LINE__);
        }
        if (acc > static_cast<std::int64_t>(kMaxInt63) - sizes[h]) {
            return Status::overflow("offset accumulation overflow", __FILE__, __LINE__);
        }
        offsets[h] = acc;
        acc += sizes[h];
    }
    return Status::make_ok();
}

Status build_layer_multipliers(const NgramHashParams& params, std::int64_t* multipliers) {
    if (multipliers == nullptr) {
        return Status::invalid_argument("multipliers must not be null", __FILE__, __LINE__);
    }
    const Status v = validate_params(params);
    if (!v.ok()) return v;

    const std::uint64_t base_seed =
        params.seed + 10007ull * static_cast<std::uint64_t>(params.ple_layer_slot);
    const std::uint64_t half_bound = std::max<std::uint64_t>(
        1ull, (kMaxInt63 / params.unigram_vocab_size) / 2ull);
    for (std::uint32_t k = 0; k < kNgramSize; ++k) {
        const std::uint64_t arg =
            (base_seed + kGoldenGamma * static_cast<std::uint64_t>(k + 1)) & kMask64;
        const std::uint64_t h = splitmix64(arg);
        multipliers[k] = static_cast<std::int64_t>(2ull * (h % half_bound) + 1ull);
    }
    return Status::make_ok();
}

Status build_ngram_id_tables(const NgramHashParams& params, NgramIdTables& out) {
    Status st = build_ngram_head_sizes(params, out.vocab_sizes);
    if (!st.ok()) return st;
    st = build_ngram_head_offsets(out.vocab_sizes, out.offsets);
    if (!st.ok()) return st;
    return build_layer_multipliers(params, out.multipliers);
}

Status shift_right_ignore_eos(const std::uint32_t* history, std::uint32_t history_len,
                              std::uint32_t shift, std::uint32_t eos_token,
                              std::uint32_t* out) {
    if (history == nullptr || out == nullptr) {
        return Status::invalid_argument("history and out must not be null", __FILE__, __LINE__);
    }
    for (std::uint32_t i = 0; i < history_len; ++i) {
        if (i < shift) {
            out[i] = eos_token;
            continue;
        }
        bool crossed = false;
        for (std::uint32_t j = i - shift; j < i; ++j) {
            if (history[j] == eos_token) {
                crossed = true;
                break;
            }
        }
        out[i] = crossed ? eos_token : history[i - shift];
    }
    return Status::make_ok();
}

Status compute_ngram_ids(const NgramIdTables& tables,
                         const std::uint32_t* context, std::uint32_t context_len,
                         const std::uint32_t* input_ids, std::uint32_t count,
                         std::uint32_t eos_token, std::uint64_t* out_ids) {
    if (context_len > kNgramContextTokens) {
        return Status::out_of_range("context_len exceeds ngram_size - 1", __FILE__, __LINE__);
    }
    if (count > 0u && (input_ids == nullptr || out_ids == nullptr)) {
        return Status::invalid_argument("input_ids and out_ids must not be null", __FILE__, __LINE__);
    }
    if (context_len > 0u && context == nullptr) {
        return Status::invalid_argument("context must not be null", __FILE__, __LINE__);
    }
    for (std::uint32_t h = 0; h < kNgramHeads; ++h) {
        if (tables.vocab_sizes[h] <= 0 || tables.offsets[h] < 0) {
            return Status::invalid_argument("invalid vocab table", __FILE__, __LINE__);
        }
    }
    for (std::uint32_t k = 0; k < kNgramSize; ++k) {
        if (tables.multipliers[k] <= 0) {
            return Status::invalid_argument("invalid multiplier table", __FILE__, __LINE__);
        }
    }

    const std::uint32_t history_len = context_len + count;
    std::vector<std::uint32_t> history(history_len);
    for (std::uint32_t i = 0; i < context_len; ++i) history[i] = context[i];
    for (std::uint32_t i = 0; i < count; ++i) history[context_len + i] = input_ids[i];

    std::vector<std::uint32_t> eos_prefix(history_len + 1u, 0u);
    for (std::uint32_t i = 0; i < history_len; ++i) {
        eos_prefix[i + 1u] = eos_prefix[i] + (history[i] == eos_token ? 1u : 0u);
    }

    for (std::uint32_t p = 0; p < count; ++p) {
        const std::uint32_t pos = context_len + p;
        std::uint64_t shifted[kNgramSize];
        for (std::uint32_t k = 0; k < kNgramSize; ++k) {
            if (pos < k) {
                shifted[k] = eos_token;
                continue;
            }
            const std::uint32_t src = pos - k;
            const bool crossed = eos_prefix[pos] > eos_prefix[src];
            shifted[k] = crossed ? eos_token : history[src];
        }
        for (std::uint32_t degree = 2; degree <= kNgramSize; ++degree) {
            std::uint64_t h = shifted[0] * static_cast<std::uint64_t>(tables.multipliers[0]);
            for (std::uint32_t k = 1; k < degree; ++k) {
                h ^= shifted[k] * static_cast<std::uint64_t>(tables.multipliers[k]);
            }
            for (std::uint32_t j = 0; j < kHeadsPerNgram; ++j) {
                const std::uint32_t head = (degree - 2u) * kHeadsPerNgram + j;
                const std::uint64_t size = static_cast<std::uint64_t>(tables.vocab_sizes[head]);
                out_ids[p * kNgramHeads + head] =
                    (h % size) + static_cast<std::uint64_t>(tables.offsets[head]);
            }
        }
    }
    return Status::make_ok();
}

void advance_ngram_context(const std::uint32_t* history, std::uint32_t history_len,
                           std::uint32_t* out_context) {
    if (history_len >= kNgramContextTokens) {
        out_context[0] = history[history_len - 2u];
        out_context[1] = history[history_len - 1u];
        return;
    }
    for (std::uint32_t i = 0; i < history_len; ++i) out_context[i] = history[i];
}

}  // namespace ps::qwen4exp::ple
