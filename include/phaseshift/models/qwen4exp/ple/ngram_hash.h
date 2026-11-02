#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>

namespace ps::qwen4exp::ple {

inline constexpr std::uint32_t kNgramSize = 3;
inline constexpr std::uint32_t kHeadsPerNgram = 8;
inline constexpr std::uint32_t kNgramHeads = (kNgramSize - 1) * kHeadsPerNgram;
inline constexpr std::uint32_t kNgramContextTokens = kNgramSize - 1;
inline constexpr std::uint64_t kNgramPrimeBase = 19999999ull;
inline constexpr std::uint64_t kNgramDefaultSeed = 1234ull;

struct NgramHashParams {
    std::uint64_t seed = kNgramDefaultSeed;
    std::uint64_t unigram_vocab_size = 248320ull;
    std::uint32_t ple_layer_slot = 0;
};

struct NgramIdTables {
    std::int64_t vocab_sizes[kNgramHeads] = {};
    std::int64_t offsets[kNgramHeads] = {};
    std::int64_t multipliers[kNgramSize] = {};
};

std::uint64_t splitmix64(std::uint64_t x);

Status build_ngram_head_sizes(const NgramHashParams& params, std::int64_t* sizes);
Status build_ngram_head_offsets(const std::int64_t* sizes, std::int64_t* offsets);
Status build_layer_multipliers(const NgramHashParams& params, std::int64_t* multipliers);
Status build_ngram_id_tables(const NgramHashParams& params, NgramIdTables& out);

Status shift_right_ignore_eos(const std::uint32_t* history, std::uint32_t history_len,
                              std::uint32_t shift, std::uint32_t eos_token,
                              std::uint32_t* out);

Status compute_ngram_ids(const NgramIdTables& tables,
                         const std::uint32_t* context, std::uint32_t context_len,
                         const std::uint32_t* input_ids, std::uint32_t count,
                         std::uint32_t eos_token, std::uint64_t* out_ids);

void advance_ngram_context(const std::uint32_t* history, std::uint32_t history_len,
                           std::uint32_t* out_context);

}  // namespace ps::qwen4exp::ple
