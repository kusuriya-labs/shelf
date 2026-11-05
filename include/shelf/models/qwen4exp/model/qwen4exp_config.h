#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/config.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace shelf {
namespace qwen4exp {

struct Qwen4ExpMoeConfig {
    std::size_t num_experts = 0;
    std::size_t num_experts_per_tok = 0;
    std::size_t moe_intermediate_size = 0;
    std::size_t shared_expert_intermediate_size = 0;
};

struct Qwen4ExpPleConfig {
    // One-based layer ids, as text_config.ple_layer_ids declares them.
    std::vector<std::size_t> layer_ids;
    std::size_t embed_dim = 0;
    std::size_t ngram_size = 0;
    std::size_t heads_per_ngram = 0;
    std::size_t ngram_vocab_size_base = 0;
    std::size_t make_ngram_vocab_size_divisible_by = 128;
    std::size_t split_ngram_parts = 0;
    std::size_t conv_kernel_size = 4;
    std::size_t seed = 1234;
    int32_t eos_token_id = 0;
};

struct Qwen4ExpIndexerConfig {
    std::size_t n_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t head_dim = 0;
    std::size_t budget = 0;
    std::size_t compress_ratio = 1;
};

// Qwen3.8-Flash-Next text config. The shared Qwen hybrid geometry is the base; the
// hyper-connection width, the indexer, the PLE hash table and the routed-expert FFN are
// this family's own fields.
struct Qwen4ExpTextConfig : shelf::qwen_hybrid::QwenHybridConfig {
    Qwen4ExpMoeConfig ffn;
    Qwen4ExpPleConfig ple;
    Qwen4ExpIndexerConfig indexer;

    std::size_t hc_count = 0;
    std::size_t hc_lowrank = 0;

    int32_t image_token_id = -1;
    int32_t video_token_id = -1;

    std::string output_gate_type;

    [[nodiscard]] bool layer_has_ple(std::size_t layer) const noexcept;
    // Zero-based decoder layer numbers from the one-based ple_layer_ids.
    [[nodiscard]] std::vector<std::size_t> ple_layers() const;
};

// Structural probe: config.json's text_config declares ple_layer_ids, hc_count and
// ngram_size. It reads no weight and never uses the model_type string to choose a code
// path. The identity gate is still the first check in the reader.
Result<bool> declares_qwen4exp_structure(const std::string& model_dir);

Result<Qwen4ExpTextConfig> read_qwen4exp_text_config(const std::string& model_dir);

// The padded n-gram row count implied by the config: the sum of the per-head prime vocab
// sizes rounded up to make_ngram_vocab_size_divisible_by.
Result<std::size_t> qwen4exp_ple_padded_rows(const Qwen4ExpTextConfig& config,
                                             std::size_t ple_layer_index = 0);

// The 16 (or (ngram_size - 1) * heads_per_ngram) per-head prime vocab sizes and their
// cumulative offsets. Pure function of the config, used to validate the checkpoint's
// metadata and to compute row ids.
Result<std::vector<std::size_t>> qwen4exp_ple_head_vocab_sizes(
    const Qwen4ExpTextConfig& config, std::size_t ple_layer_index = 0);
Result<std::vector<std::size_t>> qwen4exp_ple_head_offsets(
    const Qwen4ExpTextConfig& config, std::size_t ple_layer_index = 0);

// The per-position hash multipliers. The checkpoint stores them, but they are a pure
// function of vocab_size, ngram_size, the PLE layer index and seed.
Result<std::vector<std::int64_t>> qwen4exp_ple_layer_multipliers(
    const Qwen4ExpTextConfig& config, std::size_t ple_layer_index = 0);

}  // namespace qwen4exp
}  // namespace shelf
