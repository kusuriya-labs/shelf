#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/io/safetensors_reader.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace shelf {
namespace qwen4exp {

// Host-resident PLE n-gram table. Shards are shared by the tensor-parallel rank threads,
// which live in the same process. The HF shard path maps the checkpoint tensors and
// copies them into host RAM.
class PleTable {
public:
    // Packaged row layout held in host RAM.
    enum class TableFormat : std::uint8_t {
        Fp8E4M3 = 1,
        Ngram = 3,
    };

    static Result<PleTable> load(const std::string& model_dir,
                                 const Qwen4ExpTextConfig& config);

    PleTable() = default;
    PleTable(PleTable&&) noexcept = default;
    PleTable& operator=(PleTable&&) noexcept = default;
    PleTable(const PleTable&) = delete;
    PleTable& operator=(const PleTable&) = delete;
    ~PleTable() = default;

    [[nodiscard]] std::uint64_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::uint32_t row_width() const noexcept { return row_width_; }
    [[nodiscard]] std::uint32_t row_bytes() const noexcept { return row_bytes_; }
    [[nodiscard]] TableFormat format() const noexcept { return format_; }
    [[nodiscard]] std::uint32_t shard_count() const noexcept { return shard_count_; }
    [[nodiscard]] std::uint32_t shard_rows() const noexcept { return shard_rows_; }
    [[nodiscard]] std::uint64_t payload_bytes() const noexcept { return payload_bytes_; }
    // Bytes actually held in host RAM.
    [[nodiscard]] std::uint64_t host_bytes() const noexcept;
    // Number of block slots currently filled in the pinned hot cache.
    [[nodiscard]] std::size_t cached_blocks() const noexcept;
    [[nodiscard]] std::uint32_t head_count() const noexcept {
        return static_cast<std::uint32_t>(head_offsets_.size());
    }
    [[nodiscard]] const std::vector<std::int64_t>& head_offsets() const noexcept {
        return head_offsets_;
    }
    [[nodiscard]] const std::vector<std::int64_t>& head_vocab_sizes() const noexcept {
        return head_vocab_sizes_;
    }
    [[nodiscard]] const std::vector<std::int64_t>& layer_multipliers() const noexcept {
        return layer_multipliers_;
    }

    // One packaged row. Returns nullptr for an out-of-range row. The row layout depends
    // on `format()`: FP8 E4M3 or the n-gram codebook form.
    [[nodiscard]] const std::uint8_t* row_ptr(std::uint64_t row) const noexcept;

    // CPU reference decode of one row: 160 f32 values. Used by load validation and by
    // the initial CPU-gather path before the GPU-side gather exists. `include_head_bias`
    // applies the per-head bias stored with the checkpoint table.
    Result<std::vector<float>> decode_row(std::uint64_t row,
                                          bool include_head_bias = true) const;

    // Token history -> global table row ids. history carries the (ngram_size - 1)
    // context tokens followed by the current input; out_len says how many trailing
    // positions to return. The result is out_len * head_count row ids.
    Result<std::vector<std::uint64_t>> ngram_rows(
        const std::vector<std::int32_t>& history, std::size_t out_len) const;

    // CPU reference gather: out_len * head_count rows, each decoded to 160 f32 values,
    // concatenated per position. Used to validate the row layout and the hash.
    Result<std::vector<float>> gather(const std::vector<std::int32_t>& history,
                                      std::size_t out_len) const;

    // Same gather, rounded once to BF16 for the graph's PLE external input.
    Result<std::vector<bf16_t>> gather_bf16(const std::vector<std::int32_t>& history,
                                            std::size_t out_len) const;

    // Stateless gather for a run of tokens that follow `prefix_tail` (the last
    // ngram_size - 1 tokens of the prefix). Returns [token][hidden] BF16 rows without
    // touching any caller history.
    Result<std::vector<bf16_t>> gather_bf16_stateless(
        const std::vector<std::int32_t>& prefix_tail,
        std::span<const std::int32_t> tokens) const;


private:
    struct NgramHead {
        std::unique_ptr<io::SafetensorsReader> reader;
        const std::uint8_t* codes = nullptr;
        const std::uint8_t* scales = nullptr;
        std::vector<std::uint16_t> codebook;
        std::size_t rows = 0;
        std::size_t width = 0;
        unsigned bits = 0;
    };
    struct NgramBlock {
        std::uint64_t key = 0;
        std::vector<std::uint8_t> data;
    };
    struct NgramState {
        std::vector<NgramHead> heads;
        std::vector<NgramBlock> blocks;
        std::unordered_map<std::uint64_t, std::size_t> index;
        std::size_t capacity = 0;
        std::size_t next = 0;
        std::mutex mutex;
    };
    std::shared_ptr<NgramState> ngram_;
    static Result<PleTable> load_ngram(const std::string& root,
                                       const Qwen4ExpTextConfig& config);
    static Result<PleTable> load_index_shards(const std::string& model_dir,
                                              const Qwen4ExpTextConfig& config);
    std::size_t head_for_row(std::uint64_t row) const noexcept;

    TableFormat format_ = TableFormat::Fp8E4M3;
    float fp8_scale_ = 1.0f;

    struct FreeDeleter {
        void operator()(void* p) const noexcept;
    };
    struct Shard {
        std::unique_ptr<std::uint8_t, FreeDeleter> data;
        std::size_t bytes = 0;
    };

    std::uint64_t rows_ = 0;
    std::uint32_t row_width_ = 0;
    std::uint32_t row_bytes_ = 0;
    std::uint32_t shard_count_ = 0;
    std::uint32_t shard_rows_ = 0;
    std::uint32_t ngram_size_ = 0;
    std::uint32_t heads_per_ngram_ = 0;
    std::int32_t eos_token_id_ = 0;
    std::uint64_t payload_bytes_ = 0;

    std::vector<Shard> shards_;
    std::vector<std::uint16_t> head_bias_f16_;
    std::vector<std::int64_t> head_offsets_;
    std::vector<std::int64_t> head_vocab_sizes_;
    std::vector<std::int64_t> layer_multipliers_;
};

}  // namespace qwen4exp
}  // namespace shelf
