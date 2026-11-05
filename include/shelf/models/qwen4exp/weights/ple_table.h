#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace shelf {
namespace qwen4exp {

// Host-resident PLE n-gram table. The 128 shards are read with pread into anonymous
// RAM; no GPU allocation and no pinned host allocation are made here. The table is
// shared by the tensor-parallel rank threads, which live in the same process.
class PleTable {
public:
    static Result<PleTable> load(const std::string& model_dir,
                                 const Qwen4ExpTextConfig& config,
                                 bool fadvise_dontneed = true);

    PleTable() = default;
    PleTable(PleTable&&) noexcept = default;
    PleTable& operator=(PleTable&&) noexcept = default;
    PleTable(const PleTable&) = delete;
    PleTable& operator=(const PleTable&) = delete;
    ~PleTable() = default;

    [[nodiscard]] std::uint64_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::uint32_t row_width() const noexcept { return row_width_; }
    [[nodiscard]] std::uint32_t row_bytes() const noexcept { return row_bytes_; }
    [[nodiscard]] std::uint32_t k_bits() const noexcept { return k_bits_; }
    [[nodiscard]] std::uint32_t shard_count() const noexcept { return shard_count_; }
    [[nodiscard]] std::uint32_t shard_rows() const noexcept { return shard_rows_; }
    [[nodiscard]] std::uint64_t payload_bytes() const noexcept { return payload_bytes_; }
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

    // One 162-byte (K=8) packed row. Returns nullptr for an out-of-range row.
    [[nodiscard]] const std::uint8_t* row_ptr(std::uint64_t row) const noexcept;

    // CPU reference decode of one row: 160 f32 values. Used by load validation and by
    // the initial CPU-gather path before the GPU-side gather exists.
    Result<std::vector<float>> decode_row(std::uint64_t row) const;

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

    // Re-read one packaged row from the source file. Validation only.
    Result<std::vector<std::uint8_t>> read_source_row(const std::string& model_dir,
                                                      std::uint64_t row) const;

private:
    struct FreeDeleter {
        void operator()(void* p) const noexcept;
    };
    struct Shard {
        std::unique_ptr<std::uint8_t, FreeDeleter> data;
        std::size_t bytes = 0;
        std::uint64_t file_offset = 0;
    };

    std::string file_path_;
    std::uint64_t data_base_ = 0;
    std::uint64_t rows_ = 0;
    std::uint32_t row_width_ = 0;
    std::uint32_t row_bytes_ = 0;
    std::uint32_t k_bits_ = 0;
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
