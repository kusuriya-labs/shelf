#pragma once

#include <shelf/core/status.h>
#include <shelf/models/qwen35/model/qwen35_config.h>

#include <cstddef>
#include <string>

namespace shelf {
namespace qwen35qsa {

// The indexer the checkpoint declares at the root of config.json. It scores pooled key
// blocks and every full-attention layer keeps the `budget` most important tokens, so the
// number of kept blocks is `budget / block_size`.
struct QsaIndexerSpec {
    bool declared = false;
    std::size_t budget = 0;
    // Key compression ratio: `block_size` consecutive keys form one scored block.
    std::size_t block_size = 0;
    std::size_t heads = 0;
    std::size_t head_dim = 0;
    // Channels of each indexer head that receive partial rotary.
    std::size_t rope_dim = 0;

    [[nodiscard]]
    std::size_t kept_blocks() const noexcept {
        return block_size == 0 ? 0 : budget / block_size;
    }
};

// A Qwen3.5 QSA checkpoint: the dense backbone (hybrid GDN/full attention plus the dense
// FFN) with the block-sparse indexer declared on top of it.
struct Qwen35QsaTextConfig : shelf::qwen35::Qwen35TextConfig {
    QsaIndexerSpec indexer;
};

// Reads config.json through the dense reader and then the root `qsa` declaration. A
// checkpoint that does not declare QSA is rejected here: this family only reads the
// sparse-attention contract, the dense reader is the one that does not implement it.
Result<Qwen35QsaTextConfig> read_qwen35qsa_text_config(const std::string& model_dir);

Status validate_qsa_indexer_spec(const QsaIndexerSpec& spec);

}
}
