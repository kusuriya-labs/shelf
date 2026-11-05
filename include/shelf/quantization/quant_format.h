#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shelf::quantization {

constexpr uint32_t kQuantBlockElements = 32;
constexpr uint32_t kFp8ScaleBlockN = 128;
constexpr uint32_t kFp8ScaleBlockK = 128;
constexpr uint32_t kMxfp4ScaleBlockK = 32;
constexpr uint32_t kMxfp4WeightsPerCode = 2;

// Metadata placement relative to the codes matrix.
//
//   PerBlock        one element per (row, K block)
//   PerSuperBlock   one element per (row, superblock)
//   MatrixBlock     one element per (N block, K block) tile, shared by every
//                   row inside the tile. `block_n` / `block_k` describe the
//                   tile geometry; `block_k == 0` means "format block".
enum class MetaCountMode : uint8_t {
    None = 0,
    PerBlock = 1,
    PerSuperBlock = 2,
    MatrixBlock = 3,
    PerRow = 4,
};

enum class MetaStorage : uint8_t {
    U8 = 0,
    F32 = 1,
};

struct QuantMetaDesc {
    uint32_t element_bytes = 0;
    MetaCountMode count_mode = MetaCountMode::None;
    MetaStorage storage = MetaStorage::U8;
    uint32_t block_n = 1;
    uint32_t block_k = 0;
};

struct QuantFormatDesc {
    uint32_t block_elements = 0;
    uint32_t blocks_per_superblock = 1;
    uint32_t code_bytes_per_block = 0;
    QuantMetaDesc meta[4];
};

enum class QuantFormatId : uint8_t {
    None = 0,
    Bf16 = 1,
    Fp8Finegrained = 2,
    Mxfp4Finegrained = 3,
    Exl3Trellis = 4,
    Fp8PerChannel = 5,
};

static_assert(static_cast<uint8_t>(QuantFormatId::None) == 0);
static_assert(static_cast<uint8_t>(QuantFormatId::Bf16) == 1);
static_assert(static_cast<uint8_t>(QuantFormatId::Fp8Finegrained) == 2);
static_assert(static_cast<uint8_t>(QuantFormatId::Mxfp4Finegrained) == 3);
static_assert(static_cast<uint8_t>(QuantFormatId::Exl3Trellis) == 4);
static_assert(static_cast<uint8_t>(QuantFormatId::Fp8PerChannel) == 5);

inline constexpr QuantFormatDesc kBf16FormatDesc = {
    16, 1, 32,
    {
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
    },
};

// FP8 E4M3 codes are one byte per element; the FP32 scale tile is shared by
// 128 output rows and 128 K elements.
inline constexpr QuantFormatDesc kFp8FinegrainedFormatDesc = {
    kFp8ScaleBlockK, 1, kFp8ScaleBlockK,
    {
        {4, MetaCountMode::MatrixBlock, MetaStorage::F32, kFp8ScaleBlockN, kFp8ScaleBlockK},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
    },
};

// FP8 E4M3 with one F32 scale per output row. The activation side is one F32
// scale per token, so the GEMM applies both scales once in the epilogue.
inline constexpr QuantFormatDesc kFp8PerChannelFormatDesc = {
    kFp8ScaleBlockK, 1, kFp8ScaleBlockK,
    {
        {4, MetaCountMode::PerRow, MetaStorage::F32, 1, 0},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
    },
};

// OCP MX FP4: two E2M1 codes share one byte along K, and every output row owns
// one E8M0 scale per 32-element block.
inline constexpr QuantFormatDesc kMxfp4FinegrainedFormatDesc = {
    kMxfp4ScaleBlockK,
    1,
    kMxfp4ScaleBlockK / kMxfp4WeightsPerCode,
    {
        {1, MetaCountMode::PerBlock, MetaStorage::U8, 1, kMxfp4ScaleBlockK},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
    },
};

inline constexpr QuantFormatDesc kExl3TrellisFormatDesc = {
    128, 1, 0,
    {
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
        {0, MetaCountMode::None, MetaStorage::U8},
    },
};

inline constexpr const QuantFormatDesc* quant_format_desc(QuantFormatId id) {
    switch (id) {
        case QuantFormatId::Bf16: return &kBf16FormatDesc;
        case QuantFormatId::Fp8Finegrained: return &kFp8FinegrainedFormatDesc;
        case QuantFormatId::Mxfp4Finegrained: return &kMxfp4FinegrainedFormatDesc;
        case QuantFormatId::Exl3Trellis: return &kExl3TrellisFormatDesc;
        case QuantFormatId::Fp8PerChannel: return &kFp8PerChannelFormatDesc;
        default: return nullptr;
    }
}

inline uint64_t quant_num_blocks(uint64_t padded_k, const QuantFormatDesc& f) {
    return padded_k / f.block_elements;
}

inline uint64_t quant_num_superblocks(uint64_t padded_k, const QuantFormatDesc& f) {
    const uint64_t nb = quant_num_blocks(padded_k, f);
    return (nb + f.blocks_per_superblock - 1u) / f.blocks_per_superblock;
}

inline uint64_t quant_codes_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return rows * quant_num_blocks(padded_k, f) * f.code_bytes_per_block;
}

// Number of metadata elements for a matrix batch of `batch` matrices, each with
// `rows` output rows and `padded_k` K elements.
inline uint64_t quant_metadata_count(
    uint64_t batch, uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f, uint32_t idx) {
    const QuantMetaDesc& m = f.meta[idx];
    switch (m.count_mode) {
        case MetaCountMode::None:
            return 0;
        case MetaCountMode::PerBlock:
            return batch * rows * quant_num_blocks(padded_k, f);
        case MetaCountMode::PerSuperBlock:
            return batch * rows * quant_num_superblocks(padded_k, f);
        case MetaCountMode::PerRow:
            return batch * rows;
        case MetaCountMode::MatrixBlock: {
            const uint64_t bn = m.block_n != 0 ? m.block_n : 1;
            const uint64_t bk = m.block_k != 0 ? m.block_k : f.block_elements;
            return batch * ((rows + bn - 1u) / bn) * ((padded_k + bk - 1u) / bk);
        }
    }
    return 0;
}

inline uint64_t quant_metadata_bytes(
    uint64_t batch, uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f, uint32_t idx) {
    const QuantMetaDesc& m = f.meta[idx];
    if (m.count_mode == MetaCountMode::None || m.element_bytes == 0) return 0;
    return quant_metadata_count(batch, rows, padded_k, f, idx) * m.element_bytes;
}

inline uint64_t quant_metadata_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f, uint32_t idx) {
    return quant_metadata_bytes(1, rows, padded_k, f, idx);
}

inline uint64_t quant_metadata1_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_metadata_bytes(rows, padded_k, f, 0);
}
inline uint64_t quant_metadata2_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_metadata_bytes(rows, padded_k, f, 1);
}
inline uint64_t quant_metadata3_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_metadata_bytes(rows, padded_k, f, 2);
}
inline uint64_t quant_metadata4_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_metadata_bytes(rows, padded_k, f, 3);
}

inline uint64_t quant_payload_bytes(
    uint64_t batch, uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_codes_bytes(batch * rows, padded_k, f) +
           quant_metadata_bytes(batch, rows, padded_k, f, 0) +
           quant_metadata_bytes(batch, rows, padded_k, f, 1) +
           quant_metadata_bytes(batch, rows, padded_k, f, 2) +
           quant_metadata_bytes(batch, rows, padded_k, f, 3);
}

inline uint64_t quant_payload_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_payload_bytes(1, rows, padded_k, f);
}

inline uint64_t quant_padded_k(uint64_t k, QuantFormatId id) {
    const QuantFormatDesc* f = quant_format_desc(id);
    if (f == nullptr || id == QuantFormatId::Bf16) return k;
    if (id == QuantFormatId::Exl3Trellis) return k;
    return ((k + f->block_elements - 1u) / f->block_elements) * f->block_elements;
}

inline uint64_t quant_element_count(const std::vector<int64_t>& shape) {
    uint64_t n = 1;
    for (int64_t d : shape) {
        n *= static_cast<uint64_t>(d);
    }
    return n;
}

inline uint64_t quant_row_count(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    uint64_t rows = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        rows *= static_cast<uint64_t>(shape[i]);
    }
    return rows;
}

}  // namespace shelf::quantization
