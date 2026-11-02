#pragma once

#include <shelf/core/status.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ps::io {

struct RowRequest {
    std::uint64_t absolute_offset = 0;
    std::uint64_t byte_length = 0;
};

struct AlignedBlock {
    std::uint64_t file_offset = 0;
    std::uint64_t byte_length = 0;
};

struct RowReadPlan {
    std::vector<AlignedBlock> blocks;
    std::vector<std::uint64_t> row_block;
    std::vector<std::uint64_t> row_offset;
    std::uint64_t alignment = 0;
    std::uint64_t total_block_bytes = 0;
    bool fully_aligned = false;
};

Result<std::uint64_t> probe_odirect_alignment(const std::string& path);

Result<RowReadPlan> plan_row_reads(const std::vector<RowRequest>& rows,
                                   std::uint64_t file_size,
                                   std::uint64_t alignment);

Status read_blocks_buffered(const std::string& path,
                            const std::vector<AlignedBlock>& blocks, void* dst,
                            std::size_t dst_bytes);

Status read_blocks_direct(const std::string& path,
                          const std::vector<AlignedBlock>& blocks, void* dst,
                          std::size_t dst_bytes, std::uint64_t alignment);

Status read_rows_buffered(const std::string& path,
                          const std::vector<RowRequest>& rows, void* dst,
                          std::size_t dst_bytes, RowReadPlan* out_plan = nullptr);

}  // namespace ps::io
