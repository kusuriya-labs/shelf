#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/io/hipfile_reader.h>
#include <phaseshift/weights/weight_loader.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ps::qwen4exp::ple {

struct TableLayout {
    std::uint32_t shard_count = 0;
    std::uint64_t rows_per_shard = 0;
    std::uint64_t row_cols = 0;
    std::uint64_t row_bytes = 0;
    std::string tensor_prefix;
    std::string tensor_suffix;
};

struct RowLocation {
    std::string tensor_name;
    std::string shard_file;
    std::uint64_t global_row = 0;
    std::uint64_t row_in_shard = 0;
    std::uint64_t absolute_offset = 0;
    std::uint64_t byte_length = 0;
};

struct SsdGatherStats {
    io::SsdReadPath path = io::SsdReadPath::none;
    std::uint64_t logical_bytes = 0;
    std::uint64_t physical_bytes = 0;
    std::uint64_t block_count = 0;
    std::uint64_t hipfile_blocks = 0;
    std::uint64_t posix_blocks = 0;
    std::uint64_t fastpath_eligible_blocks = 0;
    bool fastpath_eligible = false;
    bool short_read = false;
};

TableLayout bf16_layout();

Result<TableLayout> table_layout(const weights::SafetensorsCollection& collection,
                                 const std::string& tensor_prefix,
                                 const std::string& tensor_suffix = ".weight");

Status locate_rows(const TableLayout& layout,
                   const weights::SafetensorsCollection& collection,
                   const std::uint64_t* row_ids, std::uint32_t count,
                   std::vector<RowLocation>& out);

Status gather_rows_ssd(const TableLayout& layout,
                       const weights::SafetensorsCollection& collection,
                       const std::uint64_t* row_ids, std::uint32_t count,
                       void* device_dst, hipStream_t stream, SsdGatherStats& stats,
                       bool allow_hipfile = true);

Status gather_rows_ram(const TableLayout& layout,
                       const weights::SafetensorsCollection& collection,
                       const std::uint64_t* row_ids, std::uint32_t count,
                       void* device_dst, hipStream_t stream);

Status gather_rows_vram(const TableLayout& layout, const void* device_table,
                        std::uint64_t table_rows,
                        const std::uint64_t* device_row_ids, std::uint32_t count,
                        void* device_dst, hipStream_t stream);

}  // namespace ps::qwen4exp::ple
