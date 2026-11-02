#include <phaseshift/models/qwen4exp/ple/ple_row_gather.h>

#include <phaseshift/io/row_reader.h>
#include <phaseshift/kernels/optimized/row_gather.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <map>
#include <memory>
#include <vector>

namespace ps::qwen4exp::ple {
namespace {

constexpr std::uint64_t kFastpathAlignment = 4096;


struct DeviceBuffer {
    void* ptr = nullptr;
    DeviceBuffer() = default;
    ~DeviceBuffer() {
        if (ptr != nullptr) hipFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    Status alloc(std::size_t bytes) {
        if (bytes == 0) return Status::invalid_argument("empty device buffer", __FILE__, __LINE__);
        const hipError_t e = hipMalloc(&ptr, bytes);
        if (e != hipSuccess) {
            ptr = nullptr;
            return Status::hip_error("hipMalloc", hipGetErrorString(e), __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
};

struct HostBuffer {
    void* ptr = nullptr;
    explicit HostBuffer(std::size_t bytes) {
        if (bytes > 0) ptr = std::malloc(bytes);
    }
    ~HostBuffer() { std::free(ptr); }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
};

Status sync_stream(hipStream_t stream) {
    const hipError_t e = hipStreamSynchronize(stream);
    if (e != hipSuccess) {
        return Status::hip_error("hipStreamSynchronize", hipGetErrorString(e), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

std::string shard_tensor_name(const TableLayout& layout, std::uint64_t shard) {
    return layout.tensor_prefix + std::to_string(shard) + layout.tensor_suffix;
}

}  // namespace

TableLayout bf16_layout() {
    TableLayout layout;
    layout.shard_count = 128;
    layout.rows_per_shard = 2500012;
    layout.row_cols = 160;
    layout.row_bytes = 160 * 2;
    layout.tensor_prefix =
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
    layout.tensor_suffix = ".weight";
    return layout;
}

Result<TableLayout> table_layout(const weights::SafetensorsCollection& collection,
                                 const std::string& tensor_prefix,
                                 const std::string& tensor_suffix) {
    TableLayout layout;
    layout.tensor_prefix = tensor_prefix;
    layout.tensor_suffix = tensor_suffix;

    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t unit_bytes = 0;
    bool have_shard = false;
    for (std::uint32_t shard = 0; ; ++shard) {
        const std::string name = shard_tensor_name(layout, shard);
        auto spec = collection.tensor_spec(name);
        if (!spec.ok()) break;
        if (spec.value().shape.size() != 2) {
            return Status::invalid_argument(
                ("PLE shard must be rank-2: " + name).c_str(), __FILE__, __LINE__);
        }
        auto span = collection.axis_span(name, 0, 1);
        if (!span.ok()) return span.status();
        if (!have_shard) {
            rows = spec.value().shape[0];
            cols = spec.value().shape[1];
            unit_bytes = span.value().unit_bytes;
            have_shard = true;
        } else if (spec.value().shape[0] != rows || spec.value().shape[1] != cols ||
                   span.value().unit_bytes != unit_bytes) {
            return Status::invalid_argument(
                ("PLE shard geometry mismatch: " + name).c_str(), __FILE__, __LINE__);
        }
        layout.shard_count = shard + 1;
    }
    if (!have_shard) {
        return Status::invalid_argument("no PLE shard matches the prefix",
                                    __FILE__, __LINE__);
    }
    layout.rows_per_shard = rows;
    layout.row_cols = cols;
    layout.row_bytes = unit_bytes;
    return layout;
}

Status locate_rows(const TableLayout& layout,
                   const weights::SafetensorsCollection& collection,
                   const std::uint64_t* row_ids, std::uint32_t count,
                   std::vector<RowLocation>& out) {
    if (row_ids == nullptr || count == 0) {
        return Status::invalid_argument("empty row id list", __FILE__, __LINE__);
    }
    if (layout.rows_per_shard == 0 || layout.shard_count == 0) {
        return Status::invalid_argument("invalid table layout", __FILE__, __LINE__);
    }
    const std::uint64_t total_rows =
        static_cast<std::uint64_t>(layout.shard_count) * layout.rows_per_shard;

    out.clear();
    out.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint64_t global = row_ids[i];
        if (global >= total_rows) {
            return Status::out_of_range("PLE row id is out of the table", __FILE__, __LINE__);
        }
        const std::uint64_t shard = global / layout.rows_per_shard;
        const std::uint64_t in_shard = global % layout.rows_per_shard;

        RowLocation loc;
        loc.global_row = global;
        loc.row_in_shard = in_shard;
        loc.tensor_name = shard_tensor_name(layout, shard);

        auto shard_result = collection.tensor_shard(loc.tensor_name);
        if (!shard_result.ok()) return shard_result.status();
        loc.shard_file = shard_result.value();

        auto span_result = collection.axis_span(loc.tensor_name, in_shard, 1);
        if (!span_result.ok()) return span_result.status();
        if (span_result.value().byte_length != layout.row_bytes) {
            return Status::invalid_argument("PLE row byte length mismatch", __FILE__, __LINE__);
        }
        loc.absolute_offset = span_result.value().absolute_offset;
        loc.byte_length = span_result.value().byte_length;
        out.push_back(std::move(loc));
    }
    return Status::make_ok();
}

Status gather_rows_ssd(const TableLayout& layout,
                       const weights::SafetensorsCollection& collection,
                       const std::uint64_t* row_ids, std::uint32_t count,
                       void* device_dst, hipStream_t stream, SsdGatherStats& stats,
                       bool allow_hipfile) {
    stats = SsdGatherStats{};

    if (device_dst == nullptr) {
        return Status::invalid_argument("null device destination", __FILE__, __LINE__);
    }
    std::vector<RowLocation> locations;
    Status st = locate_rows(layout, collection, row_ids, count, locations);
    if (!st.ok()) return st;

    std::map<std::string, std::vector<std::uint32_t>> by_file;
    for (std::uint32_t i = 0; i < count; ++i) by_file[locations[i].shard_file].push_back(i);

    const std::size_t total_rows_bytes =
        static_cast<std::size_t>(count) * static_cast<std::size_t>(layout.row_bytes);

    std::vector<kernel::RowGatherItem> items(count);

    std::size_t staging_total = 0;
    struct GroupPlan {
        std::string shard_file;
        io::RowReadPlan plan;
        std::vector<std::uint32_t> rows;
        std::size_t staging_base = 0;
    };
    std::vector<GroupPlan> groups;

    for (const auto& kv : by_file) {
        GroupPlan group;
        group.shard_file = kv.first;
        group.rows = kv.second;

        auto size_result = collection.shard_file_size(group.shard_file);
        if (!size_result.ok()) return size_result.status();

        std::vector<io::RowRequest> requests;
        requests.reserve(kv.second.size());
        for (std::uint32_t idx : kv.second) {
            io::RowRequest req;
            req.absolute_offset = locations[idx].absolute_offset;
            req.byte_length = locations[idx].byte_length;
            requests.push_back(req);
        }

        std::uint64_t alignment = 4096;
        const std::string full_path = collection.model_dir() + "/" + group.shard_file;
        auto probe = io::probe_odirect_alignment(full_path);
        if (probe.ok() && probe.value() > alignment) alignment = probe.value();

        auto plan_result = io::plan_row_reads(requests, size_result.value(), alignment);
        if (!plan_result.ok()) return plan_result.status();
        group.plan = plan_result.release();
        group.staging_base = staging_total;
        staging_total += static_cast<std::size_t>(group.plan.total_block_bytes);
        groups.push_back(std::move(group));
    }

    for (GroupPlan& group : groups) {
        for (std::size_t k = 0; k < group.rows.size(); ++k) {
            const std::uint32_t row_index = group.rows[k];
            std::uint64_t block_base = 0;
            const std::uint64_t target_block = group.plan.row_block[k];
            for (std::uint64_t b = 0; b < target_block; ++b) {
                block_base += group.plan.blocks[b].byte_length;
            }
            kernel::RowGatherItem item;
            item.src_offset = group.staging_base + block_base + group.plan.row_offset[k];
            item.dst_offset = static_cast<std::uint64_t>(row_index) * layout.row_bytes;
            item.bytes = static_cast<std::uint32_t>(layout.row_bytes);
            items[row_index] = item;
        }
        stats.block_count += group.plan.blocks.size();
        stats.physical_bytes += group.plan.total_block_bytes;
        for (const io::AlignedBlock& block : group.plan.blocks) {
            if (block.file_offset % kFastpathAlignment == 0 &&
                block.byte_length % kFastpathAlignment == 0) {
                ++stats.fastpath_eligible_blocks;
            }
        }
    }
    stats.logical_bytes = total_rows_bytes;
    stats.fastpath_eligible = stats.fastpath_eligible_blocks == stats.block_count;

    DeviceBuffer device_staging;
    st = device_staging.alloc(staging_total);
    if (!st.ok()) return st;

    bool use_hipfile = allow_hipfile && io::hipfile_available();

    if (use_hipfile) {
        for (GroupPlan& group : groups) {
            const std::string full_path = collection.model_dir() + "/" + group.shard_file;
            io::HipFileReader reader;
            st = reader.open(full_path, false);
            if (!st.ok()) {
                use_hipfile = false;
                break;
            }
            st = reader.register_buffer(device_staging.ptr, staging_total);
            if (!st.ok()) {
                use_hipfile = false;
                break;
            }
            auto* base = static_cast<char*>(device_staging.ptr);
            bool group_ok = true;
            for (std::size_t b = 0; b < group.plan.blocks.size(); ++b) {
                const io::AlignedBlock& block = group.plan.blocks[b];
                std::uint64_t block_base = 0;
                for (std::size_t j = 0; j < b; ++j) block_base += group.plan.blocks[j].byte_length;
                io::SsdReadResult read_result;
                st = reader.read(base + group.staging_base + block_base, block.byte_length,
                                 block.file_offset, 0, read_result);
                if (!st.ok() || read_result.short_read) {
                    if (!st.ok() && read_result.path == io::SsdReadPath::none) {
                        group_ok = false;
                        break;
                    }
                    if (read_result.short_read) stats.short_read = true;
                }
                if (read_result.path == io::SsdReadPath::hipfile) {
                    ++stats.hipfile_blocks;
                } else if (read_result.path == io::SsdReadPath::posix) {
                    ++stats.posix_blocks;
                }
                if (!st.ok()) {
                    group_ok = false;
                    break;
                }
            }
            reader.shutdown();
            if (!group_ok) {
                use_hipfile = false;
                break;
            }
        }
    }

    if (use_hipfile) {
        stats.path = io::SsdReadPath::hipfile;

    } else {
        HostBuffer host_staging(staging_total);
        if (host_staging.ptr == nullptr) {
            return Status::insufficient_memory("host staging", __FILE__, __LINE__);
        }
        auto* hbase = static_cast<char*>(host_staging.ptr);
        for (GroupPlan& group : groups) {
            const std::string full_path = collection.model_dir() + "/" + group.shard_file;
            std::vector<char> group_bytes(static_cast<std::size_t>(group.plan.total_block_bytes));
            Status read_status = Status::make_ok();
            if (group.plan.fully_aligned) {
                void* aligned = nullptr;
                if (posix_memalign(&aligned, group.plan.alignment,
                                    group_bytes.size()) == 0) {
                    read_status = io::read_blocks_direct(full_path, group.plan.blocks, aligned,
                                                         group_bytes.size(),
                                                         group.plan.alignment);
                    if (read_status.ok()) {
                        std::memcpy(group_bytes.data(), aligned, group_bytes.size());
                    }
                    std::free(aligned);
                } else {
                    read_status = io::read_blocks_buffered(full_path, group.plan.blocks,
                                                           group_bytes.data(),
                                                           group_bytes.size());
                }
            } else {
                read_status = io::read_blocks_buffered(full_path, group.plan.blocks,
                                                       group_bytes.data(),
                                                       group_bytes.size());
            }
            if (!read_status.ok()) return read_status;
            std::memcpy(hbase + group.staging_base, group_bytes.data(), group_bytes.size());
        }
        const hipError_t he = hipMemcpy(device_staging.ptr, host_staging.ptr, staging_total,
                                        hipMemcpyHostToDevice);
        if (he != hipSuccess) {
            return Status::hip_error("hipMemcpy", hipGetErrorString(he), __FILE__, __LINE__);
        }
        stats.path = io::SsdReadPath::posix;
    }

    DeviceBuffer device_items;
    st = device_items.alloc(items.size() * sizeof(kernel::RowGatherItem));
    if (!st.ok()) return st;
    const hipError_t ie = hipMemcpy(device_items.ptr, items.data(),
                                    items.size() * sizeof(kernel::RowGatherItem),
                                    hipMemcpyHostToDevice);
    if (ie != hipSuccess) {
        return Status::hip_error("hipMemcpy", hipGetErrorString(ie), __FILE__, __LINE__);
    }

    const hipError_t ge = kernel::launch_row_gather(
        device_staging.ptr, static_cast<const kernel::RowGatherItem*>(device_items.ptr),
        count, device_dst, stream);
    if (ge != hipSuccess) {
        return Status::hip_error("launch_row_gather", hipGetErrorString(ge), __FILE__, __LINE__);
    }
    return sync_stream(stream);
}

Status gather_rows_ram(const TableLayout& layout,
                       const weights::SafetensorsCollection& collection,
                       const std::uint64_t* row_ids, std::uint32_t count,
                       void* device_dst, hipStream_t stream) {
    if (device_dst == nullptr) {
        return Status::invalid_argument("null device destination", __FILE__, __LINE__);
    }
    std::vector<RowLocation> locations;
    Status st = locate_rows(layout, collection, row_ids, count, locations);
    if (!st.ok()) return st;

    const std::size_t total_bytes =
        static_cast<std::size_t>(count) * static_cast<std::size_t>(layout.row_bytes);
    HostBuffer staging(total_bytes);
    if (staging.ptr == nullptr) {
        return Status::insufficient_memory("host staging", __FILE__, __LINE__);
    }
    auto* out = static_cast<char*>(staging.ptr);

    std::map<std::string, std::vector<std::uint32_t>> by_file;
    for (std::uint32_t i = 0; i < count; ++i) by_file[locations[i].shard_file].push_back(i);

    for (const auto& kv : by_file) {
        for (std::uint32_t idx : kv.second) {
            const RowLocation& loc = locations[idx];
            std::size_t tensor_bytes = 0;
            auto data_result = collection.tensor_data(loc.tensor_name, tensor_bytes);
            if (!data_result.ok()) return data_result.status();
            const std::uint64_t inner = loc.row_in_shard * layout.row_bytes;
            if (inner > tensor_bytes || layout.row_bytes > tensor_bytes - inner) {
                return Status::out_of_range("PLE row outside tensor payload", __FILE__, __LINE__);
            }
            const auto* src = static_cast<const char*>(data_result.value()) + inner;
            std::memcpy(out + static_cast<std::size_t>(idx) * layout.row_bytes, src,
                        layout.row_bytes);
        }
    }

    const hipError_t he =
        hipMemcpy(device_dst, staging.ptr, total_bytes, hipMemcpyHostToDevice);
    if (he != hipSuccess) {
        return Status::hip_error("hipMemcpy", hipGetErrorString(he), __FILE__, __LINE__);
    }
    return sync_stream(stream);
}

Status gather_rows_vram(const TableLayout& layout, const void* device_table,
                        std::uint64_t table_rows, const std::uint64_t* device_row_ids,
                        std::uint32_t count, void* device_dst, hipStream_t stream) {
    if (device_table == nullptr || device_row_ids == nullptr || device_dst == nullptr) {
        return Status::invalid_argument("null VRAM gather operand", __FILE__, __LINE__);
    }
    if (count == 0) {
        return Status::invalid_argument("empty VRAM gather", __FILE__, __LINE__);
    }

    struct VramRequest {
        std::uint64_t row_id = 0;
        std::uint64_t dst_offset = 0;
    };
    (void)sizeof(VramRequest);
    std::vector<std::uint64_t> row_ids(count);
    const hipError_t re = hipMemcpy(row_ids.data(), device_row_ids,
                                    count * sizeof(std::uint64_t), hipMemcpyDeviceToHost);
    if (re != hipSuccess) {
        return Status::hip_error("hipMemcpy", hipGetErrorString(re), __FILE__, __LINE__);
    }

    std::vector<kernel::RowGatherItem> items(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        if (row_ids[i] >= table_rows) {
            return Status::out_of_range("VRAM row id outside the fixture table",
                                        __FILE__, __LINE__);
        }
        items[i].src_offset = row_ids[i] * layout.row_bytes;
        items[i].dst_offset = static_cast<std::uint64_t>(i) * layout.row_bytes;
        items[i].bytes = static_cast<std::uint32_t>(layout.row_bytes);
    }

    DeviceBuffer device_items;
    Status st = device_items.alloc(items.size() * sizeof(kernel::RowGatherItem));
    if (!st.ok()) return st;
    const hipError_t ie = hipMemcpy(device_items.ptr, items.data(),
                                    items.size() * sizeof(kernel::RowGatherItem),
                                    hipMemcpyHostToDevice);
    if (ie != hipSuccess) {
        return Status::hip_error("hipMemcpy", hipGetErrorString(ie), __FILE__, __LINE__);
    }

    const hipError_t ge = kernel::launch_row_gather(
        device_table, static_cast<const kernel::RowGatherItem*>(device_items.ptr), count,
        device_dst, stream);
    if (ge != hipSuccess) {
        return Status::hip_error("launch_row_gather", hipGetErrorString(ge), __FILE__, __LINE__);
    }
    return sync_stream(stream);
}

}  // namespace ps::qwen4exp::ple
