#include <phaseshift/io/row_reader.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace ps::io {
namespace {

std::uint64_t align_down(std::uint64_t v, std::uint64_t a) {
    return v - (v % a);
}

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) {
    const std::uint64_t r = v % a;
    return r == 0 ? v : v + (a - r);
}

Status open_plain(const std::string& path, int flags, int& fd) {
    fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        return Status::invalid_argument(
            ("cannot open file: " + path + " (" + std::string(std::strerror(errno)) + ")")
                .c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status pread_full(int fd, void* dst, std::size_t bytes, std::uint64_t offset,
                  const std::string& path) {
    auto* p = static_cast<char*>(dst);
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::pread(fd, p + done, bytes - done,
                                  static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return Status::invalid_argument(
                ("pread failed: " + path + " (" + std::string(std::strerror(errno)) + ")")
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (n == 0) {
            return Status::invalid_argument(
                ("short read: " + path).c_str(), __FILE__, __LINE__);
        }
        done += static_cast<std::size_t>(n);
    }
    return Status::make_ok();
}

}  // namespace

Result<std::uint64_t> probe_odirect_alignment(const std::string& path) {
    int fd = -1;
    Status st = open_plain(path, O_RDONLY | O_DIRECT, fd);
    if (!st.ok()) return st;

    struct stat sb {};
    if (::fstat(fd, &sb) != 0) {
        ::close(fd);
        return Status::invalid_argument("fstat failed", __FILE__, __LINE__);
    }
    const std::uint64_t size = static_cast<std::uint64_t>(sb.st_size);
    ::close(fd);
    if (size == 0) {
        return Status::invalid_argument("file is empty", __FILE__, __LINE__);
    }

    const std::uint64_t candidates[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    for (std::uint64_t a : candidates) {
        if (a > size) break;
        st = open_plain(path, O_RDONLY | O_DIRECT, fd);
        if (!st.ok()) return st;
        void* buf = nullptr;
        if (posix_memalign(&buf, a, a) != 0) {
            ::close(fd);
            return Status::insufficient_memory("posix_memalign", __FILE__, __LINE__);
        }
        const ssize_t n = ::pread(fd, buf, a, 0);
        const bool ok = (n == static_cast<ssize_t>(a));
        ::close(fd);
        ::free(buf);
        if (ok) return a;
    }
    return Status::unsupported("O_DIRECT alignment not determined", __FILE__, __LINE__);
}

Result<RowReadPlan> plan_row_reads(const std::vector<RowRequest>& rows,
                                   std::uint64_t file_size, std::uint64_t alignment) {
    if (rows.empty()) {
        return Status::invalid_argument("empty row request", __FILE__, __LINE__);
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return Status::invalid_argument("alignment must be a power of two",
                                        __FILE__, __LINE__);
    }

    RowReadPlan plan;
    plan.alignment = alignment;
    plan.fully_aligned = true;
    plan.row_block.resize(rows.size());
    plan.row_offset.resize(rows.size());

    struct Window {
        std::uint64_t begin;
        std::uint64_t end;
    };
    std::vector<Window> windows;
    windows.reserve(rows.size());

    for (std::size_t i = 0; i < rows.size(); ++i) {
        const RowRequest& r = rows[i];
        if (r.byte_length == 0) {
            return Status::invalid_argument("zero-length row", __FILE__, __LINE__);
        }
        if (r.absolute_offset > file_size ||
            r.byte_length > file_size - r.absolute_offset) {
            return Status::out_of_range("row exceeds file bounds", __FILE__, __LINE__);
        }
        if (r.absolute_offset > std::numeric_limits<std::uint64_t>::max() - r.byte_length) {
            return Status::overflow("row range overflow", __FILE__, __LINE__);
        }
        const std::uint64_t row_end = r.absolute_offset + r.byte_length;
        std::uint64_t begin = align_down(r.absolute_offset, alignment);
        std::uint64_t end = align_up(row_end, alignment);
        if (end > file_size) {
            end = file_size;
            plan.fully_aligned = false;
        }
        if (end <= begin) {
            return Status::invalid_argument("aligned window is empty", __FILE__, __LINE__);
        }
        windows.push_back({begin, end});
    }

    std::vector<std::uint64_t> order(windows.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::uint64_t a, std::uint64_t b) {
        if (windows[a].begin != windows[b].begin) return windows[a].begin < windows[b].begin;
        return windows[a].end < windows[b].end;
    });

    std::vector<std::uint64_t> merged_begin;
    std::vector<std::uint64_t> merged_end;
    std::vector<std::uint64_t> src_of_first;
    for (std::uint64_t idx : order) {
        const Window& w = windows[idx];
        if (!merged_begin.empty() && w.begin <= merged_end.back()) {
            if (w.end > merged_end.back()) merged_end.back() = w.end;
        } else {
            merged_begin.push_back(w.begin);
            merged_end.push_back(w.end);
            src_of_first.push_back(idx);
        }
    }

    for (std::size_t b = 0; b < merged_begin.size(); ++b) {
        AlignedBlock blk;
        blk.file_offset = merged_begin[b];
        blk.byte_length = merged_end[b] - merged_begin[b];
        if (blk.byte_length > std::numeric_limits<std::uint64_t>::max() - plan.total_block_bytes) {
            return Status::overflow("total block bytes overflow", __FILE__, __LINE__);
        }
        plan.total_block_bytes += blk.byte_length;
        if (merged_begin[b] % alignment != 0) plan.fully_aligned = false;
        if (blk.byte_length % alignment != 0) plan.fully_aligned = false;
        plan.blocks.push_back(blk);
    }

    for (std::size_t i = 0; i < windows.size(); ++i) {
        const std::uint64_t begin = windows[i].begin;
        std::size_t lo = 0, hi = merged_begin.size();
        while (lo + 1 < hi) {
            const std::size_t mid = (lo + hi) / 2;
            if (merged_begin[mid] <= begin) lo = mid;
            else hi = mid;
        }
        plan.row_block[i] = lo;
        plan.row_offset[i] = rows[i].absolute_offset - merged_begin[lo];
        (void)src_of_first;
    }

    return plan;
}

Status read_blocks_buffered(const std::string& path,
                            const std::vector<AlignedBlock>& blocks, void* dst,
                            std::size_t dst_bytes) {
    if (dst == nullptr) {
        return Status::invalid_argument("null destination", __FILE__, __LINE__);
    }
    int fd = -1;
    Status st = open_plain(path, O_RDONLY, fd);
    if (!st.ok()) return st;

    std::size_t total = 0;
    for (const AlignedBlock& b : blocks) total += b.byte_length;
    if (total != dst_bytes) {
        ::close(fd);
        return Status::invalid_argument("block size mismatch", __FILE__, __LINE__);
    }

    auto* out = static_cast<char*>(dst);
    std::size_t cursor = 0;
    for (const AlignedBlock& b : blocks) {
        st = pread_full(fd, out + cursor, b.byte_length, b.file_offset, path);
        if (!st.ok()) {
            ::close(fd);
            return st;
        }
        cursor += b.byte_length;
    }
    ::close(fd);
    return Status::make_ok();
}

Status read_blocks_direct(const std::string& path,
                          const std::vector<AlignedBlock>& blocks, void* dst,
                          std::size_t dst_bytes, std::uint64_t alignment) {
    if (dst == nullptr) {
        return Status::invalid_argument("null destination", __FILE__, __LINE__);
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return Status::invalid_argument("alignment must be a power of two",
                                        __FILE__, __LINE__);
    }
    if (reinterpret_cast<std::uintptr_t>(dst) % alignment != 0) {
        return Status::invalid_argument("destination is not aligned", __FILE__, __LINE__);
    }

    int fd = -1;
    Status st = open_plain(path, O_RDONLY | O_DIRECT, fd);
    if (!st.ok()) return st;

    std::size_t total = 0;
    for (const AlignedBlock& b : blocks) {
        if (b.file_offset % alignment != 0 || b.byte_length % alignment != 0) {
            ::close(fd);
            return Status::invalid_argument("block is not aligned for O_DIRECT",
                                            __FILE__, __LINE__);
        }
        total += b.byte_length;
    }
    if (total != dst_bytes) {
        ::close(fd);
        return Status::invalid_argument("block size mismatch", __FILE__, __LINE__);
    }

    auto* out = static_cast<char*>(dst);
    std::size_t cursor = 0;
    for (const AlignedBlock& b : blocks) {
        st = pread_full(fd, out + cursor, b.byte_length, b.file_offset, path);
        if (!st.ok()) {
            ::close(fd);
            return st;
        }
        cursor += b.byte_length;
    }
    ::close(fd);
    return Status::make_ok();
}

Status read_rows_buffered(const std::string& path,
                          const std::vector<RowRequest>& rows, void* dst,
                          std::size_t dst_bytes, RowReadPlan* out_plan) {
    struct stat sb {};
    if (::stat(path.c_str(), &sb) != 0) {
        return Status::invalid_argument(("stat failed: " + path).c_str(), __FILE__, __LINE__);
    }
    auto plan_result = plan_row_reads(rows, static_cast<std::uint64_t>(sb.st_size), 4096);
    if (!plan_result.ok()) return plan_result.status();
    RowReadPlan plan = plan_result.release();
    if (out_plan != nullptr) *out_plan = plan;

    std::size_t row_bytes = 0;
    for (const RowRequest& r : rows) row_bytes += r.byte_length;
    if (row_bytes != dst_bytes) {
        return Status::invalid_argument("row size mismatch", __FILE__, __LINE__);
    }

    std::vector<char> staging(plan.total_block_bytes);
    Status st = read_blocks_buffered(path, plan.blocks, staging.data(),
                                     static_cast<std::size_t>(plan.total_block_bytes));
    if (!st.ok()) return st;

    std::size_t block_base_cursor = 0;
    std::vector<std::uint64_t> block_base(plan.blocks.size());
    for (std::size_t b = 0; b < plan.blocks.size(); ++b) {
        block_base[b] = block_base_cursor;
        block_base_cursor += plan.blocks[b].byte_length;
    }

    auto* out = static_cast<char*>(dst);
    std::size_t out_cursor = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::uint64_t src = block_base[plan.row_block[i]] + plan.row_offset[i];
        std::memcpy(out + out_cursor, staging.data() + src, rows[i].byte_length);
        out_cursor += rows[i].byte_length;
    }
    return Status::make_ok();
}

}  // namespace ps::io
