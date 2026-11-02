#include <shelf/inference/execution/kv_calib_dump.h>
#include <shelf/inference/execution/kv_append_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/runtime/graph/primitive_graph.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace shelf::execution {
namespace {

struct CalibDumpState {
    bool initialized = false;
    bool enabled = false;
    uint32_t max_rows = 32768u;
    uint32_t rows_written = 0u;
    std::FILE* k_file = nullptr;
    std::FILE* v_file = nullptr;
    std::FILE* meta_file = nullptr;
};

CalibDumpState& state() {
    static CalibDumpState s;
    return s;
}

void initialize(CalibDumpState& s) {
    s.initialized = true;
    const char* dir = std::getenv("SHELF_KV_CALIB_DUMP");
    if (dir == nullptr || dir[0] == '\0') return;
    const char* max_rows_env = std::getenv("SHELF_KV_CALIB_MAX_ROWS");
    if (max_rows_env != nullptr && max_rows_env[0] != '\0') {
        const long parsed = std::strtol(max_rows_env, nullptr, 10);
        if (parsed > 0) s.max_rows = static_cast<uint32_t>(parsed);
    }
    std::string base(dir);
    if (!base.empty() && base.back() == '/') base.pop_back();
    const std::string k_path = base + "/k.bf16";
    const std::string v_path = base + "/v.bf16";
    const std::string meta_path = base + "/meta.jsonl";
    s.k_file = std::fopen(k_path.c_str(), "wb");
    s.v_file = std::fopen(v_path.c_str(), "wb");
    s.meta_file = std::fopen(meta_path.c_str(), "wb");
    if (s.k_file == nullptr || s.v_file == nullptr || s.meta_file == nullptr) {
        if (s.k_file != nullptr) std::fclose(s.k_file);
        if (s.v_file != nullptr) std::fclose(s.v_file);
        if (s.meta_file != nullptr) std::fclose(s.meta_file);
        s.k_file = nullptr;
        s.v_file = nullptr;
        s.meta_file = nullptr;
        std::fprintf(stderr, "kv-calib: failed to open dump files in %s\n", dir);
        return;
    }
    s.enabled = true;
    std::fprintf(stderr,
                 "kv-calib: dumping K/V to %s (max_rows=%u)\n",
                 base.c_str(), s.max_rows);
}

}  // namespace

void kv_calib_dump_maybe(
    const ::shelf::runtime::Program& program,
    std::size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    CalibDumpState& s = state();
    if (!s.initialized) initialize(s);
    if (!s.enabled) return;
    if (dispatch_index >= program.dispatches.size()) return;
    if (program.dispatches[dispatch_index].kernel_id != ::shelf::runtime::KernelId::KV_APPEND)
        return;

    auto resolved = resolve_kv_append_args(program, dispatch_index, ctx);
    if (!resolved.ok()) return;
    const KvAppendResolvedArgs& a = resolved.value();
    const auto& c = a.common;
    if (c.rows == 0u || c.head_dim == 0u || c.kv_heads == 0u) return;
    const uint32_t feature_count = c.kv_heads * c.head_dim;
    if (c.k_row_stride < feature_count || c.v_row_stride < feature_count) return;
    const uint32_t remaining = s.rows_written >= s.max_rows ? 0u : s.max_rows - s.rows_written;
    const uint32_t dump_rows = c.rows < remaining ? c.rows : remaining;
    if (dump_rows == 0u) return;

    hipError_t herr = hipStreamSynchronize(stream);
    if (herr != hipSuccess) {
        std::fprintf(stderr, "kv-calib: stream sync failed: %s\n", hipGetErrorString(herr));
        return;
    }

    std::vector<uint8_t> k_host(static_cast<std::size_t>(dump_rows) * c.k_row_stride * 2u);
    std::vector<uint8_t> v_host(static_cast<std::size_t>(dump_rows) * c.v_row_stride * 2u);
    herr = hipMemcpy(k_host.data(), c.k_input,
                     k_host.size(), hipMemcpyDeviceToHost);
    if (herr != hipSuccess) {
        std::fprintf(stderr, "kv-calib: K copy failed: %s\n", hipGetErrorString(herr));
        return;
    }
    herr = hipMemcpy(v_host.data(), c.v_input,
                     v_host.size(), hipMemcpyDeviceToHost);
    if (herr != hipSuccess) {
        std::fprintf(stderr, "kv-calib: V copy failed: %s\n", hipGetErrorString(herr));
        return;
    }

    const std::size_t row_bytes = static_cast<std::size_t>(feature_count) * 2u;
    std::vector<uint8_t> k_compact(static_cast<std::size_t>(dump_rows) * row_bytes);
    std::vector<uint8_t> v_compact(static_cast<std::size_t>(dump_rows) * row_bytes);
    for (uint32_t r = 0; r < dump_rows; ++r) {
        std::memcpy(k_compact.data() + static_cast<std::size_t>(r) * row_bytes,
                    k_host.data() + static_cast<std::size_t>(r) * c.k_row_stride * 2u,
                    row_bytes);
        std::memcpy(v_compact.data() + static_cast<std::size_t>(r) * row_bytes,
                    v_host.data() + static_cast<std::size_t>(r) * c.v_row_stride * 2u,
                    row_bytes);
    }

    const std::size_t wrote_k =
        std::fwrite(k_compact.data(), 1u, k_compact.size(), s.k_file);
    const std::size_t wrote_v =
        std::fwrite(v_compact.data(), 1u, v_compact.size(), s.v_file);
    if (wrote_k != k_compact.size() || wrote_v != v_compact.size()) {
        std::fprintf(stderr, "kv-calib: short write\n");
        return;
    }
    std::fprintf(s.meta_file,
                 "{\"layer\":%u,\"rows\":%u,\"kv_heads\":%u,\"head_dim\":%u,"
                 "\"feature_count\":%u,\"row_offset\":%u}\n",
                 c.layer, dump_rows, c.kv_heads, c.head_dim, feature_count, s.rows_written);
    std::fflush(s.k_file);
    std::fflush(s.v_file);
    std::fflush(s.meta_file);
    s.rows_written += dump_rows;
}

}
