#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/runtime/program/kernel_id.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

using shelf::execution::DecodePerfStats;
using shelf::execution::KernelDispatchCounters;
using shelf::runtime::KernelId;

const KernelDispatchCounters& slot(const DecodePerfStats& s, KernelId id) {
    return s.kernels[static_cast<std::size_t>(id)];
}

uint64_t sum_logical(const DecodePerfStats& s) {
    uint64_t n = 0;
    for (std::size_t i = 0; i < shelf::execution::kKernelIdCount; ++i)
        n += s.kernels[i].logical;
    return n;
}

uint64_t sum_fallback(const DecodePerfStats& s) {
    uint64_t n = 0;
    for (std::size_t i = 0; i < shelf::execution::kKernelIdCount; ++i)
        n += s.kernels[i].fallback;
    return n;
}

}  // namespace

int main() {
    ::setenv("SHELF_QWEN35_PERF_STATS", "1", 1);

    check(shelf::execution::perf_stats_enabled(), "perf stats enabled by env");

    shelf::execution::perf_stats_reset();

    for (int i = 0; i < 3; ++i) {
        shelf::execution::record_logical_dispatch(KernelId::RMS_NORM);
        shelf::execution::record_optimized_dispatch(KernelId::RMS_NORM);
    }
    shelf::execution::record_logical_dispatch(KernelId::PAGED_ATTENTION);
    shelf::execution::record_fallback_dispatch(KernelId::PAGED_ATTENTION, 64u);
    shelf::execution::record_logical_dispatch(KernelId::PAGED_ATTENTION);
    shelf::execution::record_fallback_dispatch(KernelId::PAGED_ATTENTION, 1u);
    shelf::execution::record_logical_dispatch(KernelId::LINEAR_BF16);
    shelf::execution::record_fallback_dispatch(KernelId::LINEAR_BF16, 128u);
    shelf::execution::record_logical_dispatch(KernelId::ROPE);
    shelf::execution::record_optimized_dispatch(KernelId::ROPE);
    shelf::execution::record_degraded_dispatch(KernelId::ROPE);

    const DecodePerfStats& s = shelf::execution::perf_stats_instance();

    check(slot(s, KernelId::RMS_NORM).logical == 3u, "rms_norm logical");
    check(slot(s, KernelId::RMS_NORM).optimized == 3u, "rms_norm optimized");
    check(slot(s, KernelId::RMS_NORM).fallback == 0u, "rms_norm fallback");

    check(slot(s, KernelId::PAGED_ATTENTION).fallback == 2u,
          "paged_attention fallback counted per kernel, not only for linear");
    check(slot(s, KernelId::PAGED_ATTENTION).fallback_rows_min == 1u, "fallback rows min");
    check(slot(s, KernelId::PAGED_ATTENTION).fallback_rows_max == 64u, "fallback rows max");

    check(slot(s, KernelId::LINEAR_BF16).fallback == 1u, "linear_bf16 fallback");
    check(slot(s, KernelId::ROPE).degraded == 1u, "rope degraded");
    check(slot(s, KernelId::ROPE).fallback == 0u, "degraded is not a correctness fallback");

    check(s.logical_dispatches == 7u, "total logical dispatches");
    check(s.correctness_fallbacks == sum_fallback(s), "total equals per-kernel sum");
    check(s.correctness_fallbacks == 3u, "total correctness fallbacks");
    check(s.logical_dispatches == sum_logical(s), "logical total equals per-kernel sum");

    bool invariant = true;
    for (std::size_t i = 0; i < shelf::execution::kKernelIdCount; ++i) {
        const KernelDispatchCounters& k = s.kernels[i];
        if (k.logical != k.optimized + k.fallback)
            invariant = false;
    }
    check(invariant, "per kernel: logical == optimized + fallback");

    const char* dump_path = "test_decode_perf_stats_dump.json";
    shelf::execution::perf_stats_dump(dump_path, 2u);
    std::ifstream in(dump_path);
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(!body.empty(), "dump written");
    check(body.find("\"kernels\"") != std::string::npos, "dump has the kernels table");
    check(body.find("PAGED_ATTENTION") != std::string::npos, "dump names paged_attention");
    check(body.find("RMS_NORM") != std::string::npos, "dump names rms_norm");
    check(body.find("LINEAR_BF16") != std::string::npos, "dump names linear_bf16");
    check(body.find("\"fallback_rows\": [1, 64]") != std::string::npos, "dump has fallback rows");
    check(body.find("\"degraded\": 1") != std::string::npos, "dump has degraded count");
    check(body.find("fallback_reasons") == std::string::npos, "no second fallback breakdown");

    shelf::execution::perf_stats_reset();
    check(slot(shelf::execution::perf_stats_instance(), KernelId::PAGED_ATTENTION).fallback == 0u,
          "reset clears the table");
    std::remove(dump_path);

    if (failures != 0) {
        std::fprintf(stderr, "decode_perf_stats: %d failures\n", failures);
        return 1;
    }
    std::printf("decode_perf_stats: ok\n");
    return 0;
}
