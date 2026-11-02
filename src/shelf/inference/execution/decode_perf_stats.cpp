#include <shelf/inference/execution/decode_perf_stats.h>

#include <shelf/runtime/program/kernel_id.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace shelf::execution {

bool perf_stats_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("SHELF_QWEN35_PERF_STATS");
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return enabled;
}

namespace {
thread_local DecodePerfStats g_perf_stats;
}

DecodePerfStats& perf_stats_instance() {
    return g_perf_stats;
}

void perf_stats_reset() {
    g_perf_stats = DecodePerfStats{};
}

namespace {

const char* fallback_reason_name(FallbackReason r) {
    switch (r) {
        case FallbackReason::UnsupportedKernel:
            return "UnsupportedKernel";
        case FallbackReason::UnsupportedShape:
            return "UnsupportedShape";
        case FallbackReason::UnsupportedDType:
            return "UnsupportedDType";
        case FallbackReason::UnsupportedEncoding:
            return "UnsupportedEncoding";
        case FallbackReason::Alignment:
            return "Alignment";
        case FallbackReason::SelectorRejected:
            return "SelectorRejected";
        case FallbackReason::Other:
            return "Other";
        case FallbackReason::COUNT:
            break;
    }
    return "Unknown";
}

}  // namespace

void perf_stats_dump(const char* path, uint32_t tokens) {
    const DecodePerfStats& s = g_perf_stats;
    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
        std::fprintf(stderr, "perf_stats: cannot open %s\n", path);
        return;
    }
    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"tokens\": %u,\n", tokens);
    std::fprintf(f, "  \"logical_dispatches\": %llu,\n",
                 (unsigned long long)s.logical_dispatches);
    std::fprintf(f, "  \"optimized_operations\": %llu,\n",
                 (unsigned long long)s.optimized_operations);
    std::fprintf(f, "  \"correctness_fallbacks\": %llu,\n",
                 (unsigned long long)s.correctness_fallbacks);
    std::fprintf(f, "  \"physical_kernel_launches\": %llu,\n",
                 (unsigned long long)s.physical_kernel_launches);
    std::fprintf(f, "  \"linear\": {\n");
    std::fprintf(f, "    \"fp8\": {\"gemm\": %llu, \"fallback\": %llu},\n",
                 (unsigned long long)s.fp8_gemm_hits,
                 (unsigned long long)s.fp8_fallbacks);
    std::fprintf(f, "    \"bf16\": {\"gemm\": %llu, \"fallback\": %llu}\n",
                 (unsigned long long)s.bf16_gemm_hits,
                 (unsigned long long)s.bf16_fallbacks);
    std::fprintf(f, "  },\n");
    std::fprintf(f, "  \"m_bucket\": [");
    for (int i = 0; i < 6; ++i) {
        if (i > 0)
            std::fprintf(f, ", ");
        std::fprintf(f, "%llu", (unsigned long long)s.m_bucket[i]);
    }
    std::fprintf(f, "],\n");
    std::fprintf(f, "  \"fallback_reasons\": {");
    bool first = true;
    for (uint32_t i = 0; i < static_cast<uint32_t>(FallbackReason::COUNT); ++i) {
        if (s.fallback_reasons[i] == 0)
            continue;
        if (!first)
            std::fprintf(f, ", ");
        std::fprintf(f, "\"%s\": %llu", fallback_reason_name(static_cast<FallbackReason>(i)),
                     (unsigned long long)s.fallback_reasons[i]);
        first = false;
    }
    std::fprintf(f, "},\n");
    std::fprintf(f, "  \"kernel_ids\": {");
    first = true;
    for (uint32_t i = 0; i < static_cast<uint32_t>(::shelf::runtime::KernelId::COUNT); ++i) {
        if (s.kernel_id_calls[i] == 0)
            continue;
        if (!first)
            std::fprintf(f, ", ");
        std::fprintf(f, "\"%s\": %llu",
                     ::shelf::runtime::kernel_to_name(static_cast<::shelf::runtime::KernelId>(i)),
                     (unsigned long long)s.kernel_id_calls[i]);
        first = false;
    }
    std::fprintf(f, "}\n");
    std::fprintf(f, "}\n");
    std::fclose(f);
}

bool semantic_timing_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("SHELF_QWEN35_PERF_TIMING");
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return enabled;
}

namespace {
struct SemState {
    bool begun = false;
    uint32_t max_pairs = 0;
    uint32_t next = 0;
    std::vector<hipEvent_t> events;
    struct Rec {
        uint32_t seq = 0;
        uint32_t kernel_id = 0;
        uint32_t path = 0;
        uint32_t logical_consumed = 0;
    };
    std::vector<Rec> recs;
};
thread_local SemState g_sem;
}  // namespace

void semantic_timing_begin(hipStream_t stream, uint32_t max_pairs) {
    (void)stream;
    g_sem.max_pairs = max_pairs;
    g_sem.next = 0;
    g_sem.recs.clear();
    g_sem.events.resize(2u * max_pairs);
    for (uint32_t i = 0; i < 2u * max_pairs; ++i)
        (void)hipEventCreate(&g_sem.events[i]);
    g_sem.begun = true;
}

uint32_t semantic_timing_start(hipStream_t stream) {
    if (!g_sem.begun || g_sem.next >= g_sem.max_pairs)
        return 0xFFFFFFFFu;
    const uint32_t seq = g_sem.next++;
    (void)hipEventRecord(g_sem.events[2u * seq], stream);
    return seq;
}

void semantic_timing_finish(hipStream_t stream, uint32_t seq, uint32_t kernel_id,
                            uint32_t path, uint32_t logical_consumed) {
    if (!g_sem.begun || seq == 0xFFFFFFFFu)
        return;
    (void)hipEventRecord(g_sem.events[2u * seq + 1u], stream);
    g_sem.recs.push_back(SemState::Rec{seq, kernel_id, path, logical_consumed});
}

void semantic_timing_end(hipStream_t stream, const char* csv_path) {
    if (!g_sem.begun)
        return;
    (void)hipStreamSynchronize(stream);
    std::FILE* f = std::fopen(csv_path, "w");
    if (f != nullptr) {
        std::fprintf(f, "seq,kernel_id,path,implementation,logical_consumed,duration_us\n");
        for (const SemState::Rec& r : g_sem.recs) {
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, g_sem.events[2u * r.seq], g_sem.events[2u * r.seq + 1u]);
            const double us = static_cast<double>(ms) * 1000.0;
            const char* kid = ::shelf::runtime::kernel_to_name(
                static_cast<::shelf::runtime::KernelId>(r.kernel_id));
            const char* path =
                r.path == static_cast<uint32_t>(SemPath::Correctness) ? "correctness"
                                                                      : "optimized";
            std::fprintf(f, "%u,%s,%s,%s,%u,%.3f\n", r.seq, kid, path,
                         (r.path == static_cast<uint32_t>(SemPath::Correctness))
                             ? "model_dispatch_correctness"
                             : kid,
                         r.logical_consumed, us);
        }
        std::fclose(f);
    }
    for (hipEvent_t ev : g_sem.events)
        (void)hipEventDestroy(ev);
    g_sem.events.clear();
    g_sem.begun = false;
}

}  // namespace shelf::execution
