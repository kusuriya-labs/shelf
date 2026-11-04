#pragma once

#include <shelf/runtime/program/kernel_id.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::execution {

// Correctness-path visibility. This is the only mechanism that reports which kernels run
// the correctness interpreter: every dispatch passes the single decision point in
// execute_program_range, and that point records the per-kernel counters below.
//
// Diagnostic-only (SHELF_QWEN35_PERF_STATS, dumped by perf_stats_dump). Not part of the
// production dispatch contract.

constexpr uint32_t kKernelIdCount = static_cast<uint32_t>(::shelf::runtime::KernelId::COUNT);

struct KernelDispatchCounters {
    uint64_t logical = 0;     // dispatches bound to this kernel id
    uint64_t optimized = 0;   // ran the optimized kernel
    uint64_t fallback = 0;    // ran the correctness interpreter
    uint64_t degraded = 0;    // optimized kernel took its generic variant
    uint32_t fallback_rows_min = 0;
    uint32_t fallback_rows_max = 0;
};

struct DecodePerfStats {
    uint64_t logical_dispatches = 0;
    uint64_t optimized_operations = 0;
    uint64_t correctness_fallbacks = 0;
    uint64_t physical_kernel_launches = 0;

    KernelDispatchCounters kernels[kKernelIdCount];
};

bool perf_stats_enabled();
DecodePerfStats& perf_stats_instance();
void perf_stats_reset();
void perf_stats_dump(const char* path, uint32_t tokens);

inline uint32_t kernel_slot(::shelf::runtime::KernelId id) {
    const uint32_t k = static_cast<uint32_t>(id);
    return k < kKernelIdCount ? k : kKernelIdCount;
}

// Records one logical dispatch and returns its counter slot, or kKernelIdCount when the
// kernel id is outside the table.
inline uint32_t record_logical_dispatch(::shelf::runtime::KernelId id) {
    if (!perf_stats_enabled())
        return kKernelIdCount;
    const uint32_t slot = kernel_slot(id);
    if (slot == kKernelIdCount)
        return slot;
    DecodePerfStats& s = perf_stats_instance();
    s.logical_dispatches++;
    s.kernels[slot].logical++;
    return slot;
}

inline void record_optimized_dispatch(::shelf::runtime::KernelId id) {
    const uint32_t slot = kernel_slot(id);
    if (slot == kKernelIdCount || !perf_stats_enabled())
        return;
    perf_stats_instance().kernels[slot].optimized++;
}

inline void record_fallback_dispatch(::shelf::runtime::KernelId id, uint32_t rows) {
    const uint32_t slot = kernel_slot(id);
    if (slot == kKernelIdCount || !perf_stats_enabled())
        return;
    DecodePerfStats& s = perf_stats_instance();
    KernelDispatchCounters& k = s.kernels[slot];
    k.fallback++;
    s.correctness_fallbacks++;
    if (k.fallback_rows_min == 0u || rows < k.fallback_rows_min)
        k.fallback_rows_min = rows;
    if (rows > k.fallback_rows_max)
        k.fallback_rows_max = rows;
}

// The optimized kernel ran, but on its generic variant: not the correctness interpreter,
// and not the measured fast path either.
inline void record_degraded_dispatch(::shelf::runtime::KernelId id) {
    const uint32_t slot = kernel_slot(id);
    if (slot == kKernelIdCount || !perf_stats_enabled())
        return;
    perf_stats_instance().kernels[slot].degraded++;
}

inline void record_physical_launch(uint64_t n = 1) {
    if (!perf_stats_enabled())
        return;
    perf_stats_instance().physical_kernel_launches += n;
}

inline void record_optimized_op() {
    if (!perf_stats_enabled())
        return;
    perf_stats_instance().optimized_operations++;
}

enum class SemPath : uint32_t { Correctness = 0, Optimized = 1 };

bool semantic_timing_enabled();
void semantic_timing_begin(hipStream_t stream, uint32_t max_pairs);
uint32_t semantic_timing_start(hipStream_t stream);
void semantic_timing_finish(hipStream_t stream, uint32_t seq, uint32_t kernel_id,
                            uint32_t path, uint32_t logical_consumed);
void semantic_timing_end(hipStream_t stream, const char* csv_path);

}  // namespace shelf::execution
