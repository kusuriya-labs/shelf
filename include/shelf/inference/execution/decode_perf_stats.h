#pragma once

#include <shelf/runtime/program/kernel_id.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::execution {

// Diagnostic-only decode counters. Default OFF (SHELF_QWEN35_PERF_STATS).
// Not part of the production dispatch contract; only observed by the bench.

enum class FallbackReason : uint32_t {
    UnsupportedKernel = 0,
    UnsupportedShape,
    UnsupportedDType,
    UnsupportedEncoding,
    Alignment,
    SelectorRejected,
    Other,
    COUNT,
};

struct DecodePerfStats {
    uint64_t logical_dispatches = 0;
    uint64_t optimized_operations = 0;
    uint64_t correctness_fallbacks = 0;
    uint64_t physical_kernel_launches = 0;

    uint64_t kernel_id_calls[static_cast<size_t>(::ps::runtime::KernelId::COUNT)] = {};

    uint64_t fp8_gemm_hits = 0, fp8_fallbacks = 0;
    uint64_t bf16_gemm_hits = 0, bf16_fallbacks = 0;

    uint64_t fallback_reasons[static_cast<size_t>(FallbackReason::COUNT)] = {};
    uint64_t m_bucket[6] = {};
};

bool perf_stats_enabled();
DecodePerfStats& perf_stats_instance();
void perf_stats_reset();
void perf_stats_dump(const char* path, uint32_t tokens);

// Count a group of logically-consecutive dispatches as covered by one execution
// (a fused launch consumes `covered` logical dispatches; a single op covers 1).
inline void record_logical_covered(
    const uint32_t* covered_kernel_ids,
    uint32_t covered) {
    if (!perf_stats_enabled())
        return;
    DecodePerfStats& s = perf_stats_instance();
    for (uint32_t j = 0; j < covered; ++j) {
        const uint32_t kid = covered_kernel_ids[j];
        if (kid < static_cast<uint32_t>(::ps::runtime::KernelId::COUNT))
            s.kernel_id_calls[kid]++;
    }
    s.logical_dispatches += covered;
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

inline void record_correctness_fallback(FallbackReason reason) {
    if (!perf_stats_enabled())
        return;
    DecodePerfStats& s = perf_stats_instance();
    s.correctness_fallbacks++;
    s.fallback_reasons[static_cast<size_t>(reason)]++;
}

enum class SemPath : uint32_t { Correctness = 0, Optimized = 1 };

bool semantic_timing_enabled();
void semantic_timing_begin(hipStream_t stream, uint32_t max_pairs);
uint32_t semantic_timing_start(hipStream_t stream);
void semantic_timing_finish(hipStream_t stream, uint32_t seq, uint32_t kernel_id,
                            uint32_t path, uint32_t logical_consumed);
void semantic_timing_end(hipStream_t stream, const char* csv_path);

enum class LinearFamilyRef : uint32_t { Fp8, Bf16 };

inline void linear_record_hit(LinearFamilyRef fam, uint32_t rows) {
    if (!perf_stats_enabled())
        return;
    DecodePerfStats& s = perf_stats_instance();
    if (rows <= 1u)
        s.m_bucket[0]++;
    else if (rows == 2u)
        s.m_bucket[1]++;
    else if (rows <= 4u)
        s.m_bucket[2]++;
    else if (rows <= 8u)
        s.m_bucket[3]++;
    else if (rows <= 16u)
        s.m_bucket[4]++;
    else
        s.m_bucket[5]++;
    if (fam == LinearFamilyRef::Fp8)
        s.fp8_gemm_hits++;
    else
        s.bf16_gemm_hits++;
}

inline void linear_record_fallback(LinearFamilyRef fam, FallbackReason reason) {
    if (!perf_stats_enabled())
        return;
    DecodePerfStats& s = perf_stats_instance();
    if (fam == LinearFamilyRef::Fp8)
        s.fp8_fallbacks++;
    else
        s.bf16_fallbacks++;
    record_correctness_fallback(reason);
}

}  // namespace ps::execution
