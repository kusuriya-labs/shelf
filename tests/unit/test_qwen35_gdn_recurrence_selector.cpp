#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "shelf/inference/execution/selectors/gdn_recurrence_selector.h"

using shelf::execution::GdnRecurrenceImplementation;
using shelf::execution::GdnRecurrenceSelectorInput;

namespace {
int g_checks = 0;
int g_failures = 0;

GdnRecurrenceImplementation sel(uint32_t rows, uint32_t num_requests,
                                uint32_t max_request_rows, uint32_t key_heads,
                                uint32_t num_v_heads, uint32_t head_k,
                                uint32_t head_v) {
    GdnRecurrenceSelectorInput si;
    si.rows = rows;
    si.num_requests = num_requests;
    si.max_request_rows = max_request_rows;
    si.key_heads = key_heads;
    si.num_v_heads = num_v_heads;
    si.head_k = head_k;
    si.head_v = head_v;
    return shelf::execution::select_gdn_recurrence_implementation(si);
}

bool check(const char* name, bool ok) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", name);
    } else {
        std::printf("PASS: %s\n", name);
    }
    return ok;
}
}  // namespace

int main() {
    // 4B profile: key_heads=16 num_v_heads=32 head_k=128 head_v=128, rows[1..2048].
    // 27B shares the same geometry rule (16/48/128/128).
    check("4b rows=1 nr=1 mr=1 optimized",
          sel(1, 1, 1, 16, 32, 128, 128) == GdnRecurrenceImplementation::Optimized);
    check("4b rows=64 nr=4 mr=16 optimized",
          sel(64, 4, 16, 16, 32, 128, 128) == GdnRecurrenceImplementation::Optimized);
    check("4b rows=2048 nr=8 mr=64 optimized",
          sel(2048, 8, 64, 16, 32, 128, 128) == GdnRecurrenceImplementation::Optimized);
    check("4b rows=8193 out of range correctness",
          sel(8193, 8, 64, 16, 32, 128, 128) == GdnRecurrenceImplementation::Correctness);
    check("27b rows=64 nr=2 mr=32 optimized",
          sel(64, 2, 32, 16, 48, 128, 128) == GdnRecurrenceImplementation::Optimized);
    check("27b tp2 rows=1 nr=1 mr=1 optimized",
          sel(1, 1, 1, 8, 24, 128, 128) == GdnRecurrenceImplementation::Optimized);
    check("27b tp2 rows=64 nr=1 mr=64 optimized",
          sel(64, 1, 64, 8, 24, 128, 128) == GdnRecurrenceImplementation::Optimized);
    check("27b tp2 rows=2048 nr=1 mr=2048 optimized",
          sel(2048, 1, 2048, 8, 24, 128, 128) == GdnRecurrenceImplementation::Optimized);
    check("27b tp2 rows=8193 out of range correctness",
          sel(8193, 1, 8193, 8, 24, 128, 128) == GdnRecurrenceImplementation::Correctness);
    check("unknown heads correctness",
          sel(4, 2, 3, 8, 16, 128, 128) == GdnRecurrenceImplementation::Correctness);
    check("unknown head dims correctness",
          sel(4, 2, 3, 16, 32, 64, 64) == GdnRecurrenceImplementation::Correctness);

    std::printf("%s: %d checks, %d failures\n",
                g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
