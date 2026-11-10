// P1-2: the mul24 prefill backend choice is a pure host function and can be tested
// without a GPU workload.

#include <shelf/inference/execution/detail/exl3_dispatch_decision.h>
#include <shelf/kernels/optimized/linear/exl3/detail/types.h>
#include <shelf/kernels/optimized/linear/exl3.h>

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;

void check(const char* name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++g_failures;
    }
}

}  // namespace

int main() {
    using shelf::execution::exl3_mul24_use_dense;

    // Existing condition: dense_ok && rows >= min && rows*4 <= rowsum_bytes.
    check("dense not ok", !exl3_mul24_use_dense(false, 2048u, 256u, 1ull << 20));
    check("below min rows", !exl3_mul24_use_dense(true, 255u, 256u, 1ull << 20));
    check("rowsum too small", !exl3_mul24_use_dense(true, 2048u, 256u, 2048ull * 4ull - 1ull));
    check("rowsum exact", exl3_mul24_use_dense(true, 2048u, 256u, 2048ull * 4ull));
    check("min row exact", exl3_mul24_use_dense(true, 256u, 256u, 1ull << 20));
    check("decode rows", !exl3_mul24_use_dense(true, 16u, 256u, 1ull << 20));
    check("dense disabled", !exl3_mul24_use_dense(true, 2048u, 0xFFFFFFFFu, 1ull << 20));

    // The decision struct keeps the independent axes and has no fallback by default.
    shelf::execution::Exl3DispatchDecision d;
    check("default direct", d.backend == shelf::kernel::exl3_detail::Exl3Backend::Direct);
    check("default no reason", d.fallback_reason == nullptr);
    check("default mul1", d.codebook == shelf::kernel::Exl3Codebook::Mul1);
    check("default decode", d.phase == shelf::kernel::Exl3Phase::Decode);

    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
