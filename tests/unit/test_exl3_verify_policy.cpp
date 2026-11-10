// Verify reproducibility policy: with a fixed tuning capacity the split count (and so
// the K reduction order) must not change with the active row count.

#include <shelf/kernels/optimized/linear/exl3_family/launch_plan.h>
#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;

char g_scratch[16];

bool stable_mul1(uint32_t tuning, uint32_t n, uint32_t k) {
    uint32_t splits = 0, groups = 0;
    for (uint32_t rows = 1; rows <= 16u; ++rows) {
        shelf::kernel::Exl3GemmArgs a;
        a.rows = rows;
        a.out_features = n;
        a.k = k;
        a.splitk_scratch = reinterpret_cast<float*>(g_scratch);
        a.splitk_floats = 1u << 22;
        a.splitk_splits = 0u;
        a.splitk_tuning_rows = tuning;
        const auto p = shelf::kernel::exl3_detail::make_mul1_launch_plan(a);
        if (rows == 1u) {
            splits = p.splits;
            groups = p.groups;
        } else if (p.splits != splits || p.groups != groups) {
            return false;
        }
    }
    return true;
}

bool stable_mul24(uint32_t n, uint32_t k, uint32_t scratch, uint32_t bits) {
    const uint32_t first = shelf::kernel::exl3_mul24_gemm_splits(1u, n, k, scratch, bits);
    for (uint32_t rows = 2u; rows <= 16u; ++rows) {
        if (shelf::kernel::exl3_mul24_gemm_splits(rows, n, k, scratch, bits) != first) return false;
    }
    return true;
}

void check(const char* name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++g_failures;
    }
}

}  // namespace

int main() {
    // mul1: a fixed verify capacity keeps splits/groups independent of the active rows.
    for (uint32_t n : {1024u, 5120u, 17408u}) {
        for (uint32_t k : {128u, 5120u, 17408u}) {
            char name[96];
            std::snprintf(name, sizeof(name), "mul1 stable n=%u k=%u", n, k);
            check(name, stable_mul1(16u, n, k));
        }
    }

    // mul24: the decode split count only depends on n/k/scratch/bits once scratch >= rows.
    for (uint32_t n : {5120u, 17408u}) {
        for (uint32_t k : {128u, 5120u}) {
            for (uint32_t scratch : {16u, 64u, 1024u}) {
                for (uint32_t bits : {4u, 6u}) {
                    char name[96];
                    std::snprintf(name, sizeof(name), "mul24 stable n=%u k=%u scratch=%u bits=%u", n,
                                  k, scratch, bits);
                    check(name, stable_mul24(n, k, scratch, bits));
                }
            }
        }
    }

    // The split count stays within the K-group count.
    for (uint32_t n : {5120u, 17408u}) {
        for (uint32_t k : {5120u, 17408u}) {
            const uint32_t s = shelf::kernel::exl3_mul24_gemm_splits(1u, n, k, 1024u, 4u);
            if (s == 0u || s > k / 128u) {
                std::printf("FAIL mul24 split range n=%u k=%u s=%u\n", n, k, s);
                ++g_failures;
            }
        }
    }

    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
