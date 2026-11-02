#include <shelf/inference/execution/selectors/l2_normalize_selector.h>

#include <cstdio>

using shelf::execution::L2NormalizeImplementation;
using shelf::execution::L2NormalizeSelectorInput;
using DT = ::shelf::runtime::ValueDType;

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

L2NormalizeImplementation sel(DT in, DT out, uint32_t features, uint32_t group,
                              uint32_t rows) {
    L2NormalizeSelectorInput si;
    si.input_dtype = in;
    si.output_dtype = out;
    si.features = features;
    si.group_size = group;
    si.rows = rows;
    return shelf::execution::select_l2_normalize_implementation(si);
}

}  // namespace

int main() {
    check(sel(DT::BF16, DT::F32, 2048u, 128u, 1u) == L2NormalizeImplementation::Optimized,
          "27B GDN qk norm BF16->F32 F=2048 G=128 rows=1 -> Optimized");
    check(sel(DT::BF16, DT::F32, 2048u, 128u, 2048u) == L2NormalizeImplementation::Optimized,
          "27B GDN qk norm rows=2048 -> Optimized");
    check(sel(DT::BF16, DT::F32, 2048u, 128u, 0u) == L2NormalizeImplementation::Correctness,
          "rows=0 -> Correctness");
    check(sel(DT::BF16, DT::F32, 2048u, 128u, 8192u) == L2NormalizeImplementation::Optimized,
          "rows=8192 -> Optimized");
    check(sel(DT::BF16, DT::F32, 4096u, 128u, 1u) == L2NormalizeImplementation::Correctness,
          "unknown features -> Correctness");
    check(sel(DT::BF16, DT::F32, 2048u, 256u, 1u) == L2NormalizeImplementation::Correctness,
          "unknown group -> Correctness");
    check(sel(DT::F32, DT::F32, 2048u, 128u, 1u) == L2NormalizeImplementation::Correctness,
          "F32 input -> Correctness");
    check(sel(DT::BF16, DT::BF16, 2048u, 128u, 1u) == L2NormalizeImplementation::Correctness,
          "BF16 output -> Correctness");

    check(sel(DT::BF16, DT::F32, 1024u, 128u, 1u) == L2NormalizeImplementation::Optimized,
          "27b tp2 GDN qk norm F=1024 G=128 rows=1 -> Optimized");
    check(sel(DT::BF16, DT::F32, 1024u, 128u, 2048u) == L2NormalizeImplementation::Optimized,
          "27b tp2 GDN qk norm rows=2048 -> Optimized");
    check(sel(DT::BF16, DT::F32, 1024u, 128u, 8193u) == L2NormalizeImplementation::Correctness,
          "27b tp2 rows=8193 -> Correctness");
    check(sel(DT::BF16, DT::F32, 1024u, 64u, 1u) == L2NormalizeImplementation::Correctness,
          "27b tp2 wrong group -> Correctness");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
