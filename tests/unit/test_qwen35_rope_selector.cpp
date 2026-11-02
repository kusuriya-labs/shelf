#include <shelf/inference/execution/selectors/rope_selector.h>

#include <cstdio>

using ps::execution::RopeImplementation;
using ps::execution::RopeSelectorInput;
using DT = ::ps::runtime::ValueDType;

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

RopeImplementation sel(DT in, DT out, uint32_t features, uint32_t head_dim,
                       uint32_t rotary, uint32_t rows) {
    RopeSelectorInput si;
    si.input_dtype = in;
    si.output_dtype = out;
    si.features = features;
    si.head_dim = head_dim;
    si.rotary_dim = rotary;
    si.rows = rows;
    return ps::execution::select_rope_implementation(si);
}

}  // namespace

int main() {
    check(sel(DT::F32, DT::BF16, 6144u, 256u, 64u, 1u) == RopeImplementation::Optimized,
          "27B q rope -> Optimized");
    check(sel(DT::F32, DT::BF16, 1024u, 256u, 64u, 1u) == RopeImplementation::Optimized,
          "27B k rope -> Optimized");
    check(sel(DT::F32, DT::BF16, 4096u, 256u, 64u, 1u) == RopeImplementation::Optimized,
          "4B q rope -> Optimized");
    check(sel(DT::F32, DT::BF16, 6144u, 256u, 64u, 2048u) == RopeImplementation::Optimized,
          "27B q rope rows=2048 -> Optimized");
    check(sel(DT::F32, DT::BF16, 6144u, 256u, 64u, 0u) == RopeImplementation::Correctness,
          "rows=0 -> Correctness");
    check(sel(DT::F32, DT::BF16, 6144u, 256u, 64u, 8192u) == RopeImplementation::Optimized,
          "27B q rope rows=8192 -> Optimized");
    check(sel(DT::F32, DT::BF16, 6144u, 256u, 64u, 8193u) == RopeImplementation::Correctness,
          "rows=8193 -> Correctness");
    check(sel(DT::F32, DT::BF16, 5120u, 256u, 64u, 1u) == RopeImplementation::Correctness,
          "unknown features -> Correctness");
    check(sel(DT::F32, DT::BF16, 6144u, 128u, 64u, 1u) == RopeImplementation::Correctness,
          "unknown head_dim -> Correctness");
    check(sel(DT::F32, DT::BF16, 6144u, 256u, 128u, 1u) == RopeImplementation::Correctness,
          "unknown rotary -> Correctness");
    check(sel(DT::F32, DT::F32, 6144u, 256u, 64u, 1u) == RopeImplementation::Correctness,
          "F32 output -> Correctness");
    check(sel(DT::BF16, DT::BF16, 6144u, 256u, 64u, 1u) == RopeImplementation::Correctness,
          "BF16 input -> Correctness");
    check(sel(DT::F32, DT::BF16, 3072u, 256u, 64u, 1u) == RopeImplementation::Optimized,
          "27b tp2 q rope -> Optimized");
    check(sel(DT::F32, DT::BF16, 512u, 256u, 64u, 1u) == RopeImplementation::Optimized,
          "27b tp2 k rope -> Optimized");
    check(sel(DT::F32, DT::BF16, 3072u, 256u, 64u, 2048u) == RopeImplementation::Optimized,
          "27b tp2 q rope rows=2048 -> Optimized");
    check(sel(DT::F32, DT::BF16, 3072u, 256u, 64u, 8192u) == RopeImplementation::Optimized,
          "27b tp2 rows=8192 -> Optimized");
    check(sel(DT::F32, DT::BF16, 3072u, 256u, 64u, 8193u) == RopeImplementation::Correctness,
          "27b tp2 rows=8193 -> Correctness");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
