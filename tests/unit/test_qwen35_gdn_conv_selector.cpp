#include <shelf/inference/execution/selectors/gdn_conv_selector.h>

#include <cstdio>

using ps::execution::GdnConvImplementation;
using ps::execution::GdnConvSelectorInput;
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

GdnConvImplementation sel(DT in, DT out, uint32_t dim, uint32_t history, uint32_t rows,
                          uint32_t nreq, uint32_t max_rows) {
    GdnConvSelectorInput si;
    si.input_dtype = in;
    si.output_dtype = out;
    si.conv_dim = dim;
    si.history = history;
    si.rows = rows;
    si.num_requests = nreq;
    si.max_request_rows = max_rows;
    return ps::execution::select_gdn_conv_implementation(si);
}

}  // namespace

int main() {
    check(sel(DT::BF16, DT::F32, 10240u, 3u, 1u, 1u, 1u) == GdnConvImplementation::Optimized,
          "27B decode -> Optimized");
    check(sel(DT::BF16, DT::F32, 10240u, 3u, 128u, 1u, 128u) == GdnConvImplementation::Optimized,
          "27B prefill -> Optimized");
    check(sel(DT::BF16, DT::F32, 8192u, 3u, 1u, 4u, 1u) == GdnConvImplementation::Optimized,
          "4B decode multi-request -> Optimized");
    check(sel(DT::BF16, DT::F32, 10240u, 3u, 0u, 1u, 1u) == GdnConvImplementation::Correctness,
          "rows=0 -> Correctness");
    check(sel(DT::BF16, DT::F32, 10240u, 3u, 1u, 0u, 1u) == GdnConvImplementation::Correctness,
          "num_requests=0 -> Correctness");
    check(sel(DT::BF16, DT::F32, 10240u, 3u, 1u, 1u, 0u) == GdnConvImplementation::Correctness,
          "max_request_rows=0 -> Correctness");
    check(sel(DT::BF16, DT::F32, 10240u, 3u, 8193u, 1u, 1u) == GdnConvImplementation::Correctness,
          "rows=8193 -> Correctness");
    check(sel(DT::BF16, DT::F32, 4096u, 3u, 1u, 1u, 1u) == GdnConvImplementation::Correctness,
          "unknown conv_dim -> Correctness");
    check(sel(DT::BF16, DT::F32, 10240u, 2u, 1u, 1u, 1u) == GdnConvImplementation::Correctness,
          "history != 3 -> Correctness");
    check(sel(DT::F32, DT::F32, 10240u, 3u, 1u, 1u, 1u) == GdnConvImplementation::Correctness,
          "F32 input -> Correctness");
    check(sel(DT::BF16, DT::BF16, 10240u, 3u, 1u, 1u, 1u) == GdnConvImplementation::Correctness,
          "BF16 output -> Correctness");
    check(sel(DT::BF16, DT::F32, 5120u, 3u, 1u, 1u, 1u) == GdnConvImplementation::Optimized,
          "27b tp2 decode -> Optimized");
    check(sel(DT::BF16, DT::F32, 5120u, 3u, 64u, 1u, 64u) == GdnConvImplementation::Optimized,
          "27b tp2 prefill -> Optimized");
    check(sel(DT::BF16, DT::F32, 5120u, 3u, 8193u, 1u, 1u) == GdnConvImplementation::Correctness,
          "27b tp2 rows=8193 -> Correctness");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
