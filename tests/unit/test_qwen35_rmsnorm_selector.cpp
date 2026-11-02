#include <shelf/inference/execution/selectors/rmsnorm_selector.h>

#include <climits>
#include <cstdio>

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

using shelf::execution::RmsNormImplementation;
using shelf::execution::RmsNormSelectorWeightLayout;
using shelf::execution::RmsNormSelectorInput;
using DT = ::shelf::runtime::ValueDType;

shelf::execution::RmsNormImplementation
sel(DT in, DT out, DT wt, RmsNormSelectorWeightLayout layout, uint32_t mode,
    uint32_t features, uint32_t group, uint32_t rows) {
    return shelf::execution::select_rmsnorm_implementation(
        {in, out, wt, layout, mode, features, group, rows});
}

}  // namespace

int main() {
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 1) == RmsNormImplementation::Optimized,
          "hidden norm BF16->BF16 PF ONE_PLUS rows=1 -> Optimized");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 2048) == RmsNormImplementation::Optimized,
          "hidden norm rows=2048 -> Optimized");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 0) == RmsNormImplementation::Correctness,
          "hidden norm rows=0 -> Correctness");

    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 4096, 256, 1) == RmsNormImplementation::Optimized,
          "Q norm BF16->F32 PG ONE_PLUS F=4096 rows=1 -> Optimized");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 1024, 256, 1) == RmsNormImplementation::Optimized,
          "K norm BF16->F32 PG ONE_PLUS F=1024 rows=1 -> Optimized");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 4096, 128, 1) == RmsNormImplementation::Optimized,
          "GDN norm F32->F32 PG DIRECT rows=1 -> Optimized");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 8193) == RmsNormImplementation::Correctness,
          "rows=8193 (unmeasured) -> Correctness");

    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 2560, 2560, 1) == RmsNormImplementation::Correctness,
          "wrong layout (PG for hidden PF profile) -> Correctness");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              1, 2560, 2560, 1) == RmsNormImplementation::Correctness,
          "wrong mode (DIRECT for ONE_PLUS profile) -> Correctness");
    check(sel(DT::F32, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 1) == RmsNormImplementation::Correctness,
          "unsupported dtype (F32->BF16) -> Correctness");

    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 1234, 1234, 1) == RmsNormImplementation::Correctness,
          "unknown features 1234 -> Correctness");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 2560, 999, 1) == RmsNormImplementation::Correctness,
          "unknown group 999 -> Correctness");

    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 6144, 128, 1) == RmsNormImplementation::Optimized,
          "27B GDN norm F32->F32 PG DIRECT F=6144 G=128 rows=1 -> Optimized");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 6144, 128, 2048) == RmsNormImplementation::Optimized,
          "27B GDN norm rows=2048 -> Optimized");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 6144, 128, 8192) == RmsNormImplementation::Optimized,
          "27B GDN norm rows=8192 -> Optimized");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 6144, 128, 8193) == RmsNormImplementation::Correctness,
          "27B GDN norm rows=8193 -> Correctness");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 6144, 256, 1) == RmsNormImplementation::Optimized,
          "27B q_norm BF16->F32 PG ONE_PLUS F=6144 G=256 rows=1 -> Optimized");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 6144, 256, 2048) == RmsNormImplementation::Optimized,
          "27B q_norm rows=2048 -> Optimized");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 6144, 128, 1) == RmsNormImplementation::Correctness,
          "27B q_norm wrong group -> Correctness");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 6144, 256, 1) == RmsNormImplementation::Correctness,
          "27B GDN norm wrong group -> Correctness");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 3072, 128, 1) == RmsNormImplementation::Optimized,
          "27b tp2 GDN norm F=3072 G=128 rows=1 -> Optimized");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 3072, 128, 2048) == RmsNormImplementation::Optimized,
          "27b tp2 GDN norm rows=2048 -> Optimized");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 512, 256, 1) == RmsNormImplementation::Optimized,
          "27b tp2 K norm F=512 G=256 rows=1 -> Optimized");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 3072, 256, 1) == RmsNormImplementation::Optimized,
          "27b tp2 q_norm F=3072 G=256 rows=1 -> Optimized");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 3072, 256, 8193) == RmsNormImplementation::Correctness,
          "27b tp2 q_norm rows=8193 -> Correctness");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
