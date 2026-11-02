#include <shelf/inference/execution/selectors/linear_selector.h>

#include <cstdio>
#include <optional>

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

using ps::kernel::Bf16GemmConfig;
using ps::kernel::Bf16GemmConfigId;
using ps::kernel::Fp8GemmConfig;
using ps::kernel::Fp8GemmConfigId;
using ps::execution::Bf16GemmSelectorInput;
using ps::execution::Fp8GemmSelectorInput;
using ps::execution::select_bf16_gemm_config;
using ps::execution::select_fp8_gemm_config;

std::optional<Bf16GemmConfig> sel_bf16(uint32_t rows, uint32_t out_features, uint32_t k,
                                       uint32_t stride, bool weight_aligned, bool input_aligned,
                                       bool verify_exact) {
    Bf16GemmSelectorInput in;
    in.rows = rows;
    in.out_features = out_features;
    in.k = k;
    in.input_row_stride = stride != 0u ? stride : k;
    in.weight_aligned16 = weight_aligned;
    in.input_aligned16 = input_aligned;
    in.verify_exact = verify_exact;
    return select_bf16_gemm_config(in);
}

std::optional<Fp8GemmConfig> sel_fp8(uint32_t rows, uint32_t out_features, uint32_t k,
                                             uint32_t k_padded) {
    return select_fp8_gemm_config(
        Fp8GemmSelectorInput{rows, out_features, k, k_padded});
}

bool is_fp8(const std::optional<Fp8GemmConfig>& c, Fp8GemmConfigId id) {
    return c.has_value() && c->id == id;
}

bool is_bf16(const std::optional<Bf16GemmConfig>& c, Bf16GemmConfigId id,
             uint8_t exact_rows) {
    return c.has_value() && c->id == id &&
           (id != Bf16GemmConfigId::ExactRows || c->exact_rows == exact_rows);
}

void test_bf16() {
    check(!sel_bf16(1, 123, 456, 456, true, true, false).has_value(),
          "bf16 unknown geometry -> no config");
    check(!sel_bf16(0, 1024, 2560, 2560, true, true, false).has_value(),
          "bf16 rows=0 -> no config");

    check(is_bf16(sel_bf16(1, 1024, 2560, 2560, true, true, false),
                  Bf16GemmConfigId::ExactRows, 1u),
          "bf16 rows=1 aligned -> ExactRows(1)");
    check(is_bf16(sel_bf16(1, 1024, 2560, 2560, false, true, false),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 rows=1 weight unaligned -> Wmma");

    bool exact_ok = true;
    for (uint32_t rows = 2u; rows <= ps::kernel::kBf16GemmExactRowsMax; ++rows) {
        exact_ok = exact_ok &&
                   is_bf16(sel_bf16(rows, 1024, 2560, 2560, true, true, true),
                           Bf16GemmConfigId::ExactRows, static_cast<uint8_t>(rows));
    }
    check(exact_ok, "bf16 verify_exact rows 2..64 aligned -> ExactRows(rows)");

    check(is_bf16(sel_bf16(8, 1024, 2560, 2560, false, true, true),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 verify_exact weight unaligned -> Wmma");
    check(is_bf16(sel_bf16(8, 1024, 2560, 2559, true, true, true),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 verify_exact row stride unaligned -> Wmma");
    check(is_bf16(sel_bf16(65, 1024, 2560, 2560, true, true, true),
                  Bf16GemmConfigId::WmmaWide, 0u),
          "bf16 verify_exact rows=65 -> WmmaWide");
    check(is_bf16(sel_bf16(17, 1024, 2560, 2560, true, true, true),
                  Bf16GemmConfigId::ExactRows, 17u),
          "bf16 verify_exact rows=17 -> ExactRows(17)");
    check(is_bf16(sel_bf16(64, 1024, 2560, 2560, true, true, true),
                  Bf16GemmConfigId::ExactRows, 64u),
          "bf16 verify_exact rows=64 -> ExactRows(64)");

    check(is_bf16(sel_bf16(2, 32, 2560, 2560, true, true, false),
                  Bf16GemmConfigId::WmmaKPartition, 0u),
          "bf16 small output geometry -> WmmaKPartition");
    check(is_bf16(sel_bf16(65, 1024, 2560, 2560, true, true, false),
                  Bf16GemmConfigId::WmmaWide, 0u),
          "bf16 large rows -> WmmaWide");
    check(is_bf16(sel_bf16(2, 1024, 2560, 2560, true, true, false),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 default geometry -> Wmma");
}

void test_fp8() {
    check(!sel_fp8(100, 999, 8888, 8896).has_value(),
          "fp8 unknown geometry -> no config");
    check(!sel_fp8(0, 1024, 2560, 2560).has_value(),
          "fp8 rows=0 -> no config");
    check(!sel_fp8(1, 1024, 2560, 2561).has_value(),
          "fp8 k_padded not a multiple of 128 -> no config");

    check(is_fp8(sel_fp8(1, 1024, 2560, 2560), Fp8GemmConfigId::RowBlock1),
          "fp8 rows=1 -> RowBlock1");
    check(is_fp8(sel_fp8(16, 1024, 2560, 2560), Fp8GemmConfigId::RowBlock1),
          "fp8 rows=16 -> RowBlock1");
    check(is_fp8(sel_fp8(17, 1024, 2560, 2560), Fp8GemmConfigId::RowBlock2),
          "fp8 rows=17 -> RowBlock2");
    check(is_fp8(sel_fp8(64, 1024, 2560, 2560), Fp8GemmConfigId::RowBlock4),
          "fp8 rows=64 -> RowBlock4");
    check(is_fp8(sel_fp8(128, 1024, 2560, 2560), Fp8GemmConfigId::RowBlock4),
          "fp8 rows=128 -> RowBlock4");
    check(is_fp8(sel_fp8(256, 1024, 2560, 2560), Fp8GemmConfigId::RowBlock4),
          "fp8 rows=256 -> RowBlock4");
    check(is_fp8(sel_fp8(512, 1024, 2560, 2560), Fp8GemmConfigId::Prefill2D),
          "fp8 2d rows=512 out=1024 -> Prefill2D");
    check(is_fp8(sel_fp8(511, 1024, 2560, 2560), Fp8GemmConfigId::RowBlock4),
          "fp8 rows=511 < 2d threshold -> RowBlock4");
    check(is_fp8(sel_fp8(512, 32, 2560, 2560), Fp8GemmConfigId::RowBlock1),
          "fp8 small out_features -> RowBlock1");
}

void test_tp2_shapes() {
    check(sel_bf16(1, 24, 5120, 5120, true, true, false).has_value(),
          "tp2 bf16 24x5120 rows=1 -> config");
    check(!sel_bf16(1, 8704, 5120, 5120, true, true, false).has_value(),
          "tp2 bf16 8704x5120 not enabled");

    check(sel_bf16(1, 124160, 5120, 5120, true, true, false).has_value(),
          "vocab parallel bf16 124160x5120 rows=1 -> config");
    check(sel_bf16(5, 124160, 5120, 5120, true, true, false).has_value(),
          "vocab parallel bf16 124160x5120 rows=5 -> config");
    check(sel_bf16(1, 248320, 5120, 5120, true, true, false).has_value(),
          "replicated bf16 248320x5120 rows=1 -> config");
    check(!sel_fp8(1, 124160, 5120, 5120).has_value(),
          "vocab parallel fp8 shape not enabled");

    const uint32_t tp2_shapes[][2] = {
        {8704u, 5120u}, {5120u, 8704u}, {3072u, 5120u},
        {512u, 5120u},   {5120u, 3072u}, {5120u, 5120u}, {24u, 5120u},
    };
    char name[96];
    for (const auto& s : tp2_shapes) {
        std::snprintf(name, sizeof(name), "tp2 fp8 %ux%u rows=1 -> config", s[0], s[1]);
        check(sel_fp8(1, s[0], s[1], s[1]).has_value(), name);
        std::snprintf(name, sizeof(name), "tp2 fp8 %ux%u rows=5 -> config", s[0], s[1]);
        check(sel_fp8(5, s[0], s[1], s[1]).has_value(), name);
    }

    check(!sel_fp8(1, 999, 5120, 5120).has_value(), "unknown fp8 shape still rejected");
    check(sel_fp8(1, 24, 5120, 5120).has_value(), "tp2 fp8 24x5120 rows=1 -> config");
    check(!sel_fp8(1, 24, 5120, 5184).has_value(), "fp8 k_padded misaligned rejected");

    const uint32_t tp2_bf16_eagle4[][2] = {
        {8192u, 5120u}, {5120u, 8192u}, {5120u, 2560u}, {2560u, 10240u},
    };
    for (const auto& s : tp2_bf16_eagle4) {
        std::snprintf(name, sizeof(name), "tp2 bf16 eagle4 %ux%u rows=1 -> config",
                      s[0], s[1]);
        check(sel_bf16(1, s[0], s[1], s[1], true, true, false).has_value(), name);
        std::snprintf(name, sizeof(name), "tp2 bf16 eagle4 %ux%u rows=256 -> config",
                      s[0], s[1]);
        check(sel_bf16(256, s[0], s[1], s[1], true, true, false).has_value(), name);
    }
    check(!sel_bf16(256, 999, 5120, 5120, true, true, false).has_value(),
          "unknown bf16 shape still rejected");
}

}  // namespace

int main() {
    test_bf16();
    test_fp8();
    test_tp2_shapes();

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
