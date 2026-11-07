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

using shelf::kernel::Bf16GemmConfig;
using shelf::kernel::Bf16GemmConfigId;
using shelf::kernel::Fp8FinegrainedGemmConfig;
using shelf::kernel::Fp8FinegrainedGemmConfigId;
using shelf::execution::Bf16GemmSelectorInput;
using shelf::execution::Fp8FinegrainedGemmSelectorInput;
using shelf::execution::select_bf16_gemm_config;
using shelf::execution::select_fp8_finegrained_gemm_config;

std::optional<Bf16GemmConfig> sel_bf16(uint32_t rows, uint32_t out_features, uint32_t k,
                                       uint32_t stride, bool weight_aligned, bool input_aligned,
                                       bool allow_exact_rows, bool allow_dot2_rows = false) {
    Bf16GemmSelectorInput in;
    in.rows = rows;
    in.out_features = out_features;
    in.k = k;
    in.input_row_stride = stride != 0u ? stride : k;
    in.weight_aligned16 = weight_aligned;
    in.input_aligned16 = input_aligned;
    in.allow_exact_rows = allow_exact_rows;
    in.allow_dot2_rows = allow_dot2_rows;
    return select_bf16_gemm_config(in);
}

std::optional<Fp8FinegrainedGemmConfig> sel_fp8(uint32_t rows, uint32_t out_features, uint32_t k,
                                             uint32_t k_padded) {
    return select_fp8_finegrained_gemm_config(
        Fp8FinegrainedGemmSelectorInput{rows, out_features, k, k_padded});
}

bool is_fp8(const std::optional<Fp8FinegrainedGemmConfig>& c, Fp8FinegrainedGemmConfigId id) {
    return c.has_value() && c->id == id;
}

bool is_bf16(const std::optional<Bf16GemmConfig>& c, Bf16GemmConfigId id,
             uint8_t exact_rows) {
    if (!c.has_value() || c->id != id) return false;
    if (id == Bf16GemmConfigId::ExactRows || id == Bf16GemmConfigId::ExactRowsDot2)
        return c->exact_rows == exact_rows;
    return true;
}

void test_bf16() {
    check(!sel_bf16(1, 123, 456, 456, true, true, true).has_value(),
          "bf16 unknown geometry -> no config");
    check(!sel_bf16(0, 1024, 2560, 2560, true, true, true).has_value(),
          "bf16 rows=0 -> no config");

    check(is_bf16(sel_bf16(1, 1024, 2560, 2560, true, true, true),
                  Bf16GemmConfigId::ExactRows, 1u),
          "bf16 rows=1 aligned -> ExactRows(1)");
    check(is_bf16(sel_bf16(1, 1024, 2560, 2560, false, true, true),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 rows=1 weight unaligned -> Wmma");
    check(is_bf16(sel_bf16(1, 1024, 2560, 2560, true, true, false),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 rows=1 without allow_exact_rows -> Wmma");

    bool exact_ok = true;
    for (uint32_t rows = 2u; rows <= shelf::kernel::kBf16GemmExactRowsKernelMax; ++rows) {
        exact_ok = exact_ok &&
                   is_bf16(sel_bf16(rows, 1024, 2560, 2560, true, true, true),
                           Bf16GemmConfigId::ExactRows, static_cast<uint8_t>(rows));
    }
    check(exact_ok, "bf16 rows 2..exact kernel max aligned -> ExactRows(rows)");

    bool dot2_ok = true;
    for (uint32_t rows = 2u; rows <= shelf::kernel::kBf16GemmExactRowsKernelMax; ++rows) {
        dot2_ok = dot2_ok &&
                  is_bf16(sel_bf16(rows, 1024, 2560, 2560, true, true, true, true),
                          Bf16GemmConfigId::ExactRowsDot2, static_cast<uint8_t>(rows));
    }
    check(dot2_ok, "bf16 allow_dot2_rows rows 2..exact kernel max -> ExactRowsDot2(rows)");
    check(is_bf16(sel_bf16(1, 1024, 2560, 2560, true, true, true, true),
                  Bf16GemmConfigId::ExactRows, 1u),
          "bf16 rows=1 ignores allow_dot2_rows -> ExactRows(1)");

    check(is_bf16(sel_bf16(shelf::kernel::kBf16GemmExactRowsKernelMax + 1u, 1024, 2560, 2560,
                           true, true, true),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 rows past the exact kernel max -> Wmma");
    check(is_bf16(sel_bf16(8, 1024, 2560, 2560, false, true, true),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 weight unaligned -> Wmma");
    check(is_bf16(sel_bf16(8, 1024, 2560, 2559, true, true, true),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 row stride unaligned -> Wmma");
    check(is_bf16(sel_bf16(65, 1024, 2560, 2560, true, true, true),
                  Bf16GemmConfigId::WmmaWide, 0u),
          "bf16 rows=65 -> WmmaWide");

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

    check(is_fp8(sel_fp8(1, 1024, 2560, 2560), Fp8FinegrainedGemmConfigId::RowBlock1),
          "fp8 rows=1 -> RowBlock1");
    check(is_fp8(sel_fp8(16, 1024, 2560, 2560), Fp8FinegrainedGemmConfigId::RowBlock1),
          "fp8 rows=16 -> RowBlock1");
    check(is_fp8(sel_fp8(17, 1024, 2560, 2560), Fp8FinegrainedGemmConfigId::RowBlock2),
          "fp8 rows=17 -> RowBlock2");
    check(is_fp8(sel_fp8(64, 1024, 2560, 2560), Fp8FinegrainedGemmConfigId::RowBlock4),
          "fp8 rows=64 -> RowBlock4");
    check(is_fp8(sel_fp8(128, 1024, 2560, 2560), Fp8FinegrainedGemmConfigId::RowBlock4),
          "fp8 rows=128 -> RowBlock4");
    check(is_fp8(sel_fp8(256, 1024, 2560, 2560), Fp8FinegrainedGemmConfigId::RowBlock4),
          "fp8 rows=256 -> RowBlock4");
    check(is_fp8(sel_fp8(512, 1024, 2560, 2560), Fp8FinegrainedGemmConfigId::Prefill2D),
          "fp8 2d rows=512 out=1024 -> Prefill2D");
    check(is_fp8(sel_fp8(511, 1024, 2560, 2560), Fp8FinegrainedGemmConfigId::RowBlock4),
          "fp8 rows=511 < 2d threshold -> RowBlock4");
    check(is_fp8(sel_fp8(512, 32, 2560, 2560), Fp8FinegrainedGemmConfigId::RowBlock1),
          "fp8 small out_features -> RowBlock1");
}

void test_9b_shapes() {
    const uint32_t shapes[][2] = {
        {12288u, 4096u},
        {4096u, 12288u},
        {8192u, 4096u},
        {4096u, 4096u},
        {32u, 4096u},
        {1024u, 4096u},
        {248320u, 4096u},
    };
    char name[96];
    for (const auto& s : shapes) {
        for (uint32_t rows : {1u, 8u, 256u}) {
            std::snprintf(name, sizeof(name), "9b bf16 %ux%u rows=%u -> config",
                          s[0], s[1], rows);
            check(sel_bf16(rows, s[0], s[1], s[1], true, true, false).has_value(), name);
        }
    }
    check(!sel_bf16(1, 12289, 4096, 4096, true, true, false).has_value(),
          "9b bf16 unknown shape still rejected");
    check(!sel_fp8(1, 12288, 4096, 4096).has_value(),
          "9b bf16 shapes are not enabled for fp8");
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

    const uint32_t tp2_bf16_drafter[][2] = {
        {8192u, 5120u}, {5120u, 8192u}, {5120u, 2560u}, {2560u, 10240u},
    };
    for (const auto& s : tp2_bf16_drafter) {
        std::snprintf(name, sizeof(name), "tp2 bf16 drafter %ux%u rows=1 -> config",
                      s[0], s[1]);
        check(sel_bf16(1, s[0], s[1], s[1], true, true, false).has_value(), name);
        std::snprintf(name, sizeof(name), "tp2 bf16 drafter %ux%u rows=256 -> config",
                      s[0], s[1]);
        check(sel_bf16(256, s[0], s[1], s[1], true, true, false).has_value(), name);
    }
    check(!sel_bf16(256, 999, 5120, 5120, true, true, false).has_value(),
          "unknown bf16 shape still rejected");
}

void test_moe_tp2_shapes() {
    // Qwen3.6-35B-A3B MoE at tp_size 2: the rank-local backbone widths. The routed expert
    // plane splits along the expert axis, so its grouped widths are unchanged.
    using shelf::execution::linear_shape_supported;
    using shelf::execution::LinearComputeFamily;
    check(linear_shape_supported(LinearComputeFamily::Bf16, 2048u, 2048u),
          "moe tp2 bf16 2048x2048 (in_proj_z, out_proj, o_proj)");
    check(linear_shape_supported(LinearComputeFamily::Bf16, 2048u, 256u),
          "moe tp2 bf16 2048x256 (shared expert down)");
    check(linear_shape_supported(LinearComputeFamily::Bf16, 16u, 2048u),
          "moe tp2 bf16 16x2048 (GDN in_proj_a, in_proj_b)");
    check(linear_shape_supported(LinearComputeFamily::Bf16, 256u, 2048u),
          "moe tp2 bf16 256x2048 (k_proj, shared gate and up)");
    check(linear_shape_supported(LinearComputeFamily::Bf16, 4096u, 2048u),
          "moe tp2 bf16 4096x2048 (in_proj_qkv, q_proj)");
    check(!linear_shape_supported(LinearComputeFamily::Bf16, 2048u, 1024u),
          "moe tp2 bf16 2048x1024 stays unmeasured");
}

}  // namespace

int main() {
    test_bf16();
    test_fp8();
    test_9b_shapes();
    test_tp2_shapes();
    test_moe_tp2_shapes();

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
