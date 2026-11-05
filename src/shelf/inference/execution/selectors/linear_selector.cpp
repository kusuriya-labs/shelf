#include <shelf/inference/execution/selectors/linear_selector.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace shelf::execution {

namespace {

// Kernel capability allowlist: the (out_features, k) geometries the optimized
// WMMA linear kernels have been validated against. The bits identify which
// compute family can execute a geometry; Qwen model geometry validation is a
// separate, model-load-time concern.
constexpr uint8_t kFamilyBf16 = 1u << 0u;
constexpr uint8_t kFamilyFp8Finegrained = 1u << 3u;
constexpr uint8_t kFamilyMxfp4 = 1u << 4u;

struct LinearShapeRule {
    uint32_t out_features;
    uint32_t k;
    uint8_t families;
};

constexpr LinearShapeRule kLinearShapeRules[] = {
    { 32, 2560, kFamilyBf16 | kFamilyFp8Finegrained },
    { 48, 5120, kFamilyBf16 | kFamilyFp8Finegrained | kFamilyMxfp4 },
    { 1024, 2560, kFamilyBf16 | kFamilyFp8Finegrained },
    { 1024, 5120, kFamilyBf16 | kFamilyFp8Finegrained | kFamilyMxfp4 },
    { 1280, 5120, kFamilyBf16 },
    { 2560, 4096, kFamilyBf16 | kFamilyFp8Finegrained },
    { 2560, 5120, kFamilyBf16 },
    { 2560, 9216, kFamilyBf16 | kFamilyFp8Finegrained },
    { 4096, 2560, kFamilyBf16 | kFamilyFp8Finegrained },
    { 4096, 5120, kFamilyBf16 },
    { 5120, 4096, kFamilyBf16 },
    { 5120, 6144, kFamilyBf16 | kFamilyFp8Finegrained | kFamilyMxfp4 },
    { 5120, 10240, kFamilyBf16 },
    { 5120, 15360, kFamilyBf16 },
    { 5120, 16384, kFamilyBf16 },
    { 16384, 5120, kFamilyBf16 },
    { 5120, 17408, kFamilyBf16 | kFamilyFp8Finegrained | kFamilyMxfp4 },
    { 5120, 25600, kFamilyBf16 },
    { 6144, 5120, kFamilyBf16 | kFamilyFp8Finegrained | kFamilyMxfp4 },
    { 8192, 2560, kFamilyBf16 | kFamilyFp8Finegrained },
    { 9216, 2560, kFamilyBf16 | kFamilyFp8Finegrained },
    { 256, 5120, kFamilyBf16 },
    { 256, 2560, kFamilyBf16 },
    { 10240, 5120, kFamilyBf16 | kFamilyFp8Finegrained | kFamilyMxfp4 },
    { 12288, 5120, kFamilyBf16 | kFamilyFp8Finegrained | kFamilyMxfp4 },
    { 12288, 4096, kFamilyBf16 },
    { 4096, 12288, kFamilyBf16 },
    { 8192, 4096, kFamilyBf16 },
    { 4096, 4096, kFamilyBf16 },
    { 32, 4096, kFamilyBf16 },
    { 1024, 4096, kFamilyBf16 },
    { 248320, 4096, kFamilyBf16 },
    { 124160, 5120, kFamilyBf16 },
    { 17408, 5120, kFamilyBf16 | kFamilyFp8Finegrained | kFamilyMxfp4 },
    { 248320, 2560, kFamilyBf16 },
    { 248320, 5120, kFamilyBf16 },
    { 24, 5120, kFamilyBf16 | kFamilyFp8Finegrained },
    { 8704, 5120, kFamilyFp8Finegrained },
    { 5120, 8704, kFamilyFp8Finegrained },
    { 3072, 5120, kFamilyFp8Finegrained },
    { 512, 5120, kFamilyFp8Finegrained },
    { 5120, 3072, kFamilyFp8Finegrained },
    { 5120, 5120, kFamilyBf16 | kFamilyFp8Finegrained },
    { 8192, 5120, kFamilyBf16 },
    { 5120, 8192, kFamilyBf16 },
    { 5120, 2560, kFamilyBf16 },
    { 2560, 10240, kFamilyBf16 },
    // Routed MoE FFN in bf16: the router over the experts, and the shared expert next to it.
    // The routed experts themselves are grouped mxfp4 launches, which take their geometry
    // from the expert plane rather than from this table.
    { 256, 2048, kFamilyBf16 },
    { 512, 2048, kFamilyBf16 },
    { 2048, 512, kFamilyBf16 },
    { 1, 2048, kFamilyBf16 },
    // The hybrid backbone of the grouped-expert checkpoint at hidden 2048: GDN in/out
    // projections, attention q and o, and the lm_head row over the full vocabulary.
    { 8192, 2048, kFamilyBf16 },
    { 4096, 2048, kFamilyBf16 },
    { 32, 2048, kFamilyBf16 },
    { 2048, 4096, kFamilyBf16 },
    { 248320, 2048, kFamilyBf16 },
    // Qwen3.6-35B-A3B MoE at tp_size 2. The expert plane splits along the expert axis, so the
    // grouped widths are unchanged; what halves is the backbone: in_proj_z / out_proj / o_proj
    // (2048 by 2048), the shared expert down (2048 by 256) and the GDN a / b projections
    // (16 by 2048). The shared gate / up (256 by 2048) and k_proj (256 by 2048) are already here.
    { 2048, 2048, kFamilyBf16 },
    { 2048, 256, kFamilyBf16 },
    { 16, 2048, kFamilyBf16 },
    // Qwen3.8-Flash-Next (Qwen4Exp): the replicated backbone and hyper-connection widths.
    { 320, 10240, kFamilyBf16 },
    { 10240, 320, kFamilyBf16 },
    { 4, 10240, kFamilyBf16 },
    { 10240, 2560, kFamilyBf16 },
    { 6144, 2560, kFamilyBf16 },
    { 48, 2560, kFamilyBf16 },
    { 2560, 6144, kFamilyBf16 },
    { 12288, 2560, kFamilyBf16 },
    { 512, 2560, kFamilyBf16 },
    { 640, 2560, kFamilyBf16 | kFamilyMxfp4 },
    { 1, 2560, kFamilyBf16 },
    { 2560, 640, kFamilyBf16 | kFamilyMxfp4 },
    { 2560, 2560, kFamilyBf16 },
    // Qwen3.8-Flash-Next DFlash2 drafter: q/k/v/o, gate/up, down, and the fc over five
    // target taps (5 * 2560).
    { 7680, 2560, kFamilyBf16 },
    { 2560, 7680, kFamilyBf16 },
    { 2560, 12800, kFamilyBf16 },
    { 512, 2560, kFamilyBf16 },
};

uint8_t family_bit(LinearComputeFamily family) {
    switch (family) {
        case LinearComputeFamily::Bf16: return kFamilyBf16;
        case LinearComputeFamily::Fp8FinegrainedW8A8: return kFamilyFp8Finegrained;
        case LinearComputeFamily::Mxfp4W4A8: return kFamilyMxfp4;
    }
    return 0u;
}

void linear_debug_miss(LinearComputeFamily family, uint32_t out_features, uint32_t k,
                       uint32_t rows) {
    static const bool linear_debug = []() {
        const char* e = std::getenv("SHELF_LINEAR_DEBUG");
        return e != nullptr && e[0] != '\0';
    }();
    if (linear_debug) {
        std::fprintf(stderr, "LINEAR_SELECT_MISS family=%u out=%u k=%u rows=%u\n",
                     static_cast<unsigned>(family), out_features, k, rows);
    }
}

shelf::kernel::Fp8FinegrainedGemmConfig fp8_finegrained_row_block_config(uint32_t rows, uint32_t out_features) {
    using Id = shelf::kernel::Fp8FinegrainedGemmConfigId;
    if (rows >= 4u * shelf::kernel::kFp8FinegrainedGemmRowTile &&
        out_features >= shelf::kernel::kBlockScaledGemmRowBlockMinOutFeatures)
        return shelf::kernel::Fp8FinegrainedGemmConfig{Id::RowBlock4};
    if (rows > shelf::kernel::kFp8FinegrainedGemmRowTile &&
        out_features >= shelf::kernel::kBlockScaledGemmRowBlockMinOutFeatures)
        return shelf::kernel::Fp8FinegrainedGemmConfig{Id::RowBlock2};
    return shelf::kernel::Fp8FinegrainedGemmConfig{Id::RowBlock1};
}

}  // namespace

bool linear_shape_supported(
    LinearComputeFamily family,
    uint32_t out_features,
    uint32_t k) {
    const uint8_t bit = family_bit(family);
    for (const auto& rule : kLinearShapeRules) {
        if (rule.out_features == out_features
            && rule.k == k
            && (rule.families & bit) != 0u) {
            return true;
        }
    }
    return false;
}

std::optional<shelf::kernel::Bf16GemmConfig>
select_bf16_gemm_config(const Bf16GemmSelectorInput& input) {
    using Id = shelf::kernel::Bf16GemmConfigId;
    if (input.rows == 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Bf16, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Bf16, input.out_features, input.k, input.rows);
        return std::nullopt;
    }

    const bool k_aligned = (input.k & (shelf::kernel::kBf16GemmExactKAlignment - 1u)) == 0u;
    if (input.allow_exact_rows && input.rows == 1u && k_aligned && input.weight_aligned16 &&
        input.input_aligned16)
        return shelf::kernel::Bf16GemmConfig{Id::ExactRows, 1u};

    if (input.allow_exact_rows && input.verify_exact
        && input.rows >= 2u
        && input.rows <= shelf::kernel::kBf16GemmExactRowsMax
        && k_aligned
        && input.weight_aligned16
        && input.input_aligned16
        && (input.input_row_stride & (shelf::kernel::kBf16GemmExactRowStrideAlignment - 1u)) == 0u) {
        if (input.allow_dot2_rows)
            return shelf::kernel::Bf16GemmConfig{Id::ExactRowsDot2,
                                              static_cast<uint8_t>(input.rows)};
        return shelf::kernel::Bf16GemmConfig{Id::ExactRows, static_cast<uint8_t>(input.rows)};
    }

    if (input.out_features <= shelf::kernel::kBf16GemmSplitKMaxOutFeatures)
        return shelf::kernel::Bf16GemmConfig{Id::WmmaKPartition, 0u};
    if (input.rows >= shelf::kernel::kBf16GemmWideMinRows)
        return shelf::kernel::Bf16GemmConfig{Id::WmmaWide, 0u};
    return shelf::kernel::Bf16GemmConfig{Id::Wmma, 0u};
}

std::optional<shelf::kernel::Fp8FinegrainedGemmConfig>
select_fp8_finegrained_gemm_config(const Fp8FinegrainedGemmSelectorInput& input) {
    if (input.rows == 0u)
        return std::nullopt;
    if ((input.k_padded & 127u) != 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Fp8FinegrainedW8A8, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Fp8FinegrainedW8A8, input.out_features, input.k, input.rows);
        return std::nullopt;
    }
    const bool use_2d =
        (input.k_padded % shelf::kernel::kFp8FinegrainedGemmPrefill2dKChunk) == 0u &&
        (input.rows % shelf::kernel::kBlockScaledGemmPrefill2dRowsPerBlock) == 0u &&
        (input.out_features % shelf::kernel::kFp8FinegrainedGemmPrefill2dOutBlock) == 0u &&
        input.rows >= shelf::kernel::kBlockScaledGemmPrefill2dMinRows &&
        input.out_features >= shelf::kernel::kBlockScaledGemmPrefill2dMinOutFeatures;
    if (use_2d)
        return shelf::kernel::Fp8FinegrainedGemmConfig{shelf::kernel::Fp8FinegrainedGemmConfigId::Prefill2D};
    return fp8_finegrained_row_block_config(input.rows, input.out_features);
}

std::optional<shelf::kernel::Mxfp4GemmConfig>
select_mxfp4_gemm_config(const Mxfp4GemmSelectorInput& input) {
    using Id = shelf::kernel::Mxfp4GemmConfigId;
    if (input.rows == 0u)
        return std::nullopt;
    if ((input.k_padded & 127u) != 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Mxfp4W4A8, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Mxfp4W4A8, input.out_features, input.k, input.rows);
        return std::nullopt;
    }
    const bool tile_aligned = (input.out_features & 15u) == 0u;
    // Split-K wins only while the 16-output blocks cannot fill the device; the
    // measured crossover on R9700 is between 2560 and 5120 output features.
    // The 2D staged kernel pays off once the weight tile is reused across enough
    // row blocks; measured crossover on R9700 (docs/research/20261103).
    const bool prefill2d =
        input.rows >= shelf::kernel::kMxfp4GemmPrefill2dMinRows &&
        (input.rows % shelf::kernel::kMxfp4GemmPrefill2dRowsPerBlock) == 0u &&
        (input.k_padded % shelf::kernel::kMxfp4GemmPrefill2dKChunk) == 0u &&
        input.out_features >= 64u;
    if (prefill2d && (input.rows >= 1024u ||
                      static_cast<uint64_t>(input.out_features) * input.k_padded >=
                          shelf::kernel::kMxfp4GemmPrefill2dMinWeights)) {
        return shelf::kernel::Mxfp4GemmConfig{Id::Prefill2D, 0u};
    }
    if (input.rows <= shelf::kernel::kMxfp4GemmSplitKMaxRows && tile_aligned &&
        input.out_features <= shelf::kernel::kMxfp4GemmSplitKMaxOutFeatures) {
        return shelf::kernel::Mxfp4GemmConfig{
            Id::SplitK, static_cast<uint8_t>(shelf::kernel::kMxfp4GemmSplitKWays)};
    }
    if (input.rows <= shelf::kernel::kMxfp4GemmRowBlock1MaxRows)
        return shelf::kernel::Mxfp4GemmConfig{Id::RowBlock1, 0u};
    if (input.rows <= shelf::kernel::kMxfp4GemmRowBlock2MaxRows)
        return shelf::kernel::Mxfp4GemmConfig{Id::RowBlock2, 0u};
    return shelf::kernel::Mxfp4GemmConfig{Id::RowBlock4, 0u};
}

}  // namespace shelf::execution
