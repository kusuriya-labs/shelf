#include <shelf/inference/execution/selectors/linear_selector.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace ps::execution {

namespace {

// Kernel capability allowlist: the (out_features, k) geometries the optimized
// WMMA linear kernels have been validated against. The bits identify which
// compute family can execute a geometry; Qwen model geometry validation is a
// separate, model-load-time concern.
constexpr uint8_t kFamilyBf16 = 1u << 0u;
constexpr uint8_t kFamilyFp8 = 1u << 3u;
constexpr uint8_t kFamilyPsq5 = 1u << 4u;

struct LinearShapeRule {
    uint32_t out_features;
    uint32_t k;
    uint8_t families;
};

constexpr LinearShapeRule kLinearShapeRules[] = {
    { 32, 2560, kFamilyBf16 | kFamilyFp8 },
    { 48, 5120, kFamilyBf16 | kFamilyFp8 | kFamilyPsq5 },
    { 1024, 2560, kFamilyBf16 | kFamilyFp8 },
    { 1024, 5120, kFamilyBf16 | kFamilyFp8 | kFamilyPsq5 },
    { 1280, 5120, kFamilyBf16 },
    { 2560, 4096, kFamilyBf16 | kFamilyFp8 },
    { 2560, 5120, kFamilyBf16 },
    { 2560, 9216, kFamilyBf16 | kFamilyFp8 },
    { 4096, 2560, kFamilyBf16 | kFamilyFp8 },
    { 4096, 5120, kFamilyBf16 },
    { 5120, 4096, kFamilyBf16 },
    { 5120, 6144, kFamilyBf16 | kFamilyFp8 | kFamilyPsq5 },
    { 5120, 10240, kFamilyBf16 },
    { 5120, 15360, kFamilyBf16 },
    { 5120, 16384, kFamilyBf16 },
    { 16384, 5120, kFamilyBf16 },
    { 5120, 17408, kFamilyBf16 | kFamilyFp8 | kFamilyPsq5 },
    { 5120, 25600, kFamilyBf16 },
    { 6144, 5120, kFamilyBf16 | kFamilyFp8 | kFamilyPsq5 },
    { 8192, 2560, kFamilyBf16 | kFamilyFp8 },
    { 9216, 2560, kFamilyBf16 | kFamilyFp8 },
    { 256, 5120, kFamilyBf16 },
    { 10240, 5120, kFamilyBf16 | kFamilyFp8 | kFamilyPsq5 },
    { 12288, 5120, kFamilyBf16 | kFamilyFp8 | kFamilyPsq5 },
    { 124160, 5120, kFamilyBf16 },
    { 17408, 5120, kFamilyBf16 | kFamilyFp8 | kFamilyPsq5 },
    { 248320, 2560, kFamilyBf16 },
    { 248320, 5120, kFamilyBf16 },
    { 24, 5120, kFamilyBf16 | kFamilyFp8 },
    { 8704, 5120, kFamilyFp8 },
    { 5120, 8704, kFamilyFp8 },
    { 3072, 5120, kFamilyFp8 },
    { 512, 5120, kFamilyFp8 },
    { 5120, 3072, kFamilyFp8 },
    { 5120, 5120, kFamilyBf16 | kFamilyFp8 | kFamilyPsq5 },
    { 8192, 5120, kFamilyBf16 },
    { 5120, 8192, kFamilyBf16 },
    { 5120, 2560, kFamilyBf16 },
    { 2560, 10240, kFamilyBf16 },
};

uint8_t family_bit(LinearComputeFamily family) {
    switch (family) {
        case LinearComputeFamily::Bf16: return kFamilyBf16;
        case LinearComputeFamily::Fp8W8A8: return kFamilyFp8;
        case LinearComputeFamily::Psq5W8A8: return kFamilyPsq5;
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

ps::kernel::Fp8GemmConfig fp8_row_block_config(uint32_t rows, uint32_t out_features) {
    using Id = ps::kernel::Fp8GemmConfigId;
    if (rows >= 4u * ps::kernel::kFp8GemmRowTile &&
        out_features >= ps::kernel::kBlockScaledGemmRowBlockMinOutFeatures)
        return ps::kernel::Fp8GemmConfig{Id::RowBlock4};
    if (rows > ps::kernel::kFp8GemmRowTile &&
        out_features >= ps::kernel::kBlockScaledGemmRowBlockMinOutFeatures)
        return ps::kernel::Fp8GemmConfig{Id::RowBlock2};
    return ps::kernel::Fp8GemmConfig{Id::RowBlock1};
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

std::optional<ps::kernel::Bf16GemmConfig>
select_bf16_gemm_config(const Bf16GemmSelectorInput& input) {
    using Id = ps::kernel::Bf16GemmConfigId;
    if (input.rows == 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Bf16, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Bf16, input.out_features, input.k, input.rows);
        return std::nullopt;
    }

    const bool k_aligned = (input.k & (ps::kernel::kBf16GemmExactKAlignment - 1u)) == 0u;
    if (input.allow_exact_rows && input.rows == 1u && k_aligned && input.weight_aligned16 &&
        input.input_aligned16)
        return ps::kernel::Bf16GemmConfig{Id::ExactRows, 1u};

    if (input.allow_exact_rows && input.verify_exact
        && input.rows >= 2u
        && input.rows <= ps::kernel::kBf16GemmExactRowsMax
        && k_aligned
        && input.weight_aligned16
        && input.input_aligned16
        && (input.input_row_stride & (ps::kernel::kBf16GemmExactRowStrideAlignment - 1u)) == 0u) {
        if (input.allow_dot2_rows)
            return ps::kernel::Bf16GemmConfig{Id::ExactRowsDot2,
                                              static_cast<uint8_t>(input.rows)};
        return ps::kernel::Bf16GemmConfig{Id::ExactRows, static_cast<uint8_t>(input.rows)};
    }

    if (input.out_features <= ps::kernel::kBf16GemmSplitKMaxOutFeatures)
        return ps::kernel::Bf16GemmConfig{Id::WmmaKPartition, 0u};
    if (input.rows >= ps::kernel::kBf16GemmWideMinRows)
        return ps::kernel::Bf16GemmConfig{Id::WmmaWide, 0u};
    return ps::kernel::Bf16GemmConfig{Id::Wmma, 0u};
}

std::optional<ps::kernel::Fp8GemmConfig>
select_fp8_gemm_config(const Fp8GemmSelectorInput& input) {
    if (input.rows == 0u)
        return std::nullopt;
    if ((input.k_padded & 127u) != 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Fp8W8A8, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Fp8W8A8, input.out_features, input.k, input.rows);
        return std::nullopt;
    }
    const bool use_2d =
        (input.k_padded % ps::kernel::kFp8GemmPrefill2dKChunk) == 0u &&
        (input.rows % ps::kernel::kBlockScaledGemmPrefill2dRowsPerBlock) == 0u &&
        (input.out_features % ps::kernel::kFp8GemmPrefill2dOutBlock) == 0u &&
        input.rows >= ps::kernel::kBlockScaledGemmPrefill2dMinRows &&
        input.out_features >= ps::kernel::kBlockScaledGemmPrefill2dMinOutFeatures;
    if (use_2d)
        return ps::kernel::Fp8GemmConfig{ps::kernel::Fp8GemmConfigId::Prefill2D};
    return fp8_row_block_config(input.rows, input.out_features);
}

std::optional<ps::kernel::Psq5GemmConfig>
select_psq5_gemm_config(const Fp8GemmSelectorInput& input) {
    using Id = ps::kernel::Psq5GemmConfigId;
    if (input.rows == 0u)
        return std::nullopt;
    if ((input.k_padded & 127u) != 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Psq5W8A8, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Psq5W8A8, input.out_features, input.k, input.rows);
        return std::nullopt;
    }
    const bool use_wide =
        (input.k_padded % ps::kernel::kPsq5GemmPrefill2dKChunk) == 0u &&
        (input.rows % ps::kernel::kPsq5GemmPrefill2dWideRowsPerBlock) == 0u &&
        (input.out_features % ps::kernel::kPsq5GemmPrefill2dWideOutBlock) == 0u &&
        input.rows >= ps::kernel::kBlockScaledGemmPrefill2dMinRows &&
        input.out_features >= ps::kernel::kBlockScaledGemmPrefill2dMinOutFeatures;
    if (use_wide) {
        static const int mode = []() -> int {
            const char* e = std::getenv("SHELF_PSQ5_P2D");
            if (e == nullptr || e[0] == '\0') return 0;
            return std::atoi(e);
        }();
        if (mode == 0)
            return ps::kernel::Psq5GemmConfig{Id::Prefill2D};
        if (mode == 2)
            return ps::kernel::Psq5GemmConfig{Id::Prefill2DWStage};
        return ps::kernel::Psq5GemmConfig{Id::Prefill2DWide};
    }
    const bool use_2d =
        (input.k_padded % ps::kernel::kPsq5GemmPrefill2dKChunk) == 0u &&
        (input.rows % ps::kernel::kBlockScaledGemmPrefill2dRowsPerBlock) == 0u &&
        (input.out_features % ps::kernel::kPsq5GemmPrefill2dOutBlock) == 0u &&
        input.rows >= ps::kernel::kBlockScaledGemmPrefill2dMinRows &&
        input.out_features >= ps::kernel::kBlockScaledGemmPrefill2dMinOutFeatures;
    if (use_2d)
        return ps::kernel::Psq5GemmConfig{Id::Prefill2D};
    const uint32_t row_tile = ps::kernel::kPsq5GemmRowTile;
    const uint32_t min_out = ps::kernel::kBlockScaledGemmRowBlockMinOutFeatures;
    if (input.rows >= 4u * row_tile && input.out_features >= min_out)
        return ps::kernel::Psq5GemmConfig{Id::RowBlock4};
    if (input.rows > row_tile && input.out_features >= min_out)
        return ps::kernel::Psq5GemmConfig{Id::RowBlock2};
    return ps::kernel::Psq5GemmConfig{Id::RowBlock1};
}

}  // namespace ps::execution
