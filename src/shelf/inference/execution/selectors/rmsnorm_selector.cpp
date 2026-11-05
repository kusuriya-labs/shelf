#include <shelf/inference/execution/selectors/rmsnorm_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>

namespace shelf::execution {

namespace {

struct RmsNormRule {
    ::shelf::runtime::ValueDType input_dtype;
    ::shelf::runtime::ValueDType output_dtype;
    ::shelf::runtime::ValueDType weight_dtype;
    RmsNormSelectorWeightLayout weight_layout;
    uint32_t weight_mode;
    uint32_t features;
    uint32_t group_size;
    uint32_t min_rows;
    uint32_t max_rows;
};

constexpr uint32_t kMeasuredMaxRows = kOptimizedMaxRows;

constexpr RmsNormRule kRmsNormRules[] = {
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::BF16,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerFeature,
      0, 2560, 2560, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::BF16,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerFeature,
      0, 5120, 5120, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::BF16,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerFeature,
      0, 2048, 2048, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 1024, 256, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 4096, 256, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::F32, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      1, 4096, 128, 1, kMeasuredMaxRows },
    // The GDN head norm of Qwen3.6-35B-A3B at tp_size 2 (local value features 2048).
    { ::shelf::runtime::ValueDType::F32, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      1, 2048, 128, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::F32, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      1, 6144, 128, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 6144, 256, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::F32, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      1, 3072, 128, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 512, 256, 1, kMeasuredMaxRows },
    // The per-head QK norms of Qwen3.6-35B-A3B at tp_size 2: local query features 2048 with
    // head_dim 256 groups. The local key plane is one head wide, so the weight vector is as
    // long as the plane and the resolve reads it as PER_FEATURE.
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 2048, 256, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerFeature,
      0, 256, 256, 1, kMeasuredMaxRows },
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::F32,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 3072, 256, 1, kMeasuredMaxRows },
    // Qwen4Exp hyper-connection and PLE norms: four streams of 2560, weight per feature.
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::BF16,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerFeature,
      0, 10240, 2560, 1, kMeasuredMaxRows },
    // Qwen4Exp indexer query norm: four 128-wide heads, weight per group.
    { ::shelf::runtime::ValueDType::BF16, ::shelf::runtime::ValueDType::BF16,
      ::shelf::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 512, 128, 1, kMeasuredMaxRows },
};

}

RmsNormImplementation
select_rmsnorm_implementation(const RmsNormSelectorInput& input)
{
    if (input.rows == 0u)
        return RmsNormImplementation::Correctness;

    for (const auto& rule : kRmsNormRules) {
        if (rule.input_dtype != input.input_dtype
            || rule.output_dtype != input.output_dtype
            || rule.weight_dtype != input.weight_dtype
            || rule.weight_layout != input.weight_layout
            || rule.weight_mode != input.weight_mode
            || rule.features != input.features
            || rule.group_size != input.group_size) {
            continue;
        }
        if (input.rows >= rule.min_rows && input.rows <= rule.max_rows) {
            return RmsNormImplementation::Optimized;
        }
        return RmsNormImplementation::Correctness;
    }

    return RmsNormImplementation::Correctness;
}

}
