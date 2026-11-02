#include <phaseshift/inference/execution/selectors/rmsnorm_selector.h>
#include <phaseshift/inference/execution/selectors/selector_limits.h>

namespace ps::execution {

namespace {

struct RmsNormRule {
    ::ps::runtime::ValueDType input_dtype;
    ::ps::runtime::ValueDType output_dtype;
    ::ps::runtime::ValueDType weight_dtype;
    RmsNormSelectorWeightLayout weight_layout;
    uint32_t weight_mode;
    uint32_t features;
    uint32_t group_size;
    uint32_t min_rows;
    uint32_t max_rows;
};

constexpr uint32_t kMeasuredMaxRows = kOptimizedMaxRows;

constexpr RmsNormRule kRmsNormRules[] = {
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::BF16,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerFeature,
      0, 2560, 2560, 1, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::BF16,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerFeature,
      0, 5120, 5120, 1, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 1024, 256, 1, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 4096, 256, 1, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::F32, ::ps::runtime::ValueDType::F32,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      1, 4096, 128, 1, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::F32, ::ps::runtime::ValueDType::F32,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      1, 6144, 128, 1, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 6144, 256, 1, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::F32, ::ps::runtime::ValueDType::F32,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      1, 3072, 128, 1, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 512, 256, 1, kMeasuredMaxRows },
    { ::ps::runtime::ValueDType::BF16, ::ps::runtime::ValueDType::F32,
      ::ps::runtime::ValueDType::BF16, RmsNormSelectorWeightLayout::PerGroup,
      0, 3072, 256, 1, kMeasuredMaxRows },
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
