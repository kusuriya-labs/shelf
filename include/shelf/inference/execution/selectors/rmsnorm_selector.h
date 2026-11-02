#pragma once

#include <shelf/runtime/graph/value_type.h>
#include <cstdint>

namespace ps::execution {

enum class RmsNormImplementation : uint8_t {
    Correctness,
    Optimized,
};

enum class RmsNormSelectorWeightLayout : uint8_t {
    None,
    PerFeature,
    PerGroup,
};

struct RmsNormSelectorInput {
    ::ps::runtime::ValueDType input_dtype;
    ::ps::runtime::ValueDType output_dtype;
    ::ps::runtime::ValueDType weight_dtype;
    RmsNormSelectorWeightLayout weight_layout;
    uint32_t weight_mode;
    uint32_t features;
    uint32_t group_size;
    uint32_t rows;
};

RmsNormImplementation select_rmsnorm_implementation(const RmsNormSelectorInput& input);

}
