#pragma once

#include <shelf/runtime/graph/value_type.h>
#include <cstdint>

namespace ps::execution {

enum class L2NormalizeImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct L2NormalizeSelectorInput {
    ::ps::runtime::ValueDType input_dtype;
    ::ps::runtime::ValueDType output_dtype;
    uint32_t features;
    uint32_t group_size;
    uint32_t rows;
};

L2NormalizeImplementation select_l2_normalize_implementation(
    const L2NormalizeSelectorInput& in);

}  // namespace ps::execution
