#pragma once

#include <shelf/runtime/graph/value_type.h>
#include <cstdint>

namespace shelf::execution {

enum class L2NormalizeImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct L2NormalizeSelectorInput {
    ::shelf::runtime::ValueDType input_dtype;
    ::shelf::runtime::ValueDType output_dtype;
    uint32_t features;
    uint32_t group_size;
    uint32_t rows;
};

L2NormalizeImplementation select_l2_normalize_implementation(
    const L2NormalizeSelectorInput& in);

}  // namespace shelf::execution
