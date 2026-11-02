#pragma once

#include <shelf/runtime/graph/value_type.h>
#include <cstdint>

namespace shelf::execution {

enum class RopeImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct RopeSelectorInput {
    ::shelf::runtime::ValueDType input_dtype;
    ::shelf::runtime::ValueDType output_dtype;
    uint32_t features;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t rows;
};

RopeImplementation select_rope_implementation(const RopeSelectorInput& in);

}  // namespace shelf::execution
