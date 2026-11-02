#pragma once

#include <phaseshift/runtime/graph/value_type.h>
#include <cstdint>

namespace ps::execution {

enum class RopeImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct RopeSelectorInput {
    ::ps::runtime::ValueDType input_dtype;
    ::ps::runtime::ValueDType output_dtype;
    uint32_t features;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t rows;
};

RopeImplementation select_rope_implementation(const RopeSelectorInput& in);

}  // namespace ps::execution
