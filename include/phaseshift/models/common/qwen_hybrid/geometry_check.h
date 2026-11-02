#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/graph/shape_spec.h>
#include <phaseshift/weights/matrix_weight.h>

#include <cstdint>
#include <limits>

namespace ps {
namespace qwen_hybrid {

using ::ps::runtime::MatrixwiseShapeKey;

inline Status checked_u32(std::size_t value, const char* name, std::uint32_t& out) {
    if (value == 0) {
        return Status::invalid_argument(name, __FILE__, __LINE__);
    }
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        return Status::out_of_range(name, __FILE__, __LINE__);
    }
    out = static_cast<std::uint32_t>(value);
    return Status::make_ok();
}

inline Status checked_product(
    std::size_t left,
    std::size_t right,
    const char* name,
    std::uint32_t& out) {

    if (left == 0 || right == 0) {
        return Status::invalid_argument(name, __FILE__, __LINE__);
    }
    if (left > std::numeric_limits<std::uint32_t>::max() / right) {
        return Status::out_of_range(name, __FILE__, __LINE__);
    }
    out = static_cast<std::uint32_t>(left * right);
    return Status::make_ok();
}

inline Status checked_double(
    std::uint32_t value,
    const char* name,
    std::uint32_t& out) {

    if (value > std::numeric_limits<std::uint32_t>::max() / 2) {
        return Status::out_of_range(name, __FILE__, __LINE__);
    }
    out = value * 2;
    return Status::make_ok();
}

inline Status validate_matrix(
    const ::ps::weights::MatrixWeight& weight,
    MatrixwiseShapeKey expected,
    const char* name) {

    if (weight.rows == 0 || weight.cols == 0) {
        return Status::invalid_argument(name, __FILE__, __LINE__);
    }
    if (weight.rows != expected.output_features ||
        weight.cols != expected.input_features) {
        return Status::invalid_argument(name, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}
}
