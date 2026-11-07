#pragma once

#include <cstdint>

namespace shelf::runtime {

enum class ValueDType : uint8_t {
    BF16 = 0,
    F32 = 1,
    I32 = 2,
    U8 = 3,
};

enum class ValueRowDomain : uint8_t {
    TOKEN_ROWS = 0,
    OUTPUT_ROWS = 1,
    // Rows of a grouped GEMM plan. See runtime/graph/grouped_row_shape.h.
    GROUPED_ROWS = 2,
};

constexpr uint32_t value_dtype_bytes(ValueDType dtype) noexcept {
    switch (dtype) {
        case ValueDType::BF16:
            return 2;
        case ValueDType::F32:
        case ValueDType::I32:
            return 4;
        case ValueDType::U8:
            return 1;
    }
    return 0;
}

struct PrimitiveValueSpec {
    uint32_t features = 0;
    ValueDType dtype = ValueDType::BF16;
    ValueRowDomain row_domain = ValueRowDomain::TOKEN_ROWS;
};

inline const char* to_string(ValueRowDomain domain) noexcept {
    switch (domain) {
        case ValueRowDomain::TOKEN_ROWS: return "TOKEN_ROWS";
        case ValueRowDomain::OUTPUT_ROWS: return "OUTPUT_ROWS";
        case ValueRowDomain::GROUPED_ROWS: return "GROUPED_ROWS";
    }
    return "UNKNOWN";
}

}
