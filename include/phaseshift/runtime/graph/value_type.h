#pragma once

#include <cstdint>

namespace ps::runtime {

enum class ValueDType : uint8_t {
    BF16 = 0,
    F32 = 1,
    I32 = 2,
};

enum class ValueRowDomain : uint8_t {
    TOKEN_ROWS = 0,
    OUTPUT_ROWS = 1,
};

constexpr uint32_t value_dtype_bytes(ValueDType dtype) noexcept {
    switch (dtype) {
        case ValueDType::BF16:
            return 2;
        case ValueDType::F32:
        case ValueDType::I32:
            return 4;
    }
    return 0;
}

struct PrimitiveValueSpec {
    uint32_t features = 0;
    ValueDType dtype = ValueDType::BF16;
    ValueRowDomain row_domain = ValueRowDomain::TOKEN_ROWS;
};

}
