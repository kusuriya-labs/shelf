#pragma once
#include <cstdint>

namespace shelf::runtime {

enum class ExecutionDomain : uint8_t {
    TARGET,
    DRAFTER,
};

enum class ExecutionClass : uint8_t {
    DECODE,
    SPEC_VERIFY,
    PREFILL,
    SPEC_DRAFT,
};

enum class ExecutionRole : uint8_t {
    Decode = 0,
    Prefill = 1,
    Verify = 2,
};

constexpr ExecutionDomain execution_domain(
    ExecutionClass execution_class) noexcept {
    switch (execution_class) {
        case ExecutionClass::DECODE:
        case ExecutionClass::SPEC_VERIFY:
        case ExecutionClass::PREFILL:
            return ExecutionDomain::TARGET;
        case ExecutionClass::SPEC_DRAFT:
            return ExecutionDomain::DRAFTER;
    }
    return ExecutionDomain::TARGET;
}

constexpr bool is_target_execution_class(
    ExecutionClass execution_class) noexcept {
    return execution_domain(execution_class)
        == ExecutionDomain::TARGET;
}

}
