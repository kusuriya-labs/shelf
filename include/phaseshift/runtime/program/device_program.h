#pragma once
#include <phaseshift/runtime/program/program.h>
#include <cstdint>

namespace ps::runtime {

struct DeviceValueBinding {
    uint32_t logical_value = 0;
    ValueStorage storage = ValueStorage::WORKSPACE;
    ValueDType dtype = ValueDType::BF16;
    ValueRowDomain row_domain = ValueRowDomain::TOKEN_ROWS;
    uint32_t slot = 0;
    uint32_t feature_count = 0;
    uint32_t row_stride = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct DeviceWorkspaceRange {
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct DeviceStateBinding {
    uint32_t slot = 0;
    uint8_t kind = 0;
    uint32_t state_index = 0;
};

struct DeviceProgramView {
    const DeviceValueBinding* values = nullptr;
    uint32_t value_count = 0;
    const void* weight_table = nullptr;
    uint32_t weight_slot_count = 0;
    const void* parameter_table = nullptr;
    uint32_t parameter_slot_count = 0;
    const DeviceWorkspaceRange* workspace_ranges = nullptr;
    uint32_t workspace_range_count = 0;
    const DeviceStateBinding* states = nullptr;
    uint32_t state_count = 0;
    uint64_t workspace_base = 0;
};

enum class ProgramStatus : uint32_t {
    COMPLETE = 0,
    INVALID_PROGRAM = 1,
    INVALID_OPCODE = 2,
    INVALID_BINDING = 3,
    DISPATCH_FAILED = 4,
    UNSUPPORTED_P2P = 5,
};

}
