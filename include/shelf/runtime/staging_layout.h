#pragma once
#include <shelf/runtime/program/program.h>
#include <shelf/runtime/program/device_program.h>
#include <cstdint>
#include <vector>

namespace ps {
namespace runtime {

struct StagingProgramRefs {
    const void* ranges = nullptr;
    uint32_t range_count = 0;
    const void* states = nullptr;
    uint32_t state_count = 0;
};

struct StagingSectionOffsets {
    uint64_t values = 0;
    uint64_t refs = 0;
    uint64_t total = 0;
};

inline StagingSectionOffsets staging_section_offsets(uint64_t nval) {
    StagingSectionOffsets o{};
    uint64_t cur = sizeof(DispatchBinding);
    cur = (cur + 15ull) & ~15ull;
    o.values = cur;
    cur += nval * sizeof(DeviceValueBinding);
    cur = (cur + 15ull) & ~15ull;
    o.refs = cur;
    cur += sizeof(StagingProgramRefs);
    cur = (cur + 15ull) & ~15ull;
    o.total = cur;
    return o;
}

struct DispatchStagingPool {
    uint32_t max_bindings = 0;
    uint64_t pool_bytes = 0;
    void* device = nullptr;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> bytes;
    std::vector<uint8_t> uploaded;
    std::vector<uint8_t> run_seen;
    std::vector<uint8_t> bucket_ran;
    uint64_t slot(uint32_t bucket_index, uint32_t binding_index) const {
        return static_cast<uint64_t>(bucket_index) * max_bindings + binding_index;
    }
};

struct ProgramStagingMeta {
    void* ranges = nullptr;
    uint32_t range_count = 0;
    void* states = nullptr;
    uint32_t state_count = 0;
    uint64_t device_bytes = 0;
};

inline uint64_t staging_bytes_for(const DispatchBinding& binding) {
    const uint64_t nval =
        static_cast<uint64_t>(binding.input_count + binding.output_count);
    return staging_section_offsets(nval).total;
}

inline uint64_t program_staging_meta_bytes(uint32_t range_count, uint32_t state_count) {
    uint64_t cur = 0;
    if (range_count > 0) {
        cur += static_cast<uint64_t>(range_count) * sizeof(DeviceWorkspaceRange);
        cur = (cur + 15ull) & ~15ull;
    }
    cur += static_cast<uint64_t>(state_count) * sizeof(DeviceStateBinding);
    cur = (cur + 15ull) & ~15ull;
    return cur;
}

inline uint64_t max_host_staging_bytes(const Program& program) {
    uint64_t max_bytes = 0;
    for (const auto& b : program.dispatches) {
        const uint64_t bytes = staging_bytes_for(b);
        if (bytes > max_bytes) max_bytes = bytes;
    }
    return max_bytes;
}

}
}
