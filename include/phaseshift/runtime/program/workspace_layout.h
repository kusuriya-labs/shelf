#pragma once
#include <phaseshift/core/status.h>
#include <cstddef>
#include <vector>

namespace ps::runtime {

struct WorkspaceRange {
    std::size_t offset = 0;
    std::size_t bytes = 0;
    uint32_t first_command = 0;
    uint32_t last_command = 0;
};

struct WorkspaceLayout {
    std::size_t total_bytes = 0;
    std::vector<WorkspaceRange> ranges;
    Status validate() const;
};

}
