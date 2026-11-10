#pragma once

// mul1 host-only launch policy. The numbers here are the existing decision rules moved
// out of the .hip files; the kernel launch itself stays in the .hip files.

#include <shelf/kernels/optimized/linear/exl3_family/types.h>

#include <cstdint>

namespace shelf::kernel::exl3_detail::mul1 {

struct LaunchPlan {
    uint32_t n_tiles = 1;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t grid_z = 1;
    uint32_t splits = 1;
    uint32_t groups = 1;
    uint32_t threads = kExl3W16A16Threads;
};

// n_tile choice for the mul1 dense path (0 override keeps the shape heuristic).
uint32_t pick_n_tiles(const Exl3GemmArgs& args);

// Split-K / group / grid decision for launch_gemm_exl3_w16a16. It preserves the
// splitk_tuning_rows semantics and the order in which splits, groups and grid_y are
// decided.
LaunchPlan make_launch_plan(const Exl3GemmArgs& args);

}  // namespace shelf::kernel::exl3_detail::mul1
