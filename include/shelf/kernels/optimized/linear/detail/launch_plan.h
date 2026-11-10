#pragma once

// Host-only launch policy. The numbers here are the existing decision rules moved out
// of the .hip files; the kernel launch itself stays in the .hip files.

#include <shelf/kernels/optimized/linear/exl3.h>

#include <cstdint>

namespace shelf::kernel::exl3_detail {

struct Exl3LaunchPlan {
    uint32_t n_tiles = 1;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t grid_z = 1;
    uint32_t splits = 1;
    uint32_t groups = 1;
    uint32_t threads = kExl3W16A16Threads;
};

// n_tile choice for the mul1 dense path (0 override keeps the shape heuristic).
uint32_t exl3_mul1_pick_n_tiles(const Exl3GemmArgs& args);

// Split-K / group / grid decision for launch_gemm_exl3_w16a16. It preserves the
// splitk_tuning_rows semantics and the order in which splits, groups and grid_y are
// decided.
Exl3LaunchPlan make_mul1_launch_plan(const Exl3GemmArgs& args);

}  // namespace shelf::kernel::exl3_detail
