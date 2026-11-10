#include <shelf/kernels/optimized/linear/exl3_family/mul1/launch_plan.h>
#include <shelf/kernels/optimized/linear/exl3_family/types.h>

#include <cstdint>

namespace shelf::kernel::exl3_detail::mul1 {
namespace {

constexpr uint32_t kOutBlockBase = kExl3W16A16OutBlock;
constexpr uint32_t kRowTile = kExl3W16A16RowTile;

}  // namespace

uint32_t pick_n_tiles(const Exl3GemmArgs& args) {
    if (args.n_tiles != 0u) return args.n_tiles;
    if (args.rows < 4u) return 1u;
    if (args.out_features % (kOutBlockBase * 2u) != 0u) return 1u;
    if (args.out_features < kOutBlockBase * 8u) return 1u;
    return 2u;
}

LaunchPlan make_launch_plan(const Exl3GemmArgs& args) {
    LaunchPlan plan;
    plan.n_tiles = pick_n_tiles(args);
    plan.grid_x = args.out_features / (kOutBlockBase * plan.n_tiles);
    if (args.splitk_scratch != nullptr) {
        const uint32_t tuning_rows =
            args.splitk_tuning_rows != 0u ? args.splitk_tuning_rows : args.rows;
        uint32_t desired = args.splitk_splits;
        if (desired == 0u) {
            if (tuning_rows == 1u) {
                desired = plan.grid_x <= 160u ? 8u : 1u;
            } else if (tuning_rows <= 8u) {
                desired = 8u;
            } else {
                desired = plan.grid_x <= 16u ? 8u : (plan.grid_x <= 160u ? 4u : 1u);
            }
        }
        const uint32_t k_chunks = args.k / kKBlock;
        if (desired > k_chunks) desired = k_chunks;
        const uint64_t per_split =
            static_cast<uint64_t>(tuning_rows) * args.out_features + tuning_rows;
        while (desired > 1u &&
               static_cast<uint64_t>(desired) * per_split > args.splitk_floats) {
            --desired;
        }
        plan.splits = desired;
    }
    const uint32_t row_blocks2 = (args.rows + 2u * kRowTile - 1u) / (2u * kRowTile);
    if (args.groups == 1u) {
        plan.groups = 1u;
    } else if (args.groups == 2u || args.groups == 4u) {
        plan.groups = args.groups;
    } else if (plan.n_tiles == 2u && args.rows >= 16u * kRowTile && plan.grid_x >= 8u) {
        plan.groups = 4u;
    } else if (args.rows >= 2u * kRowTile &&
               plan.grid_x * row_blocks2 * plan.splits >= 128u) {
        plan.groups = 2u;
    }
    const uint32_t rows_per_block = kRowTile * plan.groups;
    plan.grid_y = (args.rows + rows_per_block - 1u) / rows_per_block;
    return plan;
}

}  // namespace shelf::kernel::exl3_detail::mul1
