#include <shelf/kernels/optimized/linear/detail/launch_plan.h>
#include <shelf/kernels/optimized/linear/exl3.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel::exl3_detail {
namespace {

constexpr uint32_t kOutBlockBase = kExl3W16A16OutBlock;
constexpr uint32_t kRowTile = kExl3W16A16RowTile;
constexpr uint32_t kKBlock = 128u;

}  // namespace

uint32_t exl3_mul1_pick_n_tiles(const Exl3GemmArgs& args) {
    if (args.n_tiles != 0u) return args.n_tiles;
    if (args.rows < 4u) return 1u;
    if (args.out_features % (kOutBlockBase * 2u) != 0u) return 1u;
    if (args.out_features < kOutBlockBase * 8u) return 1u;
    return 2u;
}

Exl3LaunchPlan make_mul1_launch_plan(const Exl3GemmArgs& args) {
    Exl3LaunchPlan plan;
    plan.n_tiles = exl3_mul1_pick_n_tiles(args);
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

}  // namespace shelf::kernel::exl3_detail

namespace shelf::kernel {

uint32_t exl3_mul24_gemm_splits(uint32_t rows, uint32_t n, uint32_t k, uint32_t scratch_rows,
                                uint32_t bits) {
    if (rows > 16u || rows == 0u || scratch_rows < rows) return 1u;
    if (bits == 4u && n == 17408u && k == 5120u && scratch_rows >= 64u) return 4u;
    // Wave slots of the device (6 resident waves per SIMD at ~215 VGPRs), queried once.
    static const uint32_t slots = [] {
        int dev = 0, wgp = 0;
        if (hipGetDevice(&dev) != hipSuccess ||
            hipDeviceGetAttribute(&wgp, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
            wgp <= 0)
            return 768u;
        return static_cast<uint32_t>(wgp) * 4u * 6u;
    }();
    const uint32_t groups = k / 128u, tiles = n / 16u;
    uint32_t best = 1u;
    double best_fill = 0.0;
    // Fill = waves / (slots * rounds): the last round of a grid that is not a multiple of the slots runs partly empty.
    // Keep >= 8 groups per split so the weight prefetch pipeline still pays off.
    for (uint32_t s = 1u; s <= 16u && s * 16u <= scratch_rows && groups / s >= 8u; ++s) {
        const uint64_t waves = static_cast<uint64_t>(tiles) * s;
        const uint64_t rounds = (waves + slots - 1u) / slots;
        const double fill = static_cast<double>(waves) / static_cast<double>(rounds * slots);
        if (fill > best_fill + 0.02) { best_fill = fill; best = s; }
    }
    return best;
}

}  // namespace shelf::kernel
