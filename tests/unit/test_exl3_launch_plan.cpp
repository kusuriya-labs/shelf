// Golden test: the host launch policy must reproduce the previous decision rules exactly.

#include <shelf/kernels/optimized/linear/exl3/detail/launch_plan.h>
#include <shelf/kernels/optimized/linear/exl3.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kOutBlockBase = shelf::kernel::kExl3W16A16OutBlock;
constexpr uint32_t kRowTile = shelf::kernel::kExl3W16A16RowTile;

uint32_t ref_pick_n_tiles(const shelf::kernel::Exl3GemmArgs& a) {
    if (a.n_tiles != 0u) return a.n_tiles;
    if (a.rows < 4u) return 1u;
    if (a.out_features % (kOutBlockBase * 2u) != 0u) return 1u;
    if (a.out_features < kOutBlockBase * 8u) return 1u;
    return 2u;
}

struct RefPlan {
    uint32_t n_tiles = 1, grid_x = 0, grid_y = 0, grid_z = 1, splits = 1, groups = 1;
};

RefPlan ref_plan(const shelf::kernel::Exl3GemmArgs& a) {
    RefPlan p;
    p.n_tiles = ref_pick_n_tiles(a);
    p.grid_x = a.out_features / (kOutBlockBase * p.n_tiles);
    if (a.splitk_scratch != nullptr) {
        const uint32_t tuning_rows = a.splitk_tuning_rows != 0u ? a.splitk_tuning_rows : a.rows;
        uint32_t desired = a.splitk_splits;
        if (desired == 0u) {
            if (tuning_rows == 1u) {
                desired = p.grid_x <= 160u ? 8u : 1u;
            } else if (tuning_rows <= 8u) {
                desired = 8u;
            } else {
                desired = p.grid_x <= 16u ? 8u : (p.grid_x <= 160u ? 4u : 1u);
            }
        }
        const uint32_t k_chunks = a.k / 128u;
        if (desired > k_chunks) desired = k_chunks;
        const uint64_t per_split = static_cast<uint64_t>(tuning_rows) * a.out_features + tuning_rows;
        while (desired > 1u && static_cast<uint64_t>(desired) * per_split > a.splitk_floats) --desired;
        p.splits = desired;
    }
    const uint32_t row_blocks2 = (a.rows + 2u * kRowTile - 1u) / (2u * kRowTile);
    if (a.groups == 1u) {
        p.groups = 1u;
    } else if (a.groups == 2u || a.groups == 4u) {
        p.groups = a.groups;
    } else if (ref_pick_n_tiles(a) == 2u && a.rows >= 16u * kRowTile && p.grid_x >= 8u) {
        p.groups = 4u;
    } else if (a.rows >= 2u * kRowTile && p.grid_x * row_blocks2 * p.splits >= 128u) {
        p.groups = 2u;
    }
    const uint32_t rows_per_block = kRowTile * p.groups;
    p.grid_y = (a.rows + rows_per_block - 1u) / rows_per_block;
    return p;
}

uint32_t ref_splits(uint32_t rows, uint32_t n, uint32_t k, uint32_t scratch_rows, uint32_t bits) {
    if (rows > 16u || rows == 0u || scratch_rows < rows) return 1u;
    if (bits == 4u && n == 17408u && k == 5120u && scratch_rows >= 64u) return 4u;
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
    for (uint32_t s = 1u; s <= 16u && s * 16u <= scratch_rows && groups / s >= 8u; ++s) {
        const uint64_t waves = static_cast<uint64_t>(tiles) * s;
        const uint64_t rounds = (waves + slots - 1u) / slots;
        const double fill = static_cast<double>(waves) / static_cast<double>(rounds * slots);
        if (fill > best_fill + 0.02) { best_fill = fill; best = s; }
    }
    return best;
}

int g_failures = 0;

void check_plan(const char* name, const shelf::kernel::Exl3GemmArgs& a) {
    const RefPlan r = ref_plan(a);
    const auto p = shelf::kernel::exl3_detail::make_mul1_launch_plan(a);
    if (p.n_tiles != r.n_tiles || p.grid_x != r.grid_x || p.grid_y != r.grid_y ||
        p.grid_z != r.grid_z || p.splits != r.splits || p.groups != r.groups) {
        std::printf("FAIL %s: plan (%u,%u,%u,%u,%u,%u) vs (%u,%u,%u,%u,%u,%u)\n", name, p.n_tiles,
                    p.grid_x, p.grid_y, p.grid_z, p.splits, p.groups, r.n_tiles, r.grid_x, r.grid_y,
                    r.grid_z, r.splits, r.groups);
        ++g_failures;
    }
}

void check_splits(uint32_t rows, uint32_t n, uint32_t k, uint32_t scratch_rows, uint32_t bits) {
    const uint32_t want = ref_splits(rows, n, k, scratch_rows, bits);
    const uint32_t got = shelf::kernel::exl3_mul24_gemm_splits(rows, n, k, scratch_rows, bits);
    if (want != got) {
        std::printf("FAIL splits rows=%u n=%u k=%u scratch=%u bits=%u: %u vs %u\n", rows, n, k,
                    scratch_rows, bits, got, want);
        ++g_failures;
    }
}

}  // namespace

int main() {
    char scratch_mem[16];
    float* scratch = reinterpret_cast<float*>(scratch_mem);

    for (uint32_t rows : {1u, 2u, 8u, 16u, 32u, 128u, 256u, 2048u}) {
        for (uint32_t n : {128u, 256u, 1024u, 5120u, 17408u}) {
            for (uint32_t k : {128u, 512u, 5120u, 17408u}) {
                for (uint32_t ntiles : {0u, 1u, 2u}) {
                    for (uint32_t groups : {0u, 1u, 2u, 4u}) {
                        shelf::kernel::Exl3GemmArgs a;
                        a.rows = rows;
                        a.out_features = n;
                        a.k = k;
                        a.n_tiles = ntiles;
                        a.groups = groups;
                        check_plan("dense", a);
                        a.splitk_scratch = scratch;
                        a.splitk_floats = 0u;
                        a.splitk_splits = 0u;
                        check_plan("split0", a);
                        a.splitk_floats = 1ull << 20;
                        check_plan("split-big", a);
                        a.splitk_floats = 64ull * n;
                        check_plan("split-small", a);
                        a.splitk_splits = 4u;
                        check_plan("split-explicit", a);
                        a.splitk_tuning_rows = 1u;
                        check_plan("tuning1", a);
                        a.splitk_tuning_rows = 8u;
                        check_plan("tuning8", a);
                        a.splitk_tuning_rows = 16u;
                        check_plan("tuning16", a);
                    }
                }
            }
        }
    }

    for (uint32_t rows : {0u, 1u, 2u, 4u, 8u, 16u, 17u, 32u}) {
        for (uint32_t n : {128u, 5120u, 17408u, 248320u}) {
            for (uint32_t k : {128u, 5120u, 17408u}) {
                for (uint32_t scratch : {1u, 16u, 64u, 1024u}) {
                    for (uint32_t bits : {4u, 6u}) {
                        check_splits(rows, n, k, scratch, bits);
                    }
                }
            }
        }
    }

    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
