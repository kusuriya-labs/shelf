#pragma once

#include <shelf/models/hy4_flash/linear.h>
#include <cstddef>

namespace shelf::hy4 {
inline constexpr int expert_tile_rows = 1024;

struct alignas(16) Hy4MoePlan {
    static constexpr int max_rows = 6;
    static constexpr int choices = 16;
    static constexpr int latent_width = 1536;
    static constexpr bool supports(int rows) { return rows >= 1 && rows <= max_rows; }
    static constexpr int capacity(int rows) { return rows * choices; }
    Hy4Route routes[max_rows * choices];
    int offsets[max_rows + 1];
    int count;

    static constexpr size_t packet_bytes(int rows) {
        return sizeof(Hy4MoePlan) + size_t(rows) * latent_width * sizeof(uint16_t);
    }

    __host__ __device__ void partition(const Hy4Route *picks, int rows, const int *owners, int group) {
        count = 0;
        for (int row = 0; row < rows; ++row) {
            offsets[row] = count;
            for (int j = 0; j < choices; ++j) {
                auto route = picks[row * choices + j];
                if (owners[route.expert] != group)
                    continue;
                int index = count++;
                while (index > offsets[row] && routes[index - 1].expert > route.expert) {
                    routes[index] = routes[index - 1];
                    --index;
                }
                routes[index] = route;
            }
        }
        offsets[rows] = count;
    }
};
struct alignas(16) Hy4GroupedPlan {
    static constexpr int max_rows = 96;
    Hy4Route routes[max_rows * 16];
    int inverse[max_rows * 16];
    Hy4ExpertBatch batches[512 + max_rows];
    int offsets[max_rows + 1];
    int count, num_batches;

    __host__ __device__ void place(Hy4Route route, const Hy4Route *row_picks, const int *owners,
                                   const unsigned masks[3][512], int first) {
        const int row = route.row, expert = route.expert;
        for (int w = 0; w < row / 32; ++w)
            first += __builtin_popcount(masks[w][expert]);
        first += __builtin_popcount(masks[row / 32][expert] & ((1u << (row % 32)) - 1));
        int original = offsets[row];
        for (int j = 0; j < 16; ++j) {
            int other = row_picks[j].expert;
            original += owners[other] == owners[expert] && other < expert;
        }
        routes[first] = route;
        inverse[original] = first;
    }

    static constexpr size_t packet_bytes(int rows) {
        return sizeof(Hy4GroupedPlan) + size_t(rows) * 1536 * sizeof(uint16_t);
    }
};
}
