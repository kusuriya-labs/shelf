#pragma once

#include <shelf/models/hy4_flash/linear.h>
#include <cstddef>

namespace shelf::hy4 {
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
}
