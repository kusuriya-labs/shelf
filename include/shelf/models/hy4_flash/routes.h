#pragma once

#include <shelf/models/hy4_flash/linear.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <vector>

namespace shelf::hy4 {
inline void partition_routes(std::vector<Hy4Route> &host, int rows, const std::array<int, 512> &owners,
                             int ranks, std::array<std::vector<Hy4Route>, 4> &routes,
                             std::array<std::vector<int>, 4> &offsets) {
    if (rows < 1 || rows > 1024 || ranks < 1 || ranks > 4 || host.size() != size_t(rows) * 16)
        throw std::runtime_error("routing の行数または rank 数が不正です");
    for (int row = 0; row < rows; ++row) {
        for (int r = 0; r < ranks; ++r)
            offsets[r].push_back(routes[r].size());
        auto first = host.begin() + row * 16;
        std::sort(first, first + 16, [](const auto &a, const auto &b) { return a.expert < b.expert; });
        for (auto it = first; it != first + 16; ++it) {
            if (it->row != row || it->expert < 0 || it->expert >= 512 || !std::isfinite(it->weight) ||
                (it != first && (it - 1)->expert == it->expert))
                throw std::runtime_error("routing の行、expert または重みが不正です");
            int r = owners[it->expert];
            if (r < 0 || r >= ranks)
                throw std::runtime_error("routing の配置先 rank が不正です");
            routes[r].push_back(*it);
        }
    }
}
inline void group_routes(std::vector<Hy4Route> &route, std::vector<int> &undo,
                         std::vector<Hy4ExpertBatch> &groups) {
    int count = route.size();
    std::vector<int> order(count);
    undo.resize(count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int i, int j) { return route[i].expert < route[j].expert; });
    std::vector<Hy4Route> sorted(count);
    for (int j = 0; j < count; ++j) {
        sorted[j] = route[order[j]];
        undo[order[j]] = j;
        if (!j || sorted[j - 1].expert != sorted[j].expert || groups.back().count == 16)
            groups.push_back({j, 0});
        ++groups.back().count;
    }
    route.swap(sorted);
}

inline void validate_expert_table(const std::vector<Hy4ExpertProjection> &table,
                                  const std::array<int, 512> &owners, int rank) {
    if (table.size() != 512 * 3)
        throw std::runtime_error("expert 射影表の長さが不正です");
    for (int e = 0; e < 512; ++e) {
        bool owned = owners[e] == rank;
        for (int kind = 0; kind < 3; ++kind) {
            const auto &entry = table[size_t(e) * 3 + kind];
            if (owned != (entry.suh && entry.svh && entry.trellis))
                throw std::runtime_error("expert 射影表の構築が配置表と一致しません");
        }
    }
}
}
