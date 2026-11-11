#pragma once

#include <shelf/models/qwen35/runtime/dflash2_tree.h>

#include <cmath>
#include <queue>
#include <utility>

namespace shelf::qwen35::runtime {

inline constexpr uint32_t auto_dflash2_tree_budget(uint32_t active) noexcept {
    return active == 1u ? 10u : 8u;
}

// A probability is conditional on the candidate's parent token. The caller
// supplies the full-vocabulary normalization; top-k-only normalization can
// overestimate the probability of a branch.
struct DFlash2TreeCandidate {
    int32_t token;
    double probability;
};

// `candidates(depth, parent_token)` yields the tokens for the next depth.
// Expanding only selected nodes makes every returned path prefix-closed.
// The anchor is included in node_budget. A false result rejects malformed
// probabilities or duplicate siblings instead of constructing an invalid tree.
template <class Candidates>
bool make_budgeted_dflash2_tree(int32_t anchor, uint32_t node_budget,
                                uint32_t max_depth, uint32_t position,
                                Candidates candidates, DFlash2Tree& tree) {
    tree = {};
    if (node_budget == 0u) return false;
    struct Edge {
        double log_probability;
        uint32_t parent;
        int32_t token;
    };
    struct Compare {
        bool operator()(const Edge& a, const Edge& b) const {
            if (a.log_probability != b.log_probability)
                return a.log_probability < b.log_probability;
            if (a.parent != b.parent) return a.parent > b.parent;
            return a.token > b.token;
        }
    };
    std::priority_queue<Edge, std::vector<Edge>, Compare> pending;
    tree.tokens.push_back(anchor);
    tree.parent.push_back(-1);
    tree.depth.push_back(0u);
    const auto expand = [&](uint32_t row, double prefix_log_probability) {
        if (tree.depth[row] == max_depth) return true;
        const auto options = candidates(tree.depth[row], tree.tokens[row]);
        double probability_sum = 0.0;
        for (size_t i = 0; i < options.size(); ++i) {
            const auto& option = options[i];
            if (option.token < 0 || !std::isfinite(option.probability) ||
                option.probability < 0.0 || option.probability > 1.0) return false;
            probability_sum += option.probability;
            if (probability_sum > 1.0 + 1e-6) return false;
            for (size_t j = 0; j < i; ++j)
                if (options[j].token == option.token) return false;
            if (option.probability > 0.0)
                pending.push({prefix_log_probability + std::log(option.probability),
                              row, option.token});
        }
        return true;
    };
    if (node_budget > 1u && !expand(0u, 0.0)) return false;
    while (tree.tokens.size() < node_budget && !pending.empty()) {
        const Edge edge = pending.top();
        pending.pop();
        const uint32_t row = static_cast<uint32_t>(tree.tokens.size());
        tree.tokens.push_back(edge.token);
        tree.parent.push_back(static_cast<int32_t>(edge.parent));
        tree.depth.push_back(tree.depth[edge.parent] + 1u);
        if (tree.tokens.size() < node_budget && !expand(row, edge.log_probability)) return false;
    }
    tree.ancestor_offsets.push_back(0u);
    for (uint32_t row = 0; row < tree.tokens.size(); ++row) {
        std::vector<uint32_t> path;
        for (int32_t node = static_cast<int32_t>(row); node >= 0; node = tree.parent[node])
            path.push_back(position + static_cast<uint32_t>(node));
        std::reverse(path.begin(), path.end());
        tree.ancestor_positions.insert(tree.ancestor_positions.end(), path.begin(), path.end());
        tree.ancestor_offsets.push_back(static_cast<uint32_t>(tree.ancestor_positions.size()));
        tree.rope_positions.push_back(position + tree.depth[row]);
    }
    return true;
}

}  // namespace shelf::qwen35::runtime
