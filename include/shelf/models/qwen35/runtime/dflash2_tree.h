#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace shelf::qwen35::runtime {

// Rows are in parent-before-child order; row zero is the anchor. Positions in
// ancestors are physical KV positions, while rope_positions encode depth.
struct DFlash2Tree {
    std::vector<int32_t> tokens;
    std::vector<int32_t> parent;
    std::vector<uint32_t> depth;
    std::vector<uint32_t> ancestor_offsets;
    std::vector<uint32_t> ancestor_positions;
    std::vector<uint32_t> rope_positions;
};

// alternatives[i] is the extra candidate for chain position i, or -1 if no
// distinct candidate exists. Physical space must accommodate all returned rows.
inline DFlash2Tree make_dflash2_tree(int32_t anchor, std::span<const int32_t> chain,
                                    std::span<const int32_t> alternatives,
                                    uint32_t extra, uint32_t position) {
    DFlash2Tree tree;
    tree.tokens.push_back(anchor);
    tree.parent.push_back(-1);
    tree.depth.push_back(0);
    for (uint32_t i = 0; i < chain.size(); ++i) {
        tree.tokens.push_back(chain[i]);
        tree.parent.push_back(static_cast<int32_t>(i));
        tree.depth.push_back(i + 1u);
    }
    for (uint32_t i = 0; i < chain.size() && i < alternatives.size() && extra; ++i) {
        if (alternatives[i] < 0 || alternatives[i] == chain[i]) continue;
        tree.tokens.push_back(alternatives[i]);
        tree.parent.push_back(static_cast<int32_t>(i));
        tree.depth.push_back(i + 1u);
        --extra;
    }
    tree.ancestor_offsets.push_back(0);
    for (uint32_t row = 0; row < tree.tokens.size(); ++row) {
        std::vector<uint32_t> path;
        for (int32_t node = static_cast<int32_t>(row); node >= 0; node = tree.parent[node])
            path.push_back(position + static_cast<uint32_t>(node));
        std::reverse(path.begin(), path.end());
        tree.ancestor_positions.insert(tree.ancestor_positions.end(), path.begin(), path.end());
        tree.ancestor_offsets.push_back(static_cast<uint32_t>(tree.ancestor_positions.size()));
        tree.rope_positions.push_back(position + tree.depth[row]);
    }
    return tree;
}

struct DFlash2TreeWalk {
    // Includes the anchor and every accepted child, but not correction.
    std::vector<uint32_t> rows;
    int32_t correction = -1;
};

// Returns false for malformed parent arrays or mismatched input lengths. Tree
// validation also rules out cycles, so traversal always terminates.
inline bool walk_dflash2_tree(std::span<const int32_t> tokens,
                              std::span<const int32_t> parents,
                              std::span<const int32_t> target,
                              DFlash2TreeWalk& out) {
    out = {};
    if (tokens.empty() || tokens.size() != parents.size() || tokens.size() != target.size() ||
        parents[0] != -1) return false;
    for (uint32_t i = 1; i < parents.size(); ++i) {
        if (parents[i] < 0 || static_cast<uint32_t>(parents[i]) >= i) return false;
        for (uint32_t j = 1; j < i; ++j)
            if (parents[j] == parents[i] && tokens[j] == tokens[i]) return false;
    }
    uint32_t row = 0;
    out.rows.push_back(row);
    for (;;) {
        uint32_t child = static_cast<uint32_t>(tokens.size());
        for (uint32_t i = row + 1; i < tokens.size(); ++i) {
            if (parents[i] == static_cast<int32_t>(row) && tokens[i] == target[row]) {
                child = i;
                break;
            }
        }
        if (child == tokens.size()) {
            out.correction = target[row];
            return true;
        }
        row = child;
        out.rows.push_back(row);
    }
}

}  // namespace shelf::qwen35::runtime
