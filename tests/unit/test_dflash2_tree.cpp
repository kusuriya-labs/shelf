#include <shelf/models/qwen35/runtime/dflash2_tree.h>
#include <shelf/models/qwen35/runtime/dflash2_tree_budget.h>

#include <cstdio>
#include <vector>

using namespace shelf::qwen35::runtime;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    CHECK(auto_dflash2_tree_budget(0) == 8);
    CHECK(auto_dflash2_tree_budget(1) == 12);
    CHECK(auto_dflash2_tree_budget(2) == 8);
    CHECK(auto_dflash2_tree_budget(3) == 8);
    CHECK(auto_dflash2_tree_budget(4) == 8);
    CHECK(auto_dflash2_tree_budget(5) == 8);
    const std::vector<int32_t> chain{11, 12, 13};
    const std::vector<int32_t> alt{21, 22, 23};
    const auto tree = make_dflash2_tree(10, chain, alt, 2, 100);
    CHECK((tree.tokens == std::vector<int32_t>{10, 11, 12, 13, 21, 22}));
    CHECK((tree.parent == std::vector<int32_t>{-1, 0, 1, 2, 0, 1}));
    CHECK((tree.depth == std::vector<uint32_t>{0, 1, 2, 3, 1, 2}));
    CHECK((tree.rope_positions == std::vector<uint32_t>{100, 101, 102, 103, 101, 102}));
    CHECK((tree.ancestor_offsets == std::vector<uint32_t>{0, 1, 3, 6, 10, 12, 15}));
    CHECK((tree.ancestor_positions == std::vector<uint32_t>{100, 100,101, 100,101,102, 100,101,102,103, 100,104, 100,101,105}));
    DFlash2TreeWalk walk;
    CHECK(walk_dflash2_tree(tree.tokens, tree.parent, std::vector<int32_t>{11,12,13,99,88,77}, walk));
    CHECK((walk.rows == std::vector<uint32_t>{0,1,2,3}) && walk.correction == 99);
    CHECK(walk_dflash2_tree(tree.tokens, tree.parent, std::vector<int32_t>{21,12,13,99,88,77}, walk));
    CHECK((walk.rows == std::vector<uint32_t>{0,4}) && walk.correction == 88);
    CHECK(walk_dflash2_tree(tree.tokens, tree.parent, std::vector<int32_t>{11,22,13,99,88,77}, walk));
    CHECK((walk.rows == std::vector<uint32_t>{0,1,5}) && walk.correction == 77);
    CHECK(walk_dflash2_tree(tree.tokens, tree.parent, std::vector<int32_t>{33,12,13,99,88,77}, walk));
    CHECK((walk.rows == std::vector<uint32_t>{0}) && walk.correction == 33);
    CHECK(!walk_dflash2_tree(std::vector<int32_t>{10,11,11}, std::vector<int32_t>{-1,0,0}, std::vector<int32_t>{11,12,13}, walk));
    CHECK(!walk_dflash2_tree(tree.tokens, std::vector<int32_t>{-1,0,1,2,4,1}, std::vector<int32_t>{11,12,13,99,88,77}, walk));
    CHECK(!walk_dflash2_tree(tree.tokens, std::vector<int32_t>{-1,0,1,2,0,1}, std::vector<int32_t>{11}, walk));
    const auto four = make_dflash2_tree(10, std::vector<int32_t>{11,12,13,14},
                                        std::vector<int32_t>{21,22,23,24}, 4, 0);
    CHECK(four.tokens.size() == 9 && four.parent.back() == 3 && four.depth.back() == 4);
    const auto no_extra = make_dflash2_tree(10, chain, alt, 0, 0);
    CHECK(no_extra.tokens.size() == 4);
    const auto skip = make_dflash2_tree(10, chain, std::vector<int32_t>{11,-1,23}, 2, 0);
    CHECK((skip.tokens == std::vector<int32_t>{10,11,12,13,23}));
    // Conditional probabilities produce both a deep continuation and a
    // competing sibling, in descending path-probability order.
    auto candidates = [](uint32_t depth, int32_t parent) {
        if (depth == 0) return std::vector<DFlash2TreeCandidate>{{11, 0.6}, {21, 0.4}};
        if (depth == 1 && parent == 11) return std::vector<DFlash2TreeCandidate>{{12, 0.9}, {22, 0.1}};
        if (depth == 1 && parent == 21) return std::vector<DFlash2TreeCandidate>{{32, 0.8}};
        return std::vector<DFlash2TreeCandidate>{};
    };
    DFlash2Tree budgeted;
    CHECK(make_budgeted_dflash2_tree(10, 5, 2, 100, candidates, budgeted));
    CHECK((budgeted.tokens == std::vector<int32_t>{10, 11, 12, 21, 32}));
    CHECK((budgeted.parent == std::vector<int32_t>{-1, 0, 1, 0, 3}));
    CHECK((budgeted.rope_positions == std::vector<uint32_t>{100, 101, 102, 101, 102}));
    CHECK((budgeted.ancestor_positions == std::vector<uint32_t>{100, 100,101, 100,101,102, 100,103, 100,103,104}));
    CHECK(walk_dflash2_tree(budgeted.tokens, budgeted.parent,
                             std::vector<int32_t>{21, 99, 99, 32, 77}, walk));
    CHECK((walk.rows == std::vector<uint32_t>{0, 3, 4}) && walk.correction == 77);
    CHECK(make_budgeted_dflash2_tree(10, 1, 2, 0, candidates, budgeted));
    CHECK(budgeted.tokens.size() == 1);
    CHECK(!make_budgeted_dflash2_tree(10, 0, 2, 0, candidates, budgeted));
    auto duplicates = [](uint32_t, int32_t) { return std::vector<DFlash2TreeCandidate>{{1, 0.5}, {1, 0.5}}; };
    CHECK(!make_budgeted_dflash2_tree(10, 2, 2, 0, duplicates, budgeted));
    CHECK(make_budgeted_dflash2_tree(10, 1, 2, 0, duplicates, budgeted));
    auto binary = [](uint32_t, int32_t parent) {
        return std::vector<DFlash2TreeCandidate>{{parent * 2, 0.7}, {parent * 2 + 1, 0.3}};
    };
    for (const uint32_t budget : {8u, 10u, 12u, 16u, 24u}) {
        CHECK(make_budgeted_dflash2_tree(1, budget, 5, 0, binary, budgeted));
        CHECK(budgeted.tokens.size() == budget);
        std::vector<double> probability(budget, 1.0);
        for (uint32_t row = 1; row < budget; ++row) {
            const uint32_t parent = static_cast<uint32_t>(budgeted.parent[row]);
            CHECK(parent < row);
            probability[row] = probability[parent] * ((budgeted.tokens[row] % 2 == 0) ? 0.7 : 0.3);
            if (row > 1) CHECK(probability[row] <= probability[row - 1] + 1e-12);
        }
    }
    return 0;
}
