#include <phaseshift/models/qwen35/runtime/eagle4_tree.h>
#include <phaseshift/models/qwen35/runtime/spec_decode.h>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace rt = ps::qwen35::runtime;

static int g_fail = 0;

static void fail(const std::string& msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static bool build_chain(rt::Eagle4DraftTree& tree, const std::vector<int32_t>& tokens) {
    rt::eagle4_tree_reset(tree, static_cast<int32_t>(tokens.size()),
                          static_cast<int32_t>(tokens.size()) - 1);
    if (!rt::eagle4_tree_insert_root(tree, tokens[0], 1.0f).ok()) return false;
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        const float prob = 0.5f;
        const uint32_t inserted = rt::eagle4_tree_insert_children(
            tree, static_cast<int32_t>(i) - 1, &tokens[i], &prob, 1u);
        if (inserted != 1u) return false;
        if (rt::eagle4_tree_select_frontier(tree, 1u) != 1u) return false;
    }
    return true;
}

static void test_chain_equivalence() {
    std::mt19937 rng(1234u);
    for (int trial = 0; trial < 300; ++trial) {
        const int32_t n = 1 + static_cast<int32_t>(rng() % 12u);
        std::vector<int32_t> tokens(static_cast<std::size_t>(n));
        std::vector<int32_t> target(static_cast<std::size_t>(n) + 1u);
        for (int32_t i = 0; i < n; ++i) tokens[static_cast<std::size_t>(i)] =
            static_cast<int32_t>(rng() % 50u);
        for (std::size_t i = 0; i < target.size(); ++i)
            target[i] = static_cast<int32_t>(rng() % 50u);

        rt::SpecVerifyResult chain;
        const ps::Status cs = rt::spec_greedy_accept(tokens.data(), static_cast<uint32_t>(n),
                                                     target.data(), true, chain);
        check(cs.ok(), "chain accept status");

        rt::Eagle4DraftTree tree;
        const bool built = build_chain(tree, tokens);
        check(built, "chain tree build");
        if (!built) continue;
        rt::Eagle4TreeAcceptResult tre;
        const ps::Status ts = rt::spec_tree_greedy_accept(tree, target.data(), tre);
        check(ts.ok(), "tree accept status");
        if (!ts.ok()) continue;

        const bool all_accept = chain.num_accepted_drafts == static_cast<uint32_t>(n);
        check(tre.num_accepted == chain.num_accepted_drafts,
              "chain/tree accepted count mismatch");
        check(tre.correction_valid == chain.correction_valid,
              "chain/tree correction flag mismatch");
        if (chain.correction_valid) {
            check(tre.correction_token == chain.correction_token,
                  "chain/tree correction token mismatch");
            check(tre.correction_row == chain.first_reject_index,
                  "chain/tree correction row mismatch");
        }
        check(tre.leaf_accept == all_accept, "tree leaf_accept vs chain bonus case");
        if (all_accept) {
            check(!tre.correction_valid, "tree leaf accept has no correction");
            check(tre.num_committed_target_tokens == chain.num_accepted_drafts,
                  "tree leaf committed count");
            check(tre.emitted_tokens == std::vector<int32_t>(tokens.begin(), tokens.end()),
                  "tree leaf emitted tokens");
        } else {
            check(tre.num_committed_target_tokens == chain.num_committed_target_tokens,
                  "chain/tree committed count mismatch");
            check(tre.emitted_tokens == chain.emitted_tokens,
                  "chain/tree emitted tokens mismatch");
        }
    }
}

static void test_branching_accept() {
    rt::Eagle4DraftTree tree;
    rt::eagle4_tree_reset(tree, 16, 3);
    check(rt::eagle4_tree_insert_root(tree, 10, 1.0f).ok(), "root insert");
    const int32_t kids1[2] = {20, 21};
    const float p1[2] = {0.9f, 0.1f};
    check(rt::eagle4_tree_insert_children(tree, 0, kids1, p1, 2u) == 2u, "root children");
    check(rt::eagle4_tree_select_frontier(tree, 4u) == 2u, "frontier select");
    const int32_t kids2[2] = {30, 31};
    const float p2[2] = {0.8f, 0.2f};
    check(rt::eagle4_tree_insert_children(tree, 1, kids2, p2, 2u) == 2u,
          "depth2 children of node1");
    const int32_t kids3[1] = {40};
    const float p3[1] = {1.0f};
    check(rt::eagle4_tree_insert_children(tree, 2, kids3, p3, 1u) == 1u,
          "depth2 children of node2");
    check(tree.num_nodes == 6, "node count after inserts");

    {
        const int32_t target[6] = {77, 999, 999, 0, 0, 0};
        rt::Eagle4TreeAcceptResult res;
        check(rt::spec_tree_greedy_accept(tree, target, res).ok(), "root reject status");
        check(res.num_accepted == 0u, "root reject accepted");
        check(res.correction_valid, "root reject correction");
        check(res.correction_token == 77, "root reject correction token");
        check(res.correction_row == 0, "root reject correction row");
        check(res.num_committed_target_tokens == 1u, "root reject committed");
        check(res.emitted_tokens == std::vector<int32_t>{77}, "root reject emitted");
    }
    {
        const int32_t target[6] = {10, 999, 999, 0, 0, 0};
        rt::Eagle4TreeAcceptResult res;
        check(rt::spec_tree_greedy_accept(tree, target, res).ok(), "child reject status");
        check(res.num_accepted == 1u, "child reject accepted");
        check(res.correction_valid, "child reject correction");
        check(res.correction_row == 1, "child reject uses first child row");
        check(res.correction_token == 999, "child reject token");
        check(res.num_committed_target_tokens == 2u, "child reject committed");
        check(res.emitted_tokens == std::vector<int32_t>({10, 999}),
              "child reject emitted");
    }
    {
        const int32_t target[6] = {10, 20, 999, 999, 999, 0};
        rt::Eagle4TreeAcceptResult res;
        check(rt::spec_tree_greedy_accept(tree, target, res).ok(), "grandchild reject status");
        check(res.num_accepted == 2u, "grandchild reject accepted");
        check(res.path_count == 2u, "grandchild path count");
        check(res.path[0] == 0 && res.path[1] == 1, "grandchild path nodes");
        check(res.correction_valid, "grandchild reject correction");
        check(res.correction_row == 3, "grandchild reject row");
        check(res.correction_token == 999, "grandchild reject token");
        check(res.emitted_tokens == std::vector<int32_t>({10, 20, 999}),
              "grandchild reject emitted");
        check(res.num_committed_target_tokens == 3u, "grandchild reject committed");
    }
    {
        const int32_t target[6] = {10, 20, 0, 0, 31, 0};
        rt::Eagle4TreeAcceptResult res;
        check(rt::spec_tree_greedy_accept(tree, target, res).ok(), "deep reject status");
        check(res.num_accepted == 3u, "deep reject accepted");
        check(res.path[2] == 4, "deep reject skips non-matching sibling");
        check(res.leaf_accept, "deep reject reaches leaf");
        check(!res.correction_valid, "deep reject no correction");
        check(res.num_committed_target_tokens == 3u, "deep reject committed");
        check(res.emitted_tokens == std::vector<int32_t>({10, 20, 31}),
              "deep reject emitted");
    }
    {
        const int32_t target[6] = {10, 20, 0, 30, 0, 0};
        rt::Eagle4TreeAcceptResult res;
        check(rt::spec_tree_greedy_accept(tree, target, res).ok(), "branch match status");
        check(res.num_accepted == 3u, "branch match accepted");
        check(res.path[2] == 3, "branch match selects matching child");
        check(res.leaf_accept, "branch match reaches leaf");
        check(res.emitted_tokens == std::vector<int32_t>({10, 20, 30}),
              "branch match emitted");
    }
    {
        const int32_t target[6] = {10, 20, 0, 555, 0, 0};
        rt::Eagle4TreeAcceptResult res;
        check(rt::spec_tree_greedy_accept(tree, target, res).ok(), "deep correction status");
        check(res.num_accepted == 2u, "deep correction accepted");
        check(res.correction_valid, "deep correction valid");
        check(res.correction_row == 3, "deep correction row");
        check(res.correction_token == 555, "deep correction token");
        check(res.emitted_tokens == std::vector<int32_t>({10, 20, 555}),
              "deep correction emitted");
    }
}

static void test_frontier_select() {
    rt::Eagle4DraftTree tree;
    rt::eagle4_tree_reset(tree, 8, 4);
    check(rt::eagle4_tree_insert_root(tree, 7, 1.0f).ok(), "select root");
    check(tree.frontier_count == 1 && tree.frontier[0] == 0, "root frontier");
    const int32_t tokens[3] = {11, 12, 13};
    const float probs[3] = {0.1f, 0.5f, 0.5f};
    check(rt::eagle4_tree_insert_children(tree, 0, tokens, probs, 3u) == 3u, "select kids");
    check(tree.pending_begin == 1 && tree.pending_end == 4, "pending range");
    check(rt::eagle4_tree_select_frontier(tree, 2u) == 2u, "select width2");
    check(tree.frontier[0] == 2 && tree.frontier[1] == 3, "select score order with tie");
    check(tree.pending_begin == tree.pending_end, "pending consumed");
    check(rt::eagle4_tree_select_frontier(tree, 4u) == 0u, "empty pending selects nothing");
    check(tree.frontier_count == 0, "frontier cleared");
    check(!rt::eagle4_tree_step_available(tree), "no frontier no step");
}

static void test_limits() {
    rt::Eagle4DraftTree tree;
    rt::eagle4_tree_reset(tree, 4, 1);
    check(rt::eagle4_tree_insert_root(tree, 1, 1.0f).ok(), "limit root");
    const int32_t tokens[6] = {2, 3, 4, 5, 6, 7};
    float probs[6] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f};
    check(rt::eagle4_tree_insert_children(tree, 0, tokens, probs, 6u) == 3u,
          "capacity truncates children");
    check(tree.num_nodes == 4, "capacity respected");
    check(rt::eagle4_tree_select_frontier(tree, 8u) == 3u, "limit frontier");
    check(!rt::eagle4_tree_step_available(tree), "depth limit stops steps");
    check(rt::eagle4_tree_insert_children(tree, 0, tokens, probs, 6u) == 0u,
          "depth blocked insert");
    check(rt::eagle4_tree_insert_children(tree, 99, tokens, probs, 1u) == 0u,
          "invalid parent insert");
    rt::Eagle4DraftTree full;
    rt::eagle4_tree_reset(full, 1, 3);
    check(rt::eagle4_tree_insert_root(full, 5, 1.0f).ok(), "full root");
    check(rt::eagle4_tree_insert_children(full, 0, tokens, probs, 1u) == 0u,
          "capacity one blocks children");
    check(!rt::eagle4_tree_insert_root(full, 6, 1.0f).ok(), "double root rejected");
}

static void test_ancestors() {
    rt::Eagle4DraftTree tree;
    rt::eagle4_tree_reset(tree, 16, 3);
    check(rt::eagle4_tree_insert_root(tree, 10, 1.0f).ok(), "anc root");
    const int32_t kids1[2] = {20, 21};
    const float p1[2] = {0.6f, 0.4f};
    check(rt::eagle4_tree_insert_children(tree, 0, kids1, p1, 2u) == 2u, "anc l1");
    check(rt::eagle4_tree_select_frontier(tree, 2u) == 2u, "anc frontier1");
    const int32_t kids2[3] = {30, 31, 32};
    const float p2[3] = {0.5f, 0.3f, 0.2f};
    check(rt::eagle4_tree_insert_children(tree, 1, kids2, p2, 3u) == 3u, "anc l2");
    check(rt::eagle4_tree_select_frontier(tree, 8u) == 3u, "anc frontier2");

    const int32_t stride = tree.max_depth + 1;
    std::vector<int32_t> rows(static_cast<std::size_t>(tree.frontier_count) *
                              static_cast<std::size_t>(stride), -2);
    check(rt::eagle4_tree_ancestors_flat(tree, stride, rows.data()) == 3u, "anc count");
    check(rows[0] == 0 && rows[1] == 1 && rows[2] == 3, "anc row0 path");
    check(rows[static_cast<std::size_t>(stride)] == 0 &&
              rows[static_cast<std::size_t>(stride) + 1] == 1 &&
              rows[static_cast<std::size_t>(stride) + 2] == 4,
          "anc row1 path");
    check(rows[static_cast<std::size_t>(2 * stride)] == 0 &&
              rows[static_cast<std::size_t>(2 * stride) + 1] == 1 &&
              rows[static_cast<std::size_t>(2 * stride) + 2] == 5,
          "anc row2 path");
    check(rt::eagle4_tree_ancestors_flat(tree, 1, rows.data()) == 0u, "anc stride guard");
}

static void check_invariants(const rt::Eagle4DraftTree& tree) {
    check(tree.num_nodes >= 1, "tree has root");
    check(tree.capacity >= tree.num_nodes, "capacity bound");
    for (int32_t i = 0; i < tree.num_nodes; ++i) {
        check(tree.verify_order[i] == i, "verify order is insertion order");
        check(tree.token[i] >= 0, "token set");
        if (i == 0) {
            check(tree.parent[i] == -1 && tree.depth[i] == 0, "root shape");
            continue;
        }
        check(tree.parent[i] >= 0 && tree.parent[i] < i, "parent index precedes child");
        check(tree.depth[i] == tree.depth[tree.parent[i]] + 1, "depth increments");
        check(tree.depth[i] <= tree.max_depth, "depth within bound");
        check(tree.score[i] <= tree.score[tree.parent[i]] && tree.score[i] > 0.0f,
              "score is parent score times probability");
    }
    for (int32_t i = 0; i < tree.frontier_count; ++i) {
        const int32_t node = tree.frontier[i];
        check(node >= 0 && node < tree.num_nodes, "frontier node in range");
        if (i > 0) {
            check(tree.score[tree.frontier[i - 1]] >= tree.score[node],
                  "frontier sorted by score");
        }
    }
    if (!rt::eagle4_tree_step_available(tree)) {
        const bool depth_blocked =
            tree.frontier_count > 0 && tree.num_nodes >= tree.capacity;
        const bool frontier_deep = tree.frontier_count > 0;
        bool all_deep = frontier_deep;
        for (int32_t i = 0; i < tree.frontier_count; ++i) {
            if (tree.depth[tree.frontier[i]] < tree.max_depth) all_deep = false;
        }
        check(tree.frontier_count == 0 || depth_blocked || all_deep, "step stop reason");
    } else {
        check(tree.frontier_count > 0 && tree.num_nodes < tree.capacity,
              "step available preconditions");
    }
}

static void check_accept_invariants(const rt::Eagle4DraftTree& tree,
                                    const int32_t* target,
                                    const rt::Eagle4TreeAcceptResult& res) {
    if (res.num_accepted == 0u) {
        check(res.path_count == 0u, "reject path empty");
        check(res.correction_valid && res.correction_row == 0, "reject uses root row");
        check(res.correction_token == target[0], "reject token from root row");
        check(res.num_committed_target_tokens == 1u, "reject commits one");
        check(res.emitted_tokens.size() == 1u, "reject emits one");
        return;
    }
    check(res.path_count == res.num_accepted, "path equals accepted");
    check(res.path[0] == 0, "path starts at root");
    for (uint32_t i = 0; i < res.path_count; ++i) {
        const int32_t node = res.path[i];
        check(res.emitted_tokens[i] == tree.token[node], "emitted follows path tokens");
        if (i > 0) check(tree.parent[node] == res.path[i - 1u], "path is contiguous");
    }
    const bool terminated = res.correction_valid || res.leaf_accept;
    check(terminated, "walk terminates with correction or leaf");
    check(!(res.correction_valid && res.leaf_accept), "termination is exclusive");
    if (res.correction_valid) {
        check(res.correction_row >= 0 && res.correction_row < tree.num_nodes,
              "correction row in range");
        check(tree.parent[res.correction_row] == res.path[res.path_count - 1u],
              "correction row is child of path tail");
        check(res.correction_token == target[res.correction_row], "correction from its row");
        check(res.num_committed_target_tokens == res.num_accepted + 1u,
              "correction commits accepted plus one");
        check(res.emitted_tokens.size() == res.path_count + 1u, "emitted includes correction");
    } else {
        const int32_t tail = res.path[res.path_count - 1u];
        bool has_child = false;
        for (int32_t c = 0; c < tree.num_nodes; ++c) {
            if (tree.parent[c] == tail) has_child = true;
        }
        check(!has_child, "leaf termination has no children");
        check(res.num_committed_target_tokens == res.num_accepted, "leaf commits path");
        check(res.emitted_tokens.size() == res.path_count, "leaf emits path only");
    }
}

static void test_random_trees() {
    std::mt19937 rng(987654u);
    for (int trial = 0; trial < 400; ++trial) {
        rt::Eagle4DraftTree tree;
        const int32_t capacity = 1 + static_cast<int32_t>(rng() % 40u);
        const int32_t max_depth = 1 + static_cast<int32_t>(rng() % 5u);
        rt::eagle4_tree_reset(tree, capacity, max_depth);
        const bool root_ok = rt::eagle4_tree_insert_root(tree, static_cast<int32_t>(rng() % 100u), 1.0f).ok();
        check(root_ok, "random root");
        const uint32_t width = 1u + rng() % 4u;
        int steps = 0;
        while (rt::eagle4_tree_step_available(tree) && steps < 12) {
            ++steps;
            for (int32_t f = 0; f < tree.frontier_count; ++f) {
                const int32_t parent = tree.frontier[f];
                uint32_t k = 1u + rng() % 4u;
                std::vector<int32_t> tokens(k);
                std::vector<float> probs(k);
                for (uint32_t i = 0; i < k; ++i) {
                    tokens[i] = static_cast<int32_t>(rng() % 200u);
                    probs[i] = 0.05f + 0.95f * (static_cast<float>(rng() % 1000u) / 1000.0f);
                }
                const uint32_t inserted = rt::eagle4_tree_insert_children(
                    tree, parent, tokens.data(), probs.data(), k);
                check(inserted <= k, "insert bound");
                check(tree.num_nodes <= tree.capacity, "capacity never exceeded");
            }
            (void)rt::eagle4_tree_select_frontier(tree, width);
        }
        check_invariants(tree);

        std::vector<int32_t> target(static_cast<std::size_t>(tree.num_nodes));
        for (int32_t i = 0; i < tree.num_nodes; ++i)
            target[static_cast<std::size_t>(i)] = static_cast<int32_t>(rng() % 200u);
        rt::Eagle4TreeAcceptResult res;
        check(rt::spec_tree_greedy_accept(tree, target.data(), res).ok(),
              "random accept status");
        check_accept_invariants(tree, target.data(), res);
    }
}

int main() {
    test_chain_equivalence();
    test_branching_accept();
    test_frontier_select();
    test_limits();
    test_ancestors();
    test_random_trees();
    if (g_fail == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL (%d)\n", g_fail);
    return 1;
}
