#include <phaseshift/models/qwen35/runtime/eagle4_tree.h>

#include <algorithm>
#include <array>

namespace ps::qwen35::runtime {

void eagle4_tree_reset(Eagle4DraftTree& tree, int32_t capacity, int32_t max_depth) {
    tree.num_nodes = 0;
    tree.capacity = capacity < 1 ? 1 : (capacity > kEagle4TreeMaxNodes ? kEagle4TreeMaxNodes
                                                                      : capacity);
    tree.max_depth = max_depth < 0 ? 0 : max_depth;
    tree.frontier_count = 0;
    tree.pending_begin = 0;
    tree.pending_end = 0;
    for (int32_t i = 0; i < kEagle4TreeMaxNodes; ++i) {
        tree.parent[i] = -1;
        tree.depth[i] = 0;
        tree.token[i] = -1;
        tree.score[i] = 0.0f;
        tree.verify_order[i] = i;
        tree.frontier[i] = -1;
    }
}

Status eagle4_tree_insert_root(Eagle4DraftTree& tree, int32_t token, float prob) {
    if (tree.num_nodes != 0) {
        return Status::invalid_argument("eagle4_tree_insert_root: tree not empty", __FILE__,
                                        __LINE__);
    }
    if (tree.capacity < 1) {
        return Status::invalid_state("eagle4_tree_insert_root: capacity is zero", __FILE__,
                                     __LINE__);
    }
    tree.parent[0] = -1;
    tree.depth[0] = 0;
    tree.token[0] = token;
    tree.score[0] = prob;
    tree.verify_order[0] = 0;
    tree.num_nodes = 1;
    tree.frontier[0] = 0;
    tree.frontier_count = 1;
    tree.pending_begin = 1;
    tree.pending_end = 1;
    return Status::make_ok();
}

uint32_t eagle4_tree_insert_children(Eagle4DraftTree& tree,
                                     int32_t parent,
                                     const int32_t* tokens,
                                     const float* probs,
                                     uint32_t count) {
    if (parent < 0 || parent >= tree.num_nodes) return 0u;
    if (tokens == nullptr || probs == nullptr || count == 0u) return 0u;
    if (tree.depth[parent] >= tree.max_depth) return 0u;
    if (tree.pending_begin == tree.pending_end) tree.pending_begin = tree.num_nodes;
    uint32_t inserted = 0u;
    for (uint32_t i = 0; i < count; ++i) {
        if (tree.num_nodes >= tree.capacity) break;
        const int32_t idx = tree.num_nodes;
        tree.parent[idx] = parent;
        tree.depth[idx] = tree.depth[parent] + 1;
        tree.token[idx] = tokens[i];
        tree.score[idx] = tree.score[parent] * probs[i];
        tree.verify_order[idx] = idx;
        tree.num_nodes = idx + 1;
        ++inserted;
    }
    tree.pending_end = tree.num_nodes;
    return inserted;
}

uint32_t eagle4_tree_select_frontier(Eagle4DraftTree& tree, uint32_t width) {
    tree.frontier_count = 0;
    const int32_t begin = tree.pending_begin;
    const int32_t end = tree.pending_end;
    tree.pending_begin = tree.num_nodes;
    tree.pending_end = tree.num_nodes;
    if (end <= begin || width == 0u) return 0u;
    const int32_t n = end - begin;
    std::array<int32_t, kEagle4TreeMaxNodes> order{};
    for (int32_t i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = begin + i;
    std::stable_sort(order.begin(), order.begin() + n, [&](int32_t a, int32_t b) {
        if (tree.score[a] != tree.score[b]) return tree.score[a] > tree.score[b];
        return a < b;
    });
    uint32_t take = width < static_cast<uint32_t>(n) ? width : static_cast<uint32_t>(n);
    for (uint32_t i = 0; i < take; ++i) {
        tree.frontier[i] = order[static_cast<std::size_t>(i)];
    }
    tree.frontier_count = static_cast<int32_t>(take);
    return take;
}

bool eagle4_tree_step_available(const Eagle4DraftTree& tree) {
    if (tree.frontier_count <= 0) return false;
    if (tree.num_nodes >= tree.capacity) return false;
    for (int32_t i = 0; i < tree.frontier_count; ++i) {
        if (tree.depth[tree.frontier[i]] < tree.max_depth) return true;
    }
    return false;
}

uint32_t eagle4_tree_node_path(const Eagle4DraftTree& tree, int32_t node, int32_t* out_path) {
    if (node < 0 || node >= tree.num_nodes || out_path == nullptr) return 0u;
    int32_t stack[kEagle4TreeMaxNodes];
    int32_t count = 0;
    int32_t cur = node;
    while (cur >= 0 && count < kEagle4TreeMaxNodes) {
        stack[count++] = cur;
        cur = tree.parent[cur];
    }
    for (int32_t i = 0; i < count; ++i) out_path[i] = stack[count - 1 - i];
    return static_cast<uint32_t>(count);
}

uint32_t eagle4_tree_ancestors_flat(const Eagle4DraftTree& tree,
                                    int32_t row_stride,
                                    int32_t* out_rows) {
    if (out_rows == nullptr || row_stride < tree.max_depth + 1) return 0u;
    for (int32_t r = 0; r < tree.frontier_count; ++r) {
        int32_t* row = out_rows + static_cast<std::size_t>(r) * row_stride;
        const uint32_t n = eagle4_tree_node_path(tree, tree.frontier[r], row);
        for (int32_t i = static_cast<int32_t>(n); i < row_stride; ++i) row[i] = -1;
    }
    return static_cast<uint32_t>(tree.frontier_count);
}

Status spec_tree_greedy_accept(const Eagle4DraftTree& tree,
                               const int32_t* target_tokens,
                               Eagle4TreeAcceptResult& out) {
    if (target_tokens == nullptr) {
        return Status::invalid_argument("spec_tree_greedy_accept target_tokens is null",
                                        __FILE__, __LINE__);
    }
    if (tree.num_nodes <= 0) {
        return Status::invalid_argument("spec_tree_greedy_accept tree is empty", __FILE__,
                                        __LINE__);
    }

    out = Eagle4TreeAcceptResult{};
    out.emitted_tokens.reserve(static_cast<std::size_t>(tree.num_nodes) + 1u);

    if (tree.token[0] != target_tokens[0]) {
        out.correction_valid = true;
        out.correction_token = target_tokens[0];
        out.correction_row = 0;
        out.num_committed_target_tokens = 1u;
        out.emitted_tokens.push_back(target_tokens[0]);
        out.num_emitted_tokens = 1u;
        return Status::make_ok();
    }

    out.path[0] = 0;
    out.path_count = 1u;
    out.num_accepted = 1u;
    out.emitted_tokens.push_back(tree.token[0]);
    int32_t cur = 0;

    while (true) {
        int32_t first_child = -1;
        int32_t matched = -1;
        for (int32_t c = 0; c < tree.num_nodes; ++c) {
            if (tree.parent[c] != cur) continue;
            if (first_child < 0) first_child = c;
            if (tree.token[c] == target_tokens[c]) {
                matched = c;
                break;
            }
        }
        if (first_child < 0) {
            out.leaf_accept = true;
            out.num_committed_target_tokens = out.num_accepted;
            out.num_emitted_tokens = static_cast<uint32_t>(out.emitted_tokens.size());
            return Status::make_ok();
        }
        if (matched < 0) {
            out.correction_valid = true;
            out.correction_token = target_tokens[first_child];
            out.correction_row = first_child;
            out.num_committed_target_tokens = out.num_accepted + 1u;
            out.emitted_tokens.push_back(out.correction_token);
            out.num_emitted_tokens = static_cast<uint32_t>(out.emitted_tokens.size());
            return Status::make_ok();
        }
        out.path[out.path_count] = matched;
        ++out.path_count;
        ++out.num_accepted;
        out.emitted_tokens.push_back(tree.token[matched]);
        cur = matched;
    }
}

}  // namespace ps::qwen35::runtime
