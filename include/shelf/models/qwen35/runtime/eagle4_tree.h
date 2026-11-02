#pragma once

#include <shelf/core/status.h>

#include <cstdint>
#include <vector>

namespace ps::qwen35::runtime {

inline constexpr int32_t kEagle4TreeMaxNodes = 512;

struct Eagle4DraftTree {
    int32_t num_nodes = 0;
    int32_t capacity = 0;
    int32_t max_depth = 0;
    int32_t parent[kEagle4TreeMaxNodes];
    int32_t depth[kEagle4TreeMaxNodes];
    int32_t token[kEagle4TreeMaxNodes];
    float score[kEagle4TreeMaxNodes];
    int32_t verify_order[kEagle4TreeMaxNodes];
    int32_t frontier[kEagle4TreeMaxNodes];
    int32_t frontier_count = 0;
    int32_t pending_begin = 0;
    int32_t pending_end = 0;
};

void eagle4_tree_reset(Eagle4DraftTree& tree, int32_t capacity, int32_t max_depth);

Status eagle4_tree_insert_root(Eagle4DraftTree& tree, int32_t token, float prob);

uint32_t eagle4_tree_insert_children(Eagle4DraftTree& tree,
                                     int32_t parent,
                                     const int32_t* tokens,
                                     const float* probs,
                                     uint32_t count);

uint32_t eagle4_tree_select_frontier(Eagle4DraftTree& tree, uint32_t width);

bool eagle4_tree_step_available(const Eagle4DraftTree& tree);

uint32_t eagle4_tree_node_path(const Eagle4DraftTree& tree, int32_t node, int32_t* out_path);

uint32_t eagle4_tree_ancestors_flat(const Eagle4DraftTree& tree,
                                    int32_t row_stride,
                                    int32_t* out_rows);

struct Eagle4TreeAcceptResult {
    uint32_t num_accepted = 0;
    uint32_t path_count = 0;
    int32_t path[kEagle4TreeMaxNodes];
    bool correction_valid = false;
    int32_t correction_token = -1;
    int32_t correction_row = -1;
    bool leaf_accept = false;
    uint32_t num_committed_target_tokens = 0;
    uint32_t num_emitted_tokens = 0;
    std::vector<int32_t> emitted_tokens;
};

Status spec_tree_greedy_accept(const Eagle4DraftTree& tree,
                               const int32_t* target_tokens,
                               Eagle4TreeAcceptResult& out);

}  // namespace ps::qwen35::runtime
