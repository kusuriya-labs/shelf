#pragma once
#include <shelf/weights/matrix_weight.h>

namespace shelf {
namespace qwen35moe {

// The FFN of one MoE layer: a router over the routed experts, the routed experts as two
// rank-3 native SoA planes, and one shared expert. expert_w13 fuses gate and up with the
// gate rows first; expert_w2 is down_proj. rows / cols / k_padded and the row strides of
// those two planes describe one expert, and codes_batch_stride_bytes /
// scales_batch_stride_bytes separate experts inside one arena allocation, so an expert is
// never its own GPU allocation. Nothing here is a dense FFN: an MoE layer has no
// intermediate_size-wide gate / up / down triple.
struct MoeFfnWeights {
    shelf::weights::MatrixWeight router;          // [experts, hidden]
    shelf::weights::MatrixWeight expert_w13;      // rank-3 fused gate / up
    shelf::weights::MatrixWeight expert_w2;       // rank-3 down_proj
    shelf::weights::MatrixWeight shared_gate;     // [1, hidden]
    shelf::weights::MatrixWeight shared_gate_proj;
    shelf::weights::MatrixWeight shared_up_proj;
    shelf::weights::MatrixWeight shared_down_proj;
};

}  // namespace qwen35moe
}  // namespace shelf
