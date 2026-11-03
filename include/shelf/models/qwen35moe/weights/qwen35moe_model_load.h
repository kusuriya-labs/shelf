#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35moe/weights/moe_expert_load.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace shelf::qwen35moe {

// The FFN of one MoE layer, on the GPU. The layer's layernorms and attention are
// shelf::qwen35::Qwen35LayerWeights and are loaded by the qwen35 load blocks; everything
// MoE-specific is held here. expert_w13 / expert_w2 are rank-3 native SoA planes
// (experts > 1): rows / cols / k_padded and the row strides describe one expert, and
// codes_batch_stride_bytes / scales_batch_stride_bytes separate experts inside one arena
// allocation. w13 fuses gate and up with the gate rows first.
struct MoeLayerWeights {
    shelf::weights::MatrixWeight router;          // [experts, hidden]
    shelf::weights::MatrixWeight expert_w13;      // fused gate / up
    shelf::weights::MatrixWeight expert_w2;       // down_proj
    shelf::weights::MatrixWeight shared_gate;     // [1, hidden]
    shelf::weights::MatrixWeight shared_gate_proj;
    shelf::weights::MatrixWeight shared_up_proj;
    shelf::weights::MatrixWeight shared_down_proj;
};

// A loaded MoE model. body carries embed_tokens / lm_head / final_norm and, per layer,
// the layernorms and attention (its mlp_* fields stay empty), and layers is index
// aligned with body.layers.
struct MoeModelWeights {
    shelf::qwen35::Qwen35ModelWeights body;
    std::vector<MoeLayerWeights> layers;
};

// Router and shared-expert checkpoint names of one layer, in the order a request list
// should carry them.
std::vector<std::string> moe_aux_tensor_names(std::size_t layer);

// Logical names one layer's MoE FFN adds to the checkpoint request list: the aux
// tensors plus the routed experts in plan order. The physical `<name>_scale` planes are
// the reader's business and are never requested as logical names.
std::vector<std::string> moe_layer_logical_names(const MoeExpertGroupPlan& plan);

// Resident GPU bytes of one layer's rank-3 expert planes (codes + scales + row_ref for
// both planes). Sums the plan's own size fields, so it cannot drift from the load.
uint64_t moe_expert_group_bytes(const MoeExpertGroupPlan& plan);

// True when every MatrixWeight of a loaded layer carries the geometry its plan says:
// expert count, per-expert rows / cols / k_padded, row and batch strides, the native
// row_ref planes, and the router / shared expert shapes.
Status moe_layer_weights_match_plan(const MoeLayerWeights& w, const MoeExpertGroupPlan& plan);

// Loads a MoE checkpoint: the qwen35 body through the qwen35 load blocks, then each
// layer's routed experts as rank-3 native SoA planes plus its router and shared expert.
// Only an HF real-quantized model dir is read; a Shelf quantized bundle is rejected
// because rank-3 canonical partitions are not part of that contract.
Result<MoeModelWeights> load_qwen35moe_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::qwen35::Qwen35LoadOptions& options = {});

}  // namespace shelf::qwen35moe
