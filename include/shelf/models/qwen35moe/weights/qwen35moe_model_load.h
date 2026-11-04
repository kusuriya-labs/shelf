#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/layer_weights.h>
#include <shelf/models/common/qwen_hybrid/model_weights.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/moe_expert_load.h>
#include <shelf/models/qwen35moe/weights/moe_ffn_weights.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace shelf::qwen35moe {

// A MoE layer: the shared hybrid backbone (layernorms and whichever attention the layer
// carries) plus the MoE FFN. The FFN is a member, so a layer of this type cannot be built
// without it and cannot hold a dense FFN.
struct Qwen35MoeLayerWeights : shelf::qwen_hybrid::HybridLayerWeights {
    MoeFfnWeights ffn;
};

// A loaded MoE model: the shared model-level weights and one MoE layer per transformer
// layer. There is no second layer array and no MTP block; speculative decode uses the
// DFlash2 drafter instead.
struct Qwen35MoeModelWeights : shelf::qwen_hybrid::HybridModelLevelWeights {
    std::vector<Qwen35MoeLayerWeights> layers;
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

// True when every MatrixWeight of one layer's FFN carries the geometry its plan and its
// config say: expert count, per-expert rows / cols / k_padded, row and batch strides, the
// native row_ref planes, the router and shared-expert gate shapes, and the shared expert's
// own intermediate width (which is not the routed experts' width).
Status moe_ffn_weights_match(const MoeFfnWeights& w, const MoeExpertGroupPlan& plan,
                             const MoeFfnConfig& ffn);

// Model invariants of a loaded MoE weight set: the shared hybrid part (embed / lm_head
// vocab, hidden width, final norm, layer count, per-layer backbone topology) plus every
// layer's routed-expert planes, router and shared expert as its plan and config state.
Status validate_qwen35moe_weights(const Qwen35MoeModelWeights& weights,
                                 const Qwen35MoeTextConfig& config);

// Loads a MoE checkpoint: the hybrid backbone through the shared loader, then each
// layer's routed experts as rank-3 native SoA planes plus its router and shared expert.
// Only an HF real-quantized model dir is read; a Shelf quantized bundle is rejected
// because rank-3 canonical partitions are not part of that contract. The options carry no
// MTP field, so an MoE load cannot ask for one.
Result<Qwen35MoeModelWeights> load_qwen35moe_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::qwen_hybrid::HybridLoadOptions& options = {});

}  // namespace shelf::qwen35moe
