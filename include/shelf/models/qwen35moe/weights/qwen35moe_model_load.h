#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/layer_weights.h>
#include <shelf/models/common/qwen_hybrid/model_weights.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
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
// layer.
struct Qwen35MoeModelWeights : shelf::qwen_hybrid::HybridModelLevelWeights {
    std::vector<Qwen35MoeLayerWeights> layers;
};

// True when any linear of this weight set is carried in the rotated-domain trellis form,
// including the LM head. The memory plan uses it to decide whether the executor workspace
// must hold that form's staging planes.
inline bool uses_exl3_linear(const Qwen35MoeModelWeights& weights) noexcept {
    if (weights.lm_head.encoding == shelf::weights::MatrixEncoding::Exl3) return true;
    for (const Qwen35MoeLayerWeights& layer : weights.layers) {
        if (shelf::qwen_hybrid::layer_uses_exl3_linear(layer) ||
            ffn_uses_exl3_linear(layer.ffn))
            return true;
    }
    return false;
}

// The staging-plane geometry of this rank's linears. The LM head and the MTP block are
// excluded: the LM head stays on the fused path, and the MTP block runs in its own executor.
inline shelf::weights::Exl3LinearShape exl3_linear_shape(
    const Qwen35MoeModelWeights& weights) noexcept {
    shelf::weights::Exl3LinearShape shape;
    for (const Qwen35MoeLayerWeights& layer : weights.layers) {
        shelf::qwen_hybrid::note_layer_exl3_linear(layer, shape);
        note_ffn_exl3_linear(layer.ffn, shape);
    }
    return shape;
}

struct Qwen35MoeLoadOptions : shelf::qwen_hybrid::HybridLoadOptions {
    bool load_mtp_layers = false;
    // Routed expert IDs follow the expert-parallel rank. The backbone, shared expert and
    // each local expert's matrix follow the tensor-partition plan's tp_size / tp_rank.
    // ep_size == 0 means EP=1.
    std::uint32_t ep_size = 0;
    std::uint32_t ep_rank = 0;
    bool ep_enabled = false;

    // A pipeline stage loads only its own layer range. The layers outside the range stay in
    // the vector and hold no tensors, and the embedding / LM head / final norm are loaded
    // only by the stage that consumes them. A stage combined with an expert-axis rank is not
    // wired: the expert planes of a stage would need both a layer range and an expert range.
    shelf::qwen_hybrid::PipelineStage stage;
};

// Router and shared-expert checkpoint names of one layer, in the order a request list
// should carry them.
std::vector<std::string> moe_aux_tensor_names(std::size_t layer);

// The same names under an explicit mlp location (a plan's mlp_prefix).
std::vector<std::string> moe_aux_tensor_names(const std::string& mlp);

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
// layer's routed-expert planes, router and shared expert as its plan and config state. An
// MTP block that is present is validated with it.
Status validate_qwen35moe_weights(const Qwen35MoeModelWeights& weights,
                                 const Qwen35MoeTextConfig& config);

// The same invariants for one pipeline stage: only the stage's own layers carry tensors, and
// the model-level tensors follow the first / last rule. An MTP block with a stage is rejected.
Status validate_qwen35moe_weights(const Qwen35MoeModelWeights& weights,
                                  const Qwen35MoeTextConfig& config,
                                  std::uint32_t tp_size,
                                  std::uint32_t tp_rank,
                                  const shelf::qwen_hybrid::PipelineStage& stage);

// The same invariants for one rank of an expert-axis split: the routed expert planes carry
// config.ffn.num_experts / tp_size experts, the shared expert is tp_size times narrower, and
// the router keeps the whole expert count.
Status validate_qwen35moe_weights(const Qwen35MoeModelWeights& weights,
                                 const Qwen35MoeTextConfig& config,
                                 std::uint32_t tp_size,
                                 std::uint32_t tp_rank);

// The routed expert IDs follow (ep_rank, ep_size); the backbone, shared expert and each
// local expert's matrix follow (tp_rank, tp_size).
Status validate_qwen35moe_weights(const Qwen35MoeModelWeights& weights,
                                 const Qwen35MoeTextConfig& config,
                                 std::uint32_t ep_size,
                                 std::uint32_t ep_rank,
                                 std::uint32_t tp_size,
                                 std::uint32_t tp_rank,
                                 const shelf::qwen_hybrid::PipelineStage& stage = {});

// Loads a MoE checkpoint: the hybrid backbone through the shared loader, then each
// layer's routed experts as rank-3 native SoA planes plus its router and shared expert.
// Only an HF real-quantized model dir is read; a Shelf quantized bundle is rejected
// because rank-3 canonical partitions are not part of that contract.
//
// With options.weights.partition_plan the load is one rank of a split: the plan carries
// tp_size / tp_rank, the backbone and the shared expert are cut along their matrix axes by
// that plan, and the routed experts are cut along the expert axis. The router and the
// shared-expert gate stay whole on every rank.
Result<Qwen35MoeModelWeights> load_qwen35moe_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35MoeLoadOptions& options = {});

}  // namespace shelf::qwen35moe
