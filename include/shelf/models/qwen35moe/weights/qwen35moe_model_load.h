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

// A loaded MoE model: the shared model-level weights, one MoE layer per transformer
// layer, and the checkpoint's MTP block when it was asked for.
struct Qwen35MoeMtpWeights {
    bool present = false;

    shelf::gpu::Tensor pre_fc_norm_embedding_weight;
    shelf::gpu::Tensor pre_fc_norm_hidden_weight;
    shelf::gpu::Tensor final_norm_weight;

    shelf::weights::MatrixWeight fc;
    Qwen35MoeLayerWeights layer;
};

struct Qwen35MoeModelWeights : shelf::qwen_hybrid::HybridModelLevelWeights {
    std::vector<Qwen35MoeLayerWeights> layers;
    Qwen35MoeMtpWeights mtp;
};

struct Qwen35MoeLoadOptions : shelf::qwen_hybrid::HybridLoadOptions {
    bool load_mtp_layers = false;
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

// The same invariants for one rank of an expert-axis split: the routed expert planes carry
// config.ffn.num_experts / tp_size experts, the shared expert is tp_size times narrower, and
// the router keeps the whole expert count.
Status validate_qwen35moe_weights(const Qwen35MoeModelWeights& weights,
                                 const Qwen35MoeTextConfig& config,
                                 std::uint32_t tp_size,
                                 std::uint32_t tp_rank);

// The MTP block's own invariants: fc is [hidden, 2 * hidden], the three block norms are
// [hidden], its layer is a full-attention body, and its FFN matches the MTP plan (the
// checkpoint keeps those experts fused, so their planes are requantized on load).
Status validate_qwen35moe_mtp_weights(const Qwen35MoeMtpWeights& mtp,
                                      const MoeExpertGroupPlan& plan,
                                      const Qwen35MoeTextConfig& config);

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
