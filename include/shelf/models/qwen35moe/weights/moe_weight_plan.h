#pragma once
#include <shelf/core/status.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace shelf::qwen35moe {

// One routed expert projection. This is the logical tensor the reader resolves: its
// payload carries both the E2M1 codes plane and the E8M0 scale plane, which on disk are
// `<name>` and `<name>_scale`.
enum class MoeExpertProjection : uint8_t {
    Down,
    Gate,
    Up,
};

struct MoeExpertTensorSource {
    std::string name;
    uint32_t expert = 0;
    MoeExpertProjection projection = MoeExpertProjection::Down;
    // Row range this source occupies inside its destination plane.
    uint32_t row_offset = 0;
    uint32_t rows = 0;
};

// The rank-3 planes one MoE layer's routed experts resolve to. w13 is the fused
// gate/up plane with the gate rows first; w2 is down_proj. sizes and strides are
// per one expert unless the name says otherwise.
struct MoeExpertGroupPlan {
    uint32_t layer = 0;
    // Experts this plan's planes hold. Without a split it is every expert of the config;
    // with an expert-axis split it is one rank's contiguous range, named by expert_begin.
    uint32_t experts = 0;
    uint32_t expert_begin = 0;
    // The expert count the router selects over. It is the whole config count whatever the
    // split is, and it is what the checkpoint's own plane depth is.
    uint32_t global_experts = 0;
    // The split this plan was built for. The shared expert divides the way a dense FFN
    // does, so the plan carries the width to check it against.
    uint32_t tp_size = 1;
    uint32_t tp_rank = 0;
    uint32_t scale_block_k = 0;

    // Checkpoint location of this group's FFN tensors: "<layer prefix>mlp.". It is not a
    // transformer layer prefix for a block that lives outside the layer array (MTP).
    std::string mlp_prefix;

    uint32_t w13_rows = 0;
    uint32_t w13_cols = 0;
    uint32_t w13_k_padded = 0;
    uint32_t w13_gate_rows = 0;
    uint32_t w13_up_rows = 0;
    uint32_t w13_codes_row_stride_bytes = 0;
    uint32_t w13_scales_row_stride_bytes = 0;
    uint64_t w13_codes_batch_stride_bytes = 0;
    uint64_t w13_scales_batch_stride_bytes = 0;
    uint64_t w13_codes_bytes = 0;
    uint64_t w13_scales_bytes = 0;

    uint32_t w2_rows = 0;
    uint32_t w2_cols = 0;
    uint32_t w2_k_padded = 0;
    uint32_t w2_codes_row_stride_bytes = 0;
    uint32_t w2_scales_row_stride_bytes = 0;
    uint64_t w2_codes_batch_stride_bytes = 0;
    uint64_t w2_scales_batch_stride_bytes = 0;
    uint64_t w2_codes_bytes = 0;
    uint64_t w2_scales_bytes = 0;

    // One E8M0 row reference byte per output row of one expert, per plane.
    uint32_t w13_row_ref_bytes_per_expert = 0;
    uint32_t w2_row_ref_bytes_per_expert = 0;

    // A checkpoint that stores the routed experts as one fused rank-3 tensor per plane
    // ([experts][rows][cols], gate rows first) has no per-expert logical names. The plane
    // names replace sources, and the payload is in the checkpoint's own dtype.
    bool fused_rank3 = false;
    std::string w13_name;
    std::string w2_name;

    // Sources in checkpoint file order (expert-major, down_proj then gate_proj then
    // up_proj). One source is one logical tensor: its codes plane and its E8M0 scale
    // plane. A layer's 6 physical tensors per expert are one contiguous run on disk.
    std::vector<MoeExpertTensorSource> sources;
};

// Routed expert tensor count of the whole model: layers x experts x 3 projections x 2 planes.
std::size_t moe_expert_tensor_count(const Qwen35MoeTextConfig& config);

// Checkpoint name of one layer's mlp tensor: "model.language_model.layers.<l>.mlp."
// followed by suffix, e.g. "gate.weight" or "shared_expert.down_proj.weight".
std::string moe_mlp_tensor_name(std::size_t layer, const std::string& suffix);

// The "<prefix>mlp." location of one group's FFN tensors. A block that is not a
// transformer layer (the MTP block) has its own prefix.
std::string moe_mlp_prefix(std::size_t layer);
std::string moe_mtp_layer_prefix();
std::string moe_mtp_mlp_prefix();

std::string moe_expert_tensor_name(std::size_t layer, uint32_t expert,
                                   const char* projection, bool scale);

// The same tensor name under an explicit mlp location (a plan's mlp_prefix).
std::string moe_expert_tensor_name(const std::string& mlp, uint32_t expert,
                                   const char* projection, bool scale);

Result<MoeExpertGroupPlan> build_moe_expert_group_plan(const Qwen35MoeTextConfig& config,
                                                        std::size_t layer);

// The same plan for one rank of an expert-axis split: the rank owns the contiguous expert
// range [tp_rank * experts/tp_size, ...), and `experts` counts that range. The plane depth
// a checkpoint stores stays the global count, which the plan keeps as global_experts.
// num_experts must divide by tp_size.
Result<MoeExpertGroupPlan> build_moe_expert_group_plan(const Qwen35MoeTextConfig& config,
                                                        std::size_t layer,
                                                        uint32_t tp_size,
                                                        uint32_t tp_rank);

// The MTP block's routed experts: the same geometry as a transformer layer, stored fused
// and in the checkpoint's own dtype, so the load requantizes them.
Result<MoeExpertGroupPlan> build_moe_mtp_expert_group_plan(const Qwen35MoeTextConfig& config);
Result<MoeExpertGroupPlan> build_moe_mtp_expert_group_plan(const Qwen35MoeTextConfig& config,
                                                           uint32_t tp_size,
                                                           uint32_t tp_rank);

}  // namespace shelf::qwen35moe
