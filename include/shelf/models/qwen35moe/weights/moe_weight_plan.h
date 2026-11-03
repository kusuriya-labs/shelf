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
    uint32_t experts = 0;
    uint32_t scale_block_k = 0;

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

std::string moe_expert_tensor_name(std::size_t layer, uint32_t expert,
                                   const char* projection, bool scale);

Result<MoeExpertGroupPlan> build_moe_expert_group_plan(const Qwen35MoeTextConfig& config,
                                                        std::size_t layer);

}  // namespace shelf::qwen35moe
