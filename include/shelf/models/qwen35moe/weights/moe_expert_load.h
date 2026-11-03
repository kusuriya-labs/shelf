#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/weights/matrix_weight.h>

#include <hip/hip_runtime.h>

namespace shelf::qwen35moe {

// GPU-resident rank-3 expert planes of one MoE layer.
struct MoeExpertGroupWeights {
    shelf::weights::MatrixWeight w13;  // fused gate/up, gate rows first
    shelf::weights::MatrixWeight w2;   // down_proj
};

// Loads one layer's routed experts as two rank-3 native SoA planes: the per-expert
// canonical payloads are aggregated on the host into w13 / w2, preshuffled to the
// MXFP4 native layout once per plane (batch = experts), and uploaded as one transfer
// per plane stream. rows / cols / k_padded and the row strides of the resulting
// MatrixWeight describe one expert; codes_batch_stride_bytes and
// scales_batch_stride_bytes separate experts.
Result<MoeExpertGroupWeights> load_moe_expert_group(
    const shelf::quantization::bundle::QuantizedModelReader& reader,
    const MoeExpertGroupPlan& plan,
    shelf::gpu::GpuArena& arena,
    hipStream_t stream);

}  // namespace shelf::qwen35moe
