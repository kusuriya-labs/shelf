#pragma once

#include <shelf/core/status.h>
#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/runtime/program/moe_plan_scratch.h>
#include <shelf/runtime/program/program.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::execution {

struct HostExecutionContext;

// The plan arrays of one routing round, read out of the program workspace range the binding
// names. Types are the kernel parameters' types, so a consumer cannot mix an int32 array
// with a uint32 one.
struct MoePlanArrays {
    int32_t* sel_expert = nullptr;
    float* sel_weight = nullptr;
    uint32_t* tile_counts = nullptr;
    uint32_t* expert_row_begin = nullptr;
    uint32_t* tile_base = nullptr;
    uint32_t* expert_group_begin = nullptr;
    uint32_t* group_expert = nullptr;
    uint32_t* group_rows = nullptr;
    uint32_t* group_count = nullptr;
    int32_t* row_token = nullptr;
    int32_t* slot_row_of = nullptr;
    // Original-group split: hits use VRAM slots, misses use host slots. `base` is the
    // original group index, so both launches write the same group-row layout.
    uint32_t* hit_slot = nullptr;
    uint32_t* hit_rows = nullptr;
    uint32_t* hit_base = nullptr;
    uint32_t* hit_count = nullptr;
    uint32_t* miss_slot = nullptr;
    uint32_t* miss_rows = nullptr;
    uint32_t* miss_base = nullptr;
    uint32_t* miss_count = nullptr;
    uint32_t* hit_expert_group_begin = nullptr;
    uint32_t* miss_expert_group_begin = nullptr;
};

// The scratch arrays plus the capacity this call actually carries. The capacity comes from
// the binding's layout geometry and the batch's real token count, so a small batch never
// walks the row bucket's full capacity.
struct MoePlanView {
    MoePlanArrays arrays;
    uint32_t tokens = 0;
    // The experts this device owns and plans rows for. With a split this is one range of the
    // model's experts, and the router's logits stay as wide as the whole expert count.
    uint32_t experts = 0;
    uint32_t logit_experts = 0;
    uint32_t expert_begin = 0;
    uint32_t top_k = 0;
    uint32_t group_capacity = 0;
    uint32_t row_capacity = 0;
};

// The INT8 activation plane a grouped EXL3 dispatch rotates into, when the row budget and
// workspace allow it.
struct MoePreRotatePlane {
    uint8_t* codes = nullptr;
    float* scales = nullptr;
};

bool moe_pre_rotate_plane(const ::shelf::runtime::Program& program,
                          const ::shelf::runtime::DispatchBinding& b,
                          HostExecutionContext& ctx,
                          MoePreRotatePlane& out);

// A SWIGLU that produces this dispatch's single input and is consumed only by it, so the
// grouped projection can compute SwiGLU in its rotation instead of reading a stored row.
struct MoeSwigluProducer {
    bool found = false;
    const void* gate = nullptr;
    const void* up = nullptr;
    void* out = nullptr;
    uint32_t gate_row_stride = 0;
    uint32_t up_row_stride = 0;
    uint32_t out_row_stride = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    bool bf16_intermediate = false;
};

MoeSwigluProducer moe_find_swiglu_producer(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx);

// True when the grouped projection can rotate SwiGLU(gate, up) directly: the producer is
// exclusive, the pre-rotation plane fits, and that plane does not alias the gate/up rows
// (which the rotation would otherwise overwrite while reading them).
bool moe_swiglu_fusion_available(const ::shelf::runtime::Program& program,
                                 const ::shelf::runtime::DispatchBinding& b,
                                 HostExecutionContext& ctx);

// The group rows a batch of ctx.actual_rows tokens produces, from the plan geometry the
// binding carries. A kernel that works on group-row values uses this instead of the token
// row count.
uint32_t moe_group_rows(const ::shelf::runtime::DispatchBinding& b,
                        const HostExecutionContext& ctx) noexcept;

::shelf::Result<MoePlanView> resolve_moe_plan(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx);

Result<OptimizedLaunchResult> try_launch_moe_router(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_moe_plan_scan(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_moe_plan_scatter(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_moe_grouped_linear(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_moe_grouped_exl3(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_moe_combine(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

}
