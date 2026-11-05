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
