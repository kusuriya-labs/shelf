#pragma once

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// MoE combine: folds a token's top_k expert outputs back into one row.
//
// Accumulation order contract (MoE combine): a token's slots are added
// in expert index ascending order, which is group row ascending order because the group
// table is built expert ascending. The kernel therefore sorts each token's slot rows and
// accumulates in f32, rounding to bf16 once.
//
// When shared_output is non-null, shared_output[row] * shared_gate_weight[row *
// shared_gate_row_stride] is added last, after the routed sum. shared_gate_weight holds the
// already activated gate (the existing bf16 -> f32 sigmoid elementwise kernel produces it),
// so this kernel does no transcendental math of its own. The gate is one column per token,
// but the plane it lives in is a padded row-major value like every other one, so its row
// stride is passed: reading it packed would use another row's bytes for every token after
// the first.
hipError_t launch_moe_combine(
    shelf::bf16_t* output,
    const shelf::bf16_t* expert_output,
    const float* sel_weight,
    const int32_t* slot_row_of,
    const shelf::bf16_t* shared_output,
    const float* shared_gate_weight,
    uint32_t rows,
    uint32_t top_k,
    uint32_t features,
    uint32_t expert_row_stride,
    uint32_t shared_row_stride,
    uint32_t shared_gate_row_stride,
    hipStream_t stream);

}  // namespace shelf::kernel
