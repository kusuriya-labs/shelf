#pragma once

#include <shelf/kernels/optimized/linear/exl3.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// One routed-expert projection over a grouped row plan. The expert planes are the
// packed EXL3 layout the Qwen4Exp loader produces:
//
//   trellis [experts, trellis_bytes]
//   suh     [experts, k]
//   svh     [experts, n]
//
// Group g owns activation and output rows [g * kExl3W16A16RowTile,
// (g + 1) * kExl3W16A16RowTile), the same row contract as the MXFP4 grouped linear.
// group_expert / group_rows / group_count are device pointers from the router plan.
struct Exl3GroupedArgs {
    const void* trellis = nullptr;
    const void* suh = nullptr;
    const void* svh = nullptr;
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t output_dtype = 0;
    uint32_t ka = 0;
    uint32_t half = 0;
    // 0: mul1 codebook (W16A16). 1: mul24 codebook (EXL3-INT8 weights, q * kExl3Int8GridStep) run as an FP16 GEMM.
    // The trellis stays in the canonical [kt][nt] tile order in both cases. mul24 requires half == 0.
    uint32_t mul24 = 0;
    std::uint64_t trellis_batch_stride = 0;
    std::uint64_t suh_batch_stride = 0;
    std::uint64_t svh_batch_stride = 0;
    const uint32_t* group_expert = nullptr;
    const uint32_t* group_rows = nullptr;
    const uint32_t* group_count = nullptr;
    // Optional per-expert group range. When non-null, consecutive groups of one expert are
    // fused into one block so the expert's weight tiles are dequantized once per fused chunk.
    const uint32_t* expert_group_begin = nullptr;
    const int32_t* row_token = nullptr;
    uint32_t group_capacity = 0;
};

// Correctness-first grouped EXL3 linear. It dispatches the existing dense EXL3
// kernel once per group, so it preserves the packed expert layout but has a launch
// per group. A fused grouped kernel replaces this for throughput later.
hipError_t launch_gemm_exl3_w16a16_grouped(const Exl3GroupedArgs& args,
                                           hipStream_t stream);

}  // namespace shelf::kernel
