#pragma once

#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

inline constexpr const char* kExl3W16A16Symbol = "shelf_gemm_exl3_w16a16";
inline constexpr uint32_t kExl3W16A16OutBlock = 128u;
inline constexpr uint32_t kExl3W16A16RowTile = 16u;
inline constexpr uint32_t kExl3W16A16Threads = 256u;
inline constexpr uint32_t kExl3GroupedFuse = 2u;

struct Exl3GemmArgs {
    const void* trellis = nullptr;
    const void* suh = nullptr;
    const void* svh = nullptr;
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t output_dtype = 0;
    uint32_t ka = 0;
    uint32_t half = 0;
    float* splitk_scratch = nullptr;
    uint32_t splitk_floats = 0;
    uint32_t splitk_splits = 0;
    void* xrot = nullptr;
    uint32_t groups = 0;
    // Optional n-tile override for the grouped path. 0 keeps pick_n_tiles' shape choice.
    uint32_t n_tiles = 0;

    // Optional routed-expert group plan. When group_expert is non-null, blockIdx.z is
    // the group, group_expert[g] selects the expert plane and group_rows[g] bounds the
    // rows this group owns. The trellis/suh/svh pointers are bases and the batch
    // strides select one expert. Input and output are the group-row planes; group g
    // owns rows [g * kExl3W16A16RowTile, ...).
    const uint32_t* group_expert = nullptr;
    const uint32_t* group_rows = nullptr;
    const uint32_t* group_count = nullptr;
    // Optional per-expert group range. When non-null, consecutive groups of one expert are
    // fused into one block: a block whose group index is the first of a kGroups stride inside
    // [expert_group_begin[e], expert_group_begin[e+1]) owns up to kGroups groups of expert e
    // and dequantizes its weight tiles once for all of them.
    const uint32_t* expert_group_begin = nullptr;
    // Optional token row for each group row. When non-null, the input plane is token rows
    // and the kernel gathers through this table; when null, the input plane is already in
    // group-row order.
    const int32_t* row_token = nullptr;
    uint32_t mul24_lds = 0;
    void* mul24_activation_workspace = nullptr;
    uint64_t trellis_batch_stride = 0;
    uint64_t suh_batch_stride = 0;
    uint64_t svh_batch_stride = 0;
};

hipError_t launch_gemm_exl3_w16a16(const Exl3GemmArgs& args, hipStream_t stream);
hipError_t launch_rotate_exl3(const Exl3GemmArgs& args, void* xrot, hipStream_t stream);
hipError_t launch_rotate_exl3_rowsum(const Exl3GemmArgs& args, void* xrot, float* rowsum,
                                     hipStream_t stream);
hipError_t launch_recon_exl3_dense(const Exl3GemmArgs& args, void* dense, hipStream_t stream);
hipError_t launch_quantize_dense_w8(const Exl3GemmArgs& args, void* dense,
                                   hipStream_t stream);
hipError_t launch_quantize_xrot_x8(const Exl3GemmArgs& args, void* xrot, float* rowsum,
                                   bool residual, hipStream_t stream);
hipError_t launch_epilogue_exl3_dense(const Exl3GemmArgs& args, const void* rowsum,
                                      const void* accum, hipStream_t stream);
// Grouped mul24 GEMM (exl3_mul24_grouped.hip). `args` is the grouped plan as launch_gemm_exl3_w16a16_grouped builds it
// (rows = one group tile, groups = fused groups per block, 1 or 2; more is run as 2). int8_act selects INT8 activations
// with an iu8 WMMA; otherwise an FP16 WMMA on the exact integer weights. n_tiles is 1 or 2 (128 output columns each).
hipError_t launch_exl3_mul24_grouped(const Exl3GemmArgs& args, bool int8_act, uint32_t n_tiles,
                                     uint32_t grid_x, uint32_t grid_z, hipStream_t stream);
hipError_t launch_gemm_exl3_w16a16_pre(const Exl3GemmArgs& args, const void* xrot,
                                       hipStream_t stream);

}
