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
hipError_t launch_gemm_exl3_w16a16_pre(const Exl3GemmArgs& args, const void* xrot,
                                       hipStream_t stream);

}
